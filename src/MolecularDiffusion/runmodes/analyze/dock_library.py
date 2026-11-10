"""Dock a library of molecules into one fixed pocket and rank them.

Backs ``MolCraftDiff analyze dock``. Where ``--metrics sbdd`` scores the
``.xyz`` files a generative model produced -- each in its own box, because a
generated ligand already sits in the pocket frame -- this scores an
*arbitrary* library (SDF, or SMILES embedded on the fly) inside a single
shared box, which is what a retrospective benchmark needs.

Two questions it answers:

* **pose recovery** -- with ``--rmsd-ref``, how far the docked pose is from a
  known crystal pose. Reported as ``rmsd_top`` (Vina's own best-scored pose),
  ``rmsd_best`` (closest pose at any rank) and ``rmsd_best_rank``. A program
  that finds the right pose but ranks it fourth passes the second and fails
  the first, and only the first is available on a real screen.
* **enrichment** -- with ``--actives``/``--decoys``, whether known actives
  rank above known inactives: ROC-AUC plus enrichment factor at 1% and 5%.

Labels come from *which file* a molecule was in, so there is no SMILES
matching and no join: that is how DUD-E and LIT-PCBA actually ship.

Needs the ``[sbdd]`` extra; every heavy import is deferred, so importing
this module costs nothing.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence
    from pathlib import Path

#: Line-based SMILES sources: one molecule per line, SMILES in column 0,
#: an optional name in column 1 (the ``load_conditioning_pool`` convention).
SMILES_SUFFIXES = (".smi", ".txt", ".smiles", ".ism")

#: Everything ``load_library`` will open.
LIBRARY_SUFFIXES = (".sdf", ".csv", *SMILES_SUFFIXES)

#: Vina affinities are negative-is-better, so ranking sorts ascending.
_LOWER_IS_BETTER = True

#: The field's conventional "this is the same pose" threshold, in angstrom.
RMSD_SAME_POSE_A = 2.0


def _embed(smiles: str) -> Any:
    """SMILES -> a mol with one 3D conformer, or ``None``.

    Same recipe as ``utils.conformer_pool._pool_from_smiles`` so a library
    embedded here matches one embedded for conditioning.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) == 0:
        return mol_h
    # Bridged ring systems (norbornane, diazabicycloheptane) written with
    # bridgehead stereo that no 3D geometry satisfies fail at every seed. The
    # ring fixes the geometry anyway, so retry without stereo instead of losing
    # the molecule. Stereo at other centres is lost too, which is why the
    # fallback is logged and the output CSV's SMILES column shows the result.
    flat = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
    flat_h = Chem.AddHs(flat)
    if AllChem.EmbedMolecule(flat_h, AllChem.ETKDGv3()) == 0:
        logging.getLogger(__name__).warning(
            "embedded without stereo (original unsatisfiable): %s", smiles)
        return flat_h
    return None


def _smiles_lines(path: Path) -> list[tuple[str, str]]:
    """``(name, smiles)`` per line; name falls back to the SMILES itself."""
    out = []
    with path.open() as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            smiles = parts[0]
            out.append((parts[1] if len(parts) > 1 else smiles, smiles))
    return out


def _csv_lines(path: Path) -> list[tuple[str, str]]:
    """``(name, smiles)`` from a CSV, preferring a ``smiles`` column.

    Mirrors ``compute_metrics._load_train_smiles``: a named column if one
    exists, else column 0.
    """
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_csv(path)
    col = next(
        (c for c in frame.columns if str(c).strip().lower() == "smiles"),
        frame.columns[0],
    )
    names = next(
        (c for c in frame.columns if str(c).strip().lower() in
         ("name", "id", "title", "mol_id")),
        None,
    )
    smiles = frame[col].astype(str).tolist()
    if names is None:
        return [(s, s) for s in smiles]
    return list(zip(frame[names].astype(str).tolist(), smiles, strict=False))


def load_library(path: Path) -> list[tuple[str, Any]]:
    """Read one library file into ``(name, mol)`` pairs.

    ``mol`` is ``None`` where a SMILES could not be parsed or embedded --
    a molecule we failed on is data, and the caller records it with an
    ``error`` rather than dropping it silently.
    """
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        from rdkit import Chem  # noqa: PLC0415

        from MolecularDiffusion.runmodes.analyze.conformer_metrics import (  # noqa: PLC0415
            _read_sdf,
        )

        mols = _read_sdf(path)
        return [
            (
                m.GetProp("_Name")
                if m.HasProp("_Name") and m.GetProp("_Name").strip()
                else f"{path.stem}_{i}",
                Chem.AddHs(m, addCoords=True),
            )
            for i, m in enumerate(mols)
        ]

    if suffix == ".csv":
        rows = _csv_lines(path)
    elif suffix in SMILES_SUFFIXES:
        rows = _smiles_lines(path)
    else:
        msg = (
            f"unsupported library {path.name!r}; expected one of "
            f"{LIBRARY_SUFFIXES}"
        )
        raise ValueError(msg)

    return [(name, _embed(smiles)) for name, smiles in rows]


def ranking_metrics(
    scores: Sequence[float],
    labels: Sequence[int],
    fractions: Sequence[float] = (0.01, 0.05),
) -> dict[str, float]:
    """ROC-AUC and enrichment factor over a scored, labelled library.

    ``scores`` are Vina affinities (lower is better), ``labels`` are 1 for
    an active and 0 for a decoy. Returns ``{}`` when either class is empty
    -- enrichment is undefined then, and a fabricated 0.5 would read as a
    real result.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    pairs = [
        (s, int(y))
        for s, y in zip(scores, labels, strict=False)
        if s is not None and math.isfinite(s) and y is not None
    ]
    n_act = sum(y for _, y in pairs)
    n_dec = len(pairs) - n_act
    if not pairs or n_act == 0 or n_dec == 0:
        return {}

    # negate: roc_auc_score wants higher-is-better
    sign = -1.0 if _LOWER_IS_BETTER else 1.0
    out = {
        "roc_auc": float(
            roc_auc_score([y for _, y in pairs], [sign * s for s, _ in pairs])
        ),
        "n_actives": float(n_act),
        "n_decoys": float(n_dec),
    }

    ranked = sorted(pairs, key=lambda p: sign * -p[0])
    base = n_act / len(pairs)
    for frac in fractions:
        k = max(1, math.ceil(frac * len(ranked)))
        hits = sum(y for _, y in ranked[:k])
        key = f"ef_{frac * 100:g}pct".replace(".", "_")
        out[key] = float((hits / k) / base)
    out["log_auc"] = log_auc(
        [s for s, _ in pairs], [y for _, y in pairs]
    )
    return out


def log_auc(
    scores: Sequence[float],
    labels: Sequence[int],
    lam: float = 0.001,
) -> float:
    """Adjusted LogAUC: early enrichment on a log-scaled x axis.

    FORMULA SOURCE UNVERIFIED. Written from memory of the adjusted LogAUC in
    the docking-benchmark literature and not checked against any paper; check
    the original before citing it. As implemented: sort by score (lower is
    better, ties keep input order), take x as the fraction of the library
    screened and y as the fraction of actives found, integrate y against
    log10(x) from ``lam`` to 1 exactly (y is a step function, so no
    interpolation), divide by ``log10(1 / lam)``, and subtract the same value
    for random ranking (y = x, 0.14462 for ``lam`` = 0.001). A perfect ranking
    gives about 0.855, a random one 0 and the worst possible about -0.145.

    With fewer than 1,000 molecules a warning is logged and ``lam`` becomes
    ``1 / n`` (an unverified convention, a labelled default). Returns NaN,
    with a logged message, when there are no actives.
    """
    pairs = [
        (s, int(y))
        for s, y in zip(scores, labels, strict=False)
        if s is not None and math.isfinite(s) and y is not None
    ]
    n = len(pairs)
    n_act = sum(y for _, y in pairs)
    if n == 0 or n_act == 0:
        logging.getLogger(__name__).warning(
            "log_auc undefined: no scored actives")
        return float("nan")
    if n < 1000:  # noqa: PLR2004
        lam = 1.0 / n
        logging.getLogger(__name__).warning(
            "log_auc: only %d molecules, using lambda = 1/n = %g", n, lam)
    ranked = sorted(pairs, key=lambda p: p[0])
    found = 0
    area = 0.0
    for rank, (_, y) in enumerate(ranked, start=1):
        found += y
        # y-curve is constant at found/n_act on [rank/n, (rank+1)/n)
        lo = max(rank / n, lam)
        hi = (rank + 1) / n if rank < n else 1.0
        if hi > lo:
            area += (found / n_act) * math.log10(hi / lo)
    span = math.log10(1.0 / lam)
    random_area = (1.0 - lam) / math.log(10.0)
    return (area - random_area) / span


def _inplace_rmsd(probe: Any, ref: Any) -> float | None:
    """Heavy-atom, symmetry-corrected RMSD **without** superposition.

    Deliberately not ``tasks_conformer._rmsd``: that uses ``GetBestRMS``,
    which rigid-body aligns the probe onto the reference first. Correct for
    conformer quality, wrong for pose recovery -- the question here is
    whether the pose landed in the right place in the pocket, and alignment
    is exactly the information you must not discard. A flipped pose 9 A from
    the crystal ligand reports ~3 A once it is allowed to move.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import rdMolAlign  # noqa: PLC0415

    if probe is None or ref is None:
        return None
    try:
        return float(
            rdMolAlign.CalcRMS(
                Chem.RemoveHs(Chem.Mol(probe)), Chem.RemoveHs(Chem.Mol(ref))
            )
        )
    except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
        logging.warning("Pose RMSD failed: %s", exc)
        return None


def _pose_rmsd(poses: list[tuple[float, Any]], ref: Any) -> dict[str, Any]:
    """RMSD of the top-scored pose, and of the closest pose at any rank."""
    values = [(rank, _inplace_rmsd(pose, ref)) for rank, (_, pose) in
              enumerate(poses)]
    scored = [(rank, v) for rank, v in values if v is not None]
    if not scored:
        return {"rmsd_top": None, "rmsd_best": None, "rmsd_best_rank": None}
    top = next((v for rank, v in scored if rank == 0), None)
    best_rank, best = min(scored, key=lambda p: p[1])
    return {
        "rmsd_top": top,
        "rmsd_best": best,
        "rmsd_best_rank": best_rank,
    }


def _write_poses(
    writer: Any,
    name: str,
    poses: list[tuple[float, Any]],
    ref: Any,
) -> None:
    """Append every pose to an SDF, tagged with its rank, score and RMSD.

    Rank is written so a viewer can tell Vina's own top pick from the pose
    that happened to be closest -- the distinction the RMSD columns exist
    to expose, which is invisible once poses are in one file.
    """
    for rank, (affinity, pose) in enumerate(poses):
        pose.SetProp("_Name", f"{name} pose{rank}")
        pose.SetProp("rank", str(rank))
        pose.SetProp("vina_affinity", f"{affinity:.3f}")
        if ref is not None:
            value = _inplace_rmsd(pose, ref)
            if value is not None:
                pose.SetProp("rmsd_to_ref", f"{value:.3f}")
        writer.write(pose)


def _resolve_pocket_box(args: argparse.Namespace) -> dict[str, Any]:
    """``{center, box_size}`` kwargs for the docking calls, possibly empty.

    Empty means "no fixed box", i.e. the historical per-ligand behaviour.
    """
    from MolecularDiffusion.runmodes.analyze import docking  # noqa: PLC0415

    ref = getattr(args, "box_ref_ligand", None)
    center = getattr(args, "box_center", None)
    size = getattr(args, "box_size", None)

    if ref:
        box_center, box_size = docking.box_from_ligand(ref)
        logging.info(
            "Pocket box from %s: center %s size %s",
            os.path.basename(ref), box_center, box_size,
        )
        return {"center": box_center, "box_size": box_size}
    if center or size:
        return {"center": center or None, "box_size": size or None}
    logging.warning(
        "No --box-ref-ligand/--box-center given: each molecule is boxed on "
        "its OWN coordinates. Correct only for molecules already posed in "
        "the pocket frame."
    )
    return {}


def _collect(args: argparse.Namespace) -> list[tuple[str, Any, int | None]]:
    """Every molecule to dock, as ``(name, mol, label)``."""
    from pathlib import Path  # noqa: PLC0415

    items: list[tuple[str, Any, int | None]] = []
    for attr, label in (("actives", 1), ("decoys", 0), ("ligands", None)):
        for path in getattr(args, attr, None) or ():
            loaded = load_library(Path(path))
            logging.info("%s: %d molecules from %s", attr, len(loaded), path)
            items += [(name, mol, label) for name, mol in loaded]
    if not items:
        msg = "nothing to dock: pass --ligands, --actives or --decoys"
        raise ValueError(msg)
    return items


def runner(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    """Dock every molecule, write a CSV, and summarise."""
    import pandas as pd  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    from MolecularDiffusion.optional import (  # noqa: PLC0415
        OptionalDependencyError,
        require_modules,
    )

    try:
        require_modules("sbdd", {"vina", "meeko"})
    except OptionalDependencyError as exc:
        logging.warning(str(exc))
        raise SystemExit(1) from exc

    from rdkit import Chem  # noqa: PLC0415

    from MolecularDiffusion.runmodes.analyze import docking  # noqa: PLC0415
    from MolecularDiffusion.runmodes.analyze.compute_metrics import (  # noqa: PLC0415
        _write_summary,
    )
    from MolecularDiffusion.utils.geom_metrics import (  # noqa: PLC0415
        compute_drug_likeness,
    )

    receptor = docking.prepare_receptor(
        args.receptor,
        out_dir=getattr(args, "out_dir", None),
        allow_bad_res=getattr(args, "allow_bad_res", False),
        default_altloc=getattr(args, "default_altloc", None),
    )
    box = _resolve_pocket_box(args)
    items = _collect(args)

    rmsd_ref_path = getattr(args, "rmsd_ref", None)
    rmsd_ref = None
    if rmsd_ref_path:
        rmsd_ref = Chem.MolFromMolFile(rmsd_ref_path, sanitize=True)
        if rmsd_ref is None:
            msg = f"could not read --rmsd-ref {rmsd_ref_path!r}"
            raise ValueError(msg)

    n_poses = int(getattr(args, "n_poses", 1) or 1)
    save_poses = getattr(args, "save_poses", None)
    mode = getattr(args, "mode", "dock")
    want_poses = rmsd_ref is not None or n_poses > 1 or bool(save_poses)
    # A fixed pocket box means the library is NOT already posed in the pocket:
    # molecules embedded from SMILES sit near the origin. score_pose calls
    # v.score() unconditionally, which Vina rejects with "ligand is outside
    # the grid box" before any search happens -- so route a full dock through
    # dock_poses, which places the ligand by searching instead of scoring it
    # where it landed. score/min stay available for pre-posed input.
    if mode == "dock" and (box.get("center") is not None):
        want_poses = True
    exhaustiveness = int(getattr(args, "exhaustiveness", 8) or 8)
    seed = int(getattr(args, "seed", 42) or 42)

    writer = Chem.SDWriter(str(save_poses)) if save_poses else None
    rows = []
    for name, mol, label in tqdm(items, desc="Docking", total=len(items)):
        row: dict[str, Any] = {"name": name, "label": label}
        if mol is None:
            row["error"] = "could not parse or embed"
            rows.append(row)
            continue
        row["smiles"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        try:
            if want_poses:
                poses = docking.dock_poses(
                    mol, receptor, exhaustiveness=exhaustiveness,
                    seed=seed, n_poses=max(n_poses, 1), **box,
                )
                if not poses:
                    row["error"] = "docking produced no pose"
                    rows.append(row)
                    continue
                row["vina_dock"] = poses[0][0]
                if rmsd_ref is not None:
                    row.update(_pose_rmsd(poses, rmsd_ref))
                if writer is not None:
                    _write_poses(writer, name, poses, rmsd_ref)
            else:
                row.update(
                    docking.score_pose(
                        mol, receptor, mode=mode,
                        exhaustiveness=exhaustiveness, seed=seed, **box,
                    )
                )
            row.update({
                k: v
                for k, v in compute_drug_likeness(mol).items()
                if k in ("QED", "SA_score")
            })
        except Exception as exc:  # noqa: BLE001 -- a failed molecule is data
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    if writer is not None:
        writer.close()
        logging.info("Poses saved to %s (open alongside the reference)",
                     save_poses)

    frame = pd.DataFrame(rows)
    out_path = getattr(args, "output", None) or "dock_library.csv"
    frame.to_csv(out_path, index=False)
    logging.info("Docking results saved to %s", out_path)

    score_col = next(
        (c for c in ("vina_dock", "vina_min", "vina_score")
         if c in frame.columns),
        None,
    )
    summary: dict[str, Any] = {
        "metrics": "dock",
        "receptor": receptor,
        "n_molecules": len(frame),
        "n_scored": (
            int(frame[score_col].notna().sum()) if score_col else 0
        ),
        "box": box or "per-ligand (no fixed box)",
        "mode": "dock" if want_poses else mode,
        "exhaustiveness": exhaustiveness,
    }
    if score_col:
        values = frame[score_col].dropna()
        summary["score_column"] = score_col
        if not values.empty:
            summary[f"{score_col}_mean"] = float(values.mean())
            summary[f"{score_col}_median"] = float(values.median())
            summary[f"{score_col}_best"] = float(values.min())

    for col in ("rmsd_top", "rmsd_best"):
        if col in frame.columns and frame[col].notna().any():
            good = frame[col].dropna()
            summary[f"{col}_median"] = float(good.median())
            summary[f"{col}_under_2A"] = float(
                (good < RMSD_SAME_POSE_A).mean()
            )

    if score_col and frame["label"].notna().any():
        labelled = frame[frame["label"].notna() & frame[score_col].notna()]
        summary.update(
            ranking_metrics(
                labelled[score_col].tolist(),
                labelled["label"].astype(int).tolist(),
            )
        )

    for key, value in summary.items():
        logging.info("%s: %s", key, value)
    _write_summary(out_path, summary)
