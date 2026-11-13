"""Paired generated-vs-reference conformer metrics.

This is the home for every metric that needs a *pair* of structures -- a
generated one and the reference it is supposed to correspond to -- rather than
a bare directory of samples. It replaces the old ``analyze compare`` command
and the dead ``--check-strain`` flag, and adds the one genuinely missing
metric: stereochemistry preservation.

Two input layouts are accepted, detected from the directory:

1. **SDF pairs** (primary), as written by
   ``runmodes/generate/tasks_conformer.py:ConformerFactory``::

       <input>/conformers.csv
       <input>/mol_0000/{conformers.sdf,reference.sdf,conformer_000.xyz,...}

   Both molecules of a pair come from the same graph3d item via
   ``build_rdkit_mol``, so atom and bond ordering match by construction --
   which is what makes stereo descriptors comparable.

2. **xyz + optimized** (legacy), what ``analyze compare`` consumed::

       <input>/*.xyz
       <input>/optimized_xyz/<stem>_opt.xyz

   Here the two molecules are perceived independently from coordinates, so
   the orderings are not guaranteed to agree and **stereo columns are not
   emitted**.

Layout 1 is never routed through xyz2mol/openbabel: re-perceiving bonds from
coordinates destroys exactly the stereochemistry this group measures.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.5094740631


# ==========================================================================
# Stereochemistry preservation
#
# Ported from LoQI's ``megalodon/metrics/preserved_stereo.py``. The scoring
# semantics are kept deliberately: a molecule is scored all-or-nothing, and
# the *full enantiomer* of the reference counts as correct, so the numbers
# stay comparable with the paper. Two upstream defects are fixed:
#   * an empty denominator returned 0.0, making "no stereocentres in the set"
#     indistinguishable from "every one wrong" -> we return ``None``;
#   * a bare ``assert`` on the list lengths -> a real ``ValueError``.
# ==========================================================================


def prepare_mol_for_conformer_eval(
    mol: Any, *, assign_from_3d: bool = True
) -> Any:
    """Sanitized, Kekule-form copy with CIP tags assigned. ``None`` if bad."""
    from rdkit import Chem  # noqa: PLC0415

    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        if assign_from_3d and mol.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        else:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:  # noqa: BLE001 -- chemistry, not a bug
        return None
    return mol


def get_stereochemistry_descriptor(mol: Any) -> tuple[str, str, str]:
    """``(rs, inverted_rs, ez)`` descriptor strings for one molecule."""
    from rdkit import Chem  # noqa: PLC0415

    rs = [
        atom.GetProp("_CIPCode")
        for atom in mol.GetAtoms()
        if atom.HasProp("_CIPCode")
    ]
    inv_rs = "".join("R" if c == "S" else "S" for c in rs)
    ez = [
        "E" if bond.GetStereo() == Chem.BondStereo.STEREOE else "Z"
        for bond in mol.GetBonds()
        if bond.GetStereo()
        in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
    ]
    return "".join(rs), inv_rs, "".join(ez)


def compare_stereo(mol: Any, ref_mol: Any) -> dict | None:
    """Score one generated molecule against its reference.

    Returns ``None`` when either molecule cannot be prepared. ``rs_ok`` /
    ``ez_ok`` are ``None`` when the *reference* has nothing of that kind to
    preserve -- that is "not applicable", not "wrong".
    """
    mol = prepare_mol_for_conformer_eval(mol)
    ref_mol = prepare_mol_for_conformer_eval(ref_mol)
    if mol is None or ref_mol is None:
        return None

    rs, _, ez = get_stereochemistry_descriptor(mol)
    ref_rs, inv_ref_rs, ref_ez = get_stereochemistry_descriptor(ref_mol)

    return {
        # the full enantiomer counts as correct -- upstream semantics
        "rs_ok": (rs in (ref_rs, inv_ref_rs)) if ref_rs else None,
        "ez_ok": (ez == ref_ez) if ref_ez else None,
        "n_stereocentres": len(ref_rs),
        "n_stereobonds": len(ref_ez),
    }


def stereo_scores(molecules: list, reference_molecules: list) -> dict:
    """RS / EZ preservation scores over paired lists.

    Scores are ``None`` when nothing in the reference set carried that kind of
    stereochemistry, so an empty denominator can never masquerade as 0.0.
    """
    if len(molecules) != len(reference_molecules):
        msg = (
            f"Molecule lists must have the same length, got "
            f"{len(molecules)} and {len(reference_molecules)}."
        )
        raise ValueError(msg)

    scored = [
        r
        for r in (
            compare_stereo(m, ref)
            for m, ref in zip(molecules, reference_molecules, strict=True)
        )
        if r is not None
    ]
    rs = [r["rs_ok"] for r in scored if r["rs_ok"] is not None]
    ez = [r["ez_ok"] for r in scored if r["ez_ok"] is not None]
    return {
        "rs_score": float(np.mean(rs)) if rs else None,
        "ez_score": float(np.mean(ez)) if ez else None,
        "n_rs_scored": len(rs),
        "n_ez_scored": len(ez),
    }


# ==========================================================================
# Paired geometry / strain
# ==========================================================================


def _weighted_mean(stats: dict) -> float | None:
    """Collapse ``{key: (avg, std, weight)}`` to one weighted mean."""
    total = sum(w for _, _, w in stats.values())
    if total <= 0:
        return None
    return float(sum(avg * w for avg, _, w in stats.values()) / total)


def _geometry_diffs(gen: Any, ref: Any) -> dict:
    """Mean bond-length / angle / torsion deviation for one pair."""
    from MolecularDiffusion.utils.metrics.geom_stability import (  # noqa: PLC0415
        compute_bond_angles_diff,
        compute_bond_lengths_diff,
        compute_differences,
        compute_torsion_angles_diff,
    )

    pair = (gen, ref)
    out = {}
    for name, fn in (
        ("bond_length_mean", compute_bond_lengths_diff),
        ("bond_angle_mean", compute_bond_angles_diff),
        ("torsion_angle_mean", compute_torsion_angles_diff),
    ):
        try:
            out[name] = _weighted_mean(compute_differences([pair], fn))
        except Exception as exc:  # noqa: BLE001, PERF203 -- one bad pair
            logger.debug("%s failed: %s", name, exc)
            out[name] = None
    return out


def _mmff_strain(mol: Any) -> float | None:
    from MolecularDiffusion.utils.metrics.geom_stability import (  # noqa: PLC0415
        compute_mmff_energy_drop,
    )

    if mol is None:
        return None
    return compute_mmff_energy_drop(mol)


def xtb_available() -> bool:
    """Explicit probe -- ``get_xtb_energy`` swallows a missing binary."""
    return shutil.which("xtb") is not None


def _xtb_strain(
    xyz_path: str, charge: int, level: str, timeout: int
) -> float | None:
    """E(pose) - E(xTB-relaxed pose), kcal/mol. ``None`` if anything fails."""
    from MolecularDiffusion.runmodes.analyze.compare_to_optimized import (  # noqa: PLC0415
        get_xtb_energy,
    )
    from MolecularDiffusion.runmodes.analyze.xtb_optimization import (  # noqa: PLC0415
        optimize_molecule,
    )

    if not Path(xyz_path).exists():
        return None
    # ponytail: optimize_molecule writes its output into the CWD, so run it in
    # a scratch dir. chdir is process-global -- fine for this serial loop, swap
    # for a cwd= subprocess call if this is ever parallelised.
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        local = shutil.copy(xyz_path, tmp)
        e_pose = get_xtb_energy(local, charge, level, timeout)
        opt = optimize_molecule(local, charge, level, timeout)
        e_opt = get_xtb_energy(opt, charge, level, timeout) if opt else None
    if e_pose is None or e_opt is None:
        return None
    return float((e_pose - e_opt) * HARTREE_TO_KCAL)


# ==========================================================================
# Layout detection and readers
# ==========================================================================


def detect_layout(input_path: str | Path) -> str:
    """``"sdf_pairs"`` or ``"xyz_optimized"``; raises for anything else."""
    path = Path(input_path)
    if path.is_dir():
        if (path / "conformers.csv").is_file() and any(
            path.glob("mol_*/reference.sdf")
        ):
            return "sdf_pairs"
        if any((path / "optimized_xyz").glob("*_opt.xyz")):
            return "xyz_optimized"
    msg = (
        f"'{input_path}' is not a conformer-metrics input. --metrics "
        "conformer needs paired structures, in one of two layouts:\n"
        "  1. conformers.csv + mol_XXXX/{conformers.sdf,reference.sdf} "
        "(what 'MolCraftDiff generate' writes in conformer mode); or\n"
        "  2. *.xyz + optimized_xyz/<stem>_opt.xyz "
        "(what 'MolCraftDiff analyze optimize' writes).\n"
        "A plain directory of generated .xyz files has no reference to pair "
        "against -- use --metrics core for that."
    )
    raise ValueError(msg)


def _read_sdf(path: Path) -> list:
    from rdkit import Chem  # noqa: PLC0415

    if not path.is_file():
        return []
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    return [m for m in supplier if m is not None]


def _align_to_csv(mol_index: int, records: list, rows: pd.DataFrame) -> list:
    """Attach a ``conformer_index`` to each SDF record.

    ``conformers.sdf`` skips records whose rebuild failed, so it is *not*
    index-aligned with ``conformer_index``. Never zip blindly.
    """
    idx = list(rows["conformer_index"])
    if len(records) == len(idx):
        return list(zip(records, idx, strict=True))
    built = list(rows.loc[rows["rmsd"].notna(), "conformer_index"])
    if len(records) == len(built):
        logger.info(
            "mol_%04d: %d/%d conformers rebuilt; aligned on the rows that "
            "carry an rmsd.",
            mol_index,
            len(records),
            len(idx),
        )
        return list(zip(records, built, strict=True))
    logger.warning(
        "mol_%04d: conformers.sdf has %d records but conformers.csv has %d "
        "rows (%d with rmsd); the mapping is ambiguous, so conformer_index is "
        "left empty for this molecule.",
        mol_index,
        len(records),
        len(idx),
        len(built),
    )
    return [(rec, None) for rec in records]


def _rows_sdf_pairs(
    root: Path, charge: int, level: str, timeout: int, *, want_xtb: bool
) -> tuple[list[dict], pd.DataFrame]:
    from tqdm import tqdm  # noqa: PLC0415

    csv = pd.read_csv(root / "conformers.csv")
    rows: list[dict] = []
    mol_dirs = sorted(p for p in root.glob("mol_*") if p.is_dir())
    for mol_dir in tqdm(mol_dirs, desc="Conformer metrics"):
        mol_index = int(mol_dir.name.split("_")[-1])
        ref_list = _read_sdf(mol_dir / "reference.sdf")
        if not ref_list:
            logger.warning("%s: no readable reference.sdf; skipped.", mol_dir)
            continue
        ref = ref_list[0]
        records = _read_sdf(mol_dir / "conformers.sdf")
        sub = csv[csv["mol_index"] == mol_index]
        by_index = sub.set_index("conformer_index")

        for gen, conf_index in _align_to_csv(mol_index, records, sub):
            row: dict[str, Any] = {
                "mol_index": mol_index,
                "conformer_index": conf_index,
                "rmsd": None,
                "energy_hartree": None,
            }
            if conf_index is not None and conf_index in by_index.index:
                csv_row = by_index.loc[conf_index]
                row["rmsd"] = csv_row.get("rmsd")
                row["energy_hartree"] = csv_row.get("energy_hartree")
                xyz = csv_row.get("xyz")
            else:
                xyz = None
            row.update(_geometry_diffs(gen, ref))
            row["mmff_strain_kcal"] = _mmff_strain(gen)
            row["xtb_strain_kcal"] = (
                _xtb_strain(str(root / xyz), charge, level, timeout)
                if want_xtb and isinstance(xyz, str)
                else None
            )
            stereo = compare_stereo(gen, ref)
            row.update(
                stereo
                if stereo is not None
                else {
                    "rs_ok": None,
                    "ez_ok": None,
                    "n_stereocentres": None,
                    "n_stereobonds": None,
                }
            )
            rows.append(row)
    return rows, csv


def _rows_xyz_optimized(
    root: Path, charge: int, level: str, timeout: int
) -> list[dict]:
    from tqdm import tqdm  # noqa: PLC0415

    from MolecularDiffusion.runmodes.analyze.compare_to_optimized import (  # noqa: PLC0415
        compute_all_metrics,
        xyz2mol_openbabel,
    )

    logger.warning(
        "xyz+optimized layout: bonds are re-perceived from coordinates "
        "independently for each structure, so atom ordering is not guaranteed "
        "to match and stereo columns are not emitted. Use the SDF-pair layout "
        "for stereo preservation."
    )
    # Bonds are perceived with OpenBabel, never xyz2mol -- same reason the
    # sbdd group does: xyz2mol returns a topology with no conformer, which
    # would silently discard the coordinates being measured. Not switchable.
    args = SimpleNamespace(
        charge=charge, level=level, timeout=timeout, mol_converter="openbabel"
    )
    pairs = [
        (f, root / "optimized_xyz" / f"{f.stem}_opt.xyz")
        for f in sorted(root.glob("*.xyz"))
        if not f.stem.endswith("_opt")
    ]
    pairs = [(a, b) for a, b in pairs if b.exists()]

    rows = []
    for init_f, opt_f in tqdm(pairs, desc="Conformer metrics"):
        res = compute_all_metrics(init_f, opt_f, args)
        if "error" in res:
            logger.debug("%s: %s", init_f.name, res["error"])
            continue
        # ponytail: one extra perception pass just for the MMFF strain --
        # compute_all_metrics does not hand its mols back.
        mol = xyz2mol_openbabel(str(init_f))
        rows.append(
            {
                "file": init_f.name,
                "rmsd": res.get("rmsd"),
                "energy_hartree": res.get("e_init_Ha"),
                "energy_diff_kcal": res.get("energy_diff_kcal"),
                "bond_length_mean": res.get("bond_length_mean"),
                "bond_angle_mean": res.get("bond_angle_mean"),
                "torsion_angle_mean": res.get("torsion_angle_mean"),
                "mmff_strain_kcal": _mmff_strain(mol),
                "xtb_strain_kcal": None,
            }
        )
    return rows


# ==========================================================================
# Entry point
# ==========================================================================


def compute_conformer_metrics(
    input_path: str | Path,
    rmsd_threshold: float = 0.5,
    charge: int = 0,
    level: str = "gfn2",
    timeout: int = 120,
) -> tuple[pd.DataFrame, dict]:
    """Per-conformer metrics table plus the summary payload."""
    root = Path(input_path)
    layout = detect_layout(root)
    want_xtb = xtb_available()
    if not want_xtb:
        logger.warning(
            "The 'xtb' binary is not on PATH; xtb_strain_kcal is left empty "
            "rather than filled with a made-up number. Install xtb from "
            "conda-forge to populate it."
        )

    if layout == "sdf_pairs":
        rows, csv = _rows_sdf_pairs(
            root, charge, level, timeout, want_xtb=want_xtb
        )
        n_input = len(csv)
    else:
        rows = _rows_xyz_optimized(root, charge, level, timeout)
        n_input = len(rows)

    df = pd.DataFrame(rows)
    if df.empty:
        msg = f"No conformer pairs could be scored in '{input_path}'."
        raise ValueError(msg)

    summary: dict[str, Any] = {
        "metrics": "conformer",
        "input": str(root),
        "layout": layout,
        "n_pairs": len(df),
        "n_input_rows": n_input,
        "rmsd_threshold": rmsd_threshold,
        "xtb": want_xtb,
    }

    rmsd = df["rmsd"].dropna() if "rmsd" in df else pd.Series(dtype=float)
    if not rmsd.empty:
        summary["rmsd_median"] = float(rmsd.median())
        summary["coverage_at_threshold"] = float(
            (rmsd <= rmsd_threshold).mean()
        )
        if "mol_index" in df:
            best = (
                df.dropna(subset=["rmsd"]).groupby("mol_index")["rmsd"].min()
            )
            summary["rmsd_best_per_mol_mean"] = float(best.mean())

    for col in (
        "bond_length_mean",
        "bond_angle_mean",
        "torsion_angle_mean",
        "mmff_strain_kcal",
        "xtb_strain_kcal",
        "energy_diff_kcal",
    ):
        if col in df:
            values = df[col].dropna()
            # strain distributions are heavy-tailed (one clashing pose can move
            # the mean by orders of magnitude), so report both
            summary[f"{col}_mean"] = (
                float(values.mean()) if not values.empty else None
            )
            summary[f"{col}_median"] = (
                float(values.median()) if not values.empty else None
            )

    if "rs_ok" in df:
        for kind in ("rs", "ez"):
            ok = df[f"{kind}_ok"].dropna()
            summary[f"{kind}_score"] = (
                float(ok.mean()) if not ok.empty else None
            )
            summary[f"n_{kind}_scored"] = len(ok)
        # a pair is "skipped" only when neither descriptor could be read at
        # all -- having no stereocentres is not a skip
        summary["n_skipped"] = int(df["n_stereocentres"].isna().sum())
    else:
        summary["stereo"] = (
            "not computed: the xyz+optimized layout re-perceives bonds "
            "independently, so atom ordering is not comparable"
        )

    return df, summary
