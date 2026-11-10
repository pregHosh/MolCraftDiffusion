"""AutoDock Vina scoring of generated ligands against a protein pocket.

Backs ``MolCraftDiff analyze metrics --metrics sbdd``. Needs the ``[sbdd]``
extra (``vina`` + ``meeko`` + ``gemmi``); every import is deferred so the rest
of the analysis commands keep working without it.

Protocol follows the SBDD literature (targetdiff / KGDiff / IPDiff / Apo2Mol):

* the search box is centred on the ligand being scored, sized to its own
  extent plus a 5 A buffer -- generated ligands already sit in the pocket
  frame, so their own bounding box defines the site;
* that per-ligand box is *wrong* for molecules that did not come out of a
  pocket-conditioned model: a library embedded from SMILES sits at an
  arbitrary origin, so the box would chase each ligand instead of staying
  on the site. Pass ``center``/``box_size`` to pin it, or derive one fixed
  box from a reference crystal ligand with :func:`box_from_ligand`. Omit
  them and the historical per-ligand behaviour is byte-for-byte unchanged;
* three modes, increasing cost: ``score`` (in place), ``min`` (local
  minimisation), ``dock`` (full redock). :func:`dock_poses` is ``dock``
  that hands the poses back instead of only their energies -- which is
  what lets a caller measure RMSD against a known crystal pose.

The pose matters, so molecules are perceived with OpenBabel, **not** xyz2mol:
xyz2mol returns a topology with no conformer, which silently discards the
coordinates the model generated.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

BOX_BUFFER = 5.0
DOCK_MODES = ("score", "min", "dock")


def load_pose(xyz_path):
    """Read one .xyz into a sanitized RDKit mol that still has its pose.

    Returns ``(smiles, mol)``; ``mol`` is ``None`` when the structure cannot be
    perceived, is multi-fragment, or lost its conformer.
    """
    from rdkit import Chem  # noqa: PLC0415

    from MolecularDiffusion.utils.smilify import smilify_openbabel  # noqa: PLC0415

    try:
        smiles, mol = smilify_openbabel(xyz_path)
    except Exception:  # noqa: BLE001 -- an unperceivable structure is data
        return None, None
    if isinstance(mol, (list, tuple)):
        mol = mol[0] if len(mol) == 1 else None
    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if len(smiles) == 1 else None
    if mol is None:
        return smiles, None
    if mol.GetNumConformers() == 0:
        # no 3D coordinates -> nothing to dock
        return smiles, None
    # OpenBabel hands back an unsanitized mol; AddHs and meeko both need
    # implicit valences to have been computed.
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 -- bad valences/kekulisation are data
        return smiles, None
    return smiles, mol


def ligand_pdbqt(mol):
    """Sanitized, H-added mol -> PDBQT string (meeko 0.7 API)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: PLC0415

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise ValueError("meeko produced no molecule setup")
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"meeko could not write PDBQT: {err}")
    return pdbqt


def box_for(mol, buffer=BOX_BUFFER):
    """Search box centred on this molecule: (center, size), both length 3."""
    pos = mol.GetConformer(0).GetPositions()
    center = ((pos.max(0) + pos.min(0)) / 2).tolist()
    size = ((pos.max(0) - pos.min(0)) + buffer).tolist()
    return center, size


def box_from_ligand(
    ref_ligand_sdf: str,
    buffer: float = BOX_BUFFER,
) -> tuple[list[float], list[float]]:
    """One fixed pocket box, derived from a reference crystal ligand.

    Unlike :func:`box_for`, which follows whatever molecule it is handed,
    this is meant to be computed once and reused for every molecule in a
    library, so they all compete over the same site.
    """
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromMolFile(ref_ligand_sdf, sanitize=True)
    if mol is None:
        raise ValueError(
            f"could not read reference ligand {ref_ligand_sdf!r}"
        )
    # AddHs so the box matches what score_pose computes for the same mol
    return box_for(Chem.AddHs(mol, addCoords=True), buffer)


def _resolve_box(
    mol_h: Any,
    buffer: float,
    center: Sequence[float] | None,
    box_size: Sequence[float] | None,
) -> tuple[list[float], list[float]]:
    """The fixed box where given, else this molecule's own extent."""
    box_center, box_extent = box_for(mol_h, buffer)
    if center is not None:
        box_center = list(center)
    if box_size is not None:
        box_extent = list(box_size)
    return box_center, box_extent


def prepare_receptor(
    receptor_path,
    out_dir=None,
    *,
    allow_bad_res: bool = False,
    default_altloc: str | None = None,
):
    """Return a receptor PDBQT path, converting from PDB only if needed.

    A ``.pdbqt`` is used as-is -- CrossDocked-style test sets ship them, and
    reusing one keeps our numbers comparable with the upstream papers.

    ``allow_bad_res`` and ``default_altloc`` forward the matching
    ``mk_prepare_receptor.py`` flags, which real PDB entries routinely
    need: the converter otherwise refuses a file with incomplete residues
    or alternate conformations. Both default to off, so the argv is
    unchanged from what this has always run.
    """
    if receptor_path.endswith(".pdbqt"):
        return receptor_path

    import subprocess  # noqa: PLC0415

    out_dir = out_dir or os.path.dirname(os.path.abspath(receptor_path))
    stem = os.path.splitext(os.path.basename(receptor_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.pdbqt")
    if os.path.exists(out_path):
        return out_path

    argv = [
        "mk_prepare_receptor.py",
        "--read_pdb", receptor_path,
        "-o", os.path.join(out_dir, stem),
        "-p",
    ]
    if allow_bad_res:
        argv.append("--allow_bad_res")
    if default_altloc is not None:
        argv += ["--default_altloc", str(default_altloc)]

    # meeko ships this CLI; it replaces the AutoDockTools/pdb2pqr chain
    result = subprocess.run(  # noqa: S603
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    if not os.path.exists(out_path):
        raise ValueError(
            f"could not prepare receptor {receptor_path!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}. "
            "Pass an already-prepared .pdbqt instead."
        )
    return out_path


def score_pose(
    mol,
    receptor_pdbqt,
    mode="dock",
    exhaustiveness=8,
    seed=42,
    buffer=BOX_BUFFER,
    *,
    center: Sequence[float] | None = None,
    box_size: Sequence[float] | None = None,
):
    """Vina affinities for one posed molecule.

    Returns a dict with ``vina_score`` (always), plus ``vina_min`` for
    ``mode in {"min", "dock"}`` and ``vina_dock`` for ``mode == "dock"``.

    ``center``/``box_size`` pin the search box to a fixed pocket; omit
    both and it is exactly what :func:`box_for` has always returned.
    """
    from rdkit import Chem  # noqa: PLC0415
    from vina import Vina  # noqa: PLC0415

    if mode not in DOCK_MODES:
        raise ValueError(f"mode must be one of {DOCK_MODES}, got {mode!r}")

    mol_h = Chem.AddHs(mol, addCoords=True)
    center, size = _resolve_box(mol_h, buffer, center, box_size)

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt(mol_h))
    v.compute_vina_maps(center=center, box_size=size)

    out = {"vina_score": float(v.score()[0])}
    if mode in ("min", "dock"):
        out["vina_min"] = float(v.optimize()[0])
    if mode == "dock":
        v.dock(exhaustiveness=exhaustiveness, n_poses=5)
        out["vina_dock"] = float(v.energies(n_poses=1)[0][0])
    return out


def score_reference(ref_ligand_sdf, receptor_pdbqt, **kwargs):
    """Score the reference (crystal) ligand -- the bar `high_affinity` uses."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromMolFile(ref_ligand_sdf, sanitize=True)
    if mol is None:
        raise ValueError(f"could not read reference ligand {ref_ligand_sdf!r}")
    return score_pose(mol, receptor_pdbqt, **kwargs)


def dock_poses(  # noqa: PLR0913 -- mirrors score_pose's knobs
    mol: Any,
    receptor_pdbqt: str,
    *,
    center: Sequence[float] | None = None,
    box_size: Sequence[float] | None = None,
    buffer: float = BOX_BUFFER,
    exhaustiveness: int = 8,
    seed: int = 42,
    n_poses: int = 5,
) -> list[tuple[float, Any]]:
    """Dock one molecule and hand back its poses, best-scored first.

    Returns ``[(affinity, pose_mol), ...]`` in Vina's own rank order, each
    ``pose_mol`` carrying exactly one conformer. ``score_pose`` already
    searches five poses and then reports only the best energy; this keeps
    the geometry too, which is what a caller needs to tell "found the
    right pose" from "ranked the right pose first".

    Poses come back through meeko, which returns ONE mol holding one
    conformer per pose (not one mol per pose), in the same order as
    ``Vina.energies`` -- so conformer i pairs with energy row i.
    """
    import tempfile  # noqa: PLC0415

    from meeko import PDBQTMolecule, RDKitMolCreate  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415
    from vina import Vina  # noqa: PLC0415

    mol_h = Chem.AddHs(mol, addCoords=True)
    center, size = _resolve_box(mol_h, buffer, center, box_size)

    v = Vina(sf_name="vina", seed=seed, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt(mol_h))
    v.compute_vina_maps(center=center, box_size=size)
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    energies = v.energies(n_poses=n_poses)
    with tempfile.TemporaryDirectory() as tmp:
        out_pdbqt = os.path.join(tmp, "poses.pdbqt")
        v.write_poses(out_pdbqt, n_poses=n_poses, overwrite=True)
        docked = [
            m
            for m in RDKitMolCreate.from_pdbqt_mol(
                PDBQTMolecule.from_file(out_pdbqt)
            )
            if m is not None
        ]

    if not docked:
        return []
    # one ligand in, so one mol out; its conformers are the poses
    posed = docked[0]
    out = []
    for i, conf in enumerate(posed.GetConformers()):
        if i >= len(energies):
            break
        out.append(
            (float(energies[i][0]), Chem.Mol(posed, confId=conf.GetId()))
        )
    return out
