"""Build a DiffInt complex from a raw protein PDB + reference ligand SDF.

DiffInt is DiffSBDD with a **CA pocket** whose node set is extended by
H-bond pseudo-atoms: two per detected protein-ligand hydrogen bond, placed
at 1/3 and 2/3 along the donor-acceptor vector, carrying two extra one-hot
channels (``DD`` = 20, ``AC`` = 21) on top of the 20 amino-acid classes.
They are *conditioning* -- appended to the pocket, never diffused.

This module is the single implementation of that construction. Both the
offline converter (``docs/model_integrations/diffint/scripts/
convert_dataset.py``) and the on-the-fly generation path
(``DiffIntPocketGenerator`` with ``pocket_pdb`` + ``ref_sdf``) call
:func:`complex_from_files`, so the two can never drift apart.

## Ported from ``test_single.py``, NOT from ``lightning_modules.py``

Upstream ships two novel-PDB paths and only one of them is correct:

* ``lightning_modules.py:852-890`` (``generate_ligands`` / ``prepare_pocket``)
  builds the 22-wide one-hot by zero-padding the 20-class CA encoding and
  then **never adds the particles**. Every ``DD``/``AC`` column stays zero, so
  the model silently degrades to plain DiffSBDD -- the paper's entire
  contribution is switched off, with no error.
* ``test_single.py:141-192`` (``process_data`` + ``process_data_h``) rebuilds
  the augmented pocket properly via ``hbond_double.hbond_create``. That is
  what is ported here.

## Two upstream data bugs are fixed rather than reproduced

* ``hbond_double.py:41-43`` appends a dummy empty row to ``int_id`` when a
  complex has zero H-bonds, which produces a ragged array. ``int_id`` is only
  consumed by the auxiliary interaction loss, which is out of scope (it is
  commented out in the release and the shipped checkpoint was trained without
  it), so it is simply not built here.
* ``process_crossdock.process_ligand_and_pocket`` derives ``lig_coords`` from
  every atom but ``lig_one_hot`` from a filtered atom list, and its H filter
  (``a.element != 'H'``) hits an attribute RDKit's ``Atom`` does not have.
  Both only matter for SDFs carrying explicit hydrogens; here hydrogens are
  stripped from the molecule up front, which is what upstream's
  ``--no_H`` preprocessing achieved anyway.

Heavy dependencies (Biopython, ODDT/OpenBabel, RDKit) are imported inside the
functions and gated by ``optional.require_modules``, so importing this module
never fails on a bare install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from MolecularDiffusion.optional import require_modules

# ODDT 0.7 still calls ``np.in1d``, which numpy 2.0 removed; it must live on
# the numpy module namespace because oddt reaches for it from inside
# ``oddt.toolkits.ob.Molecule._dicts``. ``np.isin`` is exactly equivalent for
# the 1-D structured arrays oddt passes it. Without the shim the failure
# surfaces as a misleading "Molecule has no such property: atom_dict".
# Identical to the shim in ``diffpharma_prep.py``; kept local so neither
# module has to import the other for a side effect.
# ponytail: shim rather than pinning numpy<2 -- this is a torch/CUDA-pinned
# env; drop it when oddt ships a numpy 2 compatible release.
if not hasattr(np, "in1d"):  # pragma: no cover - numpy>=2.0 only
    np.in1d = np.isin  # type: ignore[attr-defined]

#: ``dataset_params['crossdock_h']['aa_decoder']`` (``constants.py:184``).
#: The first 20 entries are the standard amino acids in the same order as the
#: plain ``crossdock`` table; 20/21 are the H-bond particle classes.
DIFFINT_POCKET_VOCAB: List[str] = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "DD", "AC",
]

NUM_POCKET_CLASSES = len(DIFFINT_POCKET_VOCAB)

#: Class index of the first / second H-bond particle block. The ORDER is part
#: of the checkpoint's meaning, not a free choice: ``hbond_double.py:56-58``
#: labels the ``hbond_acceptor_donor(protein, ligand)`` block 0 and the
#: ``hbond_acceptor_donor(ligand, protein)`` block 1, and those become
#: columns 20 and 21 of the pocket one-hot.
CLASS_DD = 20
CLASS_AC = 21

_AA_TO_CLASS = {aa: i for i, aa in enumerate(DIFFINT_POCKET_VOCAB[:20])}


def _three_to_one(resname: str) -> str:
    """``'ALA' -> 'A'``, across Biopython versions.

    ``Bio.PDB.Polypeptide.three_to_one`` (what ``process_crossdock.py`` uses)
    was removed in Biopython 1.80 in favour of the
    ``protein_letters_3to1`` mapping.
    """
    from Bio.PDB.Polypeptide import protein_letters_3to1  # noqa: PLC0415

    return protein_letters_3to1[resname.capitalize().upper()]


def ca_pocket(pdbfile: str, lig_coords: np.ndarray, dist_cutoff: float = 8.0):
    """One node per CA of every standard residue within ``dist_cutoff``.

    Port of ``process_crossdock.process_ligand_and_pocket``'s ``ca_only=True``
    branch (``process_crossdock.py:65-80``). Returns
    ``(coords (N, 3) float32, classes (N,) int64 in 0..19)``.
    """
    require_modules("bio", ["Bio"])
    from Bio.PDB import PDBParser  # noqa: PLC0415
    from Bio.PDB.Polypeptide import is_aa  # noqa: PLC0415

    struct = PDBParser(QUIET=True).get_structure("", pdbfile)
    coords: List[np.ndarray] = []
    classes: List[int] = []
    for residue in struct[0].get_residues():
        if not is_aa(residue.get_resname(), standard=True):
            continue
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        dist = (
            ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1)
            ** 0.5
        ).min()
        if dist >= dist_cutoff:
            continue
        for atom in residue.get_atoms():
            if atom.name == "CA":
                coords.append(atom.coord)
                classes.append(_AA_TO_CLASS[_three_to_one(residue.get_resname())])

    if not coords:
        raise ValueError(
            f"no standard residue with a CA within {dist_cutoff} A of the "
            f"ligand ({pdbfile}) -- wrong pair of files?"
        )
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(classes, dtype=np.int64),
    )


def hbond_particles(protein, ligand) -> Tuple[np.ndarray, np.ndarray]:
    """2 pseudo-nodes per H-bond, at 1/3 and 2/3 along it.

    Port of ``hbond_double.py:8-43``. Returns
    ``(coords (2K, 3) float32, classes (2K,) int64 in {20, 21})``, with the
    two particles of a bond adjacent, the ``hbond_acceptor_donor(protein,
    ligand)`` block first.

    ``data/component/diffpharma_prep.py:hbond_particles`` runs the *same*
    ODDT detection but emits DiffPharma's geometry (3 particles at 1/4, 1/2,
    3/4 with a 3-class vocabulary), which is not interchangeable with
    DiffInt's -- different node count, different one-hot width, different
    checkpoint. What is genuinely shared (the ``np.in1d`` shim, the
    ``protein.protein = True`` / ``ligand.removeh()`` preconditions) is
    reproduced above rather than imported for a side effect.
    """
    require_modules("bio", ["oddt"])
    from oddt.interactions import hbond_acceptor_donor  # noqa: PLC0415

    coords: List[np.ndarray] = []
    classes: List[int] = []
    # (protein, ligand) -> protein acceptors / ligand donors -> class DD
    # (ligand, protein) -> ligand acceptors / protein donors -> class AC
    for pair, cls in (
        (hbond_acceptor_donor(protein, ligand), CLASS_DD),
        (hbond_acceptor_donor(ligand, protein), CLASS_AC),
    ):
        for i in range(len(pair[0])):
            if not pair[2][i]:  # `strict` flag: only unambiguous H-bonds
                continue
            a, b = pair[0][i]["coords"], pair[1][i]["coords"]
            coords += [(2 * a + b) / 3, (a + 2 * b) / 3]
            classes += [cls, cls]

    if not coords:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(classes, dtype=np.int64),
    )


def complex_from_files(
    pdb_file: str, sdf_file: str, dist_cutoff: float = 8.0
) -> Dict[str, Any]:
    """PDB + reference SDF -> the raw arrays one converted db row holds.

    "Novel pocket" means "novel protein + a reference ligand pose": the SDF is
    both the 8 A pocket-selection reference and the ligand ODDT detects the
    H-bonds against. That is upstream's own framing (``test_single.py``) and
    is unavoidable -- the interactions are protein<->ligand.

    Pocket nodes come back **residues first, then particles**, which is the
    order ``num_pocket_nodes`` (residues only) assumes.
    """
    require_modules("bio", ["Bio", "oddt"])
    import oddt  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.SDMolSupplier(str(sdf_file), removeHs=True)[0]
    if mol is None:
        raise ValueError(f"cannot read sdf mol ({sdf_file})")
    conf = mol.GetConformer(0)
    lig_coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    lig_z = np.array(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64
    )

    res_coords, res_class = ca_pocket(pdb_file, lig_coords, dist_cutoff)

    protein = next(oddt.toolkit.readfile("pdb", str(pdb_file)))
    lig = next(oddt.toolkit.readfile("sdf", str(sdf_file)))
    # both mandatory: without them ODDT silently finds zero interactions
    protein.protein = True
    lig.removeh()
    part_coords, part_class = hbond_particles(protein, lig)

    return {
        "name": Path(pdb_file).stem,
        "lig_coords": lig_coords,
        "lig_z": lig_z,
        "pocket_coords": np.concatenate([res_coords, part_coords], axis=0),
        "pocket_class": np.concatenate([res_class, part_class], axis=0),
        "n_residues": len(res_coords),
        "n_particles": len(part_coords),
    }
