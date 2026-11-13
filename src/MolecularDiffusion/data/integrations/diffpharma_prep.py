"""Build a DiffPharma complex from a raw protein PDB + reference ligand SDF.

Port of the only working novel-pocket path in the upstream repo:
``test_single.py``'s ``process_data`` + ``process_data_h``, i.e.
``process_crossdock.process_ligand_and_pocket`` (full-atom branch) plus
``interaction_construct.{hbond_create,hydrophobic_data}``.

Not ported: the ``ca_only=True`` pocket branch (needs
``Bio.PDB.Polypeptide.three_to_one``, removed in Biopython >= 1.80, and the
released weights are full-atom anyway), and
``lightning_modules.py:921 generate_ligands`` (stale DiffSBDD leftover that
calls ``sample_given_pocket`` with a pre-DiffPharma signature).

"Novel pocket" here means "novel protein + a reference ligand pose": the SDF
is both the 8 A pocket-selection reference and the ligand ODDT detects the
protein-ligand interactions against. That is upstream's own framing.

Heavy dependencies (Biopython, ODDT/OpenBabel, RDKit) are imported inside the
functions and gated by ``optional.require_modules("bio", ...)``, so importing
this module never fails on a bare install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from MolecularDiffusion.optional import require_modules

# ODDT 0.7 still calls ``np.in1d``, which numpy 2.0 removed. It must live on
# the numpy module namespace (oddt hits the alias from several places inside
# ``oddt.toolkits.ob.Molecule._dicts``), and ``np.isin`` is exactly equivalent
# for the 1-D structured arrays oddt passes it -- ``in1d`` differed only by
# flattening non-1-D input, which never happens here. Without this, the
# failure surfaces as a misleading
# ``AttributeError: Molecule has no such property: atom_dict``, because the
# ``atom_dict`` property's getter raises AttributeError internally and Python
# falls back to ``__getattr__``.
# ponytail: shim rather than pinning numpy<2 -- this is a torch/CUDA-pinned
# env; drop it when oddt ships a numpy 2 compatible release.
if not hasattr(np, "in1d"):  # pragma: no cover - numpy>=2.0 only
    np.in1d = np.isin  # type: ignore[attr-defined]

#: ``full_hDSP_hpSSP`` ligand/pocket vocabulary; unknown element -> 'others'.
ATOM_DICT = {
    "C": 0,
    "N": 1,
    "O": 2,
    "S": 3,
    "B": 4,
    "Br": 5,
    "Cl": 6,
    "P": 7,
    "I": 8,
    "F": 9,
    "others": 10,
}
#: hydrophobic particle classes; anything else -> index 5 ('others').
HYDROPHOBIC_TYPES = {"SP": 0, "C.ar": 1, "C.3": 2, "C.2": 3, "C.1": 4}


def process_ligand_and_pocket(
    pdbfile: str, sdffile: str, dist_cutoff: float = 8.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Ligand atoms + every atom of each residue within ``dist_cutoff`` of it."""
    require_modules("bio", ["Bio"])
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from rdkit import Chem

    pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)
    ligand = Chem.SDMolSupplier(str(sdffile))[0]
    if ligand is None:
        raise ValueError(f"cannot read sdf mol ({sdffile})")

    lig_atoms = [
        a.GetSymbol()
        for a in ligand.GetAtoms()
        if (a.GetSymbol().capitalize() in ATOM_DICT or a.GetSymbol() != "H")
    ]
    lig_coords = np.array(
        [
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx in range(ligand.GetNumAtoms())
        ]
    )
    lig_one_hot = np.stack(
        [
            np.eye(1, len(ATOM_DICT), ATOM_DICT[a.capitalize()]).squeeze()
            for a in lig_atoms
        ]
    )

    pocket_residues = [
        residue
        for residue in pdb_struct[0].get_residues()
        if is_aa(residue.get_resname(), standard=True)
        and (
            (
                (
                    np.array([a.get_coord() for a in residue.get_atoms()])[:, None, :]
                    - lig_coords[None, :, :]
                )
                ** 2
            ).sum(-1)
            ** 0.5
        ).min()
        < dist_cutoff
    ]
    if not pocket_residues:
        raise ValueError(
            f"No standard residue within {dist_cutoff} A of the ligand "
            f"({pdbfile} / {sdffile}) -- wrong pair of files?"
        )

    full_atoms = np.concatenate(
        [np.array([a.element for a in res.get_atoms()]) for res in pocket_residues],
        axis=0,
    )
    full_coords = np.concatenate(
        [np.array([a.coord for a in res.get_atoms()]) for res in pocket_residues],
        axis=0,
    )

    pocket_one_hot = []
    encoded = None
    for a in full_atoms:
        if a in ATOM_DICT:
            encoded = np.eye(1, len(ATOM_DICT), ATOM_DICT[a.capitalize()]).squeeze()
        elif a != "H":
            encoded = np.eye(1, len(ATOM_DICT), len(ATOM_DICT) - 1).squeeze()
        elif encoded is None:
            raise ValueError(
                f"{pdbfile} starts the pocket with a hydrogen; upstream's "
                "preprocessing cannot type it (its training PDBs carry no H). "
                "Strip hydrogens from the pocket PDB."
            )
        # NOTE: for a == 'H' the previous atom's encoding is re-appended.
        # That is upstream's behaviour and the training data was built with
        # it; CrossDocked pocket PDBs contain no hydrogens, so it never fires.
        pocket_one_hot.append(encoded)

    return (
        {"lig_coords": lig_coords, "lig_one_hot": lig_one_hot},
        {
            "pocket_coords": full_coords,
            "pocket_one_hot": np.stack(pocket_one_hot),
        },
    )


def hbond_particles(protein, ligand) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3 pseudo-nodes per H-bond, at 1/4, 1/2 and 3/4 along it.

    Classes are ``[SP, DD, AC]`` tiled ``[0, 1, 1]`` for protein-donor bonds
    and ``[0, 2, 2]`` for protein-acceptor bonds.
    """
    require_modules("bio", ["oddt"])
    from oddt.interactions import hbond_acceptor_donor

    li_do = hbond_acceptor_donor(protein, ligand)
    li_ac = hbond_acceptor_donor(ligand, protein)

    do = [i for i in range(len(li_do[0])) if li_do[2][i]]
    ac = [i for i in range(len(li_ac[0])) if li_ac[2][i]]

    int_id = [[li_do[1][i]["id"], 0, li_do[0][i]["id"], 0] for i in do]
    int_id += [[0, li_ac[0][i]["id"], 0, li_ac[1][i]["id"]] for i in ac]

    def _triplet(pair, i):
        a, b = pair[0][i]["coords"], pair[1][i]["coords"]
        return [(3 * a + b) / 4, (a + b) / 2, (a + 3 * b) / 4]

    coords = [c for i in do for c in _triplet(li_do, i)]
    coords += [c for i in ac for c in _triplet(li_ac, i)]

    if not int_id:
        empty = np.zeros((0, 3), dtype=np.float32)
        return np.zeros((0, 4), dtype=np.int64), empty, empty
    classes = np.concatenate(
        [np.tile([0, 1, 1], len(do)), np.tile([0, 2, 2], len(ac))]
    ).astype(int)
    return (
        np.array(int_id, dtype=np.int64),
        np.array(coords, dtype=np.float32),
        np.identity(3)[classes].astype(np.float32),
    )


def hydrophobic_particles(
    protein, ligand
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 pseudo-nodes per hydrophobic contact, at 1/3 and 2/3 along it."""
    require_modules("bio", ["oddt"])
    from oddt.interactions import hydrophobic_contacts

    hydo = hydrophobic_contacts(protein, ligand)
    if len(hydo[0]) == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
        )

    int_id = np.stack((hydo[1]["id"], hydo[0]["id"]), axis=1).astype(np.int64)
    c1 = (2 * hydo[0]["coords"] + hydo[1]["coords"]) / 3
    c2 = (hydo[0]["coords"] + 2 * hydo[1]["coords"]) / 3
    coords = np.array([item for pair in zip(c1, c2) for item in pair])

    atom_type = [x["atomtype"] for x in hydo[1]]
    labels = [x for t in atom_type for x in ("SP", t)]
    classes = [HYDROPHOBIC_TYPES.get(x, 5) for x in labels]
    return (
        int_id,
        coords.astype(np.float32),
        np.identity(6)[classes].astype(np.float32),
    )


def complex_from_files(
    pdb_file: str, sdf_file: str, center: bool = False, dist_cutoff: float = 8.0
) -> Dict[str, Any]:
    """PDB + reference SDF -> the same per-complex dict a converted db row yields."""
    require_modules("bio", ["Bio", "oddt"])
    import oddt

    ligand_data, pocket_data = process_ligand_and_pocket(
        pdb_file, sdf_file, dist_cutoff=dist_cutoff
    )

    protein = next(oddt.toolkit.readfile("pdb", str(pdb_file)))
    lig = next(oddt.toolkit.readfile("sdf", str(sdf_file)))
    # both are mandatory: without them ODDT silently finds zero interactions
    protein.protein = True
    lig.removeh()

    _h_id, h_coords, h_one_hot = hbond_particles(protein, lig)
    _hp_id, hp_coords, hp_one_hot = hydrophobic_particles(protein, lig)

    arrays = {
        "lig_coords": ligand_data["lig_coords"],
        "lig_one_hot": ligand_data["lig_one_hot"],
        "pocket_coords": pocket_data["pocket_coords"],
        "pocket_one_hot": pocket_data["pocket_one_hot"],
        "interh_coords": h_coords,
        "interh_one_hot": h_one_hot,
        "interhp_coords": hp_coords,
        "interhp_one_hot": hp_one_hot,
    }
    if center:
        lig_c, pocket_c = arrays["lig_coords"], arrays["pocket_coords"]
        mean = (lig_c.sum(0) + pocket_c.sum(0)) / (len(lig_c) + len(pocket_c))
        for key in ("lig_coords", "pocket_coords", "interh_coords", "interhp_coords"):
            if len(arrays[key]):
                arrays[key] = arrays[key] - mean

    item: Dict[str, Any] = {"name": Path(pdb_file).stem}
    for key, arr in arrays.items():
        width = 3 if key.endswith("_coords") else arr.shape[-1]
        item[key] = torch.from_numpy(
            np.ascontiguousarray(arr.reshape(-1, width), dtype=np.float32)
        )
    return item
