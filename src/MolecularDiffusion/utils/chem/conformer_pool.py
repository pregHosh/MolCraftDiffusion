"""Conditioning-molecule loading for conformer generation.

Shared by every conformer-generating model. Lifted verbatim out of
``modules/tasks/diffusion_loqi.py``, which is where it was first written and
which still re-exports it for backward compatibility -- a loader used by
three models does not belong in one model's task module, and
``runmodes/generate/tasks_conformer.py`` must be able to reach it without
importing a heavy model task.

``load_conditioning_pool`` is the single documented way to turn a user's
``sample_input`` (a ``.sdf``, a ``.smi``/``.txt``, a graph3d ASE ``.db``, or
an inline list of SMILES) into the platform's ``graph3d`` per-item ``Data``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _graph3d():
    """Import ``graph3d_dataset`` safely, and only when actually needed.

    ``data.component.graph3d_dataset`` and ``data.dataset`` form a
    pre-existing cycle that only bites when graph3d_dataset is the FIRST of
    the two imported -- which is exactly what happens here, since ``runmodes``
    and ``utils`` reach this module before any dataset is built. Importing
    ``data.dataset`` first breaks the tie. Pre-existing platform behaviour,
    not something to fix from here.
    """
    import MolecularDiffusion.data.dataset  # noqa: F401

    from MolecularDiffusion.data.component import graph3d_dataset

    return graph3d_dataset


# ---------------------------------------------------------------------------


def _mol_to_data(mol, atom_vocab: list[str]) -> Data | None:
    """RDKit mol with a conformer -> the platform's ``graph3d`` per-item fields."""
    BOND_ORDER_TO_CLASS = _graph3d().BOND_ORDER_TO_CLASS

    if mol is None or mol.GetNumConformers() == 0:
        return None
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(s not in atom_vocab for s in symbols):
        return None

    bond_i, bond_t = [], []
    for bond in mol.GetBonds():
        cls = BOND_ORDER_TO_CLASS.get(float(bond.GetBondTypeAsDouble()))
        if cls is None:
            return None
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bond_i.append((i, j))
        bond_t.append(cls)

    bond_index = (
        torch.tensor(bond_i, dtype=torch.long).T
        if bond_i
        else torch.zeros((2, 0), dtype=torch.long)
    )
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    return Data(
        pos=pos - pos.mean(dim=0, keepdim=True),
        z=torch.tensor(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
        ),
        atom_idx=torch.tensor(
            [atom_vocab.index(s) for s in symbols], dtype=torch.long
        ),
        fc=torch.tensor(
            [a.GetFormalCharge() for a in mol.GetAtoms()], dtype=torch.long
        ),
        bond_index=bond_index,
        bond_type=torch.tensor(bond_t, dtype=torch.long),
        n_nodes=mol.GetNumAtoms(),
        smiles=None,
    )


def _pool_from_smiles(
    smiles: Sequence[str], atom_vocab: list[str], limit: int | None = None
) -> list[Data]:
    """Embed SMILES into conditioning items via ETKDG.

    SMILES carry no geometry, so a conformer is embedded for each -- not as a
    prediction (the model overwrites the coordinates with noise) but because
    the stereo-edge derivation reads stereochemistry *from 3D*. Stereo
    annotations in the SMILES have to be embedded first or they are silently
    lost.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pool: list[Data] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Unparsable SMILES %r; skipping", smi)
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            logger.warning("ETKDG failed for %s; skipping", smi)
            continue
        Chem.Kekulize(mol, clearAromaticFlags=True)
        item = _mol_to_data(mol, atom_vocab)
        if item is not None:
            pool.append(item)
        if limit and len(pool) >= limit:
            break
    return pool


def load_conditioning_pool(  # noqa: C901, PLR0912
    sample_input: str | Sequence[str],
    atom_vocab: list[str],
    limit: int | None = None,
) -> list[Data]:
    """Load conditioning molecules.

    ``sample_input`` is either a path -- ``.sdf``, ``.smi``/``.txt``, or an
    ASE ``.db`` written by the graph3d converter -- or a plain list of
    SMILES strings given inline in the generate config.

    Both SMILES forms go through :func:`_pool_from_smiles`.
    """
    from rdkit import Chem

    if not isinstance(sample_input, str):
        pool = _pool_from_smiles(list(sample_input), atom_vocab, limit)
        if not pool:
            msg = "sample_input SMILES list yielded no usable molecules"
            raise ValueError(msg)
        return pool

    ext = os.path.splitext(sample_input)[1].lower()
    pool: list[Data] = []

    if ext == ".db":
        from ase.db import connect

        build_rdkit_mol = _graph3d().build_rdkit_mol

        for row in connect(sample_input).select():
            data = dict(row.data)
            atoms = row.toatoms()
            try:
                mol = build_rdkit_mol(
                    atoms.get_atomic_numbers(),
                    np.asarray(data["bond_index"]).reshape(2, -1),
                    data["bond_type"],
                    data.get("formal_charge"),
                    coords=atoms.get_positions(),
                )
            except Exception as exc:  # noqa: BLE001 - chemistry failures are data
                logger.debug("Skipping db row %s: %s", row.id, exc)
                continue
            item = _mol_to_data(mol, atom_vocab)
            if item is not None:
                pool.append(item)
            if limit and len(pool) >= limit:
                break

    elif ext == ".sdf":
        for mol in Chem.SDMolSupplier(sample_input, removeHs=False):
            if mol is None:
                continue
            Chem.Kekulize(mol, clearAromaticFlags=True)
            item = _mol_to_data(mol, atom_vocab)
            if item is not None:
                pool.append(item)
            if limit and len(pool) >= limit:
                break

    elif ext in (".smi", ".txt", ".smiles"):
        with open(sample_input) as handle:
            smiles = [line.split()[0] for line in handle if line.strip()]
        pool = _pool_from_smiles(smiles, atom_vocab, limit)

    else:
        msg = (
            f"sample_input '{sample_input}': expected .sdf, .smi/.txt, or an "
            "ASE .db written by the graph3d converter"
        )
        raise ValueError(msg)

    if not pool:
        msg = f"sample_input '{sample_input}' yielded no usable molecules"
        raise ValueError(msg)
    return pool
