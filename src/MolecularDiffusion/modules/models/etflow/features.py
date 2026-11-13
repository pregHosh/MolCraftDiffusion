"""ET-Flow's 10-column atom featurization and its chiral-centre tensors.

Ported from ``etflow/commons/utils.py`` (MIT, (c) 2024 Majdi Hassan, Nikhil
Shenoy, Jungyoon Lee). The vocabularies below are OGB's; **their order is the
offset the released weights were trained with**, so nothing here may be
reordered or extended.

Formal charge is column 2: ``safe_index`` into ``[-5..5] + ["misc"]``, i.e.
offset +5 and 12 classes, fed to the network as a raw float index (not a
one-hot) through one shared ``node_mlp``. That is the whole of ET-Flow's
charge handling -- there is no categorical head.

Aromaticity and hybridization are columns 7 and 5. This is how bond ORDER
reaches a network whose edge channel is a bare bonded/not-bonded flag, and it
is why the dataset config must keep ``kekulize: false``.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Number of columns ``atom_to_feature_vector`` emits. Must equal the task's
#: ``node_attr_dim`` (10 in every shipped ET-Flow config).
NODE_ATTR_DIM = 10

_CHIRALITY_SIGN = {
    "CHI_TETRAHEDRAL_CW": -1.0,
    "CHI_TETRAHEDRAL_CCW": 1.0,
}
_CHIRALITY_NAMES = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
    "misc",
]
_DEGREE = [*range(11), "misc"]
_FORMAL_CHARGE = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"]
_NUM_H = [*range(9), "misc"]
_IMPLICIT_VALENCE = [*range(7), "misc"]
_NUM_RADICAL_E = [*range(5), "misc"]
_HYBRIDIZATION = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"]
_TETRAHEDRAL_DEGREE = 4


def _safe_index(vocab: list, value) -> int:
    """Index of ``value`` in ``vocab``, or the trailing ``"misc"`` slot."""
    try:
        return vocab.index(value)
    except ValueError:
        return len(vocab) - 1


def chirality_sign(atom) -> float:
    """+1 / -1 for a tagged tetrahedral centre, 0 otherwise."""
    return _CHIRALITY_SIGN.get(str(atom.GetChiralTag()), 0.0)


def atom_to_feature_vector(atom) -> list[int]:
    """The 10 OGB-style integer columns ET-Flow feeds its ``node_mlp``."""
    return [
        # DEGENERATE UPSTREAM, reproduced deliberately. Upstream maps the tag
        # to +-1.0/0 and then looks that NUMBER up in a list of tag NAMES
        # (commons/utils.py:85-88), which never matches, so `safe_index` falls
        # through to "misc" and this column is the constant 4 for every atom of
        # every molecule. Chirality therefore does not reach the network
        # through node_attr at all -- only through the post-hoc parity switch
        # (and, in the so3 variant, the network's own cross-product term). The
        # released weights were trained with a constant here; "fixing" it would
        # shift the node_mlp's input distribution and invalidate them.
        _safe_index(_CHIRALITY_NAMES, chirality_sign(atom)),
        _safe_index(_DEGREE, atom.GetTotalDegree()),
        _safe_index(_FORMAL_CHARGE, atom.GetFormalCharge()),
        _safe_index(_IMPLICIT_VALENCE, atom.GetImplicitValence()),
        _safe_index(_NUM_H, atom.GetTotalNumHs()),
        _safe_index(_HYBRIDIZATION, str(atom.GetHybridization())),
        _safe_index(_NUM_RADICAL_E, atom.GetNumRadicalElectrons()),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        sum(atom.IsInRingSize(i) for i in range(3, 7)),
    ]


def get_chiral_tensors(mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tetrahedral centres with exactly 4 neighbours, for the parity switch.

    Returns ``(chiral_index (1, C), chiral_nbr_index (1, 4C), chiral_tag (C,))``
    -- upstream's shapes. ``C == 0`` for an achiral molecule, which every
    downstream consumer must treat as a no-op rather than an error.
    """
    centers = [
        atom
        for atom in mol.GetAtoms()
        if chirality_sign(atom) != 0
        and len(atom.GetNeighbors()) == _TETRAHEDRAL_DEGREE
    ]
    chiral_index = np.array(
        [atom.GetIdx() for atom in centers], dtype=np.int64
    ).reshape(1, -1)
    chiral_nbr_index = np.array(
        [n.GetIdx() for atom in centers for n in atom.GetNeighbors()],
        dtype=np.int64,
    ).reshape(1, -1)
    chiral_tag = np.array(
        [chirality_sign(atom) for atom in centers], dtype=np.float32
    )
    return chiral_index, chiral_nbr_index, chiral_tag


class ETFlowFeatureCache:
    """Per-item featurization, cached on the item's exact bytes.

    Both outputs depend on the COORDINATES as well as the graph: the platform
    stores no chiral tags, so :func:`build_rdkit_mol` recovers them with
    ``AssignStereochemistryFrom3D`` from the input conformer. Upstream instead
    reads them off the GEOM mol. That is the first thing to check if a
    converted pretrained checkpoint underperforms -- and it is why ``pos`` is
    part of the cache key.

    Conformer generation tiles ONE item into a batch, so the key is identical
    across the batch and the cache is a straight hit after the first row.
    """

    def __init__(self, maxsize: int = 100_000) -> None:
        self._compute = functools.lru_cache(maxsize=maxsize)(
            self._compute_uncached
        )

    def _compute_uncached(self, key: tuple):
        n_nodes, z_b, bi_b, bt_b, fc_b, pos_b = key
        z = np.frombuffer(z_b, dtype=np.int64)
        bond_index = np.frombuffer(bi_b, dtype=np.int64).reshape(2, -1)
        bond_type = np.frombuffer(bt_b, dtype=np.int64)
        fc = np.frombuffer(fc_b, dtype=np.int64)
        pos = np.frombuffer(pos_b, dtype=np.float32).reshape(n_nodes, 3)

        # Pre-import: graph3d_dataset has a latent import cycle with
        # data.dataset that only bites when graph3d_dataset is imported first.
        # Pre-existing platform behaviour, not something to fix from here.
        import MolecularDiffusion.data.dataset  # noqa: F401
        from rdkit import Chem

        from MolecularDiffusion.data.datasets.graph3d_dataset import (
            build_rdkit_mol,
        )

        try:
            mol = build_rdkit_mol(
                z, bond_index, bond_type, formal_charge=fc, coords=pos
            )
        except Exception as exc:  # noqa: BLE001 - chemistry failures are data
            logger.debug("Sanitization failed, using a raw mol: %s", exc)
            mol = build_rdkit_mol(
                z,
                bond_index,
                bond_type,
                formal_charge=fc,
                coords=pos,
                sanitize=False,
            )
            # Without these, IsInRing/IsInRingSize raise on the fallback path.
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            Chem.AssignStereochemistryFrom3D(mol)

        node_attr = np.array(
            [atom_to_feature_vector(a) for a in mol.GetAtoms()],
            dtype=np.float32,
        ).reshape(-1, NODE_ATTR_DIM)
        return (node_attr, *get_chiral_tensors(mol))

    def get(self, item) -> tuple:
        """``item`` is one ``graph3d`` PyG ``Data``."""
        z = item.z.detach().cpu().numpy().astype(np.int64)
        bond_index = (
            item.bond_index.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
            .reshape(2, -1)
        )
        bond_type = (
            item.bond_type.detach().cpu().numpy().astype(np.int64).reshape(-1)
        )
        fc = item.fc.detach().cpu().numpy().astype(np.int64)
        pos = item.pos.detach().cpu().numpy().astype(np.float32)
        key = (
            int(z.shape[0]),
            z.tobytes(),
            bond_index.tobytes(),
            bond_type.tobytes(),
            fc.tobytes(),
            pos.tobytes(),
        )
        return self._compute(key)


def graph_key(item) -> bytes:
    """Cache key for the harmonic prior: the bond graph, coordinates excluded.

    The Laplacian eigendecomposition depends on the bond graph and the atom
    ORDER, and on nothing else -- so this, not the SMILES upstream uses, is the
    key that cannot collide across two different molecules or two different
    orderings of the same one.
    """
    bi = item.bond_index.detach().cpu().numpy().astype(np.int64).reshape(2, -1)
    return f"{int(item.pos.shape[0])}:".encode() + bi.tobytes()
