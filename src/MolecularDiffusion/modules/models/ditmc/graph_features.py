"""Per-molecule featurization: ``node_attr``, shortest hops, Laplacian eigen.

Port of the parts of ``dit_mc/prepare_dataset.py`` that produce what the model
consumes. Upstream computes these once at dataset-build time and caches by
SMILES; so does this module, with an in-process LRU rather than a pickle.

The three quantities:

``node_attr``
    The 64-d (QM9) / 94-d (Drugs) RDKit atom featurization, reproduced column
    for column from ``get_node_attr_from_mol`` -- see :data:`NODE_ATTR_BLOCKS`
    for the exact layout. ``one_hot_encoding`` always appends a trailing "misc"
    slot **except** for the atom-symbol block, which upstream slices with
    ``[:-1]``.

``shortest_hops``
    Floyd-Warshall over the bond adjacency, off-diagonal entries in C-order
    over ``(i, j)`` -- the same order the all-pairs edge list is built in, which
    is the only reason the two line up element for element. Unreachable pairs
    get the sentinel **510** (``algos.pyx:28-34``), hence the 512-row embedding.

``D, P``
    Eigendecomposition of the bond-graph Laplacian ``L = D - A``, with
    ``D = 1/sqrt(lambda)`` and the zero modes set to 0 -- which is what removes
    the centre of mass from the harmonic prior. The number of zero modes is the
    number of RDKit fragments.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import ChiralType
from scipy.sparse.csgraph import floyd_warshall

#: ``algos.pyx`` writes 510 for unreachable pairs.
UNREACHABLE_HOPS = 510

#: Upstream's ``atomic_types`` (``prepare_dataset.py:24-46``). The atom-symbol
#: one-hot is ``len(types)`` wide because the misc slot is sliced away.
ATOMIC_TYPES = {
    "qm9": ["H", "C", "N", "O", "F"],
    "drugs": [
        "H", "Li", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "K", "Ca", "V", "Cr", "Mn", "Cu", "Zn", "Ga", "Ge", "As", "Se",
        "Br", "Ag", "In", "Sb", "I", "Gd", "Pt", "Au", "Hg", "Bi",
    ],
}
ATOMIC_TYPES["qm9_ablation"] = ATOMIC_TYPES["drugs"]

#: Column layout of ``node_attr``, in order. Widths sum to 64 (qm9) / 94 (drugs).
NODE_ATTR_BLOCKS = (
    ("chiral_tag", 5),  # 4 tags + misc
    ("total_num_h", 6),  # [0..4] + misc
    ("num_radical_electrons", 6),  # [0..4] + misc
    ("atom_symbol", None),  # len(types), misc SLICED AWAY
    ("is_aromatic", 1),
    ("total_degree", 6),  # [0..4] + misc
    ("hybridization", 6),  # SP..SP3D2 + misc
    ("implicit_valence", 6),  # [0..4] + misc
    ("formal_charge", 12),  # [-5..5] + misc  -> offset +5, 12 columns
    ("ring_of_size", 6),  # k = 3..8
    ("num_atom_rings", 5),  # [0..3] + misc
)

_CHIRAL_TAGS = [
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_OTHER,
]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]


def node_attr_dim(dataset: str) -> int:
    """Width of ``node_attr`` for a dataset name."""
    fixed = sum(w for _, w in NODE_ATTR_BLOCKS if w is not None)
    return fixed + len(ATOMIC_TYPES[dataset])


def _one_hot_encoding(value, choices) -> list[int]:
    """Upstream's ``one_hot_encoding``: a trailing misc slot, index -1."""
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_node_attr_from_mol(mol, dataset: str) -> np.ndarray:
    """Reproduce ``prepare_dataset.get_node_attr_from_mol`` exactly."""
    types = ATOMIC_TYPES[dataset]
    ring = mol.GetRingInfo()
    rows = []
    for i, atom in enumerate(mol.GetAtoms()):
        row: list[int] = []
        row.extend(_one_hot_encoding(atom.GetChiralTag(), _CHIRAL_TAGS))
        row.extend(_one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
        row.extend(
            _one_hot_encoding(atom.GetNumRadicalElectrons(), [0, 1, 2, 3, 4])
        )
        # The ONLY block whose misc slot is dropped.
        row.extend(_one_hot_encoding(atom.GetSymbol(), types)[:-1])
        row.append(int(atom.GetIsAromatic()))
        row.extend(_one_hot_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4]))
        row.extend(_one_hot_encoding(atom.GetHybridization(), _HYBRIDIZATIONS))
        row.extend(_one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4]))
        row.extend(
            _one_hot_encoding(
                atom.GetFormalCharge(), [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            )
        )
        row.extend(int(ring.IsAtomInRingOfSize(i, k)) for k in range(3, 9))
        row.extend(_one_hot_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))
        rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def shortest_hops_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """Off-diagonal Floyd-Warshall distances, C-order over ``(i, j)``.

    ``scipy.sparse.csgraph.floyd_warshall`` replaces the Cython ``algos.pyx``;
    ``inf`` is clamped to the same 510 sentinel upstream writes.
    """
    n = adj.shape[0]
    dist = floyd_warshall(adj.astype(np.float64), directed=False, unweighted=True)
    dist = np.where(np.isinf(dist), float(UNREACHABLE_HOPS), dist)
    dist = np.minimum(dist, float(UNREACHABLE_HOPS))
    mask = ~np.eye(n, dtype=bool)
    return dist[mask].astype(np.int64)


def laplacian_eigen(
    adj: np.ndarray, num_components: int | None = None, threshold: float = 1e-4
):
    """``(D, P)`` for the harmonic prior: ``D = 1/sqrt(lambda)``, zero modes 0.

    ``eigh`` returns ascending eigenvalues, so the ``num_components`` smallest
    are the zero modes (one per connected fragment). Eigenvector sign ambiguity
    is not a fidelity risk: the prior draws ``P diag(D) z`` with ``z`` standard
    normal, whose distribution is invariant under column sign flips.
    """
    n = adj.shape[0]
    deg = np.diag(adj.sum(axis=1))
    lap = deg - adj
    evals, evecs = np.linalg.eigh(lap.astype(np.float64))
    with np.errstate(divide="ignore"):
        # eigh can return tiny negatives for the zero modes; clip before sqrt.
        d = 1.0 / np.sqrt(np.clip(evals, 0.0, None))
    if num_components is not None:
        d[:num_components] = 0.0
    else:
        total = evals.sum()
        ratio = evals / total if total > 0 else np.zeros_like(evals)
        d = np.where(ratio < threshold, 0.0, d)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    return d.astype(np.float32), evecs.astype(np.float32)


def adjacency_from_bonds(bond_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Symmetric 0/1 adjacency from an upper-triangular bond index."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    bi = np.asarray(bond_index).reshape(2, -1)
    if bi.shape[1]:
        adj[bi[0], bi[1]] = 1
        adj[bi[1], bi[0]] = 1
    return adj


class MoleculeFeatureCache:
    """SMILES-keyed cache of the three per-molecule quantities.

    Upstream caches by SMILES at dataset-build time. Here the cache is
    in-process and bounded; a miss just recomputes.
    """

    def __init__(self, dataset: str = "qm9", maxsize: int = 200_000) -> None:
        self.dataset = dataset
        self._compute = functools.lru_cache(maxsize=maxsize)(self._compute_uncached)

    def _compute_uncached(self, key: tuple):
        smiles, n_nodes, z_bytes, bi_bytes, bt_bytes, fc_bytes = key
        z = np.frombuffer(z_bytes, dtype=np.int64)
        bond_index = np.frombuffer(bi_bytes, dtype=np.int64).reshape(2, -1)
        bond_type = np.frombuffer(bt_bytes, dtype=np.int64)
        fc = np.frombuffer(fc_bytes, dtype=np.int64)
        return self._featurize(smiles, n_nodes, z, bond_index, bond_type, fc)

    def _featurize(self, smiles, n_nodes, z, bond_index, bond_type, fc):
        # Pre-import: MolecularDiffusion.data.component.graph3d_dataset has a
        # latent import cycle with MolecularDiffusion.data.dataset that only
        # bites when graph3d_dataset is the FIRST of the two imported. This is
        # pre-existing platform behaviour, not something to fix from here.
        import MolecularDiffusion.data.dataset  # noqa: F401

        from MolecularDiffusion.data.datasets.graph3d_dataset import build_rdkit_mol

        try:
            mol = build_rdkit_mol(z, bond_index, bond_type, formal_charge=fc)
        except Exception:  # noqa: BLE001 - fall back to an unsanitized mol
            mol = build_rdkit_mol(
                z, bond_index, bond_type, formal_charge=fc, sanitize=False
            )
        node_attr = get_node_attr_from_mol(mol, self.dataset)

        adj = adjacency_from_bonds(bond_index, n_nodes)
        hops = shortest_hops_from_adjacency(adj)
        try:
            num_components = len(rdmolops.GetMolFrags(mol))
        except Exception:  # noqa: BLE001
            num_components = None
        d, p = laplacian_eigen(adj, num_components=num_components)
        return node_attr, hops, d, p

    def get(self, item) -> tuple:
        """``item`` is one PyG ``Data`` from the graph3d dataset."""
        z = item.z.detach().cpu().numpy().astype(np.int64)
        bond_index = (
            item.bond_index.detach().cpu().numpy().astype(np.int64).reshape(2, -1)
        )
        bond_type = item.bond_type.detach().cpu().numpy().astype(np.int64).reshape(-1)
        fc = item.fc.detach().cpu().numpy().astype(np.int64)
        n_nodes = int(z.shape[0])
        smiles = getattr(item, "smiles", None) or ""
        key = (
            smiles,
            n_nodes,
            z.tobytes(),
            bond_index.tobytes(),
            bond_type.tobytes(),
            fc.tobytes(),
        )
        return self._compute(key)


def all_pairs_edges(n_nodes: int, device=None):
    """All ordered pairs excluding self-loops, **C-order over ``(i, j)``**.

    Returns ``(receivers, senders) = (i, j)``. Matching this ordering to
    ``shortest_hops`` is not optional -- upstream relies on both being C-order.
    """
    i, j = torch.meshgrid(
        torch.arange(n_nodes, device=device),
        torch.arange(n_nodes, device=device),
        indexing="ij",
    )
    mask = i != j
    return i[mask], j[mask]
