"""3D molecular graph dataset with EXPLICIT BONDS.

The platform's other two data shapes (``pointcloud``, ``pyg``) are point
clouds: coordinates + atom types, with at most a ``radius_graph``/k-NN
``edge_index`` as a message-passing convenience and **no edge features**. This
module adds a third shape, ``graph3d``, carrying real bond orders and formal
charges so bond-*generating* models become expressible -- FlowMol's ``e``
modality (currently dropped, see ``modules/tasks/diffusion_flowmol.py:10``) and
MiDi.

It is strictly additive. Nothing here is imported by the existing loaders; the
shared helpers flow the other way (this file imports from ``.dataset``), so no
existing path can regress.

## The one storage rule

FlowMol and MiDi look different but agree on the underlying object:

    FlowMol  disk: sparse upper-triangular real bonds   model: [upper||lower] directed
    MiDi     disk: symmetric edge_index + edge_attr     model: dense (B,N,N) one-hot

Both are recoverable from the *smaller* of the two. So:

    **Store upper-triangular real bonds only. Materialize per model in collate.**

``bond_type`` uses 5 classes -- ``0=none, 1=SINGLE, 2=DOUBLE, 3=TRIPLE,
4=AROMATIC`` -- matching ``data/component/pharmacophore.py:186``, which already
uses this vocabulary. Class 0 is *never stored*: absence from ``bond_index``
means "no bond", and the no-bond count is derived, not counted.

## Why the fields are named ``bond_index`` / ``bond_type``

Not ``edge_index``/``edge_attr``, for two independent reasons:

1. PyG's ``Batch.from_data_list`` auto-increments any key whose name contains
   ``index``. ``bond_index`` therefore batches correctly for free; a key named
   anything else is silently concatenated *without* the node offset, which
   produces wrong graphs rather than an error.
2. A sparse real-bonds-only tensor called ``edge_index`` would be misread as
   "the message-passing graph" by every existing pyg-consuming module.

## Charges are stored raw and signed

FlowMol hardcodes ``[-2,+3]`` (6 classes), MiDi QM9 uses 3 classes offset +1,
MiDi GEOM 6 offset +2. Storing the raw signed integer means a model changing
its range never forces a dataset reconversion -- the offset/one-hot is applied
model-side.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils import data as torch_data
from torch_geometric.data import Data
from tqdm import tqdm

from MolecularDiffusion import utils

from MolecularDiffusion.data.datasets.dataset import (
    BASE_ATOM_VOCAB,
    _check_any_loaded,
    _collect_db_sources,
    _drop_hydrogens,
    _empty_pyg_buf,
    _flush_chunk,
    _iter_db_rows,
    _write_chunk_meta,
)
from MolecularDiffusion.data.features.feature import NodeFeaturizer

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:  # optional; only the converter and kekulization need it
    Chem = None


# ---------------------------------------------------------------------------
# Canonical bond vocabulary
# ---------------------------------------------------------------------------

#: Index -> RDKit bond type. Index 0 is "no bond" and is never stored on disk.
BOND_VOCAB: List[Optional[str]] = [None, "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
N_BOND_CLASSES = len(BOND_VOCAB)  # 5

#: RDKit bond-order float (from ``GetAdjacencyMatrix(useBO=True)``) -> class id.
BOND_ORDER_TO_CLASS: Dict[float, int] = {1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}


def rdkit_bond_types():
    """Index -> ``Chem.BondType``, for rebuilding molecules. Requires RDKit."""
    if Chem is None:
        raise ImportError("RDKit is required to map bond classes to RDKit types.")
    return [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]


def _row_metadata(row) -> Dict[str, Any]:
    """ASE row -> flat metadata dict (``data`` payload over key-value pairs).

    Written locally rather than imported so this module works unchanged in both
    the main and dev trees, which differ on whether ``dataset.py`` has an
    equivalent helper.
    """
    metadata: Dict[str, Any] = {}
    kvp = getattr(row, "key_value_pairs", None)
    if kvp:
        metadata.update(dict(kvp))
    row_data = getattr(row, "data", None) or {}
    if row_data:
        metadata.update(dict(row_data))
    return metadata


def build_rdkit_mol(
    atomic_numbers,
    bond_index,
    bond_type,
    formal_charge=None,
    coords=None,
    sanitize: bool = True,
    assign_stereo: bool = True,
):
    """Rebuild an RDKit mol from the stored fields.

    This is the inverse of what the converter stores, and is what makes the
    round-trip testable: if a molecule survives db -> dataset -> here with its
    canonical SMILES intact, the bond pipeline is lossless.

    Pass ``coords`` to recover stereochemistry, which lives in the geometry
    rather than in the bond table.

    Measured on 2000 real QM9 molecules: **99.9%** canonical-SMILES round-trip.
    The residue is carbenes and similar radicals (``[CH]``), where ``(z, bonds,
    formal_charge)`` alone cannot distinguish a radical from a missing hydrogen
    without a radical-electron count. MiDi's ``build_molecule`` has the same
    ambiguity, and neither target model consumes such a field, so it is not
    stored.
    """
    if Chem is None:
        raise ImportError("RDKit is required to rebuild molecules.")
    bond_types = rdkit_bond_types()
    mol = Chem.RWMol()
    for idx, z in enumerate(np.asarray(atomic_numbers).tolist()):
        atom = Chem.Atom(int(z))
        if formal_charge is not None:
            atom.SetFormalCharge(int(np.asarray(formal_charge)[idx]))
        # Deliberately NOT SetNoImplicit(True). Every H is already an explicit
        # atom here, so "no implicit H" reads as the correct statement -- but it
        # breaks RDKit's H bookkeeping when the SMILES writer folds explicit H
        # into a charged atom's count, turning [NH3+] into [N+]. Measured on
        # 2000 QM9 molecules: NoImplicit=True gives 91.2%, leaving it off gives
        # 99.9%.
        mol.AddAtom(atom)

    bond_index = np.asarray(bond_index).reshape(2, -1)
    for k, cls in enumerate(np.asarray(bond_type).tolist()):
        mol.AddBond(int(bond_index[0, k]), int(bond_index[1, k]), bond_types[int(cls)])

    if coords is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx, xyz in enumerate(np.asarray(coords).tolist()):
            conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
        mol.AddConformer(conf)

    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)
    if coords is not None and assign_stereo and sanitize:
        # Stereochemistry lives in the geometry, not the bond table.
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def kekulize_bonds(atomic_numbers, bond_index, bond_type, formal_charge=None):
    """Aromatic (class 4) bonds -> alternating SINGLE/DOUBLE.

    Kekulization is a *ring-level* operation, not a per-bond relabel, so this
    round-trips through RDKit. It runs once at dataset-build time (the result is
    cached in the processed pickle / chunks), never per ``__getitem__``.

    Storage always keeps the aromatic form; this exists so a model trained on a
    Kekule distribution -- MiDi's published QM9 run reads gdb9 with
    ``sanitize=False``, so its bond class 4 is never populated -- can be
    reproduced without a second copy of the dataset.

    Returns the bond_type array unchanged if there is nothing aromatic to do, or
    if RDKit cannot kekulize the molecule.
    """
    bond_type = np.asarray(bond_type)
    if not (bond_type == 4).any():
        return bond_type
    try:
        mol = build_rdkit_mol(
            atomic_numbers, bond_index, bond_type, formal_charge, sanitize=True
        )
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as exc:  # noqa: BLE001 - chemistry failures are data, not bugs
        logger.debug("Kekulization failed, keeping aromatic bonds: %s", exc)
        return bond_type

    bond_index = np.asarray(bond_index).reshape(2, -1)
    out = bond_type.copy()
    for k in range(bond_index.shape[1]):
        bond = mol.GetBondBetweenAtoms(int(bond_index[0, k]), int(bond_index[1, k]))
        if bond is not None:
            out[k] = BOND_ORDER_TO_CLASS.get(float(bond.GetBondTypeAsDouble()), out[k])
    return out


def remap_bonds_after_atom_removal(bond_index, bond_type, keep_mask):
    """Drop bonds touching removed atoms and renumber the survivors.

    ``keep_mask`` is the boolean array over the *original* atom order returned by
    :func:`~MolecularDiffusion.data.component.dataset._drop_hydrogens`, whose
    docstring states it exists precisely so parallel per-atom data can be
    filtered identically.

    ``i < j`` is preserved because the remap is monotone on kept indices.
    """
    keep_mask = np.asarray(keep_mask, dtype=bool)
    bond_index = np.asarray(bond_index).reshape(2, -1)
    bond_type = np.asarray(bond_type)

    remap = np.full(keep_mask.shape[0], -1, dtype=np.int64)
    remap[keep_mask] = np.arange(int(keep_mask.sum()), dtype=np.int64)

    both_kept = keep_mask[bond_index[0]] & keep_mask[bond_index[1]]
    return remap[bond_index[:, both_kept]], bond_type[both_kept]


# ---------------------------------------------------------------------------
# Dataset statistics (superset of what FlowMol and MiDi each require)
# ---------------------------------------------------------------------------


@dataclass
class Graph3DStats:
    """Preprocessing-time statistics. Both target models refuse to build without
    some of these, and both compute the same things under different names.

    ``bond_type_counts`` is in **upper-triangular pair units**:
    ``[0] = sum_mols(n(n-1)/2 - n_real_bonds)``, ``[1..4] = real bond counts``.
    FlowMol's ``p_e`` uses this directly. MiDi's ``edge_counts`` wants directed
    counts, i.e. exactly ``2x`` every entry -- one multiply, stated once here so
    it is never re-derived incorrectly downstream.
    """

    n_atoms_hist: Dict[int, int] = field(default_factory=dict)
    atom_type_counts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bond_type_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(N_BOND_CLASSES, dtype=np.int64)
    )
    charge_counts: Dict[int, Dict[int, int]] = field(default_factory=dict)
    valencies: Dict[int, Dict[int, Dict[float, int]]] = field(default_factory=dict)
    n_molecules: int = 0

    @property
    def charge_range(self):
        """Observed ``(min, max)`` raw signed formal charge, or ``None``."""
        seen = [c for per_atom in self.charge_counts.values() for c in per_atom]
        return (min(seen), max(seen)) if seen else None

    def atom_type_marginal(self) -> np.ndarray:
        total = self.atom_type_counts.sum()
        return self.atom_type_counts / total if total else self.atom_type_counts

    def bond_type_marginal(self, directed: bool = False) -> np.ndarray:
        """Normalized bond-class distribution. ``directed=True`` gives MiDi's
        convention; the scaling cancels, so the result is identical -- the flag
        exists for callers that also want the raw counts to match."""
        counts = self.bond_type_counts.astype(np.float64)
        counts = counts * 2 if directed else counts
        total = counts.sum()
        return counts / total if total else counts

    def n_atoms_histogram(self):
        """``(sizes, counts)`` sorted by size, as FlowMol's sampler expects."""
        sizes = sorted(self.n_atoms_hist)
        return (
            np.array(sizes, dtype=np.int64),
            np.array([self.n_atoms_hist[s] for s in sizes], dtype=np.int64),
        )


class _StatsAccumulator:
    """Single-pass accumulator; the molecules are already parsed in ``load_db``,
    so a second pass over GEOM would be minutes of I/O for nothing."""

    def __init__(self, n_atom_types: int):
        self.stats = Graph3DStats(
            atom_type_counts=np.zeros(n_atom_types, dtype=np.int64),
            n_atoms_hist=defaultdict(int),
        )

    def add(self, atom_idx, formal_charge, bond_index, bond_type):
        s = self.stats
        n = int(len(atom_idx))
        s.n_molecules += 1
        s.n_atoms_hist[n] += 1

        counts = np.bincount(np.asarray(atom_idx), minlength=len(s.atom_type_counts))
        s.atom_type_counts += counts[: len(s.atom_type_counts)]

        bond_type = np.asarray(bond_type)
        for cls in range(1, N_BOND_CLASSES):
            s.bond_type_counts[cls] += int((bond_type == cls).sum())
        # No-bond mass is DERIVED from the pair count, never counted -- the same
        # rule FlowMol (geom.py:235) and MiDi (metrics_utils.py:67) both use.
        s.bond_type_counts[0] += n * (n - 1) // 2 - int(bond_type.shape[0])

        bond_index = np.asarray(bond_index).reshape(2, -1)
        orders = np.array(
            [{1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}[int(c)] for c in bond_type.tolist()],
            dtype=np.float64,
        )
        valence = np.zeros(n, dtype=np.float64)
        for k in range(bond_index.shape[1]):
            valence[bond_index[0, k]] += orders[k]
            valence[bond_index[1, k]] += orders[k]

        for i in range(n):
            a, c = int(atom_idx[i]), int(formal_charge[i])
            s.charge_counts.setdefault(a, defaultdict(int))[c] += 1
            s.valencies.setdefault(a, {}).setdefault(c, defaultdict(int))[
                float(valence[i])
            ] += 1

    def finalize(self) -> Graph3DStats:
        s = self.stats
        s.n_atoms_hist = dict(s.n_atoms_hist)
        s.charge_counts = {a: dict(v) for a, v in s.charge_counts.items()}
        s.valencies = {
            a: {c: dict(v) for c, v in per_charge.items()}
            for a, per_charge in s.valencies.items()
        }
        return s


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Graph3DDataset(torch_data.Dataset):
    """3D molecular graphs with explicit bonds, loaded from an ASE db.

    Exposes the same public surface as ``GraphDataset``/``LazyChunkedGraphDataset``
    (``smiles_list``, ``num_atoms``, ``get_property``, ``atom_types``,
    ``targets``) so the ``DataModule`` tail at ``runmodes/train/data.py:320-357``
    drives it with no changes.
    """

    def __init__(
        self,
        root: str = "",
        ase_db_path: Optional[str] = None,
        dataset_name: str = "graph3d",
        max_atom: int = 200,
        load_only: bool = False,
        **kwargs: Any,
    ):
        self.root = root
        self.max_atom = max_atom
        self.dataset_name = dataset_name
        self.transform = None
        self._processed_file = os.path.join(
            root, f"processed_data_{dataset_name}_graph3d.pt"
        )
        self._reset()

        chunking = kwargs.get("chunk_size") and kwargs.get("chunk_dir")
        if chunking:
            self._processed_file = None

        if self._processed_file and os.path.exists(self._processed_file):
            logger.info("Found processed file at %s, loading it.", self._processed_file)
            self.load_pickle(self._processed_file)
        elif load_only:
            raise FileNotFoundError(
                f"load_only=True but no processed cache at {self._processed_file}"
            )
        elif ase_db_path:
            self.load_db(ase_db_path, max_atom=max_atom, **kwargs)
            if self._processed_file is not None:
                self.save_pickle(self._processed_file)
        else:
            raise ValueError("ase_db_path must be provided when no cache exists.")

    def _reset(self):
        self.transform = None
        self.graph_data_list: List[Data] = []
        self.n_atoms: List[int] = []
        self.smiles_list: List[Optional[str]] = []
        self.splits: List[str] = []
        self.targets: Dict[str, List[float]] = defaultdict(list)
        self.atom_vocab: List[str] = []
        self.with_hydrogen = True
        self.graph3d_stats: Optional[Graph3DStats] = None

    # -- loading ------------------------------------------------------------

    def load_db(
        self,
        db_path: str,
        atom_vocab: Optional[List[str]] = None,
        target_fields: Optional[List[str]] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: Optional[List[str]] = None,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        center_coords: bool = False,
        kekulize: bool = False,
        compute_stats: bool = True,
        transform: Optional[Callable] = None,
        verbose: int = 0,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """Build the dataset from ASE db rows written by the graph3d converter.

        Each row must carry ``bond_index`` (2,E upper-triangular), ``bond_type``
        (E,) and ``formal_charge`` (N,) in ``row.data``; rows without them are
        counted and skipped rather than silently treated as bond-free.
        """
        atom_vocab = list(atom_vocab) if atom_vocab else list(BASE_ATOM_VOCAB)
        forbidden_atoms = forbidden_atoms or []
        self._reset()
        self.atom_vocab = atom_vocab
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.max_atom = max_atom

        featurizer = NodeFeaturizer(
            atom_vocab=atom_vocab,
            use_ohe=use_ohe_feature,
            geom_feature=None,
            allow_unknown=allow_unknown,
        )
        n_atom_types = len(atom_vocab) + (1 if allow_unknown else 0)
        accumulator = _StatsAccumulator(n_atom_types) if compute_stats else None

        chunking = chunk_size is not None and chunk_dir is not None
        buf = _empty_pyg_buf() if chunking else None
        chunk_paths: List[str] = []
        chunk_sizes: List[int] = []
        all_smiles: List[Optional[str]] = []
        all_n_atoms: List[int] = []
        chunk_idx = 0

        db_sources = _collect_db_sources(db_path)
        total_len = sum(source["length"] for source in db_sources)
        iterator = _iter_db_rows(db_sources)
        if verbose:
            iterator = tqdm(iterator, "Processing 3D-graph ASE db", total=total_len)

        skipped = defaultdict(int)
        last_error: Optional[Exception] = None

        for i, row in enumerate(iterator):
            try:
                mol_ase = row.toatoms()
                meta = _row_metadata(row)

                if "bond_index" not in meta or "bond_type" not in meta:
                    skipped["no_bond_data"] += 1
                    continue
                if any(atom.symbol in forbidden_atoms for atom in mol_ase):
                    skipped["forbidden"] += 1
                    continue

                bond_index = np.asarray(meta["bond_index"], dtype=np.int64).reshape(2, -1)
                bond_type = np.asarray(meta["bond_type"], dtype=np.int64).reshape(-1)
                formal_charge = np.asarray(
                    meta.get("formal_charge", np.zeros(len(mol_ase))), dtype=np.int64
                ).reshape(-1)

                if formal_charge.shape[0] != len(mol_ase):
                    skipped["charge_mismatch"] += 1
                    continue

                # Hydrogen stripping BEFORE the max_atom check, so max_atom
                # bounds the *stored* atom count -- same rule as load_xyz.
                mol_ase, keep_mask = _drop_hydrogens(mol_ase, with_hydrogen)
                if keep_mask is not None:
                    bond_index, bond_type = remap_bonds_after_atom_removal(
                        bond_index, bond_type, keep_mask
                    )
                    formal_charge = formal_charge[keep_mask]

                n_nodes = len(mol_ase)
                if n_nodes == 0:
                    skipped["empty"] += 1
                    continue
                if n_nodes > max_atom:
                    skipped["too_large"] += 1
                    continue

                if kekulize:
                    bond_type = kekulize_bonds(
                        mol_ase.get_atomic_numbers(),
                        bond_index,
                        bond_type,
                        formal_charge,
                    )

                symbols = mol_ase.get_chemical_symbols()
                unknown = [s for s in symbols if s not in atom_vocab]
                if unknown and not allow_unknown:
                    skipped["unknown_atom"] += 1
                    continue

                coords = torch.from_numpy(mol_ase.get_positions()).to(torch.float32)
                if center_coords:
                    coords = coords - coords.mean(dim=0, keepdim=True)
                charges = torch.from_numpy(mol_ase.get_atomic_numbers()).to(torch.long)

                node_features = (
                    featurizer.compute_ohe(symbols) if use_ohe_feature else None
                )
                atom_idx = torch.tensor(
                    [
                        atom_vocab.index(s) if s in atom_vocab else len(atom_vocab)
                        for s in symbols
                    ],
                    dtype=torch.long,
                )

                if torch.isnan(coords).any():
                    skipped["nan"] += 1
                    continue

                smiles = meta.get("smiles")
                graph = Data(
                    x=node_features,
                    pos=coords,
                    z=charges,
                    atom_idx=atom_idx,
                    fc=torch.from_numpy(formal_charge).to(torch.long),
                    bond_index=torch.from_numpy(np.ascontiguousarray(bond_index)).long(),
                    bond_type=torch.from_numpy(np.ascontiguousarray(bond_type)).long(),
                    n_nodes=n_nodes,
                    smiles=smiles,
                )

                if accumulator is not None:
                    accumulator.add(
                        atom_idx.numpy(), formal_charge, bond_index, bond_type
                    )

                resolved: Dict[str, float] = {}
                for fieldname in target_fields or []:
                    value = meta.get(fieldname, "")
                    try:
                        value = utils.literal_eval(str(value))
                    except (ValueError, SyntaxError):
                        pass
                    resolved[fieldname] = (
                        math.nan if value == "" else float(value)
                    )

                split = str(meta.get("split", "") or "")
                if chunking:
                    buf["graph_data_list"].append(graph)
                    buf["n_atoms"].append(n_nodes)
                    buf["smiles_list"].append(smiles)
                    for key, val in resolved.items():
                        buf["targets"][key].append(val)
                    all_smiles.append(smiles)
                    all_n_atoms.append(n_nodes)
                    if len(buf["graph_data_list"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, chunk_idx, buf)
                        chunk_paths.append(path)
                        chunk_sizes.append(len(buf["graph_data_list"]))
                        logger.info(
                            "Flushed graph3d chunk %d (%d graphs) -> %s",
                            chunk_idx,
                            chunk_sizes[-1],
                            path,
                        )
                        chunk_idx += 1
                        buf = _empty_pyg_buf()
                else:
                    self.graph_data_list.append(graph)
                    self.n_atoms.append(n_nodes)
                    self.smiles_list.append(smiles)
                    self.splits.append(split)
                    for key, val in resolved.items():
                        self.targets[key].append(val)

            except Exception as exc:  # noqa: BLE001
                skipped["failed"] += 1
                last_error = exc
                logger.error("Error in loading graph3d db entry %d: %s", i, exc)
                continue

        if chunking and buf["graph_data_list"]:
            path = _flush_chunk(chunk_dir, chunk_idx, buf)
            chunk_paths.append(path)
            chunk_sizes.append(len(buf["graph_data_list"]))

        if accumulator is not None:
            self.graph3d_stats = accumulator.finalize()

        if chunking:
            _write_chunk_meta(
                chunk_dir,
                chunk_paths,
                chunk_sizes,
                list(target_fields) if target_fields else [],
                atom_vocab,
                with_hydrogen,
                all_smiles,
                all_n_atoms,
                kind="graph3d",
            )
            _augment_chunk_meta_graph3d(chunk_dir, self.graph3d_stats)

        if any(skipped.values()):
            logger.warning(
                "Discarded entries: %s (last error: %s)",
                dict(skipped),
                last_error,
            )

        _check_any_loaded(
            db_path,
            total_len,
            sum(chunk_sizes) if chunking else len(self.graph_data_list),
            last_error,
        )

    # -- persistence --------------------------------------------------------

    def save_pickle(self, pkl_file: str, verbose: int = 0):
        os.makedirs(os.path.dirname(os.path.abspath(pkl_file)), exist_ok=True)
        with utils.smart_open(pkl_file, "wb") as fout:
            tasks = list(self.targets.keys())
            pickle.dump((len(self.graph_data_list), tasks), fout)
            indexes = range(len(self.graph_data_list))
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                pickle.dump(
                    (
                        self.graph_data_list[i],
                        [v[i] for v in self.targets.values()],
                        self.splits[i] if i < len(self.splits) else "",
                    ),
                    fout,
                )
            pickle.dump(
                (self.atom_vocab, self.with_hydrogen, self.smiles_list, self.graph3d_stats),
                fout,
            )

    def load_pickle(self, pkl_file: str, verbose: int = 0):
        self._reset()
        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample, tasks = pickle.load(fin)
            for task in tasks:
                self.targets[task] = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            skipped_too_large = 0
            for _ in indexes:
                graph, values, split = pickle.load(fin)
                if self.max_atom and int(graph.n_nodes) > self.max_atom:
                    skipped_too_large += 1
                    continue
                self.graph_data_list.append(graph)
                self.n_atoms.append(int(graph.n_nodes))
                self.splits.append(split)
                for task, value in zip(tasks, values):
                    self.targets[task].append(float(value))
            (
                self.atom_vocab,
                self.with_hydrogen,
                self.smiles_list,
                self.graph3d_stats,
            ) = pickle.load(fin)
            if skipped_too_large:
                logger.warning(
                    "Discarded %d entries exceeding max_atom", skipped_too_large
                )

    # -- Dataset protocol ---------------------------------------------------

    def get_item(self, index: int) -> Dict[str, Any]:
        item = {k: v[index] for k, v in self.targets.items()}
        item["graph"] = self.graph_data_list[index]
        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        return [self.get_item(i) for i in index]

    def __len__(self) -> int:
        return len(self.graph_data_list)

    def __repr__(self) -> str:
        return (
            f"Graph3DDataset(n={len(self)}, "
            f"tasks={list(self.targets.keys())}, "
            f"with_hydrogen={self.with_hydrogen})"
        )

    @property
    def tasks(self) -> List[str]:
        return list(self.targets.keys())

    def atom_types(self) -> List[int]:
        from ase.data import atomic_numbers

        types = {atomic_numbers[s] for s in self.atom_vocab if s in atomic_numbers}
        return sorted(types - {0})

    @property
    def num_atoms(self) -> torch.Tensor:
        return torch.tensor(self.n_atoms, dtype=torch.long)

    def get_property(self, task: str) -> Optional[torch.Tensor]:
        if task not in self.targets:
            return None
        return torch.tensor(self.targets[task], dtype=torch.float32)


def _augment_chunk_meta_graph3d(chunk_dir: str, stats: Optional[Graph3DStats]):
    """Append bond statistics to an already-written ``meta.pt``.

    Kept separate from ``_write_chunk_meta`` so the shared helper stays
    untouched and ``LazyChunkedGraphDataset`` -- which reads its extra keys with
    ``.get`` defaults -- can load these chunks unmodified.
    """
    meta_path = os.path.join(chunk_dir, "meta.pt")
    meta = torch.load(meta_path, weights_only=False)
    meta["graph3d_stats"] = stats
    torch.save(meta, meta_path)


# ---------------------------------------------------------------------------
# Materializer: dense (MiDi-shaped). The FlowMol-shaped one is deliberately
# absent -- `modules/models/flowmol/graph_utils.build_edge_idxs` already emits
# the exact [upper || mirrored-lower] ordering, so labelling it is
# `cat([labels, labels])` on the model side where the DGL graph is built.
# ---------------------------------------------------------------------------


def graph3d_dense_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch into MiDi's dense shapes.

    ``bond_type`` comes back as **integer class ids** ``(B,N,N)``, not a
    ``(B,N,N,5)`` one-hot: one-hotting is ``F.one_hot(E, 5)`` on the model side
    (exactly where MiDi does it, in ``to_one_hot``), and at GEOM sizes the
    int8-equivalent dense matrix is ~20x smaller than the float one-hot.
    """
    graphs = [item["graph"] for item in batch]
    sizes = [int(g.n_nodes) for g in graphs]
    bs, n_max = len(graphs), max(sizes)

    pos = torch.zeros(bs, n_max, 3, dtype=torch.float32)
    atom_idx = torch.zeros(bs, n_max, dtype=torch.long)
    charges = torch.zeros(bs, n_max, dtype=torch.long)
    node_mask = torch.zeros(bs, n_max, dtype=torch.bool)
    bond = torch.zeros(bs, n_max, n_max, dtype=torch.long)

    for b, graph in enumerate(graphs):
        n = sizes[b]
        pos[b, :n] = graph.pos
        atom_idx[b, :n] = graph.atom_idx
        charges[b, :n] = graph.fc
        node_mask[b, :n] = True
        bi, bt = graph.bond_index, graph.bond_type
        if bt.numel():
            # Fill both triangles from the upper-triangular storage; the
            # diagonal stays zero, which is what MiDi's symmetry assert wants.
            bond[b, bi[0], bi[1]] = bt
            bond[b, bi[1], bi[0]] = bt

    out: Dict[str, Any] = {
        "pos": pos,
        "atom_idx": atom_idx,
        "charges": charges,
        "bond_type": bond,
        "node_mask": node_mask,
        "natoms": torch.tensor(sizes, dtype=torch.long),
        "smiles": [g.smiles for g in graphs],
    }
    for key in batch[0]:
        if key != "graph":
            out[key] = torch.tensor([item[key] for item in batch], dtype=torch.float32)
    return out
