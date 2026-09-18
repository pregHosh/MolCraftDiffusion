
import csv
import logging
import math
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union, Any
import os
import pickle
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import torch
from torch.utils import data as torch_data
from tqdm import tqdm

from ase.data import atomic_numbers
from ase.db import connect
try:
    from rdkit import Chem
except ImportError:
    Chem = None
try:
    import lmdb
except ImportError:
    lmdb = None
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph

from MolecularDiffusion.utils import sascore
from MolecularDiffusion import core, utils
from .feature import NodeFeaturizer
from .pointcloud import PointCloud_Mol

logger = logging.getLogger(__name__)

import itertools

# ---------------------------------------------------------------------------
# Chunked streaming helpers
# ---------------------------------------------------------------------------

def _flush_chunk(chunk_dir: str, chunk_idx: int, buf: dict) -> str:
    """Persist one buffer dict to disk and return the file path."""
    os.makedirs(chunk_dir, exist_ok=True)
    path = os.path.join(chunk_dir, f"chunk_{chunk_idx:06d}.pt")
    torch.save(buf, path)
    return path


def _empty_buf() -> dict:
    return dict(
        coords_list=[], node_mask_list=[], edge_mask_list=[],
        node_feature_list=[], charges_list=[], n_atoms=[],
        xyzs=[], smiles_list=[], targets=defaultdict(list),
    )


class LazyChunkedDataset(torch_data.Dataset):
    """
    Drop-in replacement for PointCloudDataset that never loads the full
    dataset into RAM.  Chunks are loaded from disk on demand with a small
    LRU cache.

    Args:
        chunk_dir: directory containing chunk_*.pt files and meta.pt
        cache_chunks: how many chunks to keep in memory at once (default 2)
    """

    def __init__(self, chunk_dir: str, cache_chunks: int = 2):
        from collections import OrderedDict
        import bisect

        self.chunk_dir = chunk_dir
        meta = torch.load(os.path.join(chunk_dir, "meta.pt"), weights_only=False)
        self.chunk_paths: List[str] = meta["chunk_paths"]
        self.chunk_sizes: List[int] = meta["chunk_sizes"]
        self._tasks: List[str] = meta["tasks"]
        self.atom_vocab: List[str] = meta["atom_vocab"]
        self.with_hydrogen: bool = meta["with_hydrogen"]
        self.smiles_list: List[str] = meta["smiles_list"]
        self.n_atoms: List[int] = meta["n_atoms"]

        # cumulative offsets for O(log n) index lookup
        self._offsets = [0]
        for s in self.chunk_sizes:
            self._offsets.append(self._offsets[-1] + s)

        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._cache_limit = max(1, cache_chunks)
        self._bisect = bisect

    def _load_chunk(self, chunk_idx: int) -> dict:
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]
        chunk = torch.load(self.chunk_paths[chunk_idx], weights_only=False)
        if len(self._cache) >= self._cache_limit:
            self._cache.popitem(last=False)
        self._cache[chunk_idx] = chunk
        return chunk

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, index):
        if not isinstance(index, int):
            return [self[i] for i in index]
        chunk_idx = self._bisect.bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[chunk_idx]
        c = self._load_chunk(chunk_idx)
        item = {
            "coords":       c["coords_list"][local_idx],
            "node_mask":    c["node_mask_list"][local_idx],
            "edge_mask":    c["edge_mask_list"][local_idx],
            "node_feature": c["node_feature_list"][local_idx],
            "charges":      c["charges_list"][local_idx],
            "natoms":       c["n_atoms"][local_idx],
            "xyz":          c["xyzs"][local_idx],
        }
        for task in self._tasks:
            item[task] = c["targets"][task][local_idx]
        return item

    def get_item(self, index: int):
        return self[index]

    @property
    def num_atoms(self) -> torch.Tensor:
        return torch.tensor(self.n_atoms, dtype=torch.long)

    def atom_types(self) -> List[int]:
        from ase.data import atomic_numbers as _an
        types = {_an[s] for s in self.atom_vocab if s in _an} - {0}
        return sorted(types)

    def get_property(self, task: str, indices=None) -> torch.Tensor:
        if indices is None:
            indices = range(len(self))
        grouped: Dict[int, List[tuple]] = defaultdict(list)
        for out_idx, index in enumerate(indices):
            chunk_idx = self._bisect.bisect_right(self._offsets, int(index)) - 1
            local_idx = int(index) - self._offsets[chunk_idx]
            grouped[chunk_idx].append((out_idx, local_idx))

        values = [0.0] * sum(len(v) for v in grouped.values())
        for chunk_idx, positions in grouped.items():
            path = self.chunk_paths[chunk_idx]
            c = torch.load(path, weights_only=False, map_location="cpu")
            target = c["targets"].get(task, [])
            for out_idx, local_idx in positions:
                values[out_idx] = target[local_idx]
            del c
        return torch.tensor(values, dtype=torch.float32)

    @property
    def targets(self) -> dict:
        # Return key-only dict — callers only need .keys() to validate task names.
        # Loading all chunks for this is O(n_chunks) disk reads; we avoid it entirely.
        return dict.fromkeys(self._tasks)

    @property
    def tasks(self) -> List[str]:
        return self._tasks

    def __repr__(self) -> str:
        return (
            f"LazyChunkedDataset(n={len(self)}, chunks={len(self.chunk_paths)}, "
            f"tasks={self._tasks})"
        )


def _write_chunk_meta(chunk_dir: str, chunk_paths: List[str],
                      chunk_sizes: List[int], tasks: List[str],
                      atom_vocab: List[str], with_hydrogen: bool,
                      smiles_list: List[str], n_atoms: List[int],
                      kind: str = "pointcloud"):
    torch.save(
        dict(
            chunk_paths=chunk_paths, chunk_sizes=chunk_sizes,
            tasks=tasks, atom_vocab=atom_vocab,
            with_hydrogen=with_hydrogen, smiles_list=smiles_list,
            n_atoms=n_atoms, kind=kind,
        ),
        os.path.join(chunk_dir, "meta.pt"),
    )


# ---------------------------------------------------------------------------
# pyG flavour of chunked streaming
# ---------------------------------------------------------------------------

def _empty_pyg_buf() -> dict:
    return dict(
        graph_data_list=[], n_atoms=[], smiles_list=[],
        targets=defaultdict(list),
    )


# ---------------------------------------------------------------------------
# pyG compact-storage helpers.  Compaction trades a tiny CPU cost per
# __getitem__ for large memory/disk savings.  All four flags are independent.
# ---------------------------------------------------------------------------

def _compact_build_pyg(
    *,
    node_features: torch.Tensor,
    coords: torch.Tensor,
    charges: torch.Tensor,
    n_nodes: int,
    smiles: Any,
    xyz: Any,
    edge_index: torch.Tensor,
    edge_type: str,
    mol_index: int,
    compact: Dict[str, bool],
):
    """Build a (possibly slim) pyG Data for one molecule.

    `compact` keys: drop_edge_index, drop_tags, pack_ohe, int8_z.
    Returns: (Data, ohe_size) where ohe_size is None when pack_ohe is off.
    """
    from torch_geometric.data import Data as _Data

    ohe_size: Optional[int] = None
    x_store = node_features

    if compact.get("pack_ohe"):
        # Recover atom indices from OHE.  Only valid when x is purely OHE.
        ohe_size = int(node_features.shape[1])
        x_store = node_features.argmax(dim=-1).to(torch.int8)

    z_store = charges.to(torch.int8) if compact.get("int8_z") else charges

    kw: Dict[str, Any] = dict(
        x=x_store,
        pos=coords,
        atomic_numbers=z_store,
        natoms=n_nodes,
        smiles=smiles,
    )
    if xyz is not None:
        kw["xyz"] = xyz

    if not compact.get("drop_tags"):
        kw["tags"] = torch.zeros(n_nodes, dtype=torch.long) + mol_index
        kw["token_idx"] = torch.arange(n_nodes, dtype=torch.long)

    drop_ei = compact.get("drop_edge_index") and edge_type == "fully_connected"
    if not drop_ei:
        kw["edge_index"] = edge_index

    return _Data(**kw), ohe_size


def _compact_expand_pyg(
    data_in,
    compact: Dict[str, bool],
    ohe_size: Optional[int],
    edge_type: str,
    mol_index: int,
):
    """Reconstruct full pyG Data from a slim one.  Returns a NEW Data object;
    the input is not mutated so it remains safe to read-share across epochs.
    """
    from torch_geometric.data import Data as _Data

    n = int(data_in.natoms)
    kw: Dict[str, Any] = {}
    for k in data_in.keys():
        kw[k] = getattr(data_in, k)

    if compact.get("pack_ohe") and ohe_size:
        atom_idx = data_in.x
        ohe = torch.zeros(n, ohe_size, dtype=torch.float32)
        ohe.scatter_(1, atom_idx.long().unsqueeze(1), 1.0)
        kw["x"] = ohe
    if compact.get("int8_z"):
        kw["atomic_numbers"] = data_in.atomic_numbers.long()
    if compact.get("drop_tags"):
        kw["tags"] = torch.zeros(n, dtype=torch.long) + mol_index
        kw["token_idx"] = torch.arange(n, dtype=torch.long)
    if compact.get("drop_edge_index") and edge_type == "fully_connected":
        row_ = torch.arange(n).repeat_interleave(n)
        col_ = torch.arange(n).repeat(n)
        ei = torch.stack([row_, col_], dim=0)
        ei = ei[:, row_ != col_]
        kw["edge_index"] = ei

    return _Data(**kw)


def _compact_is_active(compact: Optional[Dict[str, bool]]) -> bool:
    return bool(compact) and any(bool(v) for v in compact.values())


def _augment_chunk_meta_pyg(
    chunk_dir: str,
    compact: Dict[str, bool],
    ohe_size: Optional[int],
    edge_type: str,
):
    """Append compact-reconstruction info to an already-written meta.pt."""
    meta_path = os.path.join(chunk_dir, "meta.pt")
    meta = torch.load(meta_path, weights_only=False)
    meta["compact"] = dict(compact) if compact else {}
    meta["ohe_size"] = ohe_size
    meta["edge_type"] = edge_type
    torch.save(meta, meta_path)


class LazyChunkedGraphDataset(torch_data.Dataset):
    """
    Drop-in replacement for GraphDataset that streams pyG `Data` objects
    from disk on demand.

    Each chunk file is a dict with keys:
      graph_data_list: List[torch_geometric.data.Data]
      n_atoms:         List[int]
      smiles_list:     List[str]
      targets:         Dict[str, List[float]]
    """

    def __init__(self, chunk_dir: str, cache_chunks: int = 2):
        from collections import OrderedDict
        import bisect

        self.chunk_dir = chunk_dir
        meta = torch.load(os.path.join(chunk_dir, "meta.pt"), weights_only=False)
        self.chunk_paths: List[str] = meta["chunk_paths"]
        self.chunk_sizes: List[int] = meta["chunk_sizes"]
        self._tasks: List[str] = meta["tasks"]
        self.atom_vocab: List[str] = meta["atom_vocab"]
        self.with_hydrogen: bool = meta["with_hydrogen"]
        self.smiles_list: List[str] = meta["smiles_list"]
        self.n_atoms: List[int] = meta["n_atoms"]
        self._compact: Dict[str, bool] = meta.get("compact") or {}
        self._ohe_size: Optional[int] = meta.get("ohe_size")
        self._edge_type: str = meta.get("edge_type", "fully_connected")

        self._offsets = [0]
        for s in self.chunk_sizes:
            self._offsets.append(self._offsets[-1] + s)

        self._cache: "OrderedDict[int, dict]" = OrderedDict()
        self._cache_limit = max(1, cache_chunks)
        self._bisect = bisect
        self.transform = None

    def _load_chunk(self, chunk_idx: int) -> dict:
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]
        chunk = torch.load(self.chunk_paths[chunk_idx], weights_only=False)
        if len(self._cache) >= self._cache_limit:
            self._cache.popitem(last=False)
        self._cache[chunk_idx] = chunk
        return chunk

    def __len__(self) -> int:
        return self._offsets[-1]

    def get_item(self, index: int):
        chunk_idx = self._bisect.bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[chunk_idx]
        c = self._load_chunk(chunk_idx)
        item = {task: c["targets"][task][local_idx] for task in self._tasks}
        graph = c["graph_data_list"][local_idx]
        if _compact_is_active(self._compact):
            graph = _compact_expand_pyg(
                graph, self._compact, self._ohe_size, self._edge_type, index,
            )
        item["graph"] = graph
        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0: start += len(self)
            stop = index.stop or len(self)
            if stop < 0: stop += len(self)
            step = index.step or 1
            index = range(start, stop, step)
        return [self.get_item(i) for i in index]

    @property
    def num_atoms(self) -> torch.Tensor:
        return torch.tensor(self.n_atoms, dtype=torch.long)

    def atom_types(self) -> List[int]:
        types = {atomic_numbers[s] for s in self.atom_vocab if s in atomic_numbers} - {0}
        return sorted(types)

    @property
    def num_atom_type(self) -> int:
        return len(self.atom_types())

    def get_property(self, task: str, indices=None) -> Optional[torch.Tensor]:
        if task not in self._tasks:
            return None
        if indices is None:
            indices = range(len(self))
        grouped: Dict[int, List[tuple]] = defaultdict(list)
        for out_idx, index in enumerate(indices):
            chunk_idx = self._bisect.bisect_right(self._offsets, int(index)) - 1
            local_idx = int(index) - self._offsets[chunk_idx]
            grouped[chunk_idx].append((out_idx, local_idx))

        values = [0.0] * sum(len(v) for v in grouped.values())
        for chunk_idx, positions in grouped.items():
            path = self.chunk_paths[chunk_idx]
            c = torch.load(path, weights_only=False, map_location="cpu")
            target = c["targets"].get(task, [])
            for out_idx, local_idx in positions:
                values[out_idx] = target[local_idx]
            del c
        return torch.tensor(values, dtype=torch.float32)

    @property
    def targets(self) -> dict:
        # Return key-only dict — callers only need .keys() to validate task names.
        # Loading all chunks for this is O(n_chunks) disk reads; we avoid it entirely.
        return dict.fromkeys(self._tasks)

    @property
    def tasks(self) -> List[str]:
        return self._tasks

    def __repr__(self) -> str:
        return (
            f"LazyChunkedGraphDataset(n={len(self)}, chunks={len(self.chunk_paths)}, "
            f"tasks={self._tasks})"
        )


class _ASELMDBRow:
    # Keys that are structural fields, not row data
    _STRUCTURAL = frozenset({"atoms", "energy", "forces", "stress", "natoms", "id", "source", "data"})

    def __init__(self, payload: Dict[str, Any]):
        self._atoms = payload["atoms"]
        # Start from nested 'data' sub-dict (ASE convention), then overlay any
        # extra top-level keys (e.g. 'node_features' written by annotation scripts)
        data = dict(payload.get("data", {}) or {})
        for key, val in payload.items():
            if key not in self._STRUCTURAL:
                data.setdefault(key, val)
        self.data = data
        self.id = payload.get("id")

    def toatoms(self):
        return self._atoms


def _ensure_numpy_pickle_compat():
    try:
        import numpy._core.numeric  # type: ignore[attr-defined]  # noqa: F401
    except ImportError:
        import numpy.core.numeric as numpy_core_numeric

        sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)


def _get_aselmdb_length(db_path: str) -> int:
    if lmdb is None:
        raise ImportError(
            "lmdb is required to read .aselmdb files. Install the optional dependency first."
        )

    env = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin() as txn:
            raw_length = txn.get(b"length")
            if raw_length is None:
                raise ValueError(f"Missing 'length' key in {db_path}")
            return int(pickle.loads(raw_length))
    finally:
        env.close()


def _iter_aselmdb_rows(db_path: str) -> Iterator[_ASELMDBRow]:
    if lmdb is None:
        raise ImportError(
            "lmdb is required to read .aselmdb files. Install the optional dependency first."
        )

    _ensure_numpy_pickle_compat()
    env = lmdb.open(
        db_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        with env.begin() as txn:
            raw_length = txn.get(b"length")
            if raw_length is None:
                raise ValueError(f"Missing 'length' key in {db_path}")
            total_rows = int(pickle.loads(raw_length))
            for idx in range(total_rows):
                key = str(idx).encode("ascii")
                payload = txn.get(key)
                if payload is None:
                    raise KeyError(f"Missing key {idx} in {db_path}")
                yield _ASELMDBRow(pickle.loads(payload))
    finally:
        env.close()


def _collect_db_sources(db_path: str) -> List[Dict[str, Any]]:
    if os.path.isdir(db_path):
        sqlite_files = sorted(glob(os.path.join(db_path, "*.db")))
        aselmdb_files = sorted(glob(os.path.join(db_path, "*.aselmdb")))
        db_files = [{"kind": "ase", "path": path} for path in sqlite_files]
        db_files.extend({"kind": "aselmdb", "path": path} for path in aselmdb_files)
    elif os.path.isfile(db_path):
        kind = "aselmdb" if db_path.endswith(".aselmdb") else "ase"
        db_files = [{"kind": kind, "path": db_path}]
    else:
        raise ValueError(
            f"Invalid db_path: {db_path}. It must be a .db file, a .aselmdb file, or a directory containing them."
        )

    if not db_files:
        raise FileNotFoundError(f"No .db or .aselmdb files found in {db_path}")

    for source in db_files:
        if source["kind"] == "ase":
            source["length"] = connect(source["path"]).count()
        else:
            source["length"] = _get_aselmdb_length(source["path"])

    return db_files


def _iter_db_rows(db_sources: List[Dict[str, Any]]) -> Iterator[Any]:
    for source in db_sources:
        if source["kind"] == "ase":
            yield from connect(source["path"]).select()
        else:
            yield from _iter_aselmdb_rows(source["path"])

def _drop_hydrogens(
    mol_ase: Any, with_hydrogen: bool
) -> Tuple[Any, Optional[np.ndarray]]:
    """Strip hydrogens from an ASE ``Atoms`` unless ``with_hydrogen``.

    Mirrors what ``PointCloud_Mol.from_xyz``/``from_arrays`` already do for
    the xyz/array loaders, so the ASE-db loaders honour ``with_hydrogen``
    the same way.

    Returns ``(atoms, keep_mask)``. ``keep_mask`` is ``None`` when nothing
    was dropped, otherwise a boolean array over the *original* atom order,
    so per-atom data coming from another source (e.g. an RDKit ``mol_block``)
    can be filtered identically.
    """
    if with_hydrogen:
        return mol_ase, None
    keep = np.asarray(mol_ase.get_atomic_numbers()) != 1
    if keep.all():
        return mol_ase, None
    return mol_ase[keep], keep


def _check_any_loaded(
    source: str,
    total_rows: int,
    n_loaded: int,
    last_error: Optional[Exception],
) -> None:
    """Fail loudly when every entry of a non-empty source was discarded.

    Silently returning an empty dataset only surfaces much later as an
    ``IndexError`` in an unrelated file, so a config/data mismatch is
    reported here, where it can be acted on.
    """
    if total_rows and not n_loaded:
        raise ValueError(
            f"No entries could be loaded from {source}: all {total_rows} "
            f"rows were discarded. Check `atom_vocab`, `with_hydrogen`, "
            f"`max_atom` and `forbidden_atoms` against the data "
            f"(last error: {last_error})"
        )


hybiridization_map = {
    "S": 0, 'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5, 'UNSPECIFIED': -1
}
        
BASE_ATOM_VOCAB = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Cu",
    "Zn",
    "Ge",
    "As",
    "Se",
    "Br",
    "Sn",
    "I",
]

#TODO this does not appear to handle NaN in the data
class GraphDataset(torch_data.Dataset):

    # Compaction state.  Set by load_xyz/load_npy/load_db when compact flags
    # are passed.  Used by get_item to reconstruct full Data on the fly.
    _compact: Dict[str, bool] = {}
    _ohe_size: Optional[int] = None
    _edge_type: str = "fully_connected"

    def load_smiles(self):
        pass
    
    def load_csv(
        self,
        csv_file: str,
        xyz_dir: str,
        xyz_field: str = "xyz",
        smiles_field: str = "smiles",
        target_fields: Optional[List[str]] = None,
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[str] = None,
        forbidden_atoms: List[str] = [],
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        **kwargs,
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            xyz_dir (str): directory to store XYZ files
            xyz_field (str): name of the XYZ column in the table
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            atom_vocab (list of str, optional): atom types
            node_feature_choice (str, optional): geom features to extract
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        if xyz_field is None:
            raise ValueError("xyz_field must be provided")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            logger.info("atom vocabulary not provided, using defaul in constant.py")
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            xyzs = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == xyz_field:
                        xyz_path = os.path.join(xyz_dir, f"{value}.xyz")
                        xyzs.append(xyz_path)
                    elif field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
        assert len(xyzs) > 0, "No XYZ files found"
        # TODO to deal with when xyz but absence smiles and vice versa, skip it for now
        self.load_xyz(
            xyzs,
            smiles,
            targets,
            atom_vocab,
            forbidden_atoms=forbidden_atoms,
            node_feature_choice=node_feature_choice,
            verbose=verbose,
            allow_unknown=allow_unknown,
            use_ohe_feature=use_ohe_feature,
            **kwargs,
        )

    def load_xyz(
        self,
        xyz_list: List[str],
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        edge_type: str = "distance",
        radius: float = 4.0,
        n_neigh: int = 5,
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        compact: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ):
        """
        Load the dataset from XYZ and targets.

        Parameters:
            xyz_list (list of str): XYZ file names
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature_choice (str, optional): geom features to extract
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            edge_type (str, optional): type of edge to construct the graph (default: distance, neighbor)
            radius (float, optional): radius to construct the graph (default: 4.0)
            n_neigh (int, optional): number of neighbors to consider (default: 5)
            verbose (int, optional): output verbose level
            chunk_size (int, optional): flush to disk every N molecules to limit RAM.
            chunk_dir (str, optional): directory to write chunk files.
            **kwargs
        """

        num_sample = len(xyz_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            xyz_list = tqdm(xyz_list, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            logger.info("Hydrogen atoms are considered")
        else:
            logger.info("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.targets = defaultdict(list)
        self.atom_vocab = atom_vocab
        self.graph_data_list = []
        self.n_atoms = []
        self.smiles_list = []

        # compaction state
        self._compact = dict(compact) if compact else {}
        self._edge_type = edge_type
        self._ohe_size = None

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_pyg_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        last_error: Optional[Exception] = None
        for i, xyz in enumerate(xyz_list):
            try:
                if os.path.exists(xyz):

                    mol_xyz = PointCloud_Mol.from_xyz(
                        xyz, with_hydrogen, forbidden_atoms=forbidden_atoms
                    )
                    if mol_xyz is None:
                        skipped_forbidden += 1
                        continue
                    if len(mol_xyz.atoms) > max_atom:
                        skipped_too_large += 1
                        continue
                    coords = mol_xyz.get_coord()

                    if i < len(smiles_list):
                        smiles = smiles_list[i]
                    else:
                        logger.info("Cannot find smiles for %s", xyz)
                        smiles = None

                else:
                    logger.info(f"File {xyz} does not exist")
                    continue

                # Extract atom symbols and charges
                atom_symbols = [atom.element for atom in mol_xyz.atoms]
                charges = [atomic_numbers[atom.element]
                           for atom in mol_xyz.atoms
                           if atom.element in atomic_numbers]
                charges = torch.as_tensor(charges, dtype=torch.long)
                
                # Use NodeFeaturizer for all featurization
                featurizer = NodeFeaturizer(
                    atom_vocab=atom_vocab,
                    use_ohe=use_ohe_feature,
                    geom_feature=node_feature_choice,
                    allow_unknown=allow_unknown
                )
                node_features = featurizer.featurize_all(atom_symbols, charges, coords)

                charges = torch.as_tensor(charges, dtype=torch.long)

                n_nodes = len(mol_xyz.atoms)

                if edge_type == "distance":
                    edge_index = radius_graph(coords, r=radius)
                elif edge_type == "neighbor":
                    edge_index = knn_graph(coords, k=n_neigh)
                elif edge_type == "fully_connected":
                    num_nodes = coords.size(0)
                    row_ = torch.arange(num_nodes).repeat_interleave(num_nodes)
                    col = torch.arange(num_nodes).repeat(num_nodes)
                    edge_index = torch.stack([row_, col], dim=0)
                    edge_index = edge_index[:, row_ != col]  # Remove self-loops if needed
                else:
                    raise ValueError("Unknown edge type %s" % edge_type)

                if torch.isnan(coords).any() or (
                    node_features is not None and torch.isnan(node_features).any()
                ):
                    skipped_nan += 1
                    continue

                graph_data, _ohe_sz = _compact_build_pyg(
                    node_features=node_features,
                    coords=coords,
                    charges=charges,
                    n_nodes=n_nodes,
                    smiles=smiles,
                    xyz=xyz,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    mol_index=i,
                    compact=self._compact,
                )
                if _ohe_sz is not None and self._ohe_size is None:
                    self._ohe_size = _ohe_sz

                if _chunking:
                    _buf["graph_data_list"].append(graph_data)
                    _buf["n_atoms"].append(n_nodes)
                    _buf["smiles_list"].append(smiles)
                    for field in targets:
                        _buf["targets"][field].append(float(targets[field][i]))
                    _all_smiles.append(smiles)
                    _all_n_atoms.append(n_nodes)

                    if len(_buf["graph_data_list"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                        _chunk_paths.append(path)
                        _chunk_sizes.append(len(_buf["graph_data_list"]))
                        logger.info(f"Flushed pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")
                        _chunk_idx += 1
                        _buf = _empty_pyg_buf()
                else:
                    self.graph_data_list.append(graph_data)
                    self.n_atoms.append(n_nodes)
                    self.smiles_list.append(smiles)
                    for field in targets:
                        self.targets[field].append(float(targets[field][i]))

            except Exception as e:
                last_error = e
                logger.error(f"Error in loading {xyz}: {e}")
                continue

        if _chunking and _buf["graph_data_list"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["graph_data_list"]))
            logger.info(f"Flushed final pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")

        if _chunking:
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                list(targets.keys()), atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
                kind="pyg",
            )
            _augment_chunk_meta_pyg(chunk_dir, self._compact, self._ohe_size, edge_type)
            logger.info(
                f"Chunked pyG processing complete: {sum(_chunk_sizes)} graphs "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan:
            logger.warning(
                f"Discarded entries: {skipped_too_large} too large, "
                f"{skipped_forbidden} forbidden atoms, {skipped_nan} NaN values"
            )

        _check_any_loaded(
            "the XYZ file list", num_sample,
            sum(_chunk_sizes) if _chunking else len(self.graph_data_list),
            last_error,
        )

    def load_npy(
        self,
        coords: torch.Tensor,
        natoms: torch.Tensor,
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        edge_type: str = "distance",
        radius: float = 4.0,
        n_neigh: int = 5,
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        compact: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ):
        """
        Load the dataset from npy tensors.

        Parameters:
            coords (tensor): tensor of coordinates [total_atoms, 5] with [mol_idx, Z, x, y, z]
            natoms (tensor): tensor of number of atoms per molecule
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature_choice (str, optional): geom features to extract
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule
            with_hydrogen (bool, optional): whether to include hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            edge_type (str, optional): type of edge to construct the graph
            radius (float, optional): radius to construct the graph
            n_neigh (int, optional): number of neighbors to consider
            verbose (int, optional): output verbose level
            chunk_size (int, optional): flush to disk every N molecules to limit RAM.
            chunk_dir (str, optional): directory to write chunk files.
            **kwargs
        """
        num_sample = natoms.size(0)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            natoms = tqdm(natoms, "Constructing graphs from npy data")

        if with_hydrogen:
            logger.info("Hydrogen atoms are considered")
        else:
            logger.info("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.targets = defaultdict(list)
        self.atom_vocab = atom_vocab
        self.graph_data_list = []
        self.n_atoms = []
        self.smiles_list = []

        # compaction state
        self._compact = dict(compact) if compact else {}
        self._edge_type = edge_type
        self._ohe_size = None

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_pyg_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        last_error: Optional[Exception] = None
        start_index = 0
        for i, natom in enumerate(natoms):
            try:
                end_index = start_index + natom.item()
                molecule_data = coords[start_index:end_index, :]
                start_index = end_index

                if natom > max_atom:
                    skipped_too_large += 1
                    continue

                # Parse coords tensor: [mol_idx, Z, x, y, z]
                zs = molecule_data[:, 1].long()
                mol_coords = molecule_data[:, 2:5].float()

                mol_xyz = PointCloud_Mol.from_arrays(
                    zs, mol_coords, with_hydrogen, forbidden_atoms=forbidden_atoms
                )

                if mol_xyz is None:
                    skipped_forbidden += 1
                    continue

                coords_mol = mol_xyz.get_coord()

                if i < len(smiles_list):
                    smiles = smiles_list[i]
                else:
                    logger.info("Cannot find smiles for %s", i)
                    smiles = None

                # Extract atom symbols and charges
                atom_symbols = [atom.element for atom in mol_xyz.atoms]
                charges = [atomic_numbers[atom.element]
                           for atom in mol_xyz.atoms
                           if atom.element in atomic_numbers]
                charges = torch.as_tensor(charges, dtype=torch.long)
                
                # Use NodeFeaturizer for all featurization
                featurizer = NodeFeaturizer(
                    atom_vocab=atom_vocab,
                    use_ohe=use_ohe_feature,
                    geom_feature=node_feature_choice,
                    allow_unknown=allow_unknown
                )
                node_features = featurizer.featurize_all(atom_symbols, charges, coords_mol)

                n_nodes = len(mol_xyz.atoms)

                # Build edges
                if edge_type == "distance":
                    edge_index = radius_graph(coords_mol, r=radius)
                elif edge_type == "neighbor":
                    edge_index = knn_graph(coords_mol, k=n_neigh)
                elif edge_type == "fully_connected":
                    num_nodes = coords_mol.size(0)
                    row_ = torch.arange(num_nodes).repeat_interleave(num_nodes)
                    col_ = torch.arange(num_nodes).repeat(num_nodes)
                    edge_index = torch.stack([row_, col_], dim=0)
                    edge_index = edge_index[:, row_ != col_]  # Remove self-loops
                else:
                    raise ValueError("Unknown edge type %s" % edge_type)

                if torch.isnan(coords_mol).any() or (
                    node_features is not None and torch.isnan(node_features).any()
                ):
                    skipped_nan += 1
                    continue

                graph_data, _ohe_sz = _compact_build_pyg(
                    node_features=node_features,
                    coords=coords_mol,
                    charges=charges,
                    n_nodes=n_nodes,
                    smiles=smiles,
                    xyz=None,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    mol_index=i,
                    compact=self._compact,
                )
                if _ohe_sz is not None and self._ohe_size is None:
                    self._ohe_size = _ohe_sz

                if _chunking:
                    _buf["graph_data_list"].append(graph_data)
                    _buf["n_atoms"].append(n_nodes)
                    _buf["smiles_list"].append(smiles)
                    for field in targets:
                        _buf["targets"][field].append(float(targets[field][i]))
                    _all_smiles.append(smiles)
                    _all_n_atoms.append(n_nodes)

                    if len(_buf["graph_data_list"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                        _chunk_paths.append(path)
                        _chunk_sizes.append(len(_buf["graph_data_list"]))
                        logger.info(f"Flushed pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")
                        _chunk_idx += 1
                        _buf = _empty_pyg_buf()
                else:
                    self.graph_data_list.append(graph_data)
                    self.n_atoms.append(n_nodes)
                    self.smiles_list.append(smiles)
                    for field in targets:
                        self.targets[field].append(float(targets[field][i]))

            except Exception as e:
                last_error = e
                logger.error(f"Error in loading molecule {i}: {e}")
                continue

        if _chunking and _buf["graph_data_list"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["graph_data_list"]))
            logger.info(f"Flushed final pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")

        if _chunking:
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                list(targets.keys()), atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
                kind="pyg",
            )
            _augment_chunk_meta_pyg(chunk_dir, self._compact, self._ohe_size, edge_type)
            logger.info(
                f"Chunked pyG processing complete: {sum(_chunk_sizes)} graphs "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan:
            logger.warning(
                f"Discarded entries: {skipped_too_large} too large, "
                f"{skipped_forbidden} forbidden atoms, {skipped_nan} NaN values"
            )

        _check_any_loaded(
            "the npy coordinate arrays", num_sample,
            sum(_chunk_sizes) if _chunking else len(self.graph_data_list),
            last_error,
        )

    def load_db(
        self,
        db_path: str,
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[List[str]] = None,
        target_fields: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        edge_type: str = "distance",
        radius: float = 4.0,
        n_neigh: int = 5,
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        compact: Optional[Dict[str, bool]] = None,
        use_row_data_features: bool = False,
        **kwargs: Any,
    ):
        """
        Load the dataset from an ASE db file.

        Parameters:
            db_path (str): path to ASE db file
            atom_vocab (list of str, optional): atom types
            node_feature_choice (list of str, optional): RDKit atom features to extract
            target_fields (list of str, optional): name of target columns in the table.
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            edge_type (str, optional): type of edge to construct the graph (default: distance, neighbor)
            radius (float, optional): radius to construct the graph (default: 4.0)
            n_neigh (int, optional): number of neighbors to consider (default: 5)
            verbose (int, optional): output verbose level
            null_value (float, optional): null value for missing context data
            **kwargs
        """
        if Chem is None and node_feature_choice is not None:
            if not (isinstance(node_feature_choice, str) and node_feature_choice in ["geometric_heuristic", "atom_geom_heuristic", "geometric_fast", "geometric", "geometric_full", "geometric_hybrid"]):
                raise ImportError("RDKit is required for node_feature_choice. Please install it.")

        if not atom_vocab:
            atom_vocab = BASE_ATOM_VOCAB
            logger.info("atom vocabulary not provided, using default")

        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.targets = defaultdict(list)
        self.atom_vocab = atom_vocab
        self.graph_data_list = []
        self.n_atoms = []
        self.smiles_list = []

        # compaction state
        self._compact = dict(compact) if compact else {}
        self._edge_type = edge_type
        self._ohe_size = None  # populated from first molecule below

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_pyg_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        db_sources = _collect_db_sources(db_path)
        total_len = sum(source["length"] for source in db_sources)
        iterator = _iter_db_rows(db_sources)

        if verbose:
            iterator = tqdm(iterator, "Processing ASE db files", total=total_len)

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        skipped_mol_block = 0
        skipped_rdkit = 0
        skipped_atom_mismatch = 0
        skipped_empty = 0
        skipped_missing_target = 0
        failed = 0
        last_error: Optional[Exception] = None
        for i, row in enumerate(iterator):
            try:
                mol_ase = row.toatoms()
                row_data = getattr(row, "data", {}) or {}

                if target_fields:
                    missing_fields = [
                        f for f in target_fields
                        if f not in row_data or row_data[f] == ""
                    ]
                    if missing_fields:
                        skipped_missing_target += 1
                        if skipped_missing_target <= 5:
                            logger.warning(
                                f"row id={i}: missing target field(s) {missing_fields}, dropping entry"
                                + ("" if skipped_missing_target < 5 else " (further such warnings suppressed, see summary count at end)")
                            )
                        continue

                if any(atom.symbol in forbidden_atoms for atom in mol_ase):
                    skipped_forbidden += 1
                    continue

                # Hydrogen filtering happens before the max_atom check so
                # max_atom bounds the *stored* atom count, as in load_xyz.
                mol_ase, keep_mask = _drop_hydrogens(mol_ase, with_hydrogen)
                if len(mol_ase) == 0:
                    skipped_empty += 1
                    continue

                if len(mol_ase) > max_atom:
                    skipped_too_large += 1
                    continue

                coords = torch.from_numpy(mol_ase.get_positions()).to(torch.float32)
                charges = torch.from_numpy(mol_ase.get_atomic_numbers()).to(torch.long)
                n_nodes = len(mol_ase)

                atomic_symbols = mol_ase.get_chemical_symbols()

                # Create featurizer for OHE
                featurizer = NodeFeaturizer(
                    atom_vocab=atom_vocab,
                    use_ohe=use_ohe_feature,
                    geom_feature=None,  # geom handled separately for load_db
                    allow_unknown=allow_unknown
                )

                if use_ohe_feature:
                    node_features = featurizer.compute_ohe(atomic_symbols)
                else:
                    node_features = None

                mol_rdkit = None
                if node_feature_choice:
                    if isinstance(node_feature_choice, (list, tuple)) or hasattr(node_feature_choice, '__iter__') and not isinstance(node_feature_choice, str):
                        # List: use existing RDKit scalar logic (requires mol_block)
                        if "mol_block" not in row_data:
                            skipped_mol_block += 1
                            continue

                        mol_block = row_data.get('mol_block')
                        if isinstance(mol_block, bytes):
                            mol_block = mol_block.decode('utf-8')

                        mol_rdkit = Chem.MolFromMolBlock(mol_block, removeHs=False)
                        if not mol_rdkit:
                            skipped_rdkit += 1
                            continue

                        # Filter the RDKit atoms with the same mask used on
                        # the ASE atoms so the two stay aligned.
                        rdkit_atoms = list(mol_rdkit.GetAtoms())
                        if keep_mask is not None and len(rdkit_atoms) == len(keep_mask):
                            rdkit_atoms = [
                                a for a, k in zip(rdkit_atoms, keep_mask) if k
                            ]

                        ase_atomic_num = mol_ase.get_atomic_numbers()
                        rdkit_atomic_num = np.array([atom.GetAtomicNum() for atom in rdkit_atoms])
                        if not np.array_equal(ase_atomic_num, rdkit_atomic_num):
                            skipped_atom_mismatch += 1
                            continue

                        atom_feats = defaultdict(list)
                        for atom in rdkit_atoms:
                            atom_feats['degree'].append(atom.GetDegree())
                            atom_feats['formal_charge'].append(atom.GetFormalCharge())
                            atom_feats['hybridization'].append(hybiridization_map.get(str(atom.GetHybridization()), -1))
                            atom_feats['is_aromatic'].append(atom.GetIsAromatic())
                            atom_feats['valence'].append(atom.GetTotalValence())

                        node_features_extra = torch.tensor([
                            atom_feats[key] for key in node_feature_choice
                        ], dtype=torch.float32).T
                    elif isinstance(node_feature_choice, str):
                        # String: use NodeFeaturizer for geom features
                        geom_featurizer = NodeFeaturizer(
                            atom_vocab=atom_vocab,
                            use_ohe=False,
                            geom_feature=node_feature_choice,
                            allow_unknown=allow_unknown
                        )
                        node_features_extra = geom_featurizer.compute_geom(charges, coords)
                    else:
                        raise ValueError(
                            f"node_feature_choice must be str or list, got {type(node_feature_choice)}"
                        )
                    
                    if node_features is not None:
                        node_features = torch.cat((node_features, node_features_extra), dim=1)
                    else:
                        node_features = node_features_extra

                if use_row_data_features:
                    raw = (row_data or {}).get("node_features", None)
                    if raw is not None:
                        row_feat = torch.tensor(np.array(raw), dtype=torch.float32)
                        if row_feat.shape[0] == n_nodes:
                            if node_features is not None:
                                node_features = torch.cat((node_features, row_feat), dim=1)
                            else:
                                node_features = row_feat
                        elif verbose:
                            logger.warning(
                                f"row id={i}: row.data['node_features'] shape {row_feat.shape} "
                                f"mismatches n_nodes={n_nodes}, skipping."
                            )

                if torch.isnan(coords).any() or (node_features is not None and torch.isnan(node_features).any()):
                    skipped_nan += 1
                    continue

                smiles = Chem.MolToSmiles(mol_rdkit) if mol_rdkit else None

                if edge_type == "distance":
                    edge_index = radius_graph(coords, r=radius)
                elif edge_type == "neighbor":
                    edge_index = knn_graph(coords, k=n_neigh)
                elif edge_type == "fully_connected":
                    num_nodes = coords.size(0)
                    row_ = torch.arange(num_nodes).repeat_interleave(num_nodes)
                    col_ = torch.arange(num_nodes).repeat(num_nodes)
                    edge_index = torch.stack([row_, col_], dim=0)
                    edge_index = edge_index[:, row_ != col_]
                else:
                    raise ValueError(f"Unknown edge type {edge_type}")

                graph_data, _ohe_sz = _compact_build_pyg(
                    node_features=node_features,
                    coords=coords,
                    charges=charges,
                    n_nodes=n_nodes,
                    smiles=smiles,
                    xyz=f"db_entry_{i}",
                    edge_index=edge_index,
                    edge_type=edge_type,
                    mol_index=i,
                    compact=self._compact,
                )
                if _ohe_sz is not None and self._ohe_size is None:
                    self._ohe_size = _ohe_sz

                # Resolve target values once (used by both paths)
                resolved_targets: Dict[str, float] = {}
                if target_fields:
                    for field in target_fields:
                        # presence already guaranteed by the missing-target check above
                        value = row_data[field]
                        try:
                            value = utils.literal_eval(str(value))
                        except (ValueError, SyntaxError):
                            if isinstance(value, (np.ndarray, torch.Tensor)):
                                value = value.tolist()
                        if value == "":
                            value = math.nan
                        resolved_targets[field] = float(value)

                if _chunking:
                    _buf["graph_data_list"].append(graph_data)
                    _buf["n_atoms"].append(n_nodes)
                    _buf["smiles_list"].append(smiles)
                    for field, val in resolved_targets.items():
                        _buf["targets"][field].append(val)
                    _all_smiles.append(smiles)
                    _all_n_atoms.append(n_nodes)

                    if len(_buf["graph_data_list"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                        _chunk_paths.append(path)
                        _chunk_sizes.append(len(_buf["graph_data_list"]))
                        logger.info(f"Flushed pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")
                        _chunk_idx += 1
                        _buf = _empty_pyg_buf()
                else:
                    self.graph_data_list.append(graph_data)
                    self.n_atoms.append(n_nodes)
                    self.smiles_list.append(smiles)
                    for field, val in resolved_targets.items():
                        self.targets[field].append(val)

            except Exception as e:
                failed += 1
                last_error = e
                logger.error(f"Error in loading db entry {i}: {e}")
                continue

        if _chunking and _buf["graph_data_list"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["graph_data_list"]))
            logger.info(f"Flushed final pyG chunk {_chunk_idx} ({_chunk_sizes[-1]} graphs) → {path}")

        if _chunking:
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                list(target_fields) if target_fields else [],
                atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
                kind="pyg",
            )
            _augment_chunk_meta_pyg(chunk_dir, self._compact, self._ohe_size, edge_type)
            logger.info(
                f"Chunked pyG processing complete: {sum(_chunk_sizes)} graphs "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan or skipped_mol_block or skipped_rdkit or skipped_atom_mismatch or skipped_empty or skipped_missing_target or failed:
            logger.warning(
                f"Discarded entries: {skipped_too_large} too large, {skipped_forbidden} forbidden atoms, "
                f"{skipped_nan} NaN values, {skipped_mol_block} missing mol_block, "
                f"{skipped_rdkit} RDKit parse failures, {skipped_atom_mismatch} atom order mismatches, "
                f"{skipped_empty} empty after hydrogen removal, {skipped_missing_target} missing target field(s), "
                f"{failed} errors (last: {last_error})"
            )

        _check_any_loaded(
            db_path, total_len,
            sum(_chunk_sizes) if _chunking else len(self.graph_data_list),
            last_error,
        )

    def load_pickle(self, pkl_file, verbose=0):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        self.transform = None

        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample, tasks = pickle.load(fin)

            self.graph_data_list = []
            self.targets = defaultdict(list)
            self.n_atoms = []

            for task in tasks:
                self.targets[task] = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            
            skipped_too_large = 0
            for i in indexes:
                graph_data, values = pickle.load(fin) 
                
                if hasattr(self, 'max_atom') and self.max_atom > 0 and graph_data.natoms > self.max_atom:
                    skipped_too_large += 1
                    continue
                    
                self.graph_data_list.append(graph_data)
                self.n_atoms.append(graph_data.natoms)
                for task, value in zip(tasks, values):
                    self.targets[task].append(float(value))
            trailer = pickle.load(fin)
            self.atom_vocab, self.with_hydrogen = trailer[0], trailer[1]
            # ponytail: 3rd slot added later; pickles written before it have none.
            compact_meta = trailer[2] if len(trailer) > 2 else {}
            self._compact = compact_meta.get("compact") or {}
            self._ohe_size = compact_meta.get("ohe_size")
            self._edge_type = compact_meta.get("edge_type", "fully_connected")


            if skipped_too_large:
                logger.warning(f"Discarded {skipped_too_large} entries because they exceed max_atom limit")

    def save_pickle(self, pkl_file, verbose=0):
        """
        Save the dataset to a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        os.makedirs(os.path.dirname(os.path.abspath(pkl_file)), exist_ok=True)
        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.graph_data_list)
            tasks = list(self.targets.keys())
            pickle.dump((num_sample, tasks), fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                values = [v[i] for v in self.targets.values()]
                pickle.dump(
                    (
                        self.graph_data_list[i],
                        values,
                    ),
                    fout,
                )
            pickle.dump(
                (
                    self.atom_vocab,
                    self.with_hydrogen,
                    # Without this, a compacted dataset reloads as slim Data and
                    # get_item silently hands the model packed ints, not OHE.
                    {
                        "compact": self._compact,
                        "ohe_size": self._ohe_size,
                        "edge_type": self._edge_type,
                    },
                ),
                fout,
            )

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
        # item = {"Point Cloud": self.data[index]}

        item = {k: v[index] for k, v in self.targets.items()}
        graph = self.graph_data_list[index]
        if _compact_is_active(self._compact):
            graph = _compact_expand_pyg(
                graph, self._compact, self._ohe_size, self._edge_type, index,
            )
        item["graph"] = graph
        if self.transform:
            item = self.transform(item)
        return item
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    def atom_types(self):
        """All atom types."""
        atom_types = set()
        for symbol in self.atom_vocab:
            atom_types.add(atomic_numbers[symbol])
        if 0 in atom_types:
            atom_types.discard(0)
        atom_types = sorted(atom_types)
        return atom_types

    def __len__(self):
        return len(self.graph_data_list)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: %d" % len(self.tasks),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    def get_property(self, task):
        if len(list(self.targets.keys())) == 0:
            return None
        else:
            prop = torch.tensor(self.targets[task], dtype=torch.float32)
            return prop

    @property
    def num_atoms(self):
        """Number of atoms in each molecule."""
        num_atoms = torch.tensor(self.n_atoms, dtype=torch.long)

        return num_atoms
    
    
class PointCloudDataset(torch_data.Dataset):

    def load_xyz(
        self,
        xyz_list: List[str],
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        compact: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ):
        """
        Load the dataset from XYZ and targets.

        Parameters:
            xyz_list (list of str): XYZ file names
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature_choice (str, optional): geom features to extract
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            pad_data (bool, optional): whether to pad data to max_atom
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            chunk_size (int, optional): flush to disk every N molecules to limit RAM.
            chunk_dir (str, optional): directory to write chunk files.
            **kwargs
        """

        num_sample = len(xyz_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            xyz_list = tqdm(xyz_list, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            logger.info("Hydrogen atoms are considered")
        else:
            logger.info("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []
        self.atom_vocab = atom_vocab

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        last_error: Optional[Exception] = None
        for i, xyz in enumerate(xyz_list):
            try:
                if os.path.exists(xyz):
                    mol_xyz = PointCloud_Mol.from_xyz(
                        xyz, with_hydrogen, forbidden_atoms=forbidden_atoms
                    )
                    if mol_xyz is None:
                        skipped_forbidden += 1
                        continue
                    if len(mol_xyz.atoms) > max_atom:
                        skipped_too_large += 1
                        continue
                    coords = mol_xyz.get_coord()
                    smiles = smiles_list[i] if i < len(smiles_list) else None
                else:
                    logger.info(f"File {xyz} does not exist")
                    continue

                atom_symbols = [atom.element for atom in mol_xyz.atoms]
                charges = [atomic_numbers[atom.element]
                           for atom in mol_xyz.atoms
                           if atom.element in atomic_numbers]
                charges = torch.as_tensor(charges, dtype=torch.long)

                featurizer = NodeFeaturizer(
                    atom_vocab=atom_vocab,
                    use_ohe=True,
                    geom_feature=node_feature_choice,
                    allow_unknown=allow_unknown
                )
                node_features = featurizer.featurize_all(atom_symbols, charges, coords)

                n_nodes = len(mol_xyz.atoms)
                node_mask = torch.ones(n_nodes, dtype=torch.int8)

                if pad_data:
                    coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                    charges_mask = torch.zeros(max_atom, dtype=torch.long)
                    node_mask = torch.zeros(max_atom, dtype=torch.int8)
                    coords_full[:n_nodes] = coords
                    node_mask[:n_nodes] = 1
                    node_feat_full = torch.zeros(max_atom, node_features.size(1))
                    node_feat_full[:n_nodes] = node_features
                    charges_mask[:n_nodes] = charges
                    coords = coords_full
                    node_features = node_feat_full
                    charges = charges_mask
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
                else:
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
                edge_mask *= diag_mask

                if torch.isnan(coords).any() or (node_features is not None and torch.isnan(node_features).any()):
                    skipped_nan += 1
                    continue

                if _chunking:
                    _buf["coords_list"].append(coords)
                    _buf["node_mask_list"].append(node_mask)
                    _buf["edge_mask_list"].append(edge_mask)
                    _buf["node_feature_list"].append(node_features)
                    _buf["charges_list"].append(charges)
                    _buf["xyzs"].append(xyz)
                    _buf["n_atoms"].append(n_nodes)
                    _buf["smiles_list"].append(smiles)
                    for field in targets:
                        _buf["targets"][field].append(float(targets[field][i]))
                    _all_smiles.append(smiles)
                    _all_n_atoms.append(n_nodes)

                    if len(_buf["xyzs"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                        _chunk_paths.append(path)
                        _chunk_sizes.append(len(_buf["xyzs"]))
                        logger.info(f"Flushed chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")
                        _chunk_idx += 1
                        _buf = _empty_buf()
                else:
                    self.coords_list.append(coords)
                    self.n_atoms.append(n_nodes)
                    self.node_mask_list.append(node_mask)
                    self.edge_mask_list.append(edge_mask)
                    self.node_feature_list.append(node_features)
                    self.charges_list.append(charges)
                    self.smiles_list.append(smiles)
                    self.xyzs.append(xyz)
                    for field in targets:
                        self.targets[field].append(float(targets[field][i]))

            except Exception as e:
                last_error = e
                logger.error(f"Error in loading {xyz}: {e}")
                continue

        if _chunking and _buf["xyzs"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["xyzs"]))
            logger.info(f"Flushed final chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")

        if _chunking:
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                list(targets.keys()), atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
            )
            logger.info(
                f"Chunked processing complete: {sum(_chunk_sizes)} molecules "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan:
            logger.warning(f"Discarded entries: {skipped_too_large} too large, {skipped_forbidden} forbidden atoms, {skipped_nan} NaN values")

        _check_any_loaded(
            "the XYZ file list", num_sample,
            sum(_chunk_sizes) if _chunking else len(self.coords_list),
            last_error,
        )

    def load_npy(
        self,
        coords: torch.Tensor,
        natoms: torch.Tensor,
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Load the dataset from npy and targets.

        Parameters:
            coords (tensor): tensor of coordinates
            natoms (tensor): tensor of number of atoms
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature_choice (str, optional): geom features to extract
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            pad_data (bool, optional): whether to pad data to max_atom
            verbose (int, optional): output verbose level
            chunk_size (int, optional): flush to disk every N molecules to limit RAM.
            chunk_dir (str, optional): directory to write chunk files.
            **kwargs
        """
        num_sample = natoms.size(0)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            natoms = tqdm(natoms, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            logger.info("Hydrogen atoms are considered")
        else:
            logger.info("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []
        self.atom_vocab = atom_vocab

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        start_index = 0
        for mol_i, natom in enumerate(natoms):
            end_index = start_index + natom.item()
            molecule_data = coords[start_index:end_index, :]
            start_index = end_index

            if natom > max_atom:
                skipped_too_large += 1
                continue

            zs = torch.zeros(natom, dtype=torch.long)
            coord = torch.zeros((natom, 3), dtype=torch.float32)
            for j, row in enumerate(molecule_data):
                zs[j] = int(row[1])
                coord[j] = row[2:]

            mol_xyz = PointCloud_Mol.from_arrays(
                zs, coord, with_hydrogen, forbidden_atoms=forbidden_atoms
            )
            if mol_xyz is None:
                skipped_forbidden += 1
                continue

            coords_mol = mol_xyz.get_coord()
            smiles = smiles_list[mol_i] if mol_i < len(smiles_list) else None

            atom_symbols = [atom.element for atom in mol_xyz.atoms]
            charges = [atomic_numbers[atom.element]
                       for atom in mol_xyz.atoms
                       if atom.element in atomic_numbers]
            charges = torch.as_tensor(charges, dtype=torch.long)

            featurizer = NodeFeaturizer(
                atom_vocab=atom_vocab,
                use_ohe=True,
                geom_feature=node_feature_choice,
                allow_unknown=allow_unknown
            )
            node_features = featurizer.featurize_all(atom_symbols, charges, coords_mol)

            n_nodes = len(mol_xyz.atoms)
            node_mask = torch.ones(n_nodes, dtype=torch.int8)

            if pad_data:
                coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                charges_mask = torch.zeros(max_atom, dtype=torch.long)
                node_mask = torch.zeros(max_atom, dtype=torch.int8)
                coords_full[:n_nodes] = coords_mol
                node_mask[:n_nodes] = 1
                node_feat_full = torch.zeros(max_atom, node_features.size(1))
                node_feat_full[:n_nodes] = node_features
                charges_mask[:n_nodes] = charges
                coords_mol = coords_full
                node_features = node_feat_full
                charges = charges_mask
                edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
            else:
                edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
            edge_mask *= diag_mask

            if torch.isnan(coords_mol).any() or (node_features is not None and torch.isnan(node_features).any()):
                skipped_nan += 1
                continue

            if _chunking:
                _buf["coords_list"].append(coords_mol)
                _buf["node_mask_list"].append(node_mask)
                _buf["edge_mask_list"].append(edge_mask)
                _buf["node_feature_list"].append(node_features)
                _buf["charges_list"].append(charges)
                _buf["xyzs"].append(mol_i)
                _buf["n_atoms"].append(n_nodes)
                _buf["smiles_list"].append(smiles)
                for field in targets:
                    _buf["targets"][field].append(float(targets[field][mol_i]))
                _all_smiles.append(smiles)
                _all_n_atoms.append(n_nodes)

                if len(_buf["xyzs"]) >= chunk_size:
                    path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                    _chunk_paths.append(path)
                    _chunk_sizes.append(len(_buf["xyzs"]))
                    logger.info(f"Flushed chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")
                    _chunk_idx += 1
                    _buf = _empty_buf()
            else:
                self.coords_list.append(coords_mol)
                self.n_atoms.append(n_nodes)
                self.node_mask_list.append(node_mask)
                self.edge_mask_list.append(edge_mask)
                self.node_feature_list.append(node_features)
                self.charges_list.append(charges)
                self.smiles_list.append(smiles)
                self.xyzs.append(mol_i)
                for field in targets:
                    self.targets[field].append(float(targets[field][mol_i]))

        if _chunking and _buf["xyzs"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["xyzs"]))
            logger.info(f"Flushed final chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")

        if _chunking:
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                list(targets.keys()), atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
            )
            logger.info(
                f"Chunked processing complete: {sum(_chunk_sizes)} molecules "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan:
            logger.warning(f"Discarded entries: {skipped_too_large} too large, {skipped_forbidden} forbidden atoms, {skipped_nan} NaN values")

        _check_any_loaded(
            "the npy coordinate arrays", num_sample,
            sum(_chunk_sizes) if _chunking else len(self.coords_list),
            None,
        )

    def load_csv(
        self,
        csv_file,
        xyz_dir,
        xyz_field="xyz",
        smiles_field="smiles",
        target_fields=None,
        atom_vocab=[],
        node_feature_choice=None,
        forbidden_atoms=[],
        null_value=math.nan,
        verbose=0,
        allow_unknown=False,
        use_ohe_feature=True,
        **kwargs,
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            xyz_dir (str): directory to store XYZ files
            xyz_field (str): name of the XYZ column in the table
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            atom_vocab (list of str, optional): atom types
            node_feature_choice (str, optional): geom features to extract
            forbidden_atoms (list of str, optional): forbidden atoms
            null_value (str, optional): null value for missing targets
            verbose (int, optional): output verbose level
            **kwargs
        """
        self.null_value = null_value
        if target_fields is not None:
            target_fields = set(target_fields)

        if xyz_field is None:
            raise ValueError("xyz_field must be provided")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            logger.info("atom vocabulary not provided, using defaul in constant.py")
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            xyzs = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == xyz_field:
                        xyz_path = os.path.join(xyz_dir, f"{value}.xyz")
                        xyzs.append(xyz_path)
                    elif field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = self.null_value
                        targets[field].append(value)
        assert len(xyzs) > 0, "No XYZ files found"
        # TODO to deal with when xyz but absence smiles and vice versa, skip it for now
        self.load_xyz(
            xyzs,
            smiles,
            targets,
            atom_vocab,
            forbidden_atoms=forbidden_atoms,
            node_feature_choice=node_feature_choice,
            verbose=verbose,
            allow_unknown=allow_unknown,
            use_ohe_feature=use_ohe_feature,
            **kwargs,
        )

    def load_db(
        self,
        db_path: str,
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[List[str]] = None,
        target_fields: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        null_value=math.nan,
        allow_unknown: bool = False,
        use_ohe_feature: bool = True,
        chunk_size: Optional[int] = None,
        chunk_dir: Optional[str] = None,
        use_row_data_features: bool = False,
        **kwargs: Any,
    ):
        """
        Load the dataset from an ASE db file.

        Parameters:
            db_path (str): path to ASE db file
            atom_vocab (list of str, optional): atom types
            node_feature_choice (list of str, optional): RDKit atom features to extract
            target_fields (list of str, optional): name of target columns in the table.
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            pad_data (bool, optional): whether to pad data to max_atom)
            verbose (int, optional): output verbose level
            null_value (float, optional): null value for missing context data
            chunk_size (int, optional): flush to disk every N molecules to limit RAM.
            chunk_dir (str, optional): directory to write chunk files.
            **kwargs
        """
        if Chem is None and node_feature_choice is not None:
            raise ImportError("RDKit is required for node_feature_choice. Please install it.")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            logger.info("atom vocabulary not provided, using default")

        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []
        self.atom_vocab = atom_vocab
        self.null_value = null_value

        # chunking state
        _chunking = chunk_size is not None and chunk_dir is not None
        _buf = _empty_buf() if _chunking else None
        _chunk_paths: List[str] = []
        _chunk_sizes: List[int] = []
        _all_smiles: List[str] = []
        _all_n_atoms: List[int] = []
        _chunk_idx = 0

        db_sources = _collect_db_sources(db_path)

        if verbose:
            logger.info(f"Found {len(db_sources)} database sources to load:")
            for source in db_sources:
                logger.info(
                    f"  - {source['path']} ({source['kind']}, {source['length']} rows)"
                )

        total_len = sum(source["length"] for source in db_sources)
        iterator = _iter_db_rows(db_sources)

        if verbose:
            iterator = tqdm(iterator, "Processing ASE db files", total=total_len)

        skipped_too_large = 0
        skipped_forbidden = 0
        skipped_nan = 0
        skipped_mol_block = 0
        skipped_rdkit = 0
        skipped_atom_mismatch = 0
        skipped_empty = 0
        skipped_missing_target = 0
        failed = 0
        last_error: Optional[Exception] = None
        for i, row in enumerate(iterator):
            try:
                mol_ase = row.toatoms()
                row_data = getattr(row, "data", {}) or {}

                if target_fields:
                    missing_fields = [
                        f for f in target_fields
                        if f not in row_data or row_data[f] == ""
                    ]
                    if missing_fields:
                        skipped_missing_target += 1
                        if skipped_missing_target <= 5:
                            logger.warning(
                                f"row id={i}: missing target field(s) {missing_fields}, dropping entry"
                                + ("" if skipped_missing_target < 5 else " (further such warnings suppressed, see summary count at end)")
                            )
                        continue

                if any(atom.symbol in forbidden_atoms for atom in mol_ase):
                    skipped_forbidden += 1
                    continue

                # Hydrogen filtering happens before the max_atom check so
                # max_atom bounds the *stored* atom count, as in load_xyz.
                mol_ase, keep_mask = _drop_hydrogens(mol_ase, with_hydrogen)
                if len(mol_ase) == 0:
                    skipped_empty += 1
                    continue

                if len(mol_ase) > max_atom:
                    skipped_too_large += 1
                    continue

                coords = torch.from_numpy(mol_ase.get_positions()).to(torch.float32)
                charges = torch.from_numpy(mol_ase.get_atomic_numbers()).to(torch.long)
                n_nodes = len(mol_ase)

                atomic_symbols = mol_ase.get_chemical_symbols()

                featurizer = NodeFeaturizer(
                    atom_vocab=atom_vocab,
                    use_ohe=True,
                    geom_feature=None,
                    allow_unknown=allow_unknown
                )
                node_features = featurizer.compute_ohe(atomic_symbols)

                mol_rdkit = None

                if node_feature_choice:
                    if isinstance(node_feature_choice, (list, tuple)) or hasattr(node_feature_choice, '__iter__') and not isinstance(node_feature_choice, str):
                        if "mol_block" not in row_data:
                            skipped_mol_block += 1
                            continue

                        mol_block = row_data.get('mol_block')
                        if isinstance(mol_block, bytes):
                            mol_block = mol_block.decode('utf-8')

                        if mol_block is None:
                            skipped_mol_block += 1
                            continue

                        mol_rdkit = Chem.MolFromMolBlock(mol_block, removeHs=False)
                        if not mol_rdkit:
                            skipped_rdkit += 1
                            continue

                        # Filter the RDKit atoms with the same mask used on
                        # the ASE atoms so the two stay aligned.
                        rdkit_atoms = list(mol_rdkit.GetAtoms())
                        if keep_mask is not None and len(rdkit_atoms) == len(keep_mask):
                            rdkit_atoms = [
                                a for a, k in zip(rdkit_atoms, keep_mask) if k
                            ]

                        ase_atomic_num = mol_ase.get_atomic_numbers()
                        rdkit_atomic_num = np.array([atom.GetAtomicNum() for atom in rdkit_atoms])
                        if not np.array_equal(ase_atomic_num, rdkit_atomic_num):
                            skipped_atom_mismatch += 1
                            continue

                        atom_feats = defaultdict(list)
                        for atom in rdkit_atoms:
                            atom_feats['degree'].append(atom.GetDegree())
                            atom_feats['formal_charge'].append(atom.GetFormalCharge())
                            atom_feats['hybridization'].append(hybiridization_map.get(str(atom.GetHybridization()), -1))
                            atom_feats['is_aromatic'].append(atom.GetIsAromatic())
                            atom_feats['valence'].append(atom.GetTotalValence())

                        node_features_extra = torch.tensor([
                            atom_feats[key] for key in node_feature_choice
                        ], dtype=torch.float32).T
                    elif isinstance(node_feature_choice, str):
                        geom_featurizer = NodeFeaturizer(
                            atom_vocab=atom_vocab,
                            use_ohe=False,
                            geom_feature=node_feature_choice,
                            allow_unknown=allow_unknown
                        )
                        node_features_extra = geom_featurizer.compute_geom(charges, coords)
                    else:
                        raise ValueError(
                            f"node_feature_choice must be str or list, got {type(node_feature_choice)}"
                        )

                    if node_features is not None:
                        node_features = torch.cat((node_features, node_features_extra), dim=1)
                    else:
                        node_features = node_features_extra

                if use_row_data_features:
                    raw = (row_data or {}).get("node_features", None)
                    if raw is not None:
                        row_feat = torch.tensor(np.array(raw), dtype=torch.float32)
                        if row_feat.shape[0] == n_nodes:
                            if node_features is not None:
                                node_features = torch.cat((node_features, row_feat), dim=1)
                            else:
                                node_features = row_feat
                        elif verbose:
                            logger.warning(
                                f"row id={i}: row.data['node_features'] shape {row_feat.shape} "
                                f"mismatches n_nodes={n_nodes}, skipping."
                            )

                node_mask = torch.ones(n_nodes, dtype=torch.int8)

                if pad_data:
                    coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                    charges_mask = torch.zeros(max_atom, dtype=torch.long)
                    node_mask = torch.zeros(max_atom, dtype=torch.int8)
                    coords_full[:n_nodes] = coords
                    node_mask[:n_nodes] = 1
                    node_feat_full = torch.zeros(max_atom, node_features.size(1), dtype=torch.float32)
                    node_feat_full[:n_nodes] = node_features
                    charges_mask[:n_nodes] = charges
                    coords = coords_full
                    node_features = node_feat_full
                    charges = charges_mask
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
                    edge_mask *= diag_mask
                else:
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
                    edge_mask *= diag_mask

                if torch.isnan(coords).any() or (node_features is not None and torch.isnan(node_features).any()):
                    skipped_nan += 1
                    continue

                smiles = Chem.MolToSmiles(mol_rdkit) if mol_rdkit else None

                if _chunking:
                    _buf["coords_list"].append(coords)
                    _buf["node_mask_list"].append(node_mask)
                    _buf["edge_mask_list"].append(edge_mask)
                    _buf["node_feature_list"].append(node_features)
                    _buf["charges_list"].append(charges)
                    _buf["xyzs"].append(f"db_entry_{i}")
                    _buf["n_atoms"].append(n_nodes)
                    _buf["smiles_list"].append(smiles)
                    if target_fields:
                        for field in target_fields:
                            # presence already guaranteed by the missing-target check above
                            value = row_data[field]
                            try:
                                value = utils.literal_eval(str(value))
                            except (ValueError, SyntaxError):
                                if isinstance(value, (np.ndarray, torch.Tensor)):
                                    value = value.tolist()
                            _buf["targets"][field].append(float(value))
                    _all_smiles.append(smiles)
                    _all_n_atoms.append(n_nodes)

                    if len(_buf["xyzs"]) >= chunk_size:
                        path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
                        _chunk_paths.append(path)
                        _chunk_sizes.append(len(_buf["xyzs"]))
                        logger.info(f"Flushed chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")
                        _chunk_idx += 1
                        _buf = _empty_buf()
                else:
                    self.coords_list.append(coords)
                    self.node_mask_list.append(node_mask)
                    self.edge_mask_list.append(edge_mask)
                    self.node_feature_list.append(node_features)
                    self.charges_list.append(charges)
                    self.xyzs.append(f"db_entry_{i}")
                    self.n_atoms.append(n_nodes)
                    if target_fields:
                        for field in target_fields:
                            # presence already guaranteed by the missing-target check above
                            value = row_data[field]
                            try:
                                value = utils.literal_eval(str(value))
                            except (ValueError, SyntaxError):
                                if isinstance(value, (np.ndarray, torch.Tensor)):
                                    value = value.tolist()
                            self.targets[field].append(float(value))
                    # Append unconditionally (`smiles` is None without RDKit) so
                    # smiles_list stays index-aligned with coords_list.
                    self.smiles_list.append(smiles)

            except Exception as e:
                failed += 1
                last_error = e
                logger.error(f"Error in loading db entry {i}: {e}")
                continue

        if _chunking and _buf["xyzs"]:
            path = _flush_chunk(chunk_dir, _chunk_idx, _buf)
            _chunk_paths.append(path)
            _chunk_sizes.append(len(_buf["xyzs"]))
            logger.info(f"Flushed final chunk {_chunk_idx} ({len(_buf['xyzs'])} molecules) → {path}")

        if _chunking:
            tasks = list(target_fields) if target_fields else []
            _write_chunk_meta(
                chunk_dir, _chunk_paths, _chunk_sizes,
                tasks, atom_vocab, with_hydrogen,
                _all_smiles, _all_n_atoms,
            )
            logger.info(
                f"Chunked processing complete: {sum(_chunk_sizes)} molecules "
                f"in {len(_chunk_paths)} chunks → {chunk_dir}"
            )

        if skipped_too_large or skipped_forbidden or skipped_nan or skipped_mol_block or skipped_rdkit or skipped_atom_mismatch or skipped_empty or skipped_missing_target or failed:
            logger.warning(
                f"Discarded entries: {skipped_too_large} too large, {skipped_forbidden} forbidden atoms, "
                f"{skipped_nan} NaN values, {skipped_mol_block} missing mol_block, "
                f"{skipped_rdkit} RDKit parse failures, {skipped_atom_mismatch} atom order mismatches, "
                f"{skipped_empty} empty after hydrogen removal, {skipped_missing_target} missing target field(s), "
                f"{failed} errors (last: {last_error})"
            )

        _check_any_loaded(
            db_path, total_len,
            sum(_chunk_sizes) if _chunking else len(self.coords_list),
            last_error,
        )

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):

        item = {k: v[index] for k, v in self.targets.items()}
        item.update({"coords": self.coords_list[index]})
        item.update({"node_mask": self.node_mask_list[index]})
        item.update({"edge_mask": self.edge_mask_list[index]})
        item.update({"node_feature": self.node_feature_list[index]})
        item.update({"charges": self.charges_list[index]})
        item.update({"natoms": self.n_atoms[index]})
        item.update({"xyz": self.xyzs[index]})
        if self.transform:
            item = self.transform(item)
        return item

    def load_pickle(self, pkl_file, verbose=0, cheap_data=False):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        self.transform = None

        if cheap_data:
            float_dtype = torch.float16
            long_dtype = torch.int16
            mask_dtype = torch.int8
        else:
            float_dtype = torch.float32
            long_dtype = torch.long
            mask_dtype = torch.int8

        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample, tasks = pickle.load(fin)

            self.xyzs = []
            self.smiles_list = []
            self.coords_list = []
            self.node_mask_list = []
            self.edge_mask_list = []
            self.node_feature_list = []
            self.charges_list = []
            self.targets = defaultdict(list)
            self.n_atoms = []

            for task in tasks:
                self.targets[task] = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            # To discard nmax
            skipped_too_large = 0
            for i in indexes:
                (
                    natom,
                    coord,
                    node_mask,
                    edge_mask,
                    node_feature,
                    charge,
                    xyz,
                    values,
                ) = pickle.load(fin)

                if natom > self.max_atom:
                    skipped_too_large += 1
                    continue
                else:
                    if cheap_data:
                        coord = torch.tensor(coord, dtype=float_dtype)
                        node_mask = torch.tensor(node_mask, dtype=mask_dtype)
                        edge_mask = torch.tensor(edge_mask, dtype=mask_dtype)
                        node_feature = torch.tensor(node_feature, dtype=float_dtype)
                        charge = torch.tensor(charge, dtype=long_dtype)
                        xyz = None
                    self.n_atoms.append(natom)
                    self.coords_list.append(coord)
                    self.node_mask_list.append(node_mask)
                    self.edge_mask_list.append(edge_mask)
                    self.node_feature_list.append(node_feature)
                    self.charges_list.append(charge)
                    self.xyzs.append(xyz)
                    for task, value in zip(tasks, values):
                        self.targets[task].append(float(value))
            self.smiles_list, self.atom_vocab, self.with_hydrogen = pickle.load(fin)
            if skipped_too_large:
                logger.warning(f"Discarded entries: {skipped_too_large} too large")

    def save_pickle(self, pkl_file, verbose=0, cheap_data=False):
        """
        Save the dataset to a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        os.makedirs(os.path.dirname(os.path.abspath(pkl_file)), exist_ok=True)

        if cheap_data:
            float_dtype = torch.float16
            long_dtype = torch.int16
            mask_dtype = torch.int8
            self.xyzs = [None] * len(self.xyzs)
        else:
            float_dtype = torch.float32
            long_dtype = torch.long
            mask_dtype = torch.int8

        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.xyzs)
            tasks = list(self.targets.keys())
            pickle.dump((num_sample, tasks), fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                values = [v[i] for v in self.targets.values()]
                pickle.dump(
                    (
                        self.n_atoms[i],
                        self.coords_list[i].to(float_dtype),
                        self.node_mask_list[i].to(mask_dtype),
                        self.edge_mask_list[i].to(mask_dtype),
                        self.node_feature_list[i].to(float_dtype),
                        self.charges_list[i].to(long_dtype),
                        self.xyzs[i],
                        values,
                    ),
                    fout,
                )
            pickle.dump(
                (
                    self.smiles_list,
                    self.atom_vocab,
                    self.with_hydrogen,
                ),
                fout,
            )
            
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    def atom_types(self):
        """All atom types."""

        # if len(self.smiles_list) == 0:
        #     raise ValueError(
        #         "No SMILES available in the dataset or not yet converted from XYZ."
        #     )
        atom_types = set()
        for symbol in self.atom_vocab:
            atom_types.add(atomic_numbers[symbol])
        if 0 in atom_types:
            atom_types.discard(0)
        atom_types = sorted(atom_types)
        return atom_types

    @property
    def num_atoms(self):
        """Number of atoms in each molecule."""
        num_atoms = torch.tensor(self.n_atoms, dtype=torch.long)

        return num_atoms

    # property
    def get_property(self, task):
        if len(list(self.targets.keys())) == 0:
            return None
        else:
            prop = torch.tensor(self.targets[task], dtype=torch.float32)
            return prop

    def get_tabasco_stats(self):
        """
        Get dataset statistics required for TABASCO unconditional sampling.
        
        Returns:
            Dictionary with:
                - max_atoms: Maximum number of atoms in dataset
                - num_atom_types: Number of atom types in vocabulary
                - atom_count_histogram: Histogram of molecule sizes
                - all_smiles: List of all SMILES strings
        """
        from collections import Counter
        
        atom_counts = Counter(self.n_atoms)
        
        return {
            'max_atoms': max(self.n_atoms) if self.n_atoms else 0,
            'num_atom_types': len(self.atom_vocab),
            'atom_count_histogram': dict(atom_counts),
            'all_smiles': getattr(self, 'smiles_list', [])
        }

    def __len__(self):
        return len(self.xyzs)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: %d" % len(self.tasks),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
