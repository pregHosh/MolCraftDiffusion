import os
import time
import logging
import torch
from MolecularDiffusion.data.dataset import pointcloud_dataset, pointcloud_dataset_pyG
from MolecularDiffusion.data.datasets.dataset import (
    PointCloudDataset, LazyChunkedDataset, LazyChunkedGraphDataset,
)
from MolecularDiffusion.data.datasets.graph3d_dataset import (
    Graph3DDataset, graph3d_dense_collate,
)
from MolecularDiffusion.data.dataloader import pointcloud_collate, graph_collate, pointcloud_collate_v0
from MolecularDiffusion.utils import get_vram_size

logger = logging.getLogger(__name__)


class DatasetConfigDict:
    """Picklable callable wrapper for dataset metadata."""

    def __init__(self, with_hydrogen, node_feature, max_atom):
        self.config_dict = {
            "with_hydrogen": with_hydrogen,
            "atom_feature": node_feature,
            "max_atom": max_atom,
        }

    def __call__(self):
        return self.config_dict


class DataModule:
    """
    DataModule to load, optionally save/load pickle, and split datasets for diffusion or predictive tasks.

    Usage:
        module = DataModule(
            filename="data.pkl",
            task_type="diffusion",
            atom_vocab=atom_vocab_list,
            with_hydrogen=True,
            node_feature_choice="geom",
            max_atom=50,
            xyz_dir="xyz/",
            coord_file="coords.csv",
            natoms_file="natoms.csv",
            ase_db_path=None,
            forbidden_atom=None,
            data_efficient_collator=False,
            train_ratio=0.8,
            load_pkl=None,            # path to load dataset pickle
            save_pkl="cached_dataset.pkl"  # path to save dataset pickle
        )
        module.load()
        train_ds, valid_ds, test_ds = module.train_set, module.valid_set, module.test_set
    """
    def __init__(
        self,
        root: str,
        filename: str,
        task_type: str,
        atom_vocab: list,
        with_hydrogen: bool,
        node_feature_choice = None,
        max_atom: int = 200,
        target_fields: list = None,

        xyz_dir: str = None,
        coord_file: str = None,
        natoms_file: str = None,
        ase_db_path: str = None,
        forbidden_atom: list = None,
        data_efficient_collator: bool = False,
        train_ratio: float = 0.8,
        load_pkl: str = None,
        save_pkl: str = None,
        data_type: str = "pointcloud",
        allow_unknown: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_name: str = "suisei",
        edge_type: str = "fully_connected",
        radius: float = 4.0,
        n_neigh: int = 5,
        use_ohe_feature: bool = True,
        chunk_size: int = None,
        compact_drop_edge_index: bool = False,
        compact_drop_tags: bool = False,
        compact_pack_ohe: bool = False,
        compact_int8_z: bool = False,
        use_row_data_features: bool = False,
        # --- graph3d (3D molecular graph with bonds) options; ignored by other data_types
        bond_collate: str = "raw",
        center_coords: bool = False,
        graph3d_stats: bool = True,
        kekulize: bool = False,
        use_stored_split: bool = False,
    ):
        self.root = root
        self.filename = filename
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.with_hydrogen = with_hydrogen
        self.node_feature_choice = node_feature_choice
        self.max_atom = max_atom
        self.xyz_dir = xyz_dir
        self.coord_file = coord_file
        self.natoms_file = natoms_file
        self.forbidden_atom = forbidden_atom
        self.data_efficient_collator = data_efficient_collator
        self.train_ratio = train_ratio
        self.load_pkl = load_pkl
        self.save_pkl = save_pkl
        self.data_type = data_type.lower()
        self.dataset_name = dataset_name
        self.ase_db_path = ase_db_path

        self.target_fields = target_fields
        self.allow_unknown = allow_unknown
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.edge_type = edge_type
        self.radius = radius

        self.n_neigh = n_neigh
        self.use_ohe_feature = use_ohe_feature
        self.chunk_size = chunk_size
        self.use_row_data_features = use_row_data_features
        self.bond_collate = bond_collate
        self.center_coords = center_coords
        self.graph3d_stats = graph3d_stats
        self.kekulize = kekulize
        self.use_stored_split = use_stored_split
        self.compact = dict(
            drop_edge_index=compact_drop_edge_index,
            drop_tags=compact_drop_tags,
            pack_ohe=compact_pack_ohe,
            int8_z=compact_int8_z,
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        # Cache paths were historically keyed off dataset_name alone. Keep that
        # exact string for the two original data_types so every cache already on
        # disk keeps resolving; namespace anything new so it cannot collide.
        # (The pre-existing pointcloud<->pyg collision is left alone on purpose:
        # renaming either one would orphan existing caches.)
        _cache_tag = (
            dataset_name
            if self.data_type in ("pointcloud", "pyg")
            else f"{dataset_name}_{self.data_type}"
        )
        self.root_path = os.path.join(root, f"processed_data_{_cache_tag}.pt")
        self.chunk_dir = os.path.join(root, f"chunks_{_cache_tag}")
        if self.data_type == "graph3d":
            self.collate_fn = (
                graph3d_dense_collate if self.bond_collate == "dense" else graph_collate
            )
        elif self.data_type == "pyg":
            self.collate_fn = graph_collate
        else:
            if self.data_efficient_collator:
                vram_size = get_vram_size()
                if vram_size is None:
                    vram_size = 40
                    logger.warning(
                        "Could not detect GPU VRAM; using %d GB for data-efficient collator sizing",
                        vram_size,
                    )
                self.VRAM_SIZE = int(vram_size)
                self.collate_fn = pointcloud_collate(self.VRAM_SIZE)
            else:
                self.collate_fn = pointcloud_collate_v0  # or None

    def _split_cache_path(self, total: int) -> str:
        split_seed = int(torch.initial_seed())
        train_ratio_tag = f"{self.train_ratio:.8f}"
        filename = (
            f"splits_n{total}_tr{train_ratio_tag}_seed{split_seed}.pt"
        )
        base_dir = self.chunk_dir if self.chunk_size or os.path.exists(self.chunk_dir) else self.root
        return os.path.join(base_dir, filename)

    def _stored_splits(self, dataset):
        """Honour a split label carried by the data, when one is present.

        Opt-in via `use_stored_split`. Datasets converted from an upstream with
        a published split (MiDi's 100k/val/10% at random_state=42) can then
        reproduce that split exactly, instead of the platform's own
        `random_split`, which would make published numbers incomparable.

        Returns None -- falling back to `_build_or_load_splits` -- whenever the
        data carries no usable labels, so turning the flag on can never
        silently produce an empty or lopsided train set.
        """
        splits = getattr(dataset, "splits", None)
        if not splits or len(splits) != len(dataset):
            logger.warning(
                "use_stored_split=True but the dataset carries no split labels; "
                "falling back to the random split."
            )
            return None

        buckets = {"train": [], "val": [], "test": []}
        aliases = {"valid": "val", "validation": "val"}
        for idx, label in enumerate(splits):
            key = aliases.get(str(label).lower(), str(label).lower())
            if key in buckets:
                buckets[key].append(idx)

        if not buckets["train"]:
            logger.warning(
                "use_stored_split=True but no rows are labelled 'train'; "
                "falling back to the random split."
            )
            return None

        logger.info(
            "Using stored splits: train=%d valid=%d test=%d (%d unlabelled rows ignored)",
            len(buckets["train"]),
            len(buckets["val"]),
            len(buckets["test"]),
            len(dataset) - sum(len(v) for v in buckets.values()),
        )
        return tuple(
            torch.utils.data.Subset(dataset, buckets[key])
            for key in ("train", "val", "test")
        )

    def _build_or_load_splits(self, dataset, lengths):
        cache_path = self._split_cache_path(len(dataset))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            payload = torch.load(cache_path, weights_only=False)
            if payload.get("lengths") == lengths:
                subsets = []
                for key in ("train", "valid", "test"):
                    subsets.append(torch.utils.data.Subset(dataset, payload[key]))
                return tuple(subsets)

        split_seed = int(torch.initial_seed())
        generator = torch.Generator().manual_seed(split_seed)
        train_set, valid_set, test_set = torch.utils.data.random_split(
            dataset, lengths, generator=generator
        )
        torch.save(
            {
                "lengths": lengths,
                "seed": split_seed,
                "train": train_set.indices,
                "valid": valid_set.indices,
                "test": test_set.indices,
            },
            cache_path,
        )
        return train_set, valid_set, test_set

    def load(self):
        """
        Load dataset (from pickle if available), optionally save pickle, then split into train/valid/test.
        """
        is_main_process = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )

        _t0_load = time.perf_counter()

        chunk_meta = os.path.join(self.chunk_dir, "meta.pt")
        has_processed_pt = os.path.exists(self.root_path)
        has_chunk_meta = os.path.exists(chunk_meta)

        # Reuse any existing processed cache before touching raw data.
        # If both cache formats exist, prefer the single-file processed dataset.
        if has_processed_pt or has_chunk_meta:
            if has_processed_pt:
                if is_main_process:
                    print(f"[DataModule] Loading single-file processed dataset from {self.root_path}")
                _t = time.perf_counter()
                if self.data_type == "pointcloud":
                    dataset = pointcloud_dataset(
                        root=self.root,
                        dataset_name=self.dataset_name,
                        max_atom=self.max_atom,
                        allow_unknown=self.allow_unknown,
                    )
                elif self.data_type == "pyg":
                    dataset = pointcloud_dataset_pyG(
                        root=self.root,
                        dataset_name=self.dataset_name,
                        max_atom=self.max_atom,
                        allow_unknown=self.allow_unknown,
                        use_ohe_feature=self.use_ohe_feature,
                    )
                elif self.data_type == "graph3d":
                    dataset = Graph3DDataset(
                        root=self.root,
                        dataset_name=self.dataset_name,
                        max_atom=self.max_atom,
                        load_only=True,
                    )
                if is_main_process:
                    print(f"[DataModule] Dataset loaded in {time.perf_counter() - _t:.2f}s ({len(dataset):,} samples)")

                dataset.max_atom = self.max_atom
                dataset.atom_vocab = self.atom_vocab

                dataset.config_dict = DatasetConfigDict(
                    self.with_hydrogen, self.node_feature_choice, self.max_atom
                )
            else:
                # Fast path: chunked dataset already on disk
                if is_main_process:
                    print(f"[DataModule] Loading chunked dataset from {self.chunk_dir}")
                _t = time.perf_counter()
                meta = torch.load(chunk_meta, weights_only=False)
                kind = meta.get("kind", "pointcloud")
                if kind == "graph3d":
                    # graph3d chunks are pyG Data under the hood, so the existing
                    # lazy reader loads them as-is; it reads its compaction keys
                    # with .get defaults and stays inert when they are absent.
                    dataset = LazyChunkedGraphDataset(self.chunk_dir)
                    dataset.graph3d_stats = meta.get("graph3d_stats")
                elif kind == "pyg":
                    dataset = LazyChunkedGraphDataset(self.chunk_dir)
                else:
                    dataset = LazyChunkedDataset(self.chunk_dir)
                dataset.max_atom = self.max_atom
                if is_main_process:
                    print(f"[DataModule] Chunked dataset ready in {time.perf_counter() - _t:.2f}s ({len(dataset):,} samples)")
        else:
            # Dataset construction is independent of task_type — driven only by
            # data_type (pyg vs pointcloud). Task-specific logic lives in the
            # task module, not here. Adding a new task never requires touching
            # this file.
            if is_main_process:
                print("[DataModule] No cached dataset found — building from raw data")
            _t = time.perf_counter()
            verbose_level = int(is_main_process)
            chunk_kwargs = (
                {"chunk_size": self.chunk_size, "chunk_dir": self.chunk_dir}
                if self.chunk_size
                else {}
            )
            # Compact flags apply to both chunked and in-memory pyG paths.
            pyg_extra = {"compact": self.compact}
            if self.data_type == "graph3d":
                dataset = Graph3DDataset(
                    root=self.root,
                    ase_db_path=self.ase_db_path,
                    dataset_name=self.dataset_name,
                    atom_vocab=self.atom_vocab,
                    max_atom=self.max_atom,
                    with_hydrogen=self.with_hydrogen,
                    forbidden_atoms=self.forbidden_atom,
                    target_fields=self.target_fields,
                    allow_unknown=self.allow_unknown,
                    use_ohe_feature=self.use_ohe_feature,
                    center_coords=self.center_coords,
                    kekulize=self.kekulize,
                    compute_stats=self.graph3d_stats,
                    verbose=verbose_level,
                    **chunk_kwargs,
                )
                if chunk_kwargs and os.path.exists(chunk_meta):
                    _stats = getattr(dataset, "graph3d_stats", None)
                    dataset = LazyChunkedGraphDataset(self.chunk_dir)
                    dataset.max_atom = self.max_atom
                    dataset.graph3d_stats = _stats
            elif self.data_type == "pyg":
                dataset = pointcloud_dataset_pyG(
                    root=self.root,
                    df_path=self.filename,
                    xyz_dir=self.xyz_dir,
                    coord_file=self.coord_file,
                    natoms_file=self.natoms_file,
                    ase_db_path=self.ase_db_path,
                    max_atom=self.max_atom,
                    node_feature_choice=self.node_feature_choice,
                    atom_vocab=self.atom_vocab,
                    with_hydrogen=self.with_hydrogen,
                    forbidden_atoms=self.forbidden_atom,
                    pad_data=not self.data_efficient_collator,
                    dataset_name=self.dataset_name,
                    target_fields=self.target_fields,
                    allow_unknown=self.allow_unknown,
                    verbose=verbose_level,
                    edge_type=self.edge_type,
                    radius=self.radius,
                    n_neigh=self.n_neigh,
                    use_ohe_feature=self.use_ohe_feature,
                    use_row_data_features=self.use_row_data_features,
                    **pyg_extra,
                    **chunk_kwargs,
                )
                if chunk_kwargs and os.path.exists(chunk_meta):
                    dataset = LazyChunkedGraphDataset(self.chunk_dir)
                    dataset.max_atom = self.max_atom
            else:
                dataset = pointcloud_dataset(
                    root=self.root,
                    df_path=self.filename,
                    xyz_dir=self.xyz_dir,
                    coord_file=self.coord_file,
                    natoms_file=self.natoms_file,
                    ase_db_path=self.ase_db_path,
                    max_atom=self.max_atom,
                    node_feature_choice=self.node_feature_choice,
                    atom_vocab=self.atom_vocab,
                    with_hydrogen=self.with_hydrogen,
                    forbidden_atoms=self.forbidden_atom,
                    pad_data=not self.data_efficient_collator,
                    dataset_name=self.dataset_name,
                    target_fields=self.target_fields,
                    allow_unknown=self.allow_unknown,
                    verbose=verbose_level,
                    use_row_data_features=self.use_row_data_features,
                    **chunk_kwargs,
                )
                # If chunked, swap to lazy loader (processing is now on disk)
                if chunk_kwargs and os.path.exists(chunk_meta):
                    dataset = LazyChunkedDataset(self.chunk_dir)
                    dataset.max_atom = self.max_atom
            if is_main_process:
                print(f"[DataModule] Raw data processing done in {time.perf_counter() - _t:.2f}s ({len(dataset):,} samples)")

        dataset.max_atom = self.max_atom
        dataset.atom_vocab = self.atom_vocab
        dataset.node_feature = self.node_feature_choice
        dataset.node_feature_choice = self.node_feature_choice
        dataset.config_dict = DatasetConfigDict(
            self.with_hydrogen, self.node_feature_choice, self.max_atom
        )

        # split
        total = len(dataset)
        test_ratio = (1 - self.train_ratio) / 2
        lengths = [int(self.train_ratio * total), int(test_ratio * total)]
        lengths.append(total - sum(lengths))
        if is_main_process:
            print("[DataModule] Building/loading train/valid/test splits ...")
        _t = time.perf_counter()
        stored = self._stored_splits(dataset) if self.use_stored_split else None
        if stored is not None:
            self.train_set, self.valid_set, self.test_set = stored
        else:
            self.train_set, self.valid_set, self.test_set = self._build_or_load_splits(
                dataset, lengths
            )
        if is_main_process:
            print(f"[DataModule] Splits ready in {time.perf_counter() - _t:.2f}s")

        # Keep lazy chunked subsets in storage order to avoid pathological
        # cross-chunk seeking on the first batches. Membership stays random
        # because random_split already chose the indices; this only sorts them.
        if isinstance(dataset, (LazyChunkedDataset, LazyChunkedGraphDataset)):
            for subset in (self.train_set, self.valid_set, self.test_set):
                subset.indices = sorted(subset.indices)

        # attach metadata
        for subset in (self.train_set, self.valid_set, self.test_set):
            subset.atom_types = getattr(dataset, "atom_types", [])
            subset.targets = getattr(dataset, "targets", [])
            # Bond statistics ride along for tasks that declare `train_set`
            # (the existing declarative seam in cli/train.py) -- None for every
            # non-graph3d data_type, so nothing existing sees a change.
            subset.graph3d_stats = getattr(dataset, "graph3d_stats", None)
        if any(self.task_type.startswith(p) for p in ("diffusion", "vae", "ldm")):
            for subset in (self.train_set, self.valid_set, self.test_set):
                subset.smiles_list = getattr(dataset, "smiles_list", [])
                subset.num_atoms = getattr(dataset, "num_atoms", None)
                subset.get_property = getattr(dataset, "get_property", None)


        if is_main_process:
            print(
                f"[DataModule] Total: {total}, Train: {len(self.train_set)}, Valid: {len(self.valid_set)}, Test: {len(self.test_set)}"
            )
            print(f"[DataModule] load() completed in {time.perf_counter() - _t0_load:.2f}s")
        return self.train_set, self.valid_set, self.test_set
