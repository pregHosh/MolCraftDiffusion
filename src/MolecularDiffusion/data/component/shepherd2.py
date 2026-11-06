"""ShEPhERD-2 DataModule (``data_type: shepherd2_hetero``).

ShEPhERD-2 is routed to its own PyG ``HeteroData`` representation rather than
``pointcloud`` / ``pyg`` / ``graph3d``: three of its four modalities (the 75-point
molecular surface ``x2``, the surface electrostatic potential ``x3``, and the
10-class pharmacophore points *with direction vectors* ``x4``) and the per-atom
xTB partial charges the ESP is derived from have no channel in any of the shared
representations.  See ``docs/model_integrations/shepherd2/INTEGRATION_PLAN.md``
("Representation Routing").

Nothing here re-implements the dataset.  All item construction -- surface
sampling, ESP evaluation, pharmacophore extraction, the fully-connected
upper-triangular bond edge list, the per-sample EDM forward noising and the
scaffold/pharmacophore masks -- is upstream's own
``shepherd.datasets.HeteroDataset.process_item``, vendored verbatim at
``MolecularDiffusion.modules.models.shepherd2.datasets``.  This file is a thin
seam adapter that

1. picks the on-disk source (pickle ``list[(molblock, charges)]`` or HDF5 shards),
2. translates one ``params`` dict into ``HeteroDataset``'s keyword arguments using
   *upstream's own* mapping (``training/train.py:74-114`` of the ShEPhERD-2 repo,
   reproduced key-for-key below), and
3. exposes the ``train_set``/``valid_set``/``test_set``/``collate_fn``/``batch_size``
   surface ``cli/train.py`` instantiates.

Single source of truth for ``params``
-------------------------------------
``configs/data/shepherd2_dataset.yaml`` does **not** restate the hyperparameters.
It interpolates them straight out of the task config
(``model_params: ${tasks.model_params}``), so the architecture and the dataset
can never drift apart.  Set ``checkpoint_path`` to override both from a
checkpoint's ``hyper_parameters['params']`` instead (the ShEPhERD-v1 precedent).

open3d
------
``HeteroDataset`` imports ``open3d`` lazily, inside the first ``process_item``
call, deliberately: open3d initialises thread locks that cannot be pickled under
``spawn`` multiprocessing (upstream comment at ``datasets.py:1-2, 13``).  Do not
hoist that import.  open3d is a *hard* requirement of this integration -- the
Poisson-disk surface sampling it provides is what the released weights were
trained on, and there is no acceptable fallback.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def shepherd2_collate_fn(batch: List[Any]):
    """Collate ``HeteroData`` items exactly as upstream's PyG DataLoader does.

    Upstream drives training through ``torch_geometric.loader.DataLoader``
    (``data_utils/datamodule.py:131``), whose collater is
    ``Batch.from_data_list``.  The platform's DataLoader takes a ``collate_fn``
    instead, so this is that same collater behind a module-level (picklable)
    name.
    """
    from torch_geometric.data import Batch

    return Batch.from_data_list(batch)


def hetero_dataset_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a ShEPhERD-2 ``params`` dict into ``HeteroDataset`` kwargs.

    Reproduced key-for-key from upstream ``training/train.py:74-114`` so that a
    platform-trained dataset is byte-for-byte the one upstream would have built
    from the same config.  Do not "tidy" this mapping: a silently renamed key
    would change the training distribution without changing any error message.
    """
    d = params["dataset"]
    return {
        "edm_schedule_dict": params.get("edm", {}),
        "explicit_hydrogens": d["explicit_hydrogens"],
        "formal_charge_diffusion": params["x1_formal_charge_diffusion"],
        "x1": d["compute_x1"],
        "x2": d["compute_x2"],
        "x3": d["compute_x3"],
        "x4": d["compute_x4"],
        "recenter_x1": d["x1"]["recenter"],
        "add_virtual_node_x1": d["x1"]["add_virtual_node"],
        "remove_noise_COM_x1": d["x1"]["remove_noise_COM"],
        "atom_types_x1": d["x1"]["atom_types"],
        "charge_types_x1": d["x1"]["charge_types"],
        "bond_types_x1": d["x1"]["bond_types"],
        "scale_atom_features_x1": d["x1"]["scale_atom_features"],
        "scale_bond_features_x1": d["x1"]["scale_bond_features"],
        "recenter_x2": d["x2"]["recenter"],
        "add_virtual_node_x2": d["x2"]["add_virtual_node"],
        "remove_noise_COM_x2": d["x2"]["remove_noise_COM"],
        "num_points_x2": d["x2"]["num_points"],
        "recenter_x3": d["x3"]["recenter"],
        "add_virtual_node_x3": d["x3"]["add_virtual_node"],
        "remove_noise_COM_x3": d["x3"]["remove_noise_COM"],
        "num_points_x3": d["x3"]["num_points"],
        "scale_node_features_x3": d["x3"]["scale_node_features"],
        "recenter_x4": d["x4"]["recenter"],
        "add_virtual_node_x4": d["x4"]["add_virtual_node"],
        "remove_noise_COM_x4": d["x4"]["remove_noise_COM"],
        "max_node_types_x4": d["x4"]["max_node_types"],
        "scale_node_features_x4": d["x4"]["scale_node_features"],
        "scale_vector_features_x4": d["x4"]["scale_vector_features"],
        "multivectors": d["x4"]["multivectors"],
        "check_accessibility": d["x4"]["check_accessibility"],
        "probe_radius": d["probe_radius"],  # for x2 and x3
        "include_dummy_atoms": d["x1"].get("include_dummy_atoms", False),
        "include_dummy_pharm": d["x4"].get("include_dummy_pharm", False),
        "scale_point_cloud": d.get("scale_point_cloud", 1.0),
        "scaffold_conditioning": params.get("scaffold_conditioning", False),
        "fixed_substructure_params_x1": d["x1"].get("fixed_substructure", {}),
        "fixed_substructure_params_x4": d["x4"].get("fixed_substructure", {}),
    }


def _to_plain(obj):
    """Resolve an OmegaConf node (or anything) to plain python containers."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is a core dependency
        pass
    return obj


class Shepherd2DataModule:
    """DataModule for ShEPhERD-2's four-modality ``HeteroData`` batches.

    Two on-disk sources, both funnelling into the identical
    ``HeteroDataset.process_item``:

    ``source="pkl"``
        ``pkl_files``: pickles holding ``list[(molblock, partial_charges)]`` --
        the format the target repo ships its MOSES-aq conformers in and the same
        one the platform's ShEPhERD-v1 data module already reads.

    ``source="hdf5"``
        ``hdf5_data_dir``: a directory of ``*.h5`` shards, read through upstream's
        ``HeteroDatasetHDF5``, a lazy-loading subclass that indexes the shards on
        init and delegates every item to the parent's ``process_item``.  Needs
        ``h5py``.  Shards: Zenodo doi:10.5281/zenodo.22675705.
    """

    def __init__(
        self,
        source: str = "pkl",
        pkl_files: Optional[List[str]] = None,
        hdf5_data_dir: Optional[str] = None,
        cache_molecules: bool = False,
        model_params: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        data_fraction: float = 1.0,
        batch_size: int = 2,
        num_workers: int = 0,
        train_ratio: float = 0.9,
        task_type: str = "diffusion_shepherd2",
        **kwargs: Any,
    ) -> None:
        if source not in ("pkl", "hdf5"):
            raise ValueError(f"source must be 'pkl' or 'hdf5', got {source!r}")
        self.source = source
        self.pkl_files = list(pkl_files or [])
        self.hdf5_data_dir = hdf5_data_dir
        self.cache_molecules = cache_molecules
        self.model_params = _to_plain(model_params)
        self.checkpoint_path = checkpoint_path
        self.num_samples = num_samples
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.collate_fn = shepherd2_collate_fn

    # ------------------------------------------------------------------
    def _resolve_params(self) -> Dict[str, Any]:
        if self.checkpoint_path:
            logger.info(
                "Reading ShEPhERD-2 params from checkpoint: %s", self.checkpoint_path
            )
            ckpt = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )
            return ckpt["hyper_parameters"]["params"]
        if self.model_params is None:
            raise ValueError(
                "Shepherd2DataModule needs the ShEPhERD-2 `params` dict. Either "
                "compose it with configs/tasks/diffusion_shepherd2.yaml (the "
                "bundled data config interpolates `model_params: "
                "${tasks.model_params}`) or set `checkpoint_path`."
            )
        return self.model_params

    def _load_pkl_entries(self) -> List[Any]:
        if not self.pkl_files:
            raise ValueError("source='pkl' requires a non-empty `pkl_files` list")
        raw: List[Any] = []
        for path in self.pkl_files:
            with open(path, "rb") as fh:
                raw.extend(pickle.load(fh))
        if self.data_fraction < 1.0:
            raw = raw[: int(len(raw) * self.data_fraction)]
        if self.num_samples is not None:
            raw = raw[: self.num_samples]
        if not raw:
            raise ValueError(f"No entries loaded from {self.pkl_files}")
        logger.info("Loaded %d (molblock, charges) entries", len(raw))
        return raw

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Build train/valid/test ``HeteroDataset``s."""
        from MolecularDiffusion.modules.models.shepherd2.datasets import HeteroDataset

        params = self._resolve_params()
        ds_kwargs = hetero_dataset_kwargs(params)

        if self.source == "hdf5":
            from MolecularDiffusion.modules.models.shepherd2.data_utils.hdf5_dataset import (
                HeteroDatasetHDF5,
            )

            if not self.hdf5_data_dir:
                raise ValueError("source='hdf5' requires `hdf5_data_dir`")
            full = HeteroDatasetHDF5(
                hdf5_data_dir=self.hdf5_data_dir,
                cache_molecules=self.cache_molecules,
                **ds_kwargs,
            )
            # The HDF5 subclass indexes shards itself; splitting it would mean
            # re-indexing, so the whole shard set is the train set and
            # validation is skipped (upstream does the same -- its validation is
            # unconditional sampling, not a held-out split).
            self.train_set = full
            self.valid_set = None
            self.test_set = None
            logger.info("HDF5 source: %d items indexed", len(full))
            return

        entries = self._load_pkl_entries()
        n_train = max(1, int(len(entries) * self.train_ratio))
        n_valid = max(0, len(entries) - n_train)

        self.train_set = HeteroDataset(
            molblocks_and_charges=entries[:n_train], **ds_kwargs
        )
        self.valid_set = (
            HeteroDataset(molblocks_and_charges=entries[n_train:], **ds_kwargs)
            if n_valid
            else None
        )
        self.test_set = self.valid_set
        logger.info(
            "Shepherd2DataModule: %d train / %d valid items",
            len(self.train_set),
            n_valid,
        )
