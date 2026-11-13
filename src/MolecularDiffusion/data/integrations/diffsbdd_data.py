"""Dataset / collate / DataModule for DiffSBDD ligand-pocket complexes.

DiffSBDD carries two node sets per complex -- a diffused ligand point cloud
and a protein pocket point cloud -- flat-concatenated with ``torch_scatter``
style batch indices. That is not the platform's padded single-node-set
PointCloud batch, so this module brings its own Dataset and ``collate_fn``
and is selected via ``_target_`` from ``configs/data/diffsbdd_dataset.yaml``.
Direct precedent: ``data/component/kgdiff_data.py`` and
``data/component/pmdm_data.py``, through the same seams (``cli/train.py`` ->
``.load()`` / ``.train_set`` / ``.collate_fn``, then
``data/lightning_data_module.py``'s ``collate_fn or graph_collate``).

Storage is the **kgdiff ASE db, reused unchanged** (see the integration
plan): one row per complex, written by
``docs/model_integrations/kgdiff/scripts/convert_dataset.py``.

    Atoms            -> ligand only (positions + raw element Z)
    key_value_pairs  -> name, n_lig, n_pocket, affinity (affinity unused here)
    data             -> lig_aromatic, pocket_coords, pocket_element,
                        pocket_aa_type, pocket_is_backbone

Everything there is a raw integer, so this module owns DiffSBDD's own
vocabulary expansion and nothing has to be reconverted to change it.

**Vocabulary: one 10-class element encoder shared by ligand and pocket**, and
hydrogens stripped from both. This is not a guess -- it is read off the
released full-atom CrossDocked checkpoints (Zenodo 8183747), whose
``hyper_parameters`` say ``dataset='crossdock'`` +
``pocket_representation='full-atom'``, which in ``lightning_modules.py:90-97``
resolves *both* encoders to ``dataset_params['crossdock']['atom_encoder']``
(10 classes, no ``others`` bucket), and whose ``atom_encoder`` /
``residue_encoder`` weights are both ``(20, 10)``. The 11-class
``crossdock_full`` table in ``constants.py:154`` is unused by those weights.
Upstream's ``process_crossdock.py`` is run with ``--no_H``, so hydrogens are
absent from both node sets -- hence the strip here, which is also what keeps
the pocket inside a vocabulary that has no fallback class.

The collate emits DiffSBDD's own key names (``dataset.py:53-70``), so the
task-side adapter is one dict comprehension.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from MolecularDiffusion.optional import require_modules

logger = logging.getLogger(__name__)

#: ``dataset_params['crossdock']['atom_decoder']`` (``constants.py:171``).
#: Used for BOTH the ligand and the full-atom pocket -- see the module
#: docstring.
DIFFSBDD_ATOM_VOCAB = ["C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"]

#: Atomic number of each class above, in the same order.
DIFFSBDD_ELEMENTS = [6, 7, 8, 16, 5, 35, 17, 15, 53, 9]

NUM_ATOM_CLASSES = len(DIFFSBDD_ATOM_VOCAB)

_Z_TO_CLASS = {z: i for i, z in enumerate(DIFFSBDD_ELEMENTS)}


def one_hot_from_z(z: np.ndarray) -> torch.Tensor:
    """Raw element Z -> ``(N, 10)`` float one-hot.

    Raises on an out-of-vocabulary element rather than silently emitting an
    all-zero row: this vocabulary has no ``others`` bucket, so a zero row is
    a feature vector the model never saw in training.
    """
    idx = np.array([_Z_TO_CLASS.get(int(v), -1) for v in z], dtype=np.int64)
    if (idx < 0).any():
        bad = sorted({int(v) for v, i in zip(z.tolist(), idx.tolist()) if i < 0})
        raise ValueError(
            f"element(s) Z={bad} are outside DiffSBDD's 10-class vocabulary "
            f"{DIFFSBDD_ATOM_VOCAB}. Hydrogens are stripped upstream "
            "(process_crossdock.py --no_H); anything else needs a converter "
            "change, not a silent zero row."
        )
    return torch.from_numpy(
        np.eye(NUM_ATOM_CLASSES, dtype=np.float32)[idx]
    )


class DiffSBDDDataset(Dataset):
    """One converted ASE db -> one ligand-pocket complex per index.

    Coordinates are **joint-centred** on the combined ligand+pocket centroid
    at load time, exactly as upstream's ``ProcessedLigandPocketDataset``
    (``dataset.py:35-41``, ``center=True``) does for both modes. The
    generator restores the original pocket frame afterwards.

    # ponytail: rows are read eagerly into RAM, like KGDiffDataset. 100
    # complexes here; stream by row id in __getitem__ if a full CrossDocked
    # split ever needs it.
    """

    def __init__(
        self,
        db_path: str,
        limit: Optional[int] = None,
        center: bool = True,
    ) -> None:
        require_modules("data", ["ase"])
        from ase.db import connect

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DiffSBDD db not found: {db_path}")

        self.db_path = db_path
        self.center = center
        self.entries: List[Dict[str, Any]] = []

        with connect(db_path) as db:
            for row in db.select():
                self.entries.append(self._row_to_item(row))
                if limit is not None and len(self.entries) >= limit:
                    break
        if not self.entries:
            raise ValueError(f"No complexes found in {db_path}")

    def _row_to_item(self, row) -> Dict[str, Any]:
        data = row.data
        # np.array copies: ASE hands back read-only views, and
        # torch.from_numpy on one of those yields a non-writable tensor.
        lig_pos = np.array(row.positions, dtype=np.float32).reshape(-1, 3)
        lig_z = np.array(row.numbers, dtype=np.int64)
        pocket_pos = np.array(
            data["pocket_coords"], dtype=np.float32
        ).reshape(-1, 3)
        pocket_z = np.array(data["pocket_element"], dtype=np.int64)

        # DiffSBDD is heavy-atom only on both node sets (--no_H).
        lig_keep = lig_z != 1
        pocket_keep = pocket_z != 1
        lig_pos, lig_z = lig_pos[lig_keep], lig_z[lig_keep]
        pocket_pos, pocket_z = pocket_pos[pocket_keep], pocket_z[pocket_keep]

        if self.center:
            mean = (lig_pos.sum(0) + pocket_pos.sum(0)) / (
                len(lig_pos) + len(pocket_pos)
            )
            lig_pos = lig_pos - mean
            pocket_pos = pocket_pos - mean

        return {
            "name": row.get("name", str(row.id)),
            "lig_coords": torch.from_numpy(lig_pos),
            "lig_one_hot": one_hot_from_z(lig_z),
            "pocket_coords": torch.from_numpy(pocket_pos),
            "pocket_one_hot": one_hot_from_z(pocket_z),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


def diffsbdd_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; emit DiffSBDD's own argument names."""
    n = len(batch)
    lig_sizes = [len(item["lig_coords"]) for item in batch]
    pocket_sizes = [len(item["pocket_coords"]) for item in batch]

    def cat(key: str) -> torch.Tensor:
        return torch.cat([item[key] for item in batch], dim=0)

    def batch_index(sizes: List[int]) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), torch.tensor(sizes)
        )

    return {
        "names": [item["name"] for item in batch],
        "lig_coords": cat("lig_coords"),
        "lig_one_hot": cat("lig_one_hot"),
        "lig_mask": batch_index(lig_sizes),
        "num_lig_atoms": torch.tensor(lig_sizes, dtype=torch.int64),
        "pocket_coords": cat("pocket_coords"),
        "pocket_one_hot": cat("pocket_one_hot"),
        "pocket_mask": batch_index(pocket_sizes),
        "num_pocket_nodes": torch.tensor(pocket_sizes, dtype=torch.int64),
    }


class DiffSBDDDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: one db, no split -- ``val_file``/``test_file`` default to the
    # training db. The smoke set is 100 complexes; add a real split (and LMDB
    # streaming) when a full CrossDocked run needs one.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "diffsbdd_smoke.db",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        limit: Optional[int] = None,
        center: bool = True,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_diffsbdd",
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.train_file = train_file
        self.val_file = val_file or train_file
        self.test_file = test_file or self.val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.center = center
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(DIFFSBDD_ATOM_VOCAB)
        )
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[DiffSBDDDataset] = None
        self.valid_set: Optional[DiffSBDDDataset] = None
        self.test_set: Optional[DiffSBDDDataset] = None
        self.collate_fn = diffsbdd_collate

    def load(self) -> None:
        cache: Dict[str, DiffSBDDDataset] = {}

        def _split(filename: str) -> DiffSBDDDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"DiffSBDD db not found: {path}. It is the kgdiff "
                        "converter's output, reused unchanged: "
                        "docs/model_integrations/kgdiff/scripts/"
                        "convert_dataset.py"
                    )
                cache[path] = DiffSBDDDataset(
                    path, limit=self.limit, center=self.center
                )
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "DiffSBDD splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
