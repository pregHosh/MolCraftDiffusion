"""Dataset / collate / DataModule for DiffPharma complexes.

DiffPharma needs four node sets per complex -- ligand, protein pocket,
H-bond pseudo-particles, hydrophobic pseudo-particles -- flat-concatenated
with ``torch_scatter``-style masks. That is not expressible as the platform's
padded single-node-set PointCloud batch, so this module brings its own
Dataset + ``collate_fn`` and is selected via ``_target_`` from
``configs/data/diffpharma_dataset.yaml``. Precedent:
``data/component/pharmacophore.py::PharmacophoreDataModule``, which likewise
ships a non-PointCloud container through the same DataModule seam
(``cli/train.py`` -> ``.load()`` / ``.train_set`` / ``.collate_fn``, and
``data/lightning_data_module.py``'s ``collate_fn or graph_collate``).

Storage is one ASE db row per complex, written by
``docs/model_integrations/diffpharma/scripts/convert_dataset.py``:

    Atoms            -> ligand only (positions + Z; Z=0 for 'others')
    key_value_pairs  -> name, n_lig, n_pocket, n_interh, n_interhp
    data             -> lig_one_hot + every other node set's coords/one_hot,
                        plus the (unused by the model) interaction ids

``data['lig_one_hot']`` -- not ``numbers`` -- is the authoritative ligand
atom type; Z=0 is not invertible back to 'others'.
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

GROUPS = ("lig", "pocket", "interh", "interhp")


class DiffPharmaDataset(Dataset):
    """One converted ASE db -> one complex per index.

    ``center=True`` subtracts the joint ligand+pocket centroid from all four
    node sets (upstream ``dataset.py:68-78``); generation needs
    ``center=False`` so the output is in the input pocket's frame.

    # ponytail: rows are read eagerly into RAM at construction, like
    # upstream's whole-npz load. Fine to ~100k complexes; if the full
    # CrossDocked train split ever needs to stream, make __getitem__ read the
    # row by id instead.
    """

    def __init__(
        self, db_path: str, center: bool = True, limit: Optional[int] = None
    ) -> None:
        require_modules("data", ["ase"])
        from ase.db import connect

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DiffPharma db not found: {db_path}")

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
        item: Dict[str, Any] = {"name": row.get("name", str(row.id))}
        coords = {
            "lig": np.asarray(row.positions, dtype=np.float32),
            "pocket": np.asarray(data["pocket_coords"], dtype=np.float32),
            "interh": np.asarray(data["interh_coords"], dtype=np.float32),
            "interhp": np.asarray(data["interhp_coords"], dtype=np.float32),
        }
        if self.center:
            total = len(coords["lig"]) + len(coords["pocket"])
            mean = (coords["lig"].sum(0) + coords["pocket"].sum(0)) / total
            for group in GROUPS:
                if len(coords[group]):
                    coords[group] = coords[group] - mean

        for group in GROUPS:
            # np.array(copy=True): ASE hands back read-only views
            item[f"{group}_coords"] = torch.from_numpy(
                np.array(coords[group].reshape(-1, 3), dtype=np.float32)
            )
            item[f"{group}_one_hot"] = torch.from_numpy(
                np.array(data[f"{group}_one_hot"], dtype=np.float32)
            )
        return item

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


def diffpharma_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; scatter masks are renumbered 0..B-1."""
    out: Dict[str, Any] = {"names": [item["name"] for item in batch]}
    for group in GROUPS:
        coords = [item[f"{group}_coords"] for item in batch]
        sizes = torch.tensor([len(c) for c in coords], dtype=torch.int64)
        out[f"{group}_coords"] = torch.cat(coords, dim=0)
        out[f"{group}_one_hot"] = torch.cat(
            [item[f"{group}_one_hot"] for item in batch], dim=0
        )
        out[f"{group}_mask"] = torch.repeat_interleave(
            torch.arange(len(batch), dtype=torch.int64), sizes
        )
        out[f"num_{group}"] = sizes
    return out


class DiffPharmaDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``."""

    def __init__(
        self,
        root: str,
        train_file: str = "train.db",
        val_file: str = "val.db",
        test_file: str = "test.db",
        batch_size: int = 8,
        num_workers: int = 0,
        center: bool = True,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_diffpharma",
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.center = center
        self.limit = limit
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.collate_fn = diffpharma_collate

    def load(self) -> None:
        def _split(filename: str, required: bool):
            path = os.path.join(self.root, filename)
            if not os.path.exists(path):
                if required:
                    raise FileNotFoundError(
                        f"DiffPharma db not found: {path}. Convert the npz "
                        f"dataset first (scripts/convert_dataset.py)."
                    )
                logger.warning("Split %s missing, skipping", path)
                return None
            return DiffPharmaDataset(path, center=self.center, limit=self.limit)

        self.train_set = _split(self.train_file, required=True)
        self.valid_set = _split(self.val_file, required=False)
        self.test_set = _split(self.test_file, required=False) or self.valid_set
        logger.info(
            "DiffPharma splits: train=%d valid=%d test=%d",
            len(self.train_set),
            len(self.valid_set or []),
            len(self.test_set or []),
        )
