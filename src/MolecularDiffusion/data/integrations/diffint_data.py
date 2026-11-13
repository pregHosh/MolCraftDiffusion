"""Dataset / collate / DataModule for DiffInt ligand-pocket complexes.

DiffInt is DiffSBDD with a CA pocket that has been extended by H-bond
pseudo-atoms (see ``diffint_prep``). The ligand side is byte-for-byte
DiffSBDD's -- same 10-class element vocabulary, hydrogens stripped -- so
``one_hot_from_z`` is imported from ``diffsbdd_data`` rather than copied.
Only the pocket differs: 22 classes (20 amino acids + ``DD`` + ``AC``)
instead of 10 elements.

Selected via ``_target_`` from ``configs/data/diffint_dataset.yaml``, through
the same seams ``diffsbdd_data.py`` uses (``cli/train.py`` -> ``.load()`` /
``.train_set`` / ``.collate_fn``, then ``data/lightning_data_module.py``'s
``collate_fn or graph_collate``).

Storage: one ASE db row per complex, written by
``docs/model_integrations/diffint/scripts/convert_dataset.py``.

    Atoms            -> ligand only (positions + raw element Z)
    key_value_pairs  -> name, n_lig, n_residues, n_particles
    data             -> pocket_coords  (residues THEN particles, N_p x 3)
                        pocket_class   (int 0..21)

``pocket_class`` is a raw integer, not a one-hot, so the vocabulary lives in
this module and can change without reconverting.

## ``num_pocket_nodes`` excludes the particles -- deliberately

``pocket_mask`` spans every pocket node (residues *and* particles), because
all of them condition the EGNN. ``num_pocket_nodes`` counts residues only,
reproducing ``dataset.py:65``'s ``num_pocket_nodes -= num_inter_nodes``. That
number feeds nothing but the ``log_pN`` term and the size prior
(``en_diffusion.py:1111``), and the released checkpoint's (107, 113)
histogram is on the residue scale -- counting particles there would
mis-score every sample.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from MolecularDiffusion.data.integrations.diffint_prep import (
    DIFFINT_POCKET_VOCAB,
    NUM_POCKET_CLASSES,
)
from MolecularDiffusion.data.integrations.diffsbdd_data import (
    DIFFSBDD_ATOM_VOCAB,
    one_hot_from_z,
)
from MolecularDiffusion.optional import require_modules

logger = logging.getLogger(__name__)

__all__ = [
    "DIFFINT_POCKET_VOCAB",
    "NUM_POCKET_CLASSES",
    "DiffIntDataModule",
    "DiffIntDataset",
    "diffint_collate",
    "make_item",
]


def make_item(
    name: str,
    lig_coords: np.ndarray,
    lig_z: np.ndarray,
    pocket_coords: np.ndarray,
    pocket_class: np.ndarray,
    center: bool = True,
) -> Dict[str, Any]:
    """Raw per-complex arrays -> the tensor dict the task consumes.

    Shared by the db path and the on-the-fly ``pocket_pdb`` path so the two
    cannot drift. Hydrogens are stripped from the ligand (upstream trains
    with ``--no_H``); the CA pocket has none by construction.

    ``center`` joint-centres ligand + pocket on their combined centroid, as
    upstream's ``ProcessedLigandPocketDataset`` does (``dataset.py:53-61``).
    The particles count towards that centroid: upstream centres the
    already-augmented ``pocket_coords`` array, and the centring happens
    before the ``num_pocket_nodes`` correction on the next line.
    """
    lig_coords = np.asarray(lig_coords, dtype=np.float32).reshape(-1, 3)
    lig_z = np.asarray(lig_z, dtype=np.int64)
    keep = lig_z != 1
    lig_coords, lig_z = lig_coords[keep], lig_z[keep]

    pocket_coords = np.asarray(pocket_coords, dtype=np.float32).reshape(-1, 3)
    pocket_class = np.asarray(pocket_class, dtype=np.int64)
    if (pocket_class < 0).any() or (pocket_class >= NUM_POCKET_CLASSES).any():
        bad = sorted({int(c) for c in pocket_class if not 0 <= c < NUM_POCKET_CLASSES})
        raise ValueError(
            f"pocket class(es) {bad} outside DiffInt's {NUM_POCKET_CLASSES}-"
            f"class vocabulary {DIFFINT_POCKET_VOCAB}"
        )

    if center:
        mean = (lig_coords.sum(0) + pocket_coords.sum(0)) / (
            len(lig_coords) + len(pocket_coords)
        )
        lig_coords = lig_coords - mean
        pocket_coords = pocket_coords - mean

    n_residues = int((pocket_class < 20).sum())
    return {
        "name": name,
        "lig_coords": torch.from_numpy(np.ascontiguousarray(lig_coords)),
        "lig_one_hot": one_hot_from_z(lig_z),
        "pocket_coords": torch.from_numpy(np.ascontiguousarray(pocket_coords)),
        "pocket_one_hot": torch.from_numpy(
            np.eye(NUM_POCKET_CLASSES, dtype=np.float32)[pocket_class]
        ),
        # residues only -- see the module docstring
        "n_residues": n_residues,
    }


class DiffIntDataset(Dataset):
    """One converted ASE db -> one ligand-pocket complex per index.

    # ponytail: rows are read eagerly into RAM, like DiffSBDDDataset. 100
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
        from ase.db import connect  # noqa: PLC0415

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"DiffInt db not found: {db_path}")

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
        return make_item(
            name=row.get("name", str(row.id)),
            lig_coords=np.array(row.positions),
            lig_z=np.array(row.numbers),
            pocket_coords=np.array(data["pocket_coords"]),
            pocket_class=np.array(data["pocket_class"]),
            center=self.center,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


def diffint_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; emit DiffSBDD's own argument names.

    Identical to ``diffsbdd_collate`` except that ``num_pocket_nodes`` is the
    residue count while ``pocket_mask`` spans the particles too.
    """
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
        "num_pocket_nodes": torch.tensor(
            [item["n_residues"] for item in batch], dtype=torch.int64
        ),
    }


class DiffIntDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: one db, no split -- ``val_file``/``test_file`` default to the
    # training db, exactly as DiffSBDDDataModule. Add a real split when a full
    # CrossDocked run needs one.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "diffint_smoke.db",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        limit: Optional[int] = None,
        center: bool = True,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_diffint",
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

        self.train_set: Optional[DiffIntDataset] = None
        self.valid_set: Optional[DiffIntDataset] = None
        self.test_set: Optional[DiffIntDataset] = None
        self.collate_fn = diffint_collate

    def load(self) -> None:
        cache: Dict[str, DiffIntDataset] = {}

        def _split(filename: str) -> DiffIntDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"DiffInt db not found: {path}. Build it with "
                        "docs/model_integrations/diffint/scripts/"
                        "convert_dataset.py (it is NOT the DiffSBDD db: the "
                        "pocket is CA-only and carries H-bond particles)."
                    )
                cache[path] = DiffIntDataset(
                    path, limit=self.limit, center=self.center
                )
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "DiffInt splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
