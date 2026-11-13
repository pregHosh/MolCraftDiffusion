"""Dataset / collate / DataModule for KGDiff pocket-ligand complexes.

KGDiff carries two node sets per complex -- a diffused ligand point cloud and
a fixed pocket point cloud -- flat-concatenated with ``torch_scatter``-style
batch indices, plus one scalar affinity label per complex. That is not the
platform's padded single-node-set PointCloud batch, so this module brings its
own Dataset and ``collate_fn`` and is selected via ``_target_`` from
``configs/data/kgdiff_dataset.yaml``. Direct precedent:
``data/component/pmdm_data.py`` and ``data/component/diffpharma_data.py``,
which do the same through the same seams (``cli/train.py`` -> ``.load()`` /
``.train_set`` / ``.collate_fn``, then ``data/lightning_data_module.py``'s
``collate_fn or graph_collate``).

Storage is one ASE db row per complex, written offline by
``docs/model_integrations/kgdiff/scripts/convert_dataset.py``:

    Atoms            -> ligand only (positions + raw element Z)
    key_value_pairs  -> name, n_lig, n_pocket, affinity
    data             -> lig_aromatic, pocket_coords, pocket_element,
                        pocket_aa_type, pocket_is_backbone

Everything in the db is a **raw integer** (or a bool), never one-hot. This
module owns the vocabulary expansion -- the 13-class
``(element, is_aromatic)`` ligand index and the 27-dim protein feature -- so
changing a vocabulary does not mean reconverting.

The collate emits **KGDiff's own argument names**, so the task-side adapter
is a one-line ``.to(device)`` map rather than a container conversion.
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

#: ``MAP_ATOM_TYPE_AROMATIC_TO_INDEX`` -- KGDiff's ``add_aromatic`` ligand
#: vocabulary (``utils/transforms.py:48-62``). 13 classes over 8 elements.
MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12,
}

#: Ligand class index -> atomic number. Two indices can share an element
#: (aromatic / non-aromatic), which is why the model's 13 classes collapse to
#: 8 elements before anything writes an ``.xyz``.
LIGAND_INDEX_TO_Z = [
    z for (z, _aromatic) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX
]

#: The 8 distinct elements above, in ascending-Z order -- this is the
#: ``atom_vocab`` the platform's ``save_xyz_file`` decodes against.
KGDIFF_ATOM_VOCAB = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
KGDIFF_ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17]

#: ``FeaturizeProteinAtom.atomic_numbers`` (``utils/transforms.py:122``).
#: NB this is a *different* vocabulary from the ligand's -- pockets contain
#: Se but no halogens.
PROTEIN_ELEMENTS = [1, 6, 7, 8, 16, 34]

#: ``FeaturizeProteinAtom.max_num_aa``
MAX_NUM_AA = 20

#: 6 elements + 20 amino acids + is_backbone
PROTEIN_FEATURE_DIM = len(PROTEIN_ELEMENTS) + MAX_NUM_AA + 1

#: Number of ligand classes; equals ``tasks.ligand_atom_feature_dim``.
NUM_LIGAND_CLASSES = len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)


def ligand_class_index(z: np.ndarray, aromatic: np.ndarray) -> torch.Tensor:
    """``(element Z, is_aromatic)`` -> the 13-class index, as ``int64``.

    Out-of-vocabulary pairs fall back to class 0 (``H``), which is exactly
    what upstream's ``get_index`` does (``utils/transforms.py:109-113``) --
    it prints the offender and returns ``(1, False)``.
    """
    out = np.zeros(len(z), dtype=np.int64)
    for i, (zi, ai) in enumerate(zip(z.tolist(), aromatic.tolist())):
        out[i] = MAP_ATOM_TYPE_AROMATIC_TO_INDEX.get(
            (int(zi), bool(ai)), 0
        )
    return torch.from_numpy(out)


def protein_features(
    element: np.ndarray, aa_type: np.ndarray, is_backbone: np.ndarray
) -> torch.Tensor:
    """``(M,) x3`` raw integers -> the ``(M, 27)`` float feature matrix.

    ``[element one-hot (6) | amino acid one-hot (20) | is_backbone (1)]``,
    matching ``FeaturizeProteinAtom.__call__`` exactly. An element outside
    the vocabulary yields an all-zero element block, as upstream's boolean
    ``==`` comparison does.
    """
    vocab = np.asarray(PROTEIN_ELEMENTS, dtype=np.int64)
    elem = (np.asarray(element, dtype=np.int64)[:, None] == vocab[None, :])
    aa = np.zeros((len(aa_type), MAX_NUM_AA), dtype=np.float32)
    aa[np.arange(len(aa_type)), np.asarray(aa_type, dtype=np.int64)] = 1.0
    backbone = np.asarray(is_backbone, dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(
        np.concatenate([elem.astype(np.float32), aa, backbone], axis=-1)
    )


class KGDiffDataset(Dataset):
    """One converted ASE db -> one pocket-ligand complex per index.

    Unlike the other pocket models here, coordinates are **never** re-centred
    at load time: ``ScorePosNet3D`` centres on the pocket centroid inside
    both ``get_diffusion_loss`` and ``sample_diffusion``
    (``center_pos_mode='protein'``), and ``sample_diffusion`` adds the offset
    back, so samples land in the input pocket's own frame either way.

    # ponytail: rows are read eagerly into RAM at construction, like
    # PMDMDataset. 100 complexes here, fine to ~100k. Stream by row id in
    # __getitem__ if a full CrossDocked split ever needs it.
    """

    def __init__(self, db_path: str, limit: Optional[int] = None) -> None:
        require_modules("data", ["ase"])
        from ase.db import connect

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"KGDiff db not found: {db_path}")

        self.db_path = db_path
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
        # np.array(...) copies: ASE hands back read-only views, and
        # torch.from_numpy on one of those yields a non-writable tensor.
        lig_pos = np.array(row.positions, dtype=np.float32).reshape(-1, 3)
        lig_z = np.array(row.numbers, dtype=np.int64)
        lig_aromatic = np.array(data["lig_aromatic"], dtype=bool)
        pocket_pos = np.array(
            data["pocket_coords"], dtype=np.float32
        ).reshape(-1, 3)

        return {
            "name": row.get("name", str(row.id)),
            "ligand_pos": torch.from_numpy(lig_pos),
            "ligand_v": ligand_class_index(lig_z, lig_aromatic),
            "protein_pos": torch.from_numpy(pocket_pos),
            "protein_v": protein_features(
                data["pocket_element"],
                data["pocket_aa_type"],
                data["pocket_is_backbone"],
            ),
            # normalised Vina score in [0, 1]; 1 = strongest binder
            # (NormalizeVina('pl'), utils/transforms.py:190-212)
            "affinity": torch.tensor(float(row.affinity), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


def kgdiff_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; emit KGDiff's own argument names."""
    lig_sizes = [len(item["ligand_pos"]) for item in batch]
    pocket_sizes = [len(item["protein_pos"]) for item in batch]
    n = len(batch)

    def cat(key: str) -> torch.Tensor:
        return torch.cat([item[key] for item in batch], dim=0)

    def batch_index(sizes: List[int]) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), torch.tensor(sizes)
        )

    return {
        "names": [item["name"] for item in batch],
        "ligand_pos": cat("ligand_pos"),
        "ligand_v": cat("ligand_v"),
        "ligand_batch": batch_index(lig_sizes),
        "protein_pos": cat("protein_pos"),
        "protein_v": cat("protein_v"),
        "protein_batch": batch_index(pocket_sizes),
        "affinity": torch.stack([item["affinity"] for item in batch]),
    }


class KGDiffDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: one db, no split -- ``val_file``/``test_file`` default to the
    # training db. The smoke set is 100 complexes; add a real split (and LMDB
    # streaming) when a full CrossDocked run needs one.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "kgdiff_smoke.db",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_kgdiff",
        **kwargs: Any,
    ) -> None:
        self.root = root
        self.train_file = train_file
        self.val_file = val_file or train_file
        self.test_file = test_file or self.val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit = limit
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(KGDIFF_ATOM_VOCAB)
        )
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[KGDiffDataset] = None
        self.valid_set: Optional[KGDiffDataset] = None
        self.test_set: Optional[KGDiffDataset] = None
        self.collate_fn = kgdiff_collate

    def load(self) -> None:
        cache: Dict[str, KGDiffDataset] = {}

        def _split(filename: str) -> KGDiffDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"KGDiff db not found: {path}. Convert pocket/ligand "
                        "pairs first (docs/model_integrations/kgdiff/scripts/"
                        "convert_dataset.py)."
                    )
                cache[path] = KGDiffDataset(path, limit=self.limit)
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "KGDiff splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
