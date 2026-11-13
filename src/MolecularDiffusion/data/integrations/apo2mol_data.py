"""Dataset / collate / DataModule for Apo2Mol apo-holo-ligand complexes.

Apo2Mol carries **three** things per complex: a diffused ligand point cloud, an
apo pocket, and the holo pocket the model is trained to recover. The pocket is
not just coordinates -- it also carries a per-residue rigid transform
(quaternion + translation) and five side-chain chi angles, because the pocket
conformation is *generated*, not conditioned on. None of that fits the
platform's padded single-node-set PointCloud batch, so this module brings its
own Dataset and ``collate_fn`` and is selected via ``_target_`` from
``configs/data/apo2mol_dataset.yaml``. Direct precedent:
``data/component/kgdiff_data.py``, whose schema this extends, and
``pmdm_data.py`` / ``diffpharma_data.py``, which go through the same seams
(``cli/train.py`` -> ``.load()`` / ``.train_set`` / ``.collate_fn``, then
``data/lightning_data_module.py``'s ``collate_fn or graph_collate``).

Storage is one ASE db row per complex, written offline by
``docs/model_integrations/apo2mol/scripts/convert_dataset.py``::

    Atoms            -> ligand only (positions + raw element Z)
    key_value_pairs  -> name, n_lig, n_pocket, n_res
    data             -> lig_aromatic,
                        pocket_coords (apo), pocket_coords_holo,
                        pocket_element, pocket_aa_type, pocket_is_backbone,
                        pocket_atom_name, pocket_aa_name,
                        pocket_atom_to_aa_group,
                        res_rotations, res_translations,
                        chi_apo, chi_holo, chi_mask

Everything numeric is a **raw integer** (or float), never one-hot: the
vocabularies live here, so changing one does not mean reconverting. The
13-class ligand index and the 27-dim protein feature are imported from
``kgdiff_data`` -- Apo2Mol's are byte-for-byte the same tables.

## Two invariants the collate must not break

1. ``protein_atom_name`` and ``protein_atom_to_aa_name`` stay **nested per
   complex** (``list[list[str]]``). Flattening them makes
   ``apply_transforms_tensor_batch`` index the wrong atoms and produce wrong
   coordinates *without raising*.
2. ``protein_atom_to_aa_group`` is **not** offset across complexes -- it
   restarts at 0 in every one. Both the backbone
   (``uni_transformer._global_residue_index``) and the residue-transform code
   re-derive global ids themselves, so offsetting here would double-count.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from MolecularDiffusion.data.integrations.kgdiff_data import (
    KGDIFF_ATOM_VOCAB,
    ligand_class_index,
    protein_features,
)
from MolecularDiffusion.optional import require_modules

logger = logging.getLogger(__name__)

#: ``utils/data.py:34-37`` -- chi1..chi5.
MAX_CHI = 5

#: Re-exported so the task and generator have one import site.
APO2MOL_ATOM_VOCAB = list(KGDIFF_ATOM_VOCAB)


class Apo2MolDataset(Dataset):
    """One converted ASE db -> one apo/holo/ligand complex per index.

    Coordinates are **never** re-centred at load time: ``ScorePosNet3D``
    subtracts the apo pocket centroid inside both ``get_diffusion_loss`` and
    ``sample_diffusion`` (``center_pos_mode='protein'``) and the sampler adds
    the offset back, so samples land in the input pocket's own frame.

    # ponytail: rows are read eagerly into RAM at construction, like
    # KGDiffDataset. Fine for the smoke set and up to ~100k complexes; stream
    # by row id in __getitem__ if the full 24k-complex set with per-residue
    # arrays ever gets tight.
    """

    def __init__(self, db_path: str, limit: Optional[int] = None) -> None:
        require_modules("data", ["ase"])
        from ase.db import connect  # noqa: PLC0415

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Apo2Mol db not found: {db_path}")

        self.db_path = db_path
        self.entries: List[Dict[str, Any]] = []

        with connect(db_path) as db:
            for row in db.select():
                self.entries.append(self._row_to_item(row))
                if limit is not None and len(self.entries) >= limit:
                    break
        if not self.entries:
            raise ValueError(f"No complexes found in {db_path}")

    @staticmethod
    def _row_to_item(row) -> Dict[str, Any]:
        data = row.data
        # np.array(...) copies: ASE hands back read-only views, and
        # torch.from_numpy on one of those yields a non-writable tensor.
        lig_pos = np.array(row.positions, dtype=np.float32).reshape(-1, 3)
        lig_z = np.array(row.numbers, dtype=np.int64)
        lig_aromatic = np.array(data["lig_aromatic"], dtype=bool)

        apo = np.array(data["pocket_coords"], dtype=np.float32).reshape(-1, 3)
        holo = np.array(data["pocket_coords_holo"], dtype=np.float32).reshape(-1, 3)
        if apo.shape != holo.shape:
            raise ValueError(
                f"row {row.id}: apo/holo pockets must be index-aligned "
                f"atom-for-atom, got {apo.shape} vs {holo.shape}."
            )

        group = np.array(data["pocket_atom_to_aa_group"], dtype=np.int64)
        rotations = np.array(data["res_rotations"], dtype=np.float32).reshape(-1, 4)
        n_res = len(rotations)
        if int(group.max()) + 1 != n_res:
            # Upstream's compute_residue_transforms silently DROPS residues
            # missing N/CA/C, which desynchronises the per-residue arrays from
            # protein_atom_to_aa_group. The converter rejects such complexes;
            # this is the second line of defence.
            raise ValueError(
                f"row {row.id}: {int(group.max()) + 1} residue ids but "
                f"{n_res} residue transforms -- the db is inconsistent."
            )

        return {
            "name": row.get("name", str(row.id)),
            "ligand_pos": torch.from_numpy(lig_pos),
            "ligand_v": ligand_class_index(lig_z, lig_aromatic),
            "protein_pos": torch.from_numpy(apo),
            "protein_pos_holo": torch.from_numpy(holo),
            "protein_v": protein_features(
                data["pocket_element"],
                data["pocket_aa_type"],
                data["pocket_is_backbone"],
            ),
            "protein_atom_name": [str(s) for s in data["pocket_atom_name"]],
            "protein_atom_to_aa_name": [str(s) for s in data["pocket_aa_name"]],
            "protein_atom_to_aa_group": torch.from_numpy(group),
            "protein_rotations": torch.from_numpy(rotations),
            "protein_translations": torch.from_numpy(
                np.array(data["res_translations"], dtype=np.float32).reshape(-1, 3)
            ),
            "protein_chi_apo": torch.from_numpy(
                np.array(data["chi_apo"], dtype=np.float32).reshape(-1, MAX_CHI)
            ),
            "protein_chi_holo": torch.from_numpy(
                np.array(data["chi_holo"], dtype=np.float32).reshape(-1, MAX_CHI)
            ),
            "protein_chi_mask": torch.from_numpy(
                np.array(data["chi_mask"], dtype=np.float32).reshape(-1, MAX_CHI)
            ),
        }

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]


#: Per-atom tensors concatenated along the pocket axis.
_POCKET_ATOM_KEYS = (
    "protein_pos",
    "protein_pos_holo",
    "protein_v",
    "protein_atom_to_aa_group",
)
#: Per-residue tensors concatenated along the residue axis.
_RESIDUE_KEYS = (
    "protein_rotations",
    "protein_translations",
    "protein_chi_apo",
    "protein_chi_holo",
    "protein_chi_mask",
)


def apo2mol_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate complexes; emit Apo2Mol's own argument names.

    Three scatter indices come out, one per node set:
    ``ligand_batch`` (atoms), ``protein_batch`` / ``protein_element_batch``
    (pocket atoms; the same tensor under both names, because the model reads
    it under each), and ``protein_translations_batch`` (residues).
    """
    n = len(batch)
    lig_sizes = [len(item["ligand_pos"]) for item in batch]
    pocket_sizes = [len(item["protein_pos"]) for item in batch]
    res_sizes = [len(item["protein_rotations"]) for item in batch]

    def cat(key: str) -> torch.Tensor:
        return torch.cat([item[key] for item in batch], dim=0)

    def batch_index(sizes: List[int]) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(n, dtype=torch.int64), torch.tensor(sizes)
        )

    protein_batch = batch_index(pocket_sizes)
    out: Dict[str, Any] = {
        "names": [item["name"] for item in batch],
        "ligand_pos": cat("ligand_pos"),
        "ligand_v": cat("ligand_v"),
        "ligand_batch": batch_index(lig_sizes),
        "protein_batch": protein_batch,
        # Same index under the name the ported model code uses.
        "protein_element_batch": protein_batch,
        "protein_translations_batch": batch_index(res_sizes),
        # INVARIANT: nested per complex, never flattened. See module docstring.
        "protein_atom_name": [item["protein_atom_name"] for item in batch],
        "protein_atom_to_aa_name": [
            item["protein_atom_to_aa_name"] for item in batch
        ],
    }
    for key in _POCKET_ATOM_KEYS + _RESIDUE_KEYS:
        out[key] = cat(key)
    return out


class Apo2MolDataModule:
    """DataModule contract: ``load()`` + ``train_set``/``valid_set``/``test_set``.

    # ponytail: one db, no split -- ``val_file``/``test_file`` default to the
    # training db. Point them at real split dbs for a full run.
    """

    def __init__(
        self,
        root: str,
        train_file: str = "apo2mol_smoke.db",
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        batch_size: int = 2,
        num_workers: int = 0,
        limit: Optional[int] = None,
        atom_vocab: Optional[List[str]] = None,
        task_type: str = "diffusion_apo2mol",
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
            list(atom_vocab) if atom_vocab else list(APO2MOL_ATOM_VOCAB)
        )
        self.task_type = task_type
        self.kwargs = kwargs

        self.train_set: Optional[Apo2MolDataset] = None
        self.valid_set: Optional[Apo2MolDataset] = None
        self.test_set: Optional[Apo2MolDataset] = None
        self.collate_fn = apo2mol_collate

    def load(self) -> None:
        cache: Dict[str, Apo2MolDataset] = {}

        def _split(filename: str) -> Apo2MolDataset:
            path = os.path.join(self.root, filename)
            if path not in cache:
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Apo2Mol db not found: {path}. Convert apo/holo/ligand "
                        "complexes first (docs/model_integrations/apo2mol/"
                        "scripts/convert_dataset.py)."
                    )
                cache[path] = Apo2MolDataset(path, limit=self.limit)
            return cache[path]

        self.train_set = _split(self.train_file)
        self.valid_set = _split(self.val_file)
        self.test_set = _split(self.test_file)
        logger.info(
            "Apo2Mol splits: train=%d valid=%d test=%d (root=%s)",
            len(self.train_set),
            len(self.valid_set),
            len(self.test_set),
            self.root,
        )
