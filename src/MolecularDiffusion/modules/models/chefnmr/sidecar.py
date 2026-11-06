"""Row-aligned sidecar arrays for ChefNMR.

The platform's ``pointcloud`` batch has a per-atom array channel
(``node_features``) and a per-molecule *scalar* channel (``target_fields``),
but no per-molecule *array* channel. ChefNMR needs two of those: a
``(10080,)`` binned NMR condition and a ``(C, N, 3)`` ground-truth conformer
stack. Both ride memmapped ``.npy`` files keyed by row index and are joined
**inside the task**, off ``batch["xyz"]`` -- the same pattern
``diffusion_diffsmol.py`` uses for its shape latent, and the reason the data
layer needs no change.

Why memmap and not one ``.pt`` dict like DiffSMol's: the condition is 40 kB
per molecule (27x DiffSMol's latent), and ``torch.load`` would pull the whole
map into RAM. ``np.load(..., mmap_mode="r")[i]`` is O(1) resident and reads
40 kB per item.

Layout, all written in one pass by
``docs/model_integrations/chefnmr/scripts/convert_dataset.py``::

    <prefix>.db            ASE db; row i <-> xyz == f"db_entry_{i}"
    <prefix>_cond.npy      (R, h_dim + c_dim) float32
    <prefix>_conf.npy      (R, max_C, max_n_atoms, 3) float32, zero-padded
    <prefix>_nconf.npy     (R,) int32 -- real conformers per row
    <prefix>_meta.json     R, max_C, max_n_atoms, atom_decoder, sigma_data,
                           db sha256 + path, split, sparsity report

**A miss is a hard error, not a fallback.** DiffSMol falls back to a zero
latent on a cache miss; here a zero condition *is* the classifier-free
unconditional branch, so the model would happily emit a plausible molecule of
the right formula that has nothing to do with the spectrum and the run would
look fine. Every lookup failure raises.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_XYZ_PREFIX = "db_entry_"


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    """Streamed sha256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_row_index(xyz: object, n_rows: int) -> int:
    """``"db_entry_17"`` -> ``17``, with a message naming the real cause.

    ``PointCloudDataset.save_pickle(cheap_data=True)`` nulls ``self.xyzs``,
    which destroys the join key outright; so does pointing the task at a db
    that was not the one the sidecar was built from.
    """
    if not isinstance(xyz, str) or not xyz.startswith(_XYZ_PREFIX):
        msg = (
            f"ChefNMR needs the per-row join key batch['xyz'] and got {xyz!r}. "
            "Every row must be 'db_entry_<int>'. A None here means the dataset "
            "was pickled with cheap_data=True, which discards the key; rebuild "
            "the cache without it. A different string means the data config is "
            "not reading the ASE db this sidecar was built from."
        )
        raise ValueError(msg)
    index = int(xyz[len(_XYZ_PREFIX) :])
    if not 0 <= index < n_rows:
        msg = (
            f"row index {index} from batch['xyz']={xyz!r} is out of range for "
            f"a sidecar of {n_rows} rows -- the db and the sidecar are not the "
            "same corpus. Re-run scripts/convert_dataset.py."
        )
        raise IndexError(msg)
    return index


@dataclass
class ChefNMRSidecar:
    """The three row-aligned arrays plus the conversion metadata."""

    cond: np.ndarray  # (R, cond_dim) float32 memmap
    conf: np.ndarray  # (R, max_C, max_n_atoms, 3) float32 memmap
    n_conf: np.ndarray  # (R,) int
    meta: dict

    @property
    def n_rows(self) -> int:
        return int(self.cond.shape[0])

    @property
    def cond_dim(self) -> int:
        return int(self.cond.shape[1])

    @property
    def max_n_atoms(self) -> int:
        return int(self.conf.shape[2])


def load_sidecar(
    cond_path: Optional[str],
    conf_path: Optional[str],
    meta_path: Optional[str],
) -> Optional[ChefNMRSidecar]:
    """Open the memmaps and cross-check them against ``_meta.json``.

    Returns ``None`` when no paths are configured -- which is the
    generate-from-checkpoint case, where the corpus comes from the
    interference config instead and the task never trains.
    """
    if not (cond_path or conf_path or meta_path):
        return None
    missing = [
        name
        for name, path in (
            ("cond_path", cond_path),
            ("conf_path", conf_path),
            ("meta_path", meta_path),
        )
        if not path
    ]
    if missing:
        msg = (
            f"ChefNMR sidecar is incomplete: {missing} not set. The three "
            "files are written together by scripts/convert_dataset.py and "
            "only mean anything together."
        )
        raise ValueError(msg)

    with open(meta_path) as handle:
        meta = json.load(handle)

    cond = np.load(cond_path, mmap_mode="r")
    conf = np.load(conf_path, mmap_mode="r")
    nconf_path = meta.get("nconf_path") or conf_path.replace("_conf.npy", "_nconf.npy")
    # An absolute path recorded on the converting machine is stale once the
    # corpus moves (e.g. a fetched zoo asset), so fall back to the sibling.
    if not os.path.isabs(nconf_path) or not os.path.exists(nconf_path):
        nconf_path = os.path.join(os.path.dirname(os.path.abspath(meta_path)),
                                  os.path.basename(nconf_path))
    n_conf = np.load(nconf_path)

    n_rows = int(meta["n_rows"])
    for name, arr in (("cond", cond), ("conf", conf), ("n_conf", n_conf)):
        if arr.shape[0] != n_rows:
            msg = (
                f"sidecar desync: {name} has {arr.shape[0]} rows but "
                f"{meta_path} records n_rows={n_rows}. Re-run "
                "scripts/convert_dataset.py -- the four files are written in "
                "one pass and must not be mixed between corpora."
            )
            raise ValueError(msg)

    _verify_db(meta, meta_path, n_rows)
    return ChefNMRSidecar(cond=cond, conf=conf, n_conf=np.asarray(n_conf), meta=meta)


def _verify_db(meta: dict, meta_path: str, n_rows: int) -> None:
    """Hash the recorded db and refuse a corpus that is not the converted one.

    A db that cannot be found is a *warning*, not an error: the row-count and
    per-item index checks still hold, and the db legitimately moves (a zoo
    asset, a copied tree). A db that is found and hashes differently is an
    error, because that is silent corruption of the join.
    """
    recorded = meta.get("db_sha256")
    db_path = meta.get("db_path")
    if not recorded or not db_path:
        return
    if not os.path.exists(db_path):
        db_path = os.path.join(
            os.path.dirname(os.path.abspath(meta_path)), os.path.basename(db_path)
        )
    if not os.path.exists(db_path):
        logger.warning(
            "[chefnmr] db recorded in %s was not found, so its sha256 could "
            "not be checked; the sidecar's %d rows are trusted on the "
            "row-index check alone.",
            meta_path,
            n_rows,
        )
        return
    actual = sha256_file(db_path)
    if actual != recorded:
        msg = (
            f"db {db_path} has sha256 {actual[:16]}... but {meta_path} "
            f"records {recorded[:16]}.... The sidecar rows no longer line up "
            "with the db rows; re-run scripts/convert_dataset.py."
        )
        raise ValueError(msg)
