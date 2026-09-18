"""Helpers for accelerating Task.preprocess() on large datasets.

Two ideas:

1. Bulk stats: compute n_node_dist / max_n_nodes / SMILES list from
   dataset attributes (`n_atoms`, `smiles_list`) instead of iterating
   the entire dataset, which is catastrophically slow for chunked
   datasets that load every chunk on each `__getitem__`.

2. Disk cache: persist the expensive preprocess outputs next to the
   chunk directory so subsequent runs skip recomputation entirely.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subset unwrapping (shared with diffusion_fm_pyg)
# ---------------------------------------------------------------------------

def resolve_dataset_and_indices(train_set):
    """Unwrap torch Subset-like wrappers, returning (base_dataset, indices)."""
    indices = None
    dataset = train_set
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        subset_indices = list(dataset.indices)
        if indices is None:
            indices = subset_indices
        else:
            indices = [indices[i] for i in subset_indices]
        dataset = dataset.dataset
    return dataset, indices


def subset_sequence(values, indices):
    if values is None:
        return None
    if len(values) == 0:
        return []
    if indices is None:
        return list(values)
    max_index = max(indices) if indices else -1
    min_index = min(indices) if indices else 0
    n_values = len(values)
    if max_index >= n_values or min_index < -n_values:
        logger.warning(
            "preprocess: metadata/index mismatch for sequence with %d entries "
            "(required index range [%d, %d]); dropping out-of-range indices",
            n_values,
            min_index,
            max_index,
        )
        valid_indices = [i for i in indices if -n_values <= i < n_values]
        dropped = len(indices) - len(valid_indices)
        if dropped > 0:
            logger.warning("preprocess: dropped %d out-of-range metadata indices", dropped)
        indices = valid_indices
        if not indices:
            return []
    # numpy fancy-indexing is significantly faster than a Python loop for
    # large index arrays (e.g. 8M-element smiles_list subsets).
    if len(indices) > 50_000:
        try:
            import numpy as np
            arr = np.asarray(values, dtype=object)
            idx = np.asarray(indices, dtype=np.intp)
            return arr[idx].tolist()
        except Exception:
            pass
    try:
        return [values[i] for i in indices]
    except IndexError:
        valid_indices = [i for i in indices if -n_values <= i < n_values]
        dropped = len(indices) - len(valid_indices)
        if dropped > 0:
            logger.warning("preprocess: dropped %d out-of-range metadata indices after bounds check", dropped)
        return [values[i] for i in valid_indices]


def subset_tensor(values, indices):
    if values is None:
        return None
    if indices is None:
        return values
    if torch.is_tensor(values):
        return values[torch.as_tensor(indices, dtype=torch.long)]
    return torch.tensor([values[i] for i in indices], dtype=torch.float32)


def property_sample_indices(
    total: int,
    indices=None,
    env_var: str = "MOLCRAFT_PREPROCESS_PROP_SAMPLES",
    default_max_samples: int = 200_000,
):
    """Return deterministic base-dataset indices for property preprocessing.

    Large chunked datasets can spend hours scanning every full graph chunk just
    to estimate conditioning histograms.  A deterministic prefix sample keeps
    the estimate stable while bounding preprocessing cost and chunk reads.
    """
    try:
        max_samples = int(os.environ.get(env_var, default_max_samples))
    except ValueError:
        max_samples = default_max_samples
    if max_samples <= 0 or total <= max_samples:
        return indices

    sample_local = list(range(max_samples))
    if indices is None:
        return sample_local
    return [indices[i] for i in sample_local]


def get_property_subset(dataset, task: str, indices=None) -> torch.Tensor:
    """Read one property, using dataset-level indexed loading when available."""
    try:
        return dataset.get_property(task, indices)
    except TypeError:
        return subset_tensor(dataset.get_property(task), indices)


# ---------------------------------------------------------------------------
# Bulk node stats
# ---------------------------------------------------------------------------

def bulk_n_node_stats(train_set) -> Optional[Tuple[Dict[int, int], int, List[Optional[str]]]]:
    """Try to compute n_node_dist / max / smiles from dataset attributes.

    Returns (n_node_dist, max_n_nodes, smiles_list_raw) or None when the
    dataset does not expose `n_atoms` (caller should fall back to a
    per-sample loop).
    """
    base, indices = resolve_dataset_and_indices(train_set)
    n_atoms = getattr(base, "n_atoms", None)
    if n_atoms is None:
        return None

    _t = time.perf_counter()
    n_atoms = subset_sequence(n_atoms, indices)
    smiles_raw = subset_sequence(getattr(base, "smiles_list", None), indices) or []

    counter = Counter(int(n) for n in n_atoms)
    max_n = max(counter) if counter else 0
    logger.info(
        f"preprocess: bulk_n_node_stats computed {len(n_atoms):,} samples in {time.perf_counter() - _t:.2f}s"
    )
    return dict(counter), max_n, list(smiles_raw)


# ---------------------------------------------------------------------------
# SMILES canonicalisation (parallel)
# ---------------------------------------------------------------------------

def _canon_one(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonical_smiles_set(
    smiles_list: Iterable[Optional[str]],
    n_workers: Optional[int] = None,
    parallel_threshold: int = 5000,
) -> List[str]:
    """Return the unique canonical SMILES set.

    Deduplicates raw strings before canonicalisation so that only unique
    representatives are processed.  Uses multiprocessing when the input is
    large enough to amortise the pool startup cost.  RDKit holds the GIL
    so threads do not help.
    """
    smiles_list = [s for s in smiles_list if s]
    if not smiles_list:
        return []

    # Deduplicate raw strings first — for large datasets the unique count is
    # far smaller than the full list, so this alone can cut canonicalisation
    # work by 80-90%.  dict.fromkeys preserves first-seen order.
    n_raw = len(smiles_list)
    smiles_list = list(dict.fromkeys(smiles_list))
    logger.info(
        f"preprocess: SMILES raw dedup {n_raw:,} → {len(smiles_list):,} unique strings"
    )

    if n_workers is None:
        # os.cpu_count() reports the *physical node's* CPU count, not what a
        # job scheduler (e.g. SLURM cgroups) actually allocated to this
        # process — on a shared multi-core node this can wildly overshoot
        # and fork far more worker processes than intended.
        # os.sched_getaffinity(0) reflects the process's actual usable CPU
        # set and is cgroup-aware on Linux; fall back to cpu_count() where
        # unavailable (e.g. macOS). Capped regardless: SMILES canonicalisation
        # is pickling/IPC-bound past a handful of workers, and each worker
        # forks a copy-on-write snapshot of the (potentially multi-GB)
        # dataset already resident in the parent, so more workers than
        # necessary risks OOM rather than speed.
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 2
        n_workers = max(1, min(available_cpus - 1, 8))

    _t = time.perf_counter()
    if n_workers > 1 and len(smiles_list) >= parallel_threshold:
        try:
            from multiprocessing import Pool
            with Pool(n_workers) as pool:
                canon = pool.map(_canon_one, smiles_list, chunksize=512)
            logger.info(
                f"preprocess: SMILES canonicalisation ({len(smiles_list):,} unique strings,"
                f" {n_workers} workers) done in {time.perf_counter() - _t:.2f}s"
            )
        except Exception as exc:
            logger.warning(f"Parallel SMILES canonicalisation failed ({exc}); falling back to serial")
            canon = [_canon_one(s) for s in smiles_list]
            logger.info(
                f"preprocess: SMILES canonicalisation (serial fallback, {len(smiles_list):,} strings)"
                f" done in {time.perf_counter() - _t:.2f}s"
            )
    else:
        canon = [_canon_one(s) for s in smiles_list]
        logger.info(
            f"preprocess: SMILES canonicalisation ({len(smiles_list):,} strings, serial)"
            f" done in {time.perf_counter() - _t:.2f}s"
        )

    return list({s for s in canon if s})


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_dir(train_set) -> Optional[str]:
    base, _ = resolve_dataset_and_indices(train_set)
    chunk_dir = getattr(base, "chunk_dir", None)
    if chunk_dir and os.path.isdir(chunk_dir):
        return chunk_dir
    # Single-file (.pt) datasets: fall back to the directory that contains the
    # processed file so the cache lives next to the data.
    processed_file = getattr(base, "_processed_file", None)
    if processed_file and os.path.isfile(processed_file):
        return os.path.dirname(os.path.abspath(processed_file))
    return None


def _signature(train_set, condition: List[str]) -> Dict[str, Any]:
    base, indices = resolve_dataset_and_indices(train_set)
    sig: Dict[str, Any] = {
        "n": len(train_set),
        "condition": tuple(sorted(condition)) if condition else (),
        "atom_vocab": tuple(getattr(base, "atom_vocab", []) or []),
        "property_sample_cap": os.environ.get(
            "MOLCRAFT_PREPROCESS_PROP_SAMPLES", "200000"
        ),
    }
    chunk_paths = getattr(base, "chunk_paths", None)
    chunk_sizes = getattr(base, "chunk_sizes", None)
    if chunk_paths is not None and chunk_sizes is not None:
        sig["chunks"] = tuple(
            (os.path.basename(p), int(s)) for p, s in zip(chunk_paths, chunk_sizes)
        )
    else:
        # Non-chunked: fingerprint on the processed file size + mtime so a
        # re-processed dataset correctly busts the cache.
        processed_file = getattr(base, "_processed_file", None)
        if processed_file and os.path.isfile(processed_file):
            stat = os.stat(processed_file)
            sig["processed_file"] = (
                os.path.basename(processed_file),
                stat.st_size,
                int(stat.st_mtime),
            )
    if indices is not None:
        # Cheap fingerprint of the subset: length + first/last + sum.
        sig["subset"] = (
            len(indices),
            int(indices[0]) if indices else 0,
            int(indices[-1]) if indices else 0,
            int(sum(indices)) if indices else 0,
        )
    return sig


def _cache_path(cache_dir: str, condition: List[str]) -> str:
    cond_tag = "_".join(sorted(condition)) if condition else "none"
    # Conservative filename: avoid spaces / odd chars
    cond_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in cond_tag)
    return os.path.join(cache_dir, f"preprocess_cache__{cond_tag}.pt")


def try_load(train_set, condition: List[str]) -> Optional[Dict[str, Any]]:
    """Return cached preprocess payload if signature matches, else None."""
    cache_dir = _cache_dir(train_set)
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, condition)
    if not os.path.exists(path):
        logger.info(f"preprocess: no cache found at {path} — will compute from scratch")
        return None
    _t = time.perf_counter()
    try:
        payload = torch.load(path, weights_only=False)
    except Exception as exc:
        logger.warning(f"preprocess: failed to load cache at {path}: {exc}")
        return None
    elapsed = time.perf_counter() - _t
    if payload.get("signature") != _signature(train_set, condition):
        logger.info(
            f"preprocess: cache at {path} is stale (signature mismatch); recomputing"
            f" (load attempted in {elapsed:.2f}s)"
        )
        return None
    logger.info(f"preprocess: loaded cache from {path} in {elapsed:.2f}s")
    return payload


def save(train_set, condition: List[str], payload: Dict[str, Any]) -> None:
    cache_dir = _cache_dir(train_set)
    if cache_dir is None:
        return
    path = _cache_path(cache_dir, condition)
    payload = dict(payload)
    payload["signature"] = _signature(train_set, condition)
    _t = time.perf_counter()
    try:
        torch.save(payload, path)
        logger.info(f"preprocess: saved cache to {path} in {time.perf_counter() - _t:.2f}s")
    except Exception as exc:
        logger.warning(f"preprocess: failed to save cache to {path}: {exc}")
