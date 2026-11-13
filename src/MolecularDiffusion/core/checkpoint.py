"""Checkpoint helpers shared by the CLI and the task factories.

Which weights a checkpoint yields differs by format on purpose:

* ``engine_weights``: original-``Engine`` checkpoints (``ema_model`` /
  ``model`` keys), EMA preferred.
* ``extract_clean_state_dict``: Lightning checkpoints (``state_dict`` with
  ``task.`` prefixes, new ``ema_model_state_dict`` or old embedded
  ``ema_model.*`` keys) and ``prefer_ema=False`` for raw/LoRA weights.
* ``cli.train.load_weights`` keeps its own fine-tune rules (shape adjustment
  through ``adjust_state_dict``), so it is not merged into these.
"""

import glob
import os
import re

from MolecularDiffusion.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def engine_weights(ckpt):
    """Weights of an original-Engine checkpoint: the EMA copy, else the raw model."""
    return ckpt.get("ema_model") or ckpt.get("model")


def get_ckpt_meta(checkpoint):
    """Extract task_type and condition_names from any checkpoint format."""
    hparams = (
        checkpoint.get("hyperparameters")
        or checkpoint.get("hyper_parameters")
        or {}
    )
    task_type = hparams.get("task_type") or checkpoint.get("task_type")
    condition_names = hparams.get("condition_names") or checkpoint.get("condition_names")
    return task_type, condition_names


def validate_task_type(checkpoint, expected_task_type):
    """Raise ValueError if checkpoint task_type doesn't match expected_task_type."""
    if expected_task_type is None:
        return
    ckpt_task_type, _ = get_ckpt_meta(checkpoint)
    # LDM/VAE checkpoints written before the configured task_type was propagated
    # into the task carry these generic placeholders, which match no config name.
    # They say nothing about the architecture, so they cannot be validated.
    #
    # This is only safe because neither string is a live task name: no
    # configs/tasks/*.yaml declares `task_type: ldm` or `task_type: vae` (they
    # use `diffusion_adit`, `vae_transformer`, ...). Naming a future task config
    # exactly `ldm` or `vae` would silently disable validation for it.
    if ckpt_task_type in ("ldm", "vae"):
        return
    if ckpt_task_type is not None and ckpt_task_type != expected_task_type:
        raise ValueError(
            f"Task type mismatch: checkpoint was trained as '{ckpt_task_type}' "
            f"but current config specifies '{expected_task_type}'. "
            f"Update your config to use tasks: {ckpt_task_type} or point to the correct checkpoint."
        )


def select_ckpt_file(path):
    """Resolve a checkpoint path that may be a .ckpt file or a directory.

    For a directory, pick the highest-metric checkpoint (by filename), else last.ckpt,
    else the first .ckpt found — mirroring load_model's selection logic.
    """
    if os.path.isfile(path):
        return path
    ckpt_files = glob.glob(os.path.join(path, '*.ckpt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt file found at {path}")
    best_metric = -1.0
    best = None
    for f in ckpt_files:
        m = re.search(r"(?:metric|val)[_=](\d+\.?\d*)", os.path.basename(f))
        if m:
            v = float(m.group(1))
            if v > best_metric:
                best_metric = v
                best = f
    if best is None:
        last = os.path.join(path, 'last.ckpt')
        best = last if os.path.exists(last) else ckpt_files[0]
    return best


def extract_ema_state_dict(ckpt):
    """EMA weights of a checkpoint as {bare module name: tensor}, or None if it has none.

    Looks for the new-format top-level `ema_model_state_dict` first, then old-format
    `ema_model.*` keys embedded inside `state_dict`.
    """
    # New-format top-level EMA dict (keys already unprefixed).
    ema = ckpt.get("ema_model_state_dict") or ckpt.get("ema_model")
    if ema:
        return {k[len("task."):] if k.startswith("task.") else k: v for k, v in ema.items()}

    # Old-format EMA embedded as `...ema_model.<name>` keys inside state_dict.
    sd = ckpt.get("state_dict") or ckpt.get("model")
    if sd:
        marker = "ema_model."
        ema_keys = [k for k in sd if marker in k]
        if ema_keys:
            return {k[k.index(marker) + len(marker):]: sd[k] for k in ema_keys}
    return None


def extract_clean_state_dict(ckpt, prefer_ema=True):
    """Return a {name: tensor} dict from a checkpoint, normalized to bare module names.

    Weight preference (when prefer_ema): new-format top-level `ema_model_state_dict`, then
    old-format `ema_model.*` keys embedded inside `state_dict`, then raw weights. Strips a
    leading `task.` and any `ema_model.` prefix; the raw fallback drops embedded EMA keys so
    they don't collide with the real weights.
    """
    if prefer_ema:
        ema = extract_ema_state_dict(ckpt)
        if ema:
            return ema

    sd = ckpt.get("state_dict") or ckpt.get("model")
    if sd is None:
        raise KeyError("Checkpoint has no loadable state dict (state_dict/ema_model_state_dict).")

    # Raw weights — drop any embedded EMA keys so they don't shadow the real ones.
    cleaned = {}
    for k, v in sd.items():
        if "ema_model." in k:
            continue
        cleaned[k[len("task."):] if k.startswith("task.") else k] = v
    return cleaned


def apply_total_step_override(task, total_step):
    """Override the model's sampling-step count (T / fm_num_timesteps / interpolant)."""
    if total_step <= 0:
        return
    override_occured = False
    if hasattr(task, 'model'):
        m_obj = task.model
        if hasattr(m_obj, 'T'):
            log.info(f"Overriding diffusion steps (T): {m_obj.T} -> {total_step}")
            m_obj.T = total_step
            override_occured = True
        elif hasattr(m_obj, 'fm_num_timesteps'):
            log.info(f"Overriding flow-matching steps: {m_obj.fm_num_timesteps} -> {total_step}")
            m_obj.fm_num_timesteps = total_step
            override_occured = True
    if hasattr(task, 'T') and not override_occured:
        task.T = total_step
        override_occured = True
    if hasattr(task, 'interpolant') and hasattr(task.interpolant, 'num_timesteps'):
        task.interpolant.num_timesteps = total_step
