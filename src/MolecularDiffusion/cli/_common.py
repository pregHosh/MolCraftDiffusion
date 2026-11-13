"""Helpers shared by the CLI entry points (train, generate, predict, eval_predict)."""

import torch
from omegaconf import OmegaConf


def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log_hyperparameters(
    object_dict: dict, log, *, model_params: bool = True, collapse_modules: bool = False
):
    """Log hyperparameters for debugging.

    ``log`` is the caller's logger, so lines keep the caller's logger name.
    ``model_params`` adds the parameter-count block for ``object_dict["task"].task``.
    ``collapse_modules`` prints an ``nn.Module`` attribute as its class name only.
    """
    if not is_rank_zero():
        return

    log.info("\n========== Logging Hyperparameters ==========\n")
    for name, obj in object_dict.items():
        log.info(f"{'=' * 20} {name.upper()} {'=' * 20}")
        if name == "cfg":
            if isinstance(obj, dict):
                log.info("\n" + OmegaConf.to_yaml(OmegaConf.create(obj)))
            else:
                log.info("\n" + OmegaConf.to_yaml(obj))
        else:
            if hasattr(obj, '__dict__'):
                for k, v in vars(obj).items():
                    if not k.startswith("_"):
                        if collapse_modules and isinstance(v, torch.nn.Module):
                            log.info(f"{k}: {v.__class__.__name__}")
                        else:
                            log.info(f"{k}: {v}")
        log.info(f"{'=' * (44 + len(name))}\n")

    if model_params and "task" in object_dict and hasattr(object_dict["task"], "task"):
        model = object_dict["task"].task
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{'=' * 20} MODEL PARAMS {'=' * 20}")
        log.info(f"model/params/total: {total}")
        log.info(f"model/params/trainable: {trainable}")
        log.info("=" * 54 + "\n")

    log.info("========== End of Hyperparameters ==========\n")
