"""Training command for MolCraft CLI.

Adapted from scripts/train.py for package-level execution.
"""

from typing import Any, Dict, Optional, Tuple
import inspect
import math
import os
import pickle
import logging
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.train import (
    evaluate,
    DataModule,
    Logger,
    OptimSchedulerFactory,
    get_versioned_output_path,
)
from MolecularDiffusion.runmodes.train.eval import _resolve_task_config
from MolecularDiffusion.utils import (
    RankedLogger,
    task_wrapper,
    seed_everything,
)

log = RankedLogger(__name__, rank_zero_only=True)



def is_rank_zero():
    """Check if current process is rank zero."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def load_weights(task, ckpt_path, task_module=None):
    """Load model weights from a checkpoint file (weights only).

    This loads the state_dict from the checkpoint into the task model,
    ignoring optimizer/scheduler states and other metadata.
    Useful for fine-tuning or starting from a pre-trained model.

    If a RuntimeError occurs due to size mismatches (e.g., from adding
    new conditions), delegates to task_module.adjust_state_dict() for
    model-specific dimension adjustment.

    Args:
        task: The task model to load weights into.
        ckpt_path: Path to the checkpoint file.
        task_module: Optional task factory with adjust_state_dict() method
                     for handling dimension mismatches.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    log.info(f"Loading weights from: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Validate task_type if both checkpoint and task_module expose it
    if task_module is not None and hasattr(task_module, "task_type"):
        ckpt_task_type = (
            checkpoint.get("hyperparameters", {}).get("task_type")
            or checkpoint.get("task_type")
        )
        if ckpt_task_type is not None and ckpt_task_type != task_module.task_type:
            raise ValueError(
                f"Task type mismatch: checkpoint was trained as '{ckpt_task_type}' "
                f"but current config specifies '{task_module.task_type}'. "
                f"Update your config to use tasks: {ckpt_task_type} or point to the correct checkpoint."
            )
    # Extract the model state dict — original engine uses "ema_model"/"model" keys,
    # Lightning uses "state_dict" with "task." prefix.
    cleaned_state_dict = None
    if "ema_model" in checkpoint or "model" in checkpoint:
        # Original engine format: prefer EMA weights (matches engine.load() behaviour)
        raw_state_dict = checkpoint.get("ema_model") or checkpoint.get("model")
        cleaned_state_dict = raw_state_dict
    else:
        # Lightning format: strip "task." / "task.ema_model." prefixes
        state_dict = checkpoint.get("state_dict", {})
        cleaned_state_dict = {}
        ema_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("task.ema_model."):
                ema_state_dict[key[len("task.ema_model."):]] = value
            elif key.startswith("ema_model."):
                ema_state_dict[key[len("ema_model."):]] = value
            elif key.startswith("task."):
                cleaned_state_dict[key[5:]] = value
            else:
                cleaned_state_dict[key] = value
        # Fallback to root EMA dict if present (for newer Lightning engine dumps)
        if "ema_model_state_dict" in checkpoint:
            ema_state_dict = checkpoint["ema_model_state_dict"]

        # Prefer EMA weights when available
        if ema_state_dict:
            log.info(f"Found {len(ema_state_dict)} EMA parameters in Lightning checkpoint, using them")
            cleaned_state_dict = ema_state_dict

    if not cleaned_state_dict:
        raise ValueError(f"Could not extract a model state dict from checkpoint: {ckpt_path}")

    # Detect size mismatches before loading by comparing shapes
    model_state = task.state_dict()
    mismatched = [
        k for k in cleaned_state_dict
        if k in model_state and cleaned_state_dict[k].shape != model_state[k].shape
    ]
    if mismatched:
        if task_module is not None and hasattr(task_module, 'adjust_state_dict'):
            log.warning(
                f"\n{'='*60}\n"
                f"  ARCHITECTURE EXPANSION DETECTED ({len(mismatched)} tensor(s))\n"
                f"{'='*60}"
            )
            for k in mismatched:
                ckpt_shape = cleaned_state_dict[k].shape
                model_shape = model_state[k].shape
                log.warning(f"  [EXPAND] {k}\n"
                            f"           checkpoint: {list(ckpt_shape)} -> model: {list(model_shape)}")
            log.warning(f"{'='*60}\nDelegating to task module adjust_state_dict...")
            cleaned_state_dict = task_module.adjust_state_dict(cleaned_state_dict, task)
            mismatched = [
                k for k in cleaned_state_dict
                if k in model_state and cleaned_state_dict[k].shape != model_state[k].shape
            ]
            if mismatched:
                raise RuntimeError(
                    f"Size mismatch persists after adjust_state_dict for keys: {mismatched[:5]}"
                    f"{'...' if len(mismatched) > 5 else ''}. "
                    f"Check that your model config matches the checkpoint architecture."
                )
            log.info("Architecture expansion applied successfully via adjust_state_dict.")
        else:
            raise RuntimeError(
                f"{len(mismatched)} tensor(s) have mismatched shapes between the checkpoint and "
                f"the current model (e.g. {mismatched[0]}: ckpt={cleaned_state_dict[mismatched[0]].shape} "
                f"vs model={model_state[mismatched[0]].shape}). "
                f"Ensure your model config matches the checkpoint, or use chkpt_path instead of "
                f"load_weights_from to let the task factory handle dimension adjustment."
            )

    if cleaned_state_dict:
        # PyTorch strict=False still raises on size mismatches — filter those out first.
        # Keys that still mismatch after adjust_state_dict will be handled by
        # on_load_checkpoint inside lightning_wrapper using the dimension-expansion hook.
        model_state = task.state_dict()
        still_mismatched = [
            k for k in cleaned_state_dict
            if k in model_state and cleaned_state_dict[k].shape != model_state[k].shape
        ]
        if still_mismatched:
            log.warning(
                f"{len(still_mismatched)} key(s) still have shape mismatches after adjust_state_dict. "
                f"Skipping these for now — they will be handled by on_load_checkpoint in lightning_wrapper."
            )
            cleaned_state_dict = {k: v for k, v in cleaned_state_dict.items() if k not in still_mismatched}
        missing, unexpected = task.load_state_dict(cleaned_state_dict, strict=False)
        
        # Log new modules: keys present in the current model but absent from checkpoint
        new_modules = [k for k in model_state if k not in cleaned_state_dict and
                       not any(k in um for um in (unexpected or []))]
        if missing or new_modules:
            new_params = sum(model_state[k].numel() for k in new_modules if k in model_state)
            log.warning(
                f"\n{'='*60}\n"
                f"  NEW / UNINITIALIZED MODULES ({len(new_modules)} tensor(s), {new_params:,} param(s))\n"
                f"  These tensors are NOT in the checkpoint and will train from random init:\n"
                f"{'='*60}"
            )
            for k in new_modules[:10]:
                log.warning(f"  [NEW]  {k}  shape={list(model_state[k].shape)}")
            if len(new_modules) > 10:
                log.warning(f"  ... and {len(new_modules) - 10} more.")
            log.warning(f"{'='*60}")
        if unexpected:
            log.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        log.info(f"Successfully loaded {len(cleaned_state_dict)} parameters into task.")

    # Store EMA state for later initialization by EngineLightning.on_train_start
    if locals().get("ema_state_dict"):
        task._pending_ema_state = ema_state_dict
        log.info(f"Stored {len(ema_state_dict)} EMA parameters for deferred loading")



# Lightning imports (optional)
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning.callbacks import ModelSummary
    from MolecularDiffusion.core.engine_lightning import EngineLightning
    from MolecularDiffusion.data.lightning_data_module import MolecularDiffusionDataModule
    from MolecularDiffusion.core.lightning_callbacks import GenerativeEvalCallback
    LIGHTNING_AVAILABLE = True
except ImportError as e:
    LIGHTNING_AVAILABLE = False
    log.warning(f"PyTorch Lightning not found: {e}. Only original Engine available.")


def evaluate_and_save(i, solver, task_module, trainer_module, logger_module, versioned_ckpt_path, use_amp, **kwargs):
    """Helper to run evaluation and save checkpoints."""
    if hasattr(task_module.task, "sample"):
        output_generated_dir = os.path.join(versioned_ckpt_path, "generated_molecules")
        os.makedirs(output_generated_dir, exist_ok=True)
        return evaluate(
            task_module.task_type, solver, i, kwargs.get("best_metrics", torch.inf), kwargs.get("best_checkpoints", []),
            logger_module.logger, output_generated_dir=output_generated_dir,
            generative_analysis=kwargs.get("generative_analysis", False),
            n_samples=kwargs.get("n_samples", 100),
            metric=kwargs.get("metric", "Validity Relax and connected"),
            output_path=versioned_ckpt_path,
            use_amp=use_amp, precision=trainer_module.precision,
            use_posebuster=kwargs.get("use_posebuster", False),
            batch_size=kwargs.get("batch_size", 1),
            save_top_k=getattr(trainer_module, "save_top_k", 3),
            save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
        )
    else:
        return evaluate(
            task_module.task_type, solver, i, kwargs.get("best_metrics", torch.inf), kwargs.get("best_checkpoints", []),
            logger_module.logger, output_path=versioned_ckpt_path,
            save_top_k=getattr(trainer_module, "save_top_k", 3),
            save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
        )



def evaluate_and_save(i, solver, task_module, trainer_module, logger_module, versioned_ckpt_path, use_amp, **kwargs):
    """Run evaluation for any task type — routing is handled by TASK_REGISTRY inside evaluate()."""
    output_generated_dir = os.path.join(versioned_ckpt_path, "generated_molecules")
    os.makedirs(output_generated_dir, exist_ok=True)
    return evaluate(
        task_module.task_type, solver, i,
        kwargs.get("best_metrics", torch.inf), kwargs.get("best_checkpoints", []),
        logger_module.logger,
        output_path=versioned_ckpt_path,
        output_generated_dir=output_generated_dir,
        use_amp=use_amp,
        precision=trainer_module.precision,
        generative_analysis=kwargs.get("generative_analysis", False),
        n_samples=kwargs.get("n_samples", 100),
        metric=kwargs.get("metric", "Validity Relax and connected"),
        use_posebuster=kwargs.get("use_posebuster", False),
        batch_size=kwargs.get("batch_size", 1),
        save_top_k=getattr(trainer_module, "save_top_k", 3),
        save_every_val_epoch=getattr(trainer_module, "save_every_val_epoch", False),
        eval_metric_key=kwargs.get("eval_metric_key", None),
        eval_higher_is_better=kwargs.get("eval_higher_is_better", None),
    )


def engine_wrapper(task_module, data_module, trainer_module, logger_module,
                   resume_from_checkpoint=None, tags=None, **kwargs):
    """Training loop using original Engine."""
    trainer_module.get_optimizer()
    trainer_module.get_scheduler()
    
    solver = Engine(
        task_module.task,
        data_module.train_set,
        data_module.valid_set,
        data_module.test_set,
        batch_size=data_module.batch_size,
        collate_fn=data_module.collate_fn,
        optimizer=trainer_module.optimizer,
        ema_decay=trainer_module.ema_decay,
        scheduler=trainer_module.scheduler,
        clipping_gradient=trainer_module.gradient_clip_mode,
        clip_value=trainer_module.gradnorm_queue,
        logger=logger_module.logger,
        log_interval=logger_module.log_interval,
        name_wandb=logger_module.name_wandb,
        project_wandb=logger_module.project_wandb,
        dir_wandb=trainer_module.output_path,
        tags_wandb=tags,
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = solver.resume(resume_from_checkpoint, strict=False)
        log.info(f"Resumed from epoch {start_epoch}")
    
    use_amp = trainer_module.precision in ["bf16", 16]
    
    best_checkpoints = []
    if hasattr(task_module.task, "sample"):
        models_to_save = {"node": task_module.task.node_dist_model}
        if len(getattr(task_module, "condition_names", [])) > 0:
            models_to_save["prop"] = task_module.task.prop_dist_model
        if hasattr(task_module.task, "reference_freeze_mode"):
            models_to_save["reference_freeze_mode"] = task_module.task.reference_freeze_mode
        if getattr(task_module.task, "reference_feature_stats", None) is not None:
            models_to_save["reference_feature_stats"] = task_module.task.reference_feature_stats
        if getattr(task_module.task, "reference_scaffold", None) is not None:
            models_to_save["reference_scaffold"] = task_module.task.reference_scaffold
        if is_rank_zero():
            with open(os.path.join(trainer_module.output_path, "edm_stat.pkl"), "wb") as f:
                pickle.dump(models_to_save, f)

    # Determine correct initial sentinel from registry + engine overrides
    _eval_cfg = _resolve_task_config(task_module.task_type)
    if kwargs.get("eval_higher_is_better") is not None:
        from dataclasses import replace as _dcreplace
        _eval_cfg = _dcreplace(_eval_cfg, higher_is_better=kwargs["eval_higher_is_better"])
    best_metrics = -torch.inf if _eval_cfg.higher_is_better else torch.inf
    
    # Create versioned checkpoint folder (like Lightning's version_X folders)
    versioned_ckpt_path = get_versioned_output_path(trainer_module.output_path)
    
    # Adjust loop to continue from start_epoch
    for i in range(start_epoch, trainer_module.num_epochs):
        metric = solver.train(num_epoch=1, num_step=trainer_module.num_steps, use_amp=use_amp, precision=trainer_module.precision)
        
        # Check if we should stop because num_steps was reached
        if trainer_module.num_steps is not None and solver.meter.batch_id >= trainer_module.num_steps:
            log.info(f"Terminating training loop after epoch {i} because num_steps={trainer_module.num_steps} was reached.")
            # Trigger final evaluation if not already done
            if i % trainer_module.validation_interval != 0:
                best_metrics, best_checkpoints = evaluate_and_save(
                    i, solver, task_module, trainer_module, logger_module, 
                    versioned_ckpt_path, use_amp, best_metrics=best_metrics, 
                    best_checkpoints=best_checkpoints, **kwargs
                )
            break

        if i % trainer_module.validation_interval == 0 or i == trainer_module.num_epochs - 1:
            best_metrics, best_checkpoints = evaluate_and_save(
                i, solver, task_module, trainer_module, logger_module, 
                versioned_ckpt_path, use_amp, best_metrics=best_metrics, 
                best_checkpoints=best_checkpoints, **kwargs
            )
    return best_metrics, solver


def lightning_wrapper(task_module, data_module, trainer_module, logger_module, engine_cfg, 
                      ckpt_path=None, monitor_metric=None, monitor_mode=None, model_config=None, load_weights_from=None, **kwargs):
    """Training using PyTorch Lightning Trainer."""
    if not LIGHTNING_AVAILABLE:
        raise ImportError("PyTorch Lightning required. Install with: pip install pytorch-lightning")
    
    if hasattr(task_module.task, "preprocess"):
        log.info("Calling task.preprocess() for Lightning engine")
        _t_pre = time.perf_counter()
        result = task_module.task.preprocess(data_module.train_set)
        log.info(f"task.preprocess() completed in {time.perf_counter() - _t_pre:.2f}s")
        if result is not None:
            data_module.train_set, data_module.valid_set, data_module.test_set = result
    
    pl_data_module = MolecularDiffusionDataModule(
        data_module=data_module,
        batch_size=data_module.batch_size,
        num_workers=int(OmegaConf.select(engine_cfg, "num_workers", default=0) or 0),
        pin_memory=bool(OmegaConf.select(engine_cfg, "pin_memory", default=True)),
        persistent_workers=bool(OmegaConf.select(engine_cfg, "persistent_workers", default=False)),
        prefetch_factor=int(OmegaConf.select(engine_cfg, "prefetch_factor", default=1) or 1),
    )
    
    pl_module = EngineLightning(
        task=task_module.task,
        optimizer_config={
            "optimizer_choice": trainer_module.optimizer_choice,
            "lr": trainer_module.lr,
            "weight_decay": trainer_module.weight_decay,
            "betas": trainer_module.betas,
            "eps": trainer_module.eps,
            "lora_lr": getattr(trainer_module, "lora_lr", None),
        },
        scheduler_config={
            "scheduler": trainer_module.scheduler_choice,
            "scheduler_kwargs": trainer_module.scheduler_choice_kwargs,
        },
        model_config=model_config,
        monitor_metric=monitor_metric,
        ema_decay=trainer_module.ema_decay,
        gradnorm_queue=trainer_module.gradnorm_queue,
        gradient_clip_algorithm=getattr(trainer_module, 'gradient_clip_algorithm', 'adaptive'),
        sleep_every_N=int(OmegaConf.select(engine_cfg, "sleep_every_N", default=0) or 0),
        sleep_time=float(OmegaConf.select(engine_cfg, "sleep_time", default=60.0) or 60.0),
    )

    # Resuming from a checkpoint saved before a task added/removed a
    # get_extra_state()-backed key (or any other state_dict key) would
    # otherwise hard-fail Lightning's default strict=True checkpoint
    # restore. Generic escape hatch, not task-specific.
    pl_module.strict_loading = bool(
        OmegaConf.select(engine_cfg, "strict_loading", default=True)
    )

    # Inject factory dimension adjustment logic so Lightning can use it natively
    if hasattr(task_module, "adjust_state_dict"):
        pl_module._custom_state_dict_adjuster = task_module.adjust_state_dict
        
    callbacks = []
    
    if hasattr(task_module.task, "sample") and kwargs.get("generative_analysis"):
        callbacks.append(GenerativeEvalCallback(
            n_samples=kwargs.get("n_samples", 100),
            batch_size=kwargs.get("batch_size", 100),
            metric=kwargs.get("metric", "Validity Relax and connected"),
            output_dir=os.path.join(trainer_module.output_path, "generated_molecules"),
            use_posebuster=kwargs.get("use_posebuster", False),
            monitor_metric=monitor_metric,
        ))
    
    # Checkpoint callback
    # Handle OmegaConf ListConfig properly
    if monitor_metric is not None:
        # Convert OmegaConf types to Python types
        if OmegaConf.is_list(monitor_metric):
            monitor_metric_key = str(monitor_metric[0])
        elif isinstance(monitor_metric, (list, tuple)):
            monitor_metric_key = str(monitor_metric[0])
        else:
            monitor_metric_key = str(monitor_metric)
        mode = monitor_mode or ("min" if "loss" in monitor_metric_key else "max")
    elif hasattr(task_module.task, "sample") and kwargs.get("generative_analysis"):
        monitor_metric_key = f"gen/{kwargs.get('metric', 'Validity Relax and connected')}"
        mode = "max"
    else:
        monitor_metric_key = "val/loss"
        mode = "min"
    
    # Handle save_every_val_epoch
    save_top_k = trainer_module.save_top_k
    if getattr(trainer_module, "save_every_val_epoch", False) or kwargs.get("save_every_val_epoch", False):
        log.info("save_every_val_epoch=True: Overriding save_top_k to -1 (save all checkpoints)")
        save_top_k = -1

    callbacks.append(ModelCheckpoint(
        monitor=monitor_metric_key,
        mode=mode,
        save_top_k=save_top_k,
        filename=f"epoch={{epoch}}-{monitor_metric_key.replace('/', '_').replace(' ', '_')}={{{monitor_metric_key}:.3f}}",
        save_last=True,
    ))

    # Lightning writes `last.ckpt` only when the monitored top-k save fires at
    # that same step (pytorch_lightning/callbacks/model_checkpoint.py:517), and
    # its `on_train_end` fallback is skipped once any checkpoint was saved.
    # So on a run whose monitored metric stops improving, `last.ckpt` freezes
    # at the last improvement and every later epoch is discarded -- silently,
    # exit 0. A second, unmonitored callback fixes that: `save_top_k=0` means
    # it writes nothing during training (no extra files, no extra disk), and
    # `save_last=True` makes its `on_train_end` write `last.ckpt` from the
    # final training step. `enable_version_counter=False` so it overwrites
    # the file above in place instead of creating `last-v1.ckpt`.
    # The monitored callback is left untouched and stays first in
    # `trainer.checkpoint_callbacks` (Lightning preserves the relative order of
    # checkpoint callbacks), so `trainer.checkpoint_callback`, the top-k
    # filenames and the top-k selection are all unchanged.
    callbacks.append(ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        enable_version_counter=False,
    ))

    # Learning rate monitor for wandb logging
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # PL's default ModelSummary (max_depth=1) only shows the top-level
    # LightningModule attribute ("task"), not its submodules (e.g. a frozen
    # vae vs a trainable dynamics network) -- passing our own callback here
    # overrides that default without needing to touch enable_model_summary.
    callbacks.append(ModelSummary(
        max_depth=int(OmegaConf.select(engine_cfg, "model_summary_max_depth", default=1) or 1)
    ))
    
    trainer_config = OmegaConf.to_container(engine_cfg.trainer_config, resolve=True)
    if trainer_module.num_steps is not None:
        log.info(f"Setting max_steps to {trainer_module.num_steps} for Lightning trainer")
        trainer_config["max_steps"] = trainer_module.num_steps
        steps_per_epoch = max(1, len(data_module.train_set) // data_module.batch_size)
        trainer_config["max_epochs"] = math.ceil(trainer_module.num_steps / steps_per_epoch)
        log.info(f"Estimated max_epochs={trainer_config['max_epochs']} ({steps_per_epoch} steps/epoch)")
    else:
        trainer_config["max_epochs"] = trainer_module.num_epochs

    precision_map = {32: 32, 16: "16-mixed", "16": "16-mixed", "bf16": "bf16-mixed"}
    trainer_config["precision"] = precision_map.get(trainer_config.get("precision", 32), 32)
    
    if logger_module.logger == "wandb":
        pl_logger = pl.loggers.WandbLogger(
            project=logger_module.project_wandb,
            name=logger_module.name_wandb,
            save_dir=trainer_module.output_path,
            tags=kwargs.get("tags") or None,
        )
    else:
        pl_logger = True
    
    trainer = hydra.utils.instantiate(trainer_config, callbacks=callbacks, logger=pl_logger)
    
    if ckpt_path:
        fit_kwargs = {"datamodule": pl_data_module, "ckpt_path": ckpt_path}
        if "weights_only" in inspect.signature(trainer.fit).parameters:
            fit_kwargs["weights_only"] = False
        trainer.fit(pl_module, **fit_kwargs)
    else:
        if load_weights_from:
            log.info(f"Triggering EngineLightning.on_load_checkpoint framework loader from {load_weights_from}")
            checkpoint = torch.load(load_weights_from, map_location="cpu", weights_only=False)

            # Preserve distribution models computed from the NEW dataset by preprocess().
            # on_load_checkpoint would overwrite them with the old checkpoint's versions,
            # but when fine-tuning on a new dataset the distributions must reflect the new data.
            saved_node_dist = getattr(pl_module.task, 'node_dist_model', None)
            saved_prop_dist = getattr(pl_module.task, 'prop_dist_model', None)
            saved_n_node_dist = getattr(pl_module.task, 'n_node_dist', None)
            saved_property_norms = getattr(pl_module.task, 'property_norms', None)

            pl_module.on_load_checkpoint(checkpoint)

            def _restore_task_attr(attr_name: str, value, fallback_attr: str = None) -> bool:
                """Best-effort restore for task attrs that may be read-only properties."""
                if value is None:
                    return False
                try:
                    setattr(pl_module.task, attr_name, value)
                    return True
                except AttributeError:
                    if fallback_attr is not None:
                        try:
                            setattr(pl_module.task, fallback_attr, value)
                            return True
                        except Exception:
                            return False
                    return False
                except Exception:
                    return False

            # Restore new-dataset distribution models over the checkpoint's stale ones
            if _restore_task_attr("node_dist_model", saved_node_dist, fallback_attr="_node_dist_model"):
                log.info("Restored node_dist_model from new dataset (overriding checkpoint)")
            elif saved_node_dist is not None:
                log.warning("Could not restore node_dist_model (read-only and no writable fallback found)")

            if _restore_task_attr("prop_dist_model", saved_prop_dist):
                log.info("Restored prop_dist_model from new dataset (overriding checkpoint)")
            elif saved_prop_dist is not None:
                log.warning("Could not restore prop_dist_model")

            if _restore_task_attr("n_node_dist", saved_n_node_dist):
                pass
            elif saved_n_node_dist is not None:
                log.warning("Could not restore n_node_dist")

            if _restore_task_attr("property_norms", saved_property_norms):
                pass
            elif saved_property_norms is not None:
                log.warning("Could not restore property_norms")

        trainer.fit(pl_module, datamodule=pl_data_module)
    
    return trainer.callback_metrics, trainer


def _assert_train_config(cfg: DictConfig):
    """Crash early if the user passed a non-train config to the train command."""
    has_interference = "interference" in cfg
    has_chkpt_dir = "chkpt_directory" in cfg
    has_trainer = "trainer" in cfg

    if has_interference or has_chkpt_dir:
        mismatched = []
        if has_interference:
            mismatched.append("'interference'")
        if has_chkpt_dir:
            mismatched.append("'chkpt_directory'")
        raise ValueError(
            f"Config contains {', '.join(mismatched)} — this looks like a generation config, "
            f"not a training config. Did you mean: molcraft generate <config>?"
        )
    if not has_trainer:
        raise ValueError(
            "Config is missing required 'trainer' block. "
            "Please provide a training config (e.g., configs/train.yaml)."
        )


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Main training function."""
    _assert_train_config(cfg)
    output_path = cfg.trainer.output_path
    os.makedirs(output_path, exist_ok=True)

    if is_rank_zero():
        config_path = os.path.join(output_path, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Configuration saved to {config_path}")
    
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # A config with no `engine:` block is valid (the "original" engine is the
    # default). Must be a Config, not a bare dict -- OmegaConf.select() only
    # accepts Containers, and every wrapper below selects out of this.
    engine_cfg = cfg.get("engine", None) or OmegaConf.create({})
    eval_metric_key = engine_cfg.get("eval_metric_key", None)
    eval_higher_is_better = engine_cfg.get("eval_higher_is_better", None)


    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(cfg.data, task_type=cfg.tasks.task_type)
    _t_load = time.perf_counter()
    data_module.load()
    log.info(f"data_module.load() completed in {time.perf_counter() - _t_load:.2f}s")
    
    log.info(f"Instantiating task <{cfg.tasks._target_}>")
    data_point_chk = data_module.train_set[0]
    if isinstance(data_point_chk, dict):
        node_feature_0 = data_point_chk.get("node_feature", None)
        if node_feature_0 is None and "graph" in data_point_chk:
            node_feature_0 = getattr(data_point_chk["graph"], "x", None)
    else:
        node_feature_0 = getattr(data_point_chk, "node_feature", None)
    if node_feature_0 is not None:
        n_dim = node_feature_0.shape[1]
    else:
        try:
            node_feature_0 = getattr(data_point_chk, "x", None)
            n_dim = node_feature_0.shape[1]
        except:
            n_dim = 0

    factory_cfg = cfg.tasks
    overrides = {}
    data_atom_vocab = None
    
    # Declarative seam: inject the actual train_set only into factories that
    # explicitly declare a `train_set` __init__ parameter (e.g. for building an
    # atom-count histogram at construction time). Keyed off the signature, not
    # a hardcoded _target_ substring list, so new factories opt in by declaring
    # the parameter rather than requiring a core edit. Factories that only
    # swallow **kwargs (which never names train_set) are deliberately excluded.
    factory_params = inspect.signature(
        hydra.utils.get_class(factory_cfg._target_).__init__
    ).parameters
    if "train_set" in factory_params:
        overrides["train_set"] = data_module.train_set
        if "condition_names" in factory_cfg:
            overrides["task_names"] = factory_cfg.condition_names
            
    if "atom_vocab" in cfg.data:
        data_atom_vocab = list(cfg.data.atom_vocab)
        
    if cfg.data.get("allow_unknown", False):
        if data_atom_vocab is None:
            data_atom_vocab = []
        data_atom_vocab.append("Suisei")

    if data_atom_vocab is not None:
        overrides["atom_vocab"] = data_atom_vocab

    base_feature_dim = len(data_atom_vocab or list(factory_cfg.get("atom_vocab", [])))
    if not cfg.data.get("use_ohe_feature", True):
        base_feature_dim = 0
    inferred_extra_dim = max(0, int(n_dim) - base_feature_dim)
    configured_extra_norm_values = list(factory_cfg.get("extra_norm_values", []) or [])
    if inferred_extra_dim > 0:
        if configured_extra_norm_values and len(configured_extra_norm_values) != inferred_extra_dim:
            raise ValueError(
                "Node feature dimensionality mismatch: dataset provides "
                f"{n_dim} node feature columns ({base_feature_dim} atom OHE + "
                f"{inferred_extra_dim} extra), but tasks.extra_norm_values has "
                f"{len(configured_extra_norm_values)} entries. Set it to length "
                f"{inferred_extra_dim}, or remove it to use unit scaling."
            )
        if not configured_extra_norm_values:
            overrides["extra_norm_values"] = [1.0] * inferred_extra_dim
        log.info(
            "Detected %d extra node feature dimensions from data.node_feature_choice=%s",
            inferred_extra_dim,
            cfg.data.get("node_feature_choice", None),
        )

    node_feature_choice = cfg.data.get("node_feature_choice", None)
    if node_feature_choice is not None:
        overrides["node_feature"] = node_feature_choice
        overrides["node_feature_choice"] = node_feature_choice
    overrides["node_feature_dim"] = inferred_extra_dim
    
    if cfg.tasks.get("metrics", None) == "valid_posebuster":
        overrides["use_posebuster"] = True
        try:
            import posebusters
        except ImportError:
            log.warning("PoseBuster not installed. Falling back to 'Validity Relax and connected'.")
            overrides["use_posebuster"] = False
            overrides["metrics"] = ["Validity Relax and connected"]
        
    task_module = hydra.utils.instantiate(factory_cfg, **overrides)
    task_module.build()
    task_module.task.node_feature = node_feature_choice
    task_module.task.node_feature_choice = node_feature_choice
    task_module.task.node_feature_dim = inferred_extra_dim
    
    # Optional: Load weights from checkpoint (without resuming full state)
    if cfg.trainer.get("load_weights_from"):
        load_weights(task_module.task, cfg.trainer.load_weights_from, task_module=task_module)
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(
        cfg.trainer, parameters=task_module.task.parameters()
    )

    name_wandb = trainer_module.output_path.split('/')[-1] if "/" in trainer_module.output_path else trainer_module.output_path
    log.info(f"Instantiating loggers... <{cfg.logger._target_}>")
    logger_module: Logger = hydra.utils.instantiate(cfg.logger, name_wandb=name_wandb)
    
    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "task": task_module,
        "trainer": trainer_module,
        "logger": logger_module,
    }

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    engine_type = cfg.get("engine", {}).get("engine_type", "original")
    log.info(f"Using engine: {engine_type}")



    # Extract top-level tags for wandb
    tags = cfg.get("tags", None)
    if tags is not None:
        try:
            from omegaconf import OmegaConf as _OC
            tags = _OC.to_container(tags, resolve=True)
        except Exception:
            tags = list(tags)

    # Extract generative parameters with fallbacks
    tasks_cfg = cfg.get("tasks", {})
    gen_analysis = cfg.get("generative_analysis", tasks_cfg.get("generative_analysis", False))
    n_samples = cfg.get("n_samples", tasks_cfg.get("n_samples", 100))
    metric = cfg.get("metrics", cfg.get("metric", tasks_cfg.get("metrics", "Validity Relax and connected")))
    use_posebuster = cfg.get("use_posebuster", tasks_cfg.get("use_posebuster", False))
    # Preference: cli/top-level -> tasks -> data -> default
    gen_batch_size = cfg.get("batch_size", tasks_cfg.get("batch_size", cfg.data.get("batch_size", 100)))

    if engine_type == "lightning":
        # Always save model_config for checkpoint reconstruction (VAE, LDM, etc.)
        model_config = OmegaConf.to_container(factory_cfg, resolve=True)
        for k, v in overrides.items():
            if k != "train_set":
                model_config[k] = v

        if hasattr(task_module.task, "sample"):
            metrics = lightning_wrapper(
                task_module, data_module, trainer_module, logger_module,
                engine_cfg=engine_cfg,
                generative_analysis=gen_analysis, n_samples=n_samples,
                metric=metric, use_posebuster=use_posebuster, batch_size=gen_batch_size,
                ckpt_path=cfg.trainer.get("resume_from_checkpoint", None),
                monitor_metric=cfg.trainer.get("monitor_metric", None),
                monitor_mode=cfg.trainer.get("monitor_mode", None),
                model_config=model_config,
                tags=tags,
                load_weights_from=cfg.trainer.get("load_weights_from", None),
            )
        else:
            metrics = lightning_wrapper(
                task_module, data_module, trainer_module, logger_module,
                engine_cfg=engine_cfg,
                ckpt_path=cfg.trainer.get("resume_from_checkpoint", None),
                monitor_metric=cfg.trainer.get("monitor_metric", None),
                monitor_mode=cfg.trainer.get("monitor_mode", None),
                model_config=model_config,
                tags=tags,
                load_weights_from=cfg.trainer.get("load_weights_from", None),
            )
    
    elif engine_type == "original":
        resume_ckpt = cfg.trainer.get("resume_from_checkpoint", None)
        metrics = engine_wrapper(
            task_module, data_module, trainer_module, logger_module,
            resume_from_checkpoint=resume_ckpt,
            generative_analysis=gen_analysis,
            n_samples=n_samples,
            metric=metric,
            use_posebuster=use_posebuster,
            batch_size=gen_batch_size,
            tags=tags,
            eval_metric_key=eval_metric_key,
            eval_higher_is_better=eval_higher_is_better,
        )
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")

    return metrics, object_dict


def log_hyperparameters(object_dict: dict):
    """Log hyperparameters for debugging."""
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
                        if isinstance(v, torch.nn.Module):
                            log.info(f"{k}: {v.__class__.__name__}")
                        else:
                            log.info(f"{k}: {v}")
        log.info(f"{'=' * (44 + len(name))}\n")

    if "task" in object_dict and hasattr(object_dict["task"], "task"):
        model = object_dict["task"].task
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"{'=' * 20} MODEL PARAMS {'=' * 20}")
        log.info(f"model/params/total: {total}")
        log.info(f"model/params/trainable: {trainable}")
        log.info("=" * 54 + "\n")
    
    log.info("========== End of Hyperparameters ==========\n")


def train_main(cfg: DictConfig):
    """Entry point for CLI train command."""
    metric, _ = train(cfg)
    return metric
