"""
PyTorch Lightning wrapper for MolecularDiffusion Task classes.

This module provides a LightningModule that wraps existing Task classes
(from tasks_esen.py, tasks_egt.py, tasks_egcl.py) to enable training
with PyTorch Lightning infrastructure.
"""

import copy
import logging
import time
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch import nn, optim

import hydra
from omegaconf import OmegaConf
from MolecularDiffusion import utils, core
from MolecularDiffusion.callbacks import EMA, Queue, gradient_clipping

logger = logging.getLogger(__name__)


@core.Registry.register("EngineLightning")
class EngineLightning(pl.LightningModule, core.Configurable):
    """
    Lightning wrapper for MolecularDiffusion Task classes.

    This wraps any task (regression, guidance, diffusion) and adapts it
    to Lightning's training interface.

    Parameters:
        task (nn.Module): The task module (e.g., PropertyPrediction, GeomMolecularGenerative)
        optimizer_config (dict): Configuration for optimizer
            - optimizer_choice: str, one of ["adam", "adamw", "amsgrad", "radam"]
            - lr: float
            - weight_decay: float
            - betas: tuple
            - eps: float
        scheduler_config (dict): Configuration for learning rate scheduler
            - scheduler: str or None
            - scheduler_kwargs: dict
    """

    def __init__(
        self,
        optimizer_config: Dict[str, Any],
        task: Optional[nn.Module] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        monitor_metric: Optional[Any] = None,
        model_config: Optional[Dict[str, Any]] = None,
        ema_decay: float = 0.0,
        gradnorm_queue: Optional[Queue] = None,
        gradient_clip_algorithm: str = "adaptive",
        sleep_every_N: int = 0,
        sleep_time: float = 60.0,
    ):
        super().__init__()

        # If task is not provided, try to build it from model_config (reloading from checkpoint)
        if task is None:
            if model_config is None:
                raise ValueError("Either 'task' or 'model_config' must be provided.")

            # Convert dict back to DictConfig if needed for hydration
            if isinstance(model_config, dict):
                config = OmegaConf.create(model_config)
            else:
                config = copy.deepcopy(model_config)

            # CRITICAL: Clear chkpt_path during inference/checkpoint loading.
            # The saved model_config may reference the original training checkpoint path
            # which no longer exists. Since weights come from the Lightning checkpoint's
            # state_dict, we don't need to reload from the old chkpt_path.
            OmegaConf.set_struct(config, False)
            if 'chkpt_path' in config:
                logger.info(f"Clearing stale chkpt_path from model_config: {config.chkpt_path}")
                config.chkpt_path = None

            logger.info(f"Building task from model_config: {config._target_}")
            # We don't have atom_vocab here easily unless saved in config or passed in kwargs
            # Assuming model_config is complete enough or factory handles defaults
            task_factory = hydra.utils.instantiate(config)
            task = task_factory.build()

        self.task = task
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config or {}
        self.monitor_metric = monitor_metric
        self.model_config = model_config
        self.ema_decay = ema_decay

        # Gradient clipping
        self.gradnorm_queue = gradnorm_queue
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self._clipper = gradient_clipping(m=1) if gradnorm_queue is not None else None
        self._nonfinite_grad_steps = 0
        self._consecutive_bad_batches = 0
        self.max_consecutive_bad_batches = 20

        # Periodic sleep (e.g., for GPU thermal throttling relief)
        self.sleep_every_N = sleep_every_N
        self.sleep_time = sleep_time

        # EMA model stored in list to prevent auto-registration as submodule
        # (direct assignment as self.ema_model = nn.Module causes state_dict to include ema_model.* keys)
        self._ema_models = []  # List wrapper to prevent submodule detection
        self._ema_helper = None

        # Save hyperparameters
        self.save_hyperparameters(ignore=['task', 'gradnorm_queue'])

    @property
    def ema_model(self):
        """Access EMA model from list wrapper."""
        return self._ema_models[0] if self._ema_models else None

    @ema_model.setter
    def ema_model(self, value):
        """Set EMA model in list wrapper."""
        if value is None:
            self._ema_models = []
        elif self._ema_models:
            self._ema_models[0] = value
        else:
            self._ema_models.append(value)

    def setup(self, stage: str):
        """
        Called at the beginning of fit, validate, test, or predict.
        """
        self._update_task_device()

    def on_train_start(self):
        """Ensure device is correct after model transfer and initialize EMA."""
        self._update_task_device()

        # Log key config to wandb (if active)
        self._log_config_to_wandb()

        # Initialize EMA model if not already done
        if self.ema_decay > 0 and self.ema_model is None:
            logger.info(f"Initializing EMA model with decay={self.ema_decay}")
            ema = copy.deepcopy(self.task)
            ema.eval()
            for param in ema.parameters():
                param.requires_grad = False

            # Load pending EMA state if resuming from checkpoint
            # Check both self (from on_load_checkpoint) and task (from load_weights_from)
            pending_state = None
            if hasattr(self, '_pending_ema_state') and self._pending_ema_state is not None:
                pending_state = self._pending_ema_state
                self._pending_ema_state = None
            elif hasattr(self.task, '_pending_ema_state') and self.task._pending_ema_state is not None:
                pending_state = self.task._pending_ema_state
                del self.task._pending_ema_state

            if pending_state is not None:
                # Apply dimension-expansion adjustment to EMA state before loading,
                # in case the model has been expanded (e.g. new conditions) compared to checkpoint.
                if hasattr(self, "_custom_state_dict_adjuster"):
                    orig_shapes = {k: v.shape for k, v in pending_state.items()}
                    pending_state = self._custom_state_dict_adjuster(pending_state, self.task)
                    expanded = [(k, orig_shapes[k], pending_state[k].shape)
                                for k in orig_shapes
                                if k in pending_state and pending_state[k].shape != orig_shapes[k]]
                    if expanded:
                        logger.info(
                            f"\n{'='*60}\n"
                            f"  EMA DIMENSION EXPANSION ({len(expanded)} tensor(s))\n"
                            f"{'='*60}"
                        )
                        for k, old_s, new_s in expanded:
                            logger.info(f"  [EMA EXPANDED] {k}: {list(old_s)} -> {list(new_s)}")
                        logger.info(f"{'='*60}")
                ema.load_state_dict(pending_state, strict=False)
                logger.info("Loaded EMA model state from checkpoint")

            self.ema_model = ema  # Assign via property setter
            self._ema_helper = EMA(self.ema_decay)

    def on_validation_start(self):
        """Ensure device is correct after model transfer."""
        self._update_task_device()
        # Call task's validation epoch start if available (e.g., for VAE reconstruction)
        if hasattr(self.task, 'on_validation_epoch_start'):
            self.task.on_validation_epoch_start()

    def on_test_start(self):
        """Ensure device is correct after model transfer."""
        self._update_task_device()

    def _update_task_device(self):
        """Recursively set device attribute for all submodules."""
        def set_device_recursive(module):
            # We set it even if it exists to ensure it points to the current correct device
            # Wrap in try-except because some modules might have device as a read-only property
            try:
                module.device = self.device
            except AttributeError:
                pass

            for child in module.children():
                set_device_recursive(child)

        set_device_recursive(self.task)

    def _log_config_to_wandb(self):
        """Log key config to wandb if WandbLogger is active."""
        # Only log on rank 0
        if self.trainer.global_rank != 0:
            return

        # Check if any logger is WandbLogger
        wandb_logger = None
        if self.trainer.logger is not None:
            from pytorch_lightning.loggers import WandbLogger
            if isinstance(self.trainer.logger, WandbLogger):
                wandb_logger = self.trainer.logger
            elif hasattr(self.trainer, 'loggers'):
                for lg in self.trainer.loggers:
                    if isinstance(lg, WandbLogger):
                        wandb_logger = lg
                        break

        if wandb_logger is None:
            return

        # Build config dict from hyperparameters
        key_config = {}

        # Add model_config if available
        if self.model_config is not None:
            if isinstance(self.model_config, dict):
                # Filter out non-serializable items
                for k, v in self.model_config.items():
                    if k.startswith('_'):
                        continue
                    if isinstance(v, (str, int, float, bool, list, tuple, type(None))):
                        key_config[k] = v
                    elif hasattr(v, '__name__'):  # Class or function
                        key_config[k] = v.__name__

        # Add optimizer/scheduler config
        if self.optimizer_config:
            key_config['optimizer'] = self.optimizer_config
        if self.scheduler_config:
            key_config['scheduler'] = self.scheduler_config

        # Add some additional useful info
        key_config['task_class'] = self.task.__class__.__name__
        key_config['ema_decay'] = self.ema_decay
        key_config['gradient_clip_algorithm'] = self.gradient_clip_algorithm

        # Add diffusion steps if applicable
        if hasattr(self.task, 'model') and hasattr(self.task.model, 'T'):
            key_config['diffusion_steps'] = self.task.model.T
        elif hasattr(self.task, 'T'):
            key_config['diffusion_steps'] = self.task.T

        # Log to wandb
        try:
            wandb_logger.experiment.config.update(key_config, allow_val_change=True)
            logger.info("Logged key config to wandb")
        except Exception as e:
            logger.warning(f"Failed to log config to wandb: {e}")

    def on_save_checkpoint(self, checkpoint):
        """Save task-specific distribution data to checkpoint.

        This saves:
        - For Tabasco: data_stats dictionary containing num_atoms_histogram
        - For other diffusion models: node_dist_model and prop_dist_model
        - EMA model state if enabled

        This ensures all data needed for generation is embedded in the checkpoint.
        """
        # Save task_type and condition_names for validation
        if hasattr(self.task, 'task_type'):
            checkpoint['task_type'] = self.task.task_type
        if hasattr(self.task, 'condition'):
            checkpoint['condition_names'] = list(self.task.condition)

        # Save EMA model state if available
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            logger.info("Saved EMA model state to checkpoint")

        # For Tabasco: save data_stats dictionary
        if hasattr(self.task, 'tabasco_model'):
            if hasattr(self.task.tabasco_model, 'data_stats'):
                checkpoint['data_stats'] = self.task.tabasco_model.data_stats
                logger.info("Saved Tabasco data_stats to checkpoint")

        # For other diffusion tasks: save node/prop distribution models
        if hasattr(self.task, 'node_dist_model') and self.task.node_dist_model is not None:
            checkpoint['node_dist_model'] = self.task.node_dist_model
            logger.info("Saved node_dist_model to checkpoint")

        if hasattr(self.task, 'n_node_dist'):
            checkpoint['n_node_dist'] = self.task.n_node_dist
            logger.info("Saved n_node_dist to checkpoint")

        if hasattr(self.task, 'prop_dist_model') and self.task.prop_dist_model is not None:
            checkpoint['prop_dist_model'] = self.task.prop_dist_model
            logger.info("Saved prop_dist_model to checkpoint")

        # property_norms (mean/mad/min/max per condition, used by
        # normalize_condition) is a separate, lighter-weight attribute than
        # prop_dist_model -- TABASCO's preprocess() builds only this, not a
        # full DistributionProperty. Generic: benefits any task exposing it.
        if hasattr(self.task, 'property_norms') and self.task.property_norms is not None:
            checkpoint['property_norms'] = self.task.property_norms
            logger.info("Saved property_norms to checkpoint")

        if (
            hasattr(self.task, 'reference_indices')
            and self.task.reference_indices is not None
        ):
            checkpoint['reference_indices'] = self.task.reference_indices
        if hasattr(self.task, 'reference_freeze_mode'):
            checkpoint['reference_freeze_mode'] = self.task.reference_freeze_mode
        if (
            hasattr(self.task, 'reference_feature_stats')
            and self.task.reference_feature_stats is not None
        ):
            checkpoint['reference_feature_stats'] = self.task.reference_feature_stats
        if (
            hasattr(self.task, 'reference_scaffold')
            and self.task.reference_scaffold is not None
        ):
            checkpoint['reference_scaffold'] = self.task.reference_scaffold
            logger.info(
                f"Saved reference_scaffold to checkpoint, "
                f"shape {list(self.task.reference_scaffold.shape)}"
            )

    def on_load_checkpoint(self, checkpoint):
        """Load EMA model state and distribution models from checkpoint.

        Handles:
        1. EMA state (new format: 'ema_model_state_dict' or old format: 'ema_model.*' keys)
        2. Distribution models: node_dist_model, prop_dist_model, n_node_dist
        3. Tabasco data_stats
        """
        # Restore distribution models from checkpoint ONLY if the task doesn't
        # already have them (i.e., preprocess() hasn't been called with a new dataset).
        # When fine-tuning on a new dataset, preprocess() computes fresh distribution
        # models that must NOT be overwritten by the old checkpoint's stale ones.
        # Tabasco's data_stats must be restored BEFORE the freshness probes below.
        # Its node_dist_model is a lazy property that builds a sampler from
        # tabasco_model.data_stats and caches it in _node_dist_model. Probing the
        # property while data_stats is still empty therefore caches a sampler whose
        # histogram is empty, which makes the task look like it carries a fresh
        # distribution and silently falls back to a uniform 5-29 QM9-like size range.
        # Restoring the stats first, and dropping any cached sampler, lets the
        # property rebuild from the checkpoint's own histogram.
        if 'data_stats' in checkpoint and hasattr(self.task, 'tabasco_model'):
            self.task.tabasco_model.set_data_stats(checkpoint['data_stats'])
            if getattr(self.task, '_node_dist_model', None) is not None:
                self.task._node_dist_model = None
            logger.info("Restored Tabasco data_stats from checkpoint")

        has_fresh_node_dist = getattr(self.task, 'node_dist_model', None) is not None
        has_fresh_prop_dist = getattr(self.task, 'prop_dist_model', None) is not None

        if 'node_dist_model' in checkpoint:
            if has_fresh_node_dist:
                logger.info("Skipping node_dist_model from checkpoint — using fresh model from new dataset")
            else:
                self.task.node_dist_model = checkpoint['node_dist_model']
                logger.info("Restored node_dist_model from checkpoint")

        if 'n_node_dist' in checkpoint:
            if has_fresh_node_dist:
                logger.info("Skipping n_node_dist from checkpoint — using fresh distribution from new dataset")
            else:
                self.task.n_node_dist = checkpoint['n_node_dist']
                logger.info("Restored n_node_dist from checkpoint")

        if 'prop_dist_model' in checkpoint:
            if has_fresh_prop_dist:
                logger.info("Skipping prop_dist_model from checkpoint — using fresh model from new dataset")
            else:
                self.task.prop_dist_model = checkpoint['prop_dist_model']
                logger.info("Restored prop_dist_model from checkpoint")

        has_fresh_property_norms = getattr(self.task, 'property_norms', None) is not None
        if 'property_norms' in checkpoint:
            if has_fresh_property_norms:
                logger.info("Skipping property_norms from checkpoint — using fresh stats from new dataset")
            else:
                self.task.property_norms = checkpoint['property_norms']
                logger.info("Restored property_norms from checkpoint")

        if 'reference_indices' in checkpoint:
            self.task.reference_indices = checkpoint['reference_indices']
        if 'reference_freeze_mode' in checkpoint:
            self.task.reference_freeze_mode = checkpoint['reference_freeze_mode']
        if 'reference_feature_stats' in checkpoint:
            self.task.reference_feature_stats = checkpoint['reference_feature_stats']
        if 'reference_scaffold' in checkpoint:
            self.task.reference_scaffold = checkpoint['reference_scaffold']
            logger.info("Restored reference_scaffold from checkpoint")

        # Dynamically resize tensors for architecture expansion if fine-tuning
        if hasattr(self, "_custom_state_dict_adjuster") and "state_dict" in checkpoint:
            orig_sd = {k: v.shape for k, v in checkpoint["state_dict"].items()}
            checkpoint["state_dict"] = self._custom_state_dict_adjuster(checkpoint["state_dict"], self.task)

            # Log which tensors were expanded by the adjuster
            expanded = [
                (k, orig_sd[k], checkpoint["state_dict"][k].shape)
                for k in orig_sd
                if k in checkpoint["state_dict"] and checkpoint["state_dict"][k].shape != orig_sd[k]
            ]
            if expanded:
                logger.info(
                    f"\n{'='*60}\n"
                    f"  DIMENSION EXPANSION in on_load_checkpoint ({len(expanded)} tensor(s))\n"
                    f"{'='*60}"
                )
                for k, old_s, new_s in expanded:
                    logger.info(f"  [EXPANDED] {k}: {list(old_s)} -> {list(new_s)}")
                logger.info(f"{'='*60}")

            # Manually load the adjusted tensors into self.task
            task_sd = {}
            for k, v in checkpoint["state_dict"].items():
                stripped = k[len("task."):] if k.startswith("task.") else k
                task_sd[stripped] = v
            missing, unexpected = self.task.load_state_dict(task_sd, strict=False)
            if missing:
                logger.debug(f"on_load_checkpoint: {len(missing)} missing keys (expected for partial load)")

        # Same application identically for the EMA models just in case
        if hasattr(self, "_custom_state_dict_adjuster") and "ema_model_state_dict" in checkpoint:
            checkpoint["ema_model_state_dict"] = self._custom_state_dict_adjuster(checkpoint["ema_model_state_dict"], self.task)

        # Handle EMA state - new format first
        if 'ema_model_state_dict' in checkpoint and self.ema_decay > 0:
            self._pending_ema_state = checkpoint['ema_model_state_dict']
            logger.info("Found EMA state in checkpoint (new format), will load after model init")

        # Handle old format: ema_model.* keys in state_dict from EMACallback
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            ema_keys = [k for k in state_dict.keys() if k.startswith('ema_model.')]

            if ema_keys:
                logger.info(f"Found {len(ema_keys)} EMA keys from old checkpoint format, extracting...")

                # Extract EMA state dict (strip 'ema_model.' prefix)
                ema_state = {}
                for k in ema_keys:
                    new_key = k.replace('ema_model.', '', 1)
                    ema_state[new_key] = state_dict[k]

                if self.ema_decay > 0:
                    self._pending_ema_state = ema_state
                    logger.info("Will load EMA state from old checkpoint after model init")

                # Remove ema_model.* keys from state_dict to prevent "unexpected key" errors
                for k in ema_keys:
                    del checkpoint['state_dict'][k]
                logger.info(f"Removed {len(ema_keys)} old ema_model.* keys from state_dict")


    def forward(self, batch):
        """Forward pass through the task."""
        return self.task(batch)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Calls task(batch) which returns (loss, metrics).
        """
        self.task.split = "train"

        # Some backbones assert their own invariants mid-forward (MiDi's
        # masking / finiteness checks). One bad batch shouldn't kill a run that
        # took hours -- skip it. A run whose weights have actually gone NaN
        # fails every batch, so still die once the skips stop being isolated.
        try:
            loss, metrics = self.task(batch)
        except ValueError as err:
            self._consecutive_bad_batches += 1
            if self._consecutive_bad_batches > self.max_consecutive_bad_batches:
                msg = (
                    f"{self._consecutive_bad_batches} consecutive failed "
                    f"training batches -- the model is not recovering."
                )
                raise RuntimeError(msg) from err
            logger.warning(
                f"Skipping batch {batch_idx}: {err} "
                f"({self._consecutive_bad_batches} in a row)"
            )
            return None  # Lightning skips None returns

        # Skip batches with invalid loss (matches original Engine behavior)
        if torch.isnan(loss) or torch.isinf(loss):
            self._consecutive_bad_batches += 1
            logger.warning(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
            return None  # Lightning skips None returns

        self._consecutive_bad_batches = 0

        # Infer batch size from batch (graph datasets have variable batch sizes)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch) if isinstance(batch, (list, tuple)) else 1

        # Log metrics with explicit batch_size to suppress warnings
        # Log metrics with explicit batch_size to suppress warnings
        for key, value in metrics.items():
            # Filter progress bar if monitor_metric is set
            if self.monitor_metric is not None:
                # Handle both string and list/tuple
                if isinstance(self.monitor_metric, str):
                    targets = [self.monitor_metric]
                else:
                    targets = self.monitor_metric

                # Always show loss and the monitored metric(s)
                # Normalize targets to handle _step/_epoch suffixes user might use
                normalized_targets = [t.replace("_step", "").replace("_epoch", "") for t in targets]

                # Check strict match, raw match, or normalized match
                metric_full_name = f"train/{key}"
                show_on_bar = (key == "loss") or \
                              (metric_full_name in targets) or \
                              (key in targets) or \
                              (metric_full_name in normalized_targets) or \
                              (key in normalized_targets)
            else:
                show_on_bar = True

            self.log(f"train/{key}", value, on_step=True, on_epoch=True,
                    prog_bar=show_on_bar, sync_dist=True, batch_size=batch_size)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if self._ema_helper is not None and self.ema_model is not None:
            self._ema_helper.update_model_average(self.ema_model, self.task)

        # Periodic sleep every N global steps (0 disables)
        if self.sleep_every_N > 0:
            step = self.trainer.global_step
            if step > 0 and step % self.sleep_every_N == 0:
                logger.info(f"Sleeping for {self.sleep_time}s at global step {step}")
                time.sleep(self.sleep_time)

    def _skip_nonfinite_grads(self) -> bool:
        """Zero the gradients if any is NaN/Inf. True if the step was dropped."""
        grads = [p.grad for p in self.task.parameters() if p.grad is not None]
        if not grads:
            return False
        # ponytail: one fused check over the flat norm, not a per-tensor scan.
        total = torch.norm(torch.stack([g.norm() for g in grads]))
        if torch.isfinite(total):
            return False
        self._nonfinite_grad_steps += 1
        logger.warning(
            f"Non-finite gradient (norm={total}); skipping optimizer step "
            f"({self._nonfinite_grad_steps} skipped so far)."
        )
        for g in grads:
            g.zero_()
        return True

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val, gradient_clip_algorithm
    ):
        """Override gradient clipping to support adaptive Queue-based clipping.

        Args:
            optimizer: The optimizer being used
            gradient_clip_val: Value from Trainer config (used for 'lightning' mode)
            gradient_clip_algorithm: Algorithm from Trainer config ('norm' or 'value')
        """
        # A single non-finite gradient poisons every weight it touches, and
        # from then on the forward pass is all-NaN -- the run dies on a
        # masking assert deep in some backbone instead of here. Drop the step
        # instead: zeroed grads make optimizer.step() a no-op.
        if self._skip_nonfinite_grads():
            return

        if self.gradient_clip_algorithm == "adaptive" and self._clipper is not None:
            # Use the adaptive Queue-based gradient clipping (matches original Engine)
            self._clipper(self.task, self.gradnorm_queue)
        elif self.gradient_clip_algorithm == "lightning":
            # Use Lightning's built-in gradient clipping
            super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        # else: no clipping (gradient_clip_algorithm is None or empty)

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Uses EMA model if available, otherwise uses the main task.
        Checks for task.predict_and_target(batch). If available, returns pred/target for aggregation.
        Otherwise, runs standard forward pass and logs metrics.
        """
        # Use EMA model for validation if available (matches Engine behavior)
        model = self.ema_model if self.ema_model is not None else self.task
        model.split = "valid"

        if hasattr(model, "predict_and_target"):
            pred, target = model.predict_and_target(batch)

            # Compute batch-level loss for scheduler monitoring
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch) if isinstance(batch, (list, tuple)) else 1
            if hasattr(self.task, 'criterion') and callable(self.task.criterion):
                batch_loss = self.task.criterion(pred, target)
            else:
                batch_loss = torch.nn.functional.mse_loss(pred, target)

            # Log val/loss at batch level (aggregated by Lightning at epoch end)
            self.log("val/loss", batch_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

            return {"pred": pred, "target": target, "loss": batch_loss}

        # Fallback for generative models or tasks without explicit targets
        loss, metrics = model(batch)

        # Determine batch size
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch) if isinstance(batch, (list, tuple)) else 1

        for key, value in metrics.items():
            # Filter progress bar if monitor_metric is set
            if self.monitor_metric is not None:
                if isinstance(self.monitor_metric, str):
                    targets = [self.monitor_metric]
                else:
                    targets = self.monitor_metric

                # Normalize targets
                normalized_targets = [t.replace("_step", "").replace("_epoch", "") for t in targets]

                metric_full_name = f"val/{key}"
                show_on_bar = (key == "loss") or \
                              (metric_full_name in targets) or \
                              (key in targets) or \
                              (metric_full_name in normalized_targets) or \
                              (key in normalized_targets)
            else:
                show_on_bar = True

            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=show_on_bar, sync_dist=True, batch_size=batch_size)

        # Always show val/loss on bar unless strictly excluded (rare)
        show_loss_on_bar = True
        if self.monitor_metric is not None:
            if isinstance(self.monitor_metric, str):
                 if self.monitor_metric != "val/loss" and "loss" not in self.monitor_metric:
                     # Could hide, but generally keep loss
                     pass
            # Logic for list is complex, default to showing loss


        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=show_loss_on_bar, sync_dist=True, batch_size=batch_size)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        """
        Aggregate predictions from all validation batches and compute metrics.

        Note: In Lightning v2.0+, outputs are accessed via trainer.validation_step_outputs
        """
        # Call task's validation epoch end if available (e.g., for VAE reconstruction)
        if hasattr(self.task, 'on_validation_epoch_end'):
            recon_metrics = self.task.on_validation_epoch_end(current_epoch=self.current_epoch)
            if recon_metrics:
                for key, value in recon_metrics.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.mean().item()
                    self.log(f"val/recon_{key}", value, on_epoch=True, prog_bar=True, sync_dist=True)

        # Get outputs from all validation steps
        outputs = self.trainer.logged_metrics if hasattr(self.trainer, 'logged_metrics') else []

        # Access validation step outputs stored by Lightning
        if not hasattr(self, 'validation_step_outputs'):
            return

        outputs = self.validation_step_outputs

        if len(outputs) == 0:
            return

        # Check if we have predictions/targets to aggregate
        if "pred" not in outputs[0] or "target" not in outputs[0]:
            self.validation_step_outputs.clear()
            return

        # Concatenate all predictions and targets
        preds = [out["pred"] for out in outputs if "pred" in out]
        targets = [out["target"] for out in outputs if "target" in out]

        if len(preds) == 0 or len(targets) == 0:
            # Clear outputs for next epoch
            self.validation_step_outputs.clear()
            return

        pred = utils.cat(preds)
        target = utils.cat(targets)

        # Filter out NaN/Inf values
        invalid_pred_indices = torch.isnan(pred) | torch.isinf(pred)
        invalid_target_indices = torch.isnan(target) | torch.isinf(target)
        combined_invalid_indices = invalid_pred_indices | invalid_target_indices

        if combined_invalid_indices.ndim > 1:
            combined_invalid_indices = combined_invalid_indices.flatten(start_dim=1).any(dim=1)

        valid_mask = ~combined_invalid_indices
        pred = pred[valid_mask]
        target = target[valid_mask]

        if pred.numel() == 0:
            logger.warning("All predictions or targets are NaN/Inf. Skipping validation metrics.")
            self.validation_step_outputs.clear()
            return

        # Compute metrics via task.evaluate()
        metrics = self.task.evaluate(pred, target)

        # Compute val/loss for scheduler/checkpoint monitoring
        # Use task's criterion if available and callable, otherwise compute mean of all metrics
        if hasattr(self.task, 'criterion') and callable(self.task.criterion):
            val_loss = self.task.criterion(pred, target)
        elif len(metrics) > 0:
            # Use mean of all metric values as a combined loss proxy
            metric_values = [v.item() if isinstance(v, torch.Tensor) else v for v in metrics.values()]
            val_loss = sum(metric_values) / len(metric_values)
        else:
            # Fallback to MSE if no criterion and no metrics
            val_loss = torch.nn.functional.mse_loss(pred, target)

        self.log("val/loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log metrics
        for key, value in metrics.items():
            self.log(f"val/{key}", value, on_epoch=True, prog_bar=True, sync_dist=True)

        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Store validation outputs for epoch end aggregation."""
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        self.validation_step_outputs.append(outputs)

    def test_step(self, batch, batch_idx):
        """Test step. Uses EMA model if available."""
        # Use EMA model for testing if available (matches Engine behavior)
        model = self.ema_model if self.ema_model is not None else self.task
        model.split = "test"

        if hasattr(model, "predict_and_target"):
            pred, target = model.predict_and_target(batch)
            return {"pred": pred, "target": target}

        # Fallback for generative models or tasks without explicit targets
        loss, metrics = model(batch)

        # Determine batch size
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else len(batch) if isinstance(batch, (list, tuple)) else 1

        for key, value in metrics.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return {"loss": loss}

    def on_test_epoch_end(self):
        """Same as on_validation_epoch_end for test set."""
        if not hasattr(self, 'test_step_outputs'):
            return

        outputs = self.test_step_outputs

        if len(outputs) == 0:
            return

        # Check if we have predictions/targets to aggregate
        if "pred" not in outputs[0] or "target" not in outputs[0]:
            self.test_step_outputs.clear()
            return

        preds = [out["pred"] for out in outputs if "pred" in out]
        targets = [out["target"] for out in outputs if "target" in out]

        if len(preds) == 0 or len(targets) == 0:
            self.test_step_outputs.clear()
            return

        pred = utils.cat(preds)
        target = utils.cat(targets)

        # Filter out NaN/Inf values
        invalid_pred_indices = torch.isnan(pred) | torch.isinf(pred)
        invalid_target_indices = torch.isnan(target) | torch.isinf(target)
        combined_invalid_indices = invalid_pred_indices | invalid_target_indices

        if combined_invalid_indices.ndim > 1:
            combined_invalid_indices = combined_invalid_indices.flatten(start_dim=1).any(dim=1)

        valid_mask = ~combined_invalid_indices
        pred = pred[valid_mask]
        target = target[valid_mask]

        if pred.numel() == 0:
            logger.warning("All predictions or targets are NaN/Inf. Skipping test metrics.")
            self.test_step_outputs.clear()
            return

        metrics = self.task.evaluate(pred, target)

        for key, value in metrics.items():
            self.log(f"test/{key}", value, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_step_outputs.clear()

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """Store test outputs for epoch end aggregation."""
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append(outputs)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        """
        # Build optimizer
        optimizer_choice = self.optimizer_config.get("optimizer_choice", "adamw").lower()
        lr = self.optimizer_config.get("lr", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)
        betas = tuple(self.optimizer_config.get("betas", [0.9, 0.999]))
        eps = self.optimizer_config.get("eps", 1e-8)

        if optimizer_choice == "adam":
            optimizer = optim.Adam(
                self.task.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        elif optimizer_choice == "adamw":
            optimizer = optim.AdamW(
                self.task.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        elif optimizer_choice == "amsgrad":
            optimizer = optim.Adam(
                self.task.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                amsgrad=True,
            )
        elif optimizer_choice == "radam":
            optimizer = optim.RAdam(
                self.task.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_choice}")

        # Build scheduler
        scheduler_name = self.scheduler_config.get("scheduler", None)
        if scheduler_name is None:
            return optimizer

        scheduler_kwargs = {k: v for k, v in self.scheduler_config.get("scheduler_kwargs", {}).items() if v is not None}
        scheduler_name_lower = scheduler_name.lower()

        if scheduler_name_lower == "steplr":
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
        elif scheduler_name_lower == "multisteplr":
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)
        elif scheduler_name_lower == "exponentiallr":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_kwargs)
        elif scheduler_name_lower == "cosineannealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
        elif scheduler_name_lower == "onecyclelr":
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, **scheduler_kwargs)
        elif scheduler_name_lower == "reducelronplateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)

            # Get monitor metric from config or default to val/loss
            monitor = self.scheduler_config.get("monitor", "val/loss")

            # Get other scheduler config options
            interval = self.scheduler_config.get("interval", "epoch")
            frequency = self.scheduler_config.get("frequency", 1)
            strict = self.scheduler_config.get("strict", False)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": interval,
                    "frequency": frequency,
                    "strict": strict,
                },
            }
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}. Proceeding without scheduler.")
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
