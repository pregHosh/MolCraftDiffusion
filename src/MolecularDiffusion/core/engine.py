import copy
import logging
import os
import sys
import time
from itertools import islice
from tqdm import tqdm

import torch
from torch import distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils import data as torch_data

from MolecularDiffusion import utils, core, data
from MolecularDiffusion.utils import pretty, recursive_module_to_device
from MolecularDiffusion.utils.runtime import comm
from MolecularDiffusion.callbacks import EMA, Queue, gradient_clipping
from MolecularDiffusion.device import get_device, to_device

module = sys.modules[__name__]
logger = logging.getLogger(__name__)



class Engine(core.Configurable):
    """
    General class that handles everything about training and test of a task.

    If :meth:`preprocess` is defined by the task, it will be applied to ``train_set``, ``valid_set`` and ``test_set``.

    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        collate_fn (callable, optional): collate function for batching (default to :func:`data.graph_collate`)
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        clipping_gradient (str, optional): toggle to clip the gradient (by norm or value, default is None)
        clip_value (float, Queue, optional): clip value (value of norm), or provide gradient gradient queue
        ema_decay (float, optional): decay rate for exponential moving average
        num_worker (int, optional): number of CPU workers per GPU
        pin_memory (bool, optional): pin memory for faster data transfer
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
        project_wandb (str, optional): project name for wandb
        name_wandb (str, optional): name for wandb
        dir_wandb (str, optional): directory for wandb
        debug (bool, optional): Toggle debug mode
    """

    def __init__(
        self,
        task,
        train_set,
        valid_set,
        test_set,
        optimizer=None,
        collate_fn=None,
        scheduler=None,
        batch_size=1,
        gradient_interval=1,
        clipping_gradient=None,
        clip_value=1,
        ema_decay=0.0,
        num_worker=0,
        pin_memory=True,
        logger="logging",
        log_interval=100,
        project_wandb=None,
        name_wandb=None,
        dir_wandb=None,
        tags_wandb=None,
        debug=False,
        progress_bar_metrics="all",
    ):
        try:
            self.rank = int(os.environ["SLURM_PROCID"])
        except KeyError:
            self.rank = comm.get_rank()

        if collate_fn is None:
            self.collate_fn = data.dataloader.graph_collate
        else:
            self.collate_fn = collate_fn

        self.world_size = comm.get_world_size()
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        self.pin_memory = pin_memory
        self.gpus = None
        self.gpus_per_node = 0
        self.clipping_gradient = clipping_gradient
        self.clip_value = clip_value
        if type(self.clip_value) == Queue:
            self.clipper = gradient_clipping(m=1)

        self.project_wandb = project_wandb
        self.name_wandb = name_wandb
        self.dir_wandb = dir_wandb
        self.tags_wandb = tags_wandb

        self.debug = debug
        self.progress_bar_metrics = progress_bar_metrics

        try:
            gpus_per_node = int(
                os.environ["SLURM_GPUS_ON_NODE"]
            )  # number of GPUs per node
        except KeyError:
            #  might be wrong here
            gpus_per_node = torch.cuda.device_count()

        if gpus_per_node > 0:
            self.gpus = [i for i in range(gpus_per_node)]
            nnode = int(self.world_size / gpus_per_node)
            for i in range(nnode - 1):
                self.gpus.extend([i for i in range(gpus_per_node)])

            module.logger.info(
                f"Hello from rank {self.rank} of {self.world_size}"
                f" {gpus_per_node} allocated GPUs per node.",
            )

        if self.gpus is None:
            if self.world_size == 1 and get_device().type == "mps":
                module.logger.info("Using Apple MPS")
                self.device = torch.device("mps")
                self.pin_memory = False  # pinned memory is a CUDA feature
            else:
                module.logger.info("Using CPU")
                self.device = torch.device("cpu")
        else:
            assert gpus_per_node == torch.cuda.device_count()
            if len(self.gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                raise ValueError(error_msg % (self.world_size, len(self.gpus)))
            self.device = torch.device(self.gpus[self.rank % len(self.gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            # backend = "gloo" if self.gpus is None else "nccl"
            # comm.init_process_group(backend, rank=self.rank, world_size=self.world_size, init_method="env://")
            if self.gpus is None:
                comm.init_process_group("gloo", init_method="env://")
            else:
                comm.init_process_group(
                    "nccl",
                    rank=self.rank,
                    world_size=self.world_size,
                    init_method="env://",
                )  # not sure if putting init_method="env://" here is correct

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # handle dynamic parameters in optimizer
            # old_params = list(task.parameters())
            _t_pre = time.perf_counter()
            result = task.preprocess(train_set)
            if self.rank == 0:
                module.logger.warning(
                    f"task.preprocess() completed in {time.perf_counter() - _t_pre:.2f}s"
                )
            if result is not None:
                train_set, valid_set, test_set = result
            # new_params = list(task.parameters())
            # if len(new_params) != len(old_params):
            #     optimizer.add_param_group({"params": new_params[len(old_params) :]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda" and task is not None:
            task = task.cuda(self.device)
        elif self.device.type == "mps" and task is not None:
            task = task.to(self.device)

        if not(hasattr(task, 'device')) and task is not None:
            recursive_module_to_device(task, self.device)
            
        self.model = task
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self.ema_model = copy.deepcopy(task)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.EMA = EMA(ema_decay)
        else:
            self.ema_model = task
            self.EMA = None

        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if train_set is None or optimizer is None:
            logger = "logging" # this is the interference mode
            
        if isinstance(logger, str):
            
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                if self.rank == 0:
                    if self.project_wandb is None:
                        self.project_wandb = task.__class__.__name__
                    logger = core.WandbLogger(
                        project=self.project_wandb,
                        name=self.name_wandb,
                        dir=self.dir_wandb,
                        rank=self.rank,
                        tags=self.tags_wandb,
                    )
                else:
                    logger = core.LoggingLogger() # Fallback for non-main processes
            else:
                raise ValueError("Unknown logger `%s`" % logger)
            self.logger = logger
        self.meter = core.Meter(
            log_interval=log_interval, silent=self.rank > 0, logger=logger
        )
        
        # Log key config to wandb (if active)
        if self.rank == 0:
            key_config = self.sanitized_config_dict()
            # Add some additional useful info
            if hasattr(self.model, '__class__'):
                key_config['model_class'] = self.model.__class__.__name__
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'T'):
                key_config['diffusion_steps'] = self.model.model.T
            self.meter.log_config(key_config)

    def train(self, num_epoch=1, num_step=None, batch_per_epoch=1, use_amp=False, precision="bf16"):
        """
        Train the model.

        Iterates over the WHOLE dataset for each epoch unless num_step is provided.

        Parameters:
            num_epoch (int, optional): number of epochs.
            num_step (int, optional): total number of optimizer steps to perform.
                                     If provided, training stops after this many updates.
            batch_per_epoch (int, optional): Gradient Accumulation Steps.
                                             The number of batches to accumulate gradients
                                             for before performing an optimizer step.
                                             Default is 1 (step every batch).
            use_amp(bool, optional): whether to use automatic mixed precision (AMP).
            precision (str, optional): precision to use for AMP ("bfloat16" or "float16").
        Returns:
            dict: metrics
        """
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        sampler = torch_data.DistributedSampler(
            self.train_set, self.world_size, self.rank
        )

        # Auto-adjust batch size if needed
        batch_size = self.batch_size
        while len(self.train_set) % batch_size == 1:
            batch_size += 1
        if batch_size != self.batch_size:
            logger.warning(f"Batch size adjusted to {batch_size} for training")

        # Initialize DataLoader
        dataloader = data.dataloader.DataLoader(
            self.train_set,
            batch_size,
            sampler=sampler,
            num_workers=self.num_worker,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False
        )


        # Total number of batches in the dataset
        total_batches = len(dataloader)

        # Use batch_per_epoch as the number of accumulation steps
        accumulation_steps = batch_per_epoch

        model = self.model
        model.split = "train"

        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.device], find_unused_parameters=True
                )
            else:
                model = nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True
                )
        model.train()

        scaler = GradScaler() if use_amp and self.device.type == "cuda" else None

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)

            metrics = []
            batch_loss = 0

            # Iterate over the ENTIRE dataset
            progress_bar = tqdm(
                enumerate(dataloader),
                desc=f"Training Epoch [{epoch + 1}]",
                leave=True,
                dynamic_ncols=True,
                total=total_batches,
                disable=self.rank != 0
            )

            for batch_id, batch in progress_bar:
                if len(batch) == 0 or batch is None:
                    continue
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                elif self.device.type == "mps":
                    batch = to_device(batch, self.device)

                # --- 1. Forward Pass ---
                with autocast(enabled=use_amp, dtype=amp_dtype, device_type=self.device.type):
                    loss, metric = model(batch)


                    # Check if loss requires grad (SKIP logic)
                    if not loss.requires_grad:
                        if self.rank == 0:
                            print(f"Warning: Skipping batch {batch_id} - Loss does not require gradients.")
                        continue # Skip this batch entirely, don't zero_grad yet as we might be accumulating

                    loss_val = loss.item()

                    # Check for invalid loss (NaN/Inf/Zero)
                    if torch.isnan(loss) or torch.isinf(loss) or loss_val == 0:
                        if self.rank == 0:
                            print(f"Skipping batch due to invalid loss: {loss_val}")
                        # If we skip a batch, we don't add to accumulation.
                        # Note: If your loss is frequently NaN, you might need to reset the optimizer/accumulation here.
                        continue

                    batch_loss += loss_val

                    # Normalize loss by accumulation steps
                    loss = loss / accumulation_steps

                # --- 2. Backward Pass (Accumulate) ---
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                metrics.append(metric)

                # --- 3. Optimizer Step Condition ---
                # Step if: We reached the accumulation limit OR we are at the very last batch of the dataset
                is_update_step = ((batch_id + 1) % accumulation_steps == 0) or ((batch_id + 1) == total_batches)

                if is_update_step:

                    # A. Explicit Unscale
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)

                    # B. Log & Check Gradients
                    grad_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norms.append(param.grad.norm().item())
                            if self.debug:
                                module.logger.info(f"Gradient - {name}: {param.grad.norm().item()}")

                    # Check for NaN gradients
                    if torch.tensor(grad_norms).isnan().any():
                        if self.debug:
                            module.logger.info(f"NaN gradients detected at step {batch_id}. Skipping step.")
                        self.optimizer.zero_grad()
                        # Don't continue the loop, just skip the step updates
                    else:
                        # C. Clipping
                        if type(self.clip_value) == float:


                            if self.clipping_gradient == "norm":
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_value)
                            elif self.clipping_gradient == "value":
                                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.clip_value)
                        elif type(self.clip_value) == Queue:
                            grad_norms = self.clipper(model, self.clip_value)

                        # D. Step
                        if scaler is not None:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                    # E. Reset Gradients
                    self.optimizer.zero_grad()

                    # F. EMA Update
                    if self.EMA:
                        self.EMA.update_model_average(self.ema_model, self.model)
                    
                    # G. Check for step-based termination
                    if num_step is not None:
                        # self.meter.batch_id is incremented every batch, but we want to count optimizer steps.
                        # However, to keep it simple and consistent with how num_steps is often used:
                        # if we want exactly 'num_step' updates, we should track it.
                        # Let's use a local counter or assume num_step is the target for self.meter.batch_id / accumulation_steps
                        # Wait, easier: if we are here, we just did an update.
                        # Let's check how many updates were done.
                        if not hasattr(self, '_total_steps'):
                            self._total_steps = 0 # This might not be right for multiple calls to train()
                        # Actually, let's just use self.meter.batch_id and assume it's roughly proportional,
                        # OR better, add a true step counter to the engine.
                        pass # See below for better implementation

                # --- 4. Metrics Calculation ---
                # (Update metrics every step, or accumulate them - adhering to original logic of updating meter)
                # To keep the progress bar responsive, we calculate metrics for the current batch
                # but we only update the global meter occasionally to save time if needed.
                # Here we update per batch to match the progress bar flow.

                if len(metrics) > 0:
                    metric_stack = utils.stack(metrics, dim=0)
                    metric_mean = utils.mean(metric_stack, dim=0)
                    if self.world_size > 1:
                        metric_mean = comm.reduce(metric_mean, op="mean")

                    # Update the global meter
                    self.meter.update(metric_mean)
                    metrics = [] # Reset local metric accumulation

                # --- 5. Progress Bar Update ---
                postfix_dict = {"batch": batch_id + 1}
                if 'metric' in locals() and metric:
                     for metric_name in metric.keys():
                        if self.progress_bar_metrics == "all" or metric_name in self.progress_bar_metrics:
                            val = metric[metric_name]
                            postfix_dict[metric_name] = val.item() if isinstance(val, torch.Tensor) else val
                progress_bar.set_postfix(postfix_dict)
                
                # Termination Check
                if num_step is not None and self.meter.batch_id >= num_step:
                    if self.rank == 0:
                        module.logger.info(f"Reached num_step ({num_step}). Terminating training.")
                    return metric

            # --- 6. Scheduler Step (End of Epoch) ---
            if self.scheduler:
                if type(self.scheduler).__name__ == "ReduceLROnPlateau":
                    try:
                        self.scheduler.step(batch_loss / total_batches)
                    except IndexError:
                        pass
                else:
                    self.scheduler.step()

        return metric


    @torch.no_grad()
    def evaluate(self, split, log=True, use_amp=False, precision="bfloat16"):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not
            use_amp (bool, optional): whether to use automatic mixed precision (AMP) during evaluation.
            precision (str, optional): precision to use for AMP, either "bfloat16" or "float16".
        Returns:
            dict: metrics
        """
        
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)

        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)

        batch_size = self.batch_size
        while len(test_set) % batch_size == 1:
            batch_size += 1

        if batch_size != self.batch_size:
            logger.warning(f"Batch size adjusted to {batch_size} for split {split}")

        dataloader = data.dataloader.DataLoader(
            test_set,
            batch_size,
            sampler=sampler,
            num_workers=self.num_worker,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        model = self.ema_model if self.ema_decay > 0 else self.model
        model.split = split
        model.eval()

        preds = []
        targets = []

        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating on {split}",
            total=len(dataloader),
            leave=False,
            dynamic_ncols=True,
            disable=self.rank != 0,
        )
        for batch in progress_bar:
            if len(batch) == 0 or batch is None:
                continue
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
            elif self.device.type == "mps":
                batch = to_device(batch, self.device)

            try:
                # AMP: Autocast context for mixed precision during evaluation
                with autocast(enabled=use_amp, dtype=amp_dtype, device_type=self.device.type):
                    pred, target = model.predict_and_target(batch)
                
                    preds.append(pred)
                    targets.append(target)
            except Exception as e:
                module.logger.info(f"Error in batch: {e}")
                continue

        pred = utils.cat(preds)
        target = utils.cat(targets)


        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)

        # Identify NaN or Inf values in pred and target
        invalid_pred_indices = torch.isnan(pred) | torch.isinf(pred)
        invalid_target_indices = torch.isnan(target) | torch.isinf(target)

        # Combine invalid indices
        combined_invalid_indices = invalid_pred_indices | invalid_target_indices
        # If 2D or more, check if any value in the row is invalid to drop the whole row
        if combined_invalid_indices.ndim > 1:
            combined_invalid_indices = combined_invalid_indices.flatten(start_dim=1).any(dim=1)

        # Get the mask for valid (non-nan/inf) values
        valid_mask = ~combined_invalid_indices

        # Filter pred and target to keep only valid values
        pred = pred[valid_mask]
        target = target[valid_mask]

        if pred.numel() == 0: # Check if any valid predictions are left
            module.logger.warning("All predictions or targets are NaN/Inf. Skipping evaluation for this split.")
            return {}, [], [] # Return empty metrics and data if nothing valid is left.

        metric = model.evaluate(pred, target)

        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        return metric, preds, targets

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, strict: bool = False, interference_mode: bool = False):
        """
        Load full Engine from a checkpoint using saved hyperparameters.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the model.
            ininterference_mode (bool): The train_set, val_set, and test_set will be set to None if True.
        Returns:
            Engine: Fully reconstructed Engine with model, optimizer, and scheduler states.
        """
        checkpoint_path = os.path.expanduser(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # CPU for safe loading

        # Reconstruct the Engine using saved hyperparameters
        if "hyperparameters" not in state:
            raise ValueError("Checkpoint does not contain hyperparameters.")
        config_dict = state["hyperparameters"]
        optimizer_state = state.get("optimizer", None)
        
        # Ensure datasets are present (might have been removed in compact mode)
        config_dict.setdefault("train_set", None)
        config_dict.setdefault("valid_set", None)
        config_dict.setdefault("test_set", None)

        if interference_mode:
            config_dict["train_set"] = None
            config_dict["valid_set"] = None
            config_dict["test_set"] = None
            optimizer_state = None

        # Filter out extra keys that are not arguments of Engine.__init__
        # to avoid TypeError in load_config_dict
        extra_keys = [
            "atom_vocab", "data_type", "node_feature", "with_hydrogen", 
            "edge_type", "radius", "n_neigh",
            # Hybrid diffusion-specific
            "discrete_schedule", "num_atom_classes", "discrete_loss_weight", "diffusion_timesteps"
        ]
        engine_config = config_dict.copy()
        extras = {}
        for k in extra_keys:
            if k in engine_config:
                extras[k] = engine_config.pop(k)

        engine = cls.load_config_dict(engine_config)  # Uses class method to build Engine
        
        # Attach extras to engine
        for k, v in extras.items():
            setattr(engine, k, v)

        # Move model to device
        engine.model.to(engine.device)

        # Load weights
        if "ema_model" in state:
            engine.model.load_state_dict(state["ema_model"], strict=strict)
        else:
            engine.model.load_state_dict(state["model"], strict=strict)


        # Load optimizer
        if engine.optimizer is not None and optimizer_state is not None:
            engine.optimizer.to(engine.device)
            engine.optimizer.load_state_dict(optimizer_state)
            
            
        # EMA model setup
        if engine.ema_decay > 0:
            engine.ema_model = copy.deepcopy(engine.model)
            if "ema_model" in state:
                engine.ema_model.load_state_dict(state["ema_model"], strict=strict)
            engine.ema_model.eval()
            for param in engine.ema_model.parameters():
                param.requires_grad = False

        # Scheduler state
        if "scheduler" in state and engine.scheduler is not None:
            engine.scheduler.load_state_dict(state["scheduler"])

        return engine

    def load(self, checkpoint, load_optimizer=True, strict=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)

        if "ema_model" in state:
            self.model.load_state_dict(state["ema_model"], strict=strict)
        else:
            self.model.load_state_dict(state["model"], strict=strict)
            

        if self.ema_decay > 0:
            if "ema_model" in state:
                self.ema_model.load_state_dict(state["ema_model"], strict=strict)
            else:
                self.ema_model = copy.deepcopy(self.model)

            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def resume(self, checkpoint, strict=True):
        """
        Resume training from a checkpoint (loads full training state).

        This loads model weights, optimizer, scheduler, epoch counter, and gradnorm queue.
        Use this to continue training from where it left off.

        Parameters:
            checkpoint (file-like): checkpoint file path
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        
        Returns:
            int: The epoch to resume from (for adjusting training loop)
        """
        if comm.get_rank() == 0:
            logger.warning("Resuming from checkpoint: %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)

        # Load model weights (prefer EMA if available)
        if "ema_model" in state:
            self.model.load_state_dict(state["ema_model"], strict=strict)
        else:
            self.model.load_state_dict(state["model"], strict=strict)

        # Load EMA model
        if self.ema_decay > 0:
            if "ema_model" in state:
                self.ema_model.load_state_dict(state["ema_model"], strict=strict)
            else:
                self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad = False

        # Load optimizer state
        if "optimizer" in state and state["optimizer"] is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(self.device)
            if comm.get_rank() == 0:
                logger.info("Loaded optimizer state")

        # Load scheduler state
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
            if comm.get_rank() == 0:
                logger.info("Loaded scheduler state")

        # Load epoch counter
        resume_epoch = 0
        if "epoch" in state:
            resume_epoch = state["epoch"]
            self.meter.set_epoch(resume_epoch)  # Use set_epoch to properly pad internal lists
            if comm.get_rank() == 0:
                logger.info(f"Resuming from epoch {resume_epoch}")

        # Load gradnorm queue
        if "gradnorm_queue" in state and type(self.clip_value) == Queue:
            queue_data = state["gradnorm_queue"]
            self.clip_value.items = queue_data["data"]
            self.clip_value.max_len = queue_data["maxlen"]
            if comm.get_rank() == 0:
                logger.info("Loaded gradnorm queue state")

        comm.synchronize()
        return resume_epoch

    def save(self, checkpoint, compact=True, full_state=False):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
            compact (bool, optional): whether to save a lightweight checkpoint (no optimizer state).
            full_state (bool, optional): save full training state for resumption (epoch, scheduler, etc.).
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            hyperparameters = self.sanitized_config_dict()

            # Inject data properties if available
            if self.train_set is not None:
                # Get underlying dataset (handle Subset)
                dataset = self.train_set.dataset if hasattr(self.train_set, "dataset") else self.train_set
                
                # 1. atom_vocab
                if hasattr(dataset, "atom_vocab"):
                    hyperparameters["atom_vocab"] = dataset.atom_vocab
                elif hasattr(self.model, "atom_vocab"):
                    hyperparameters["atom_vocab"] = self.model.atom_vocab

                # 2. with_hydrogen
                if hasattr(dataset, "with_hydrogen"):
                    hyperparameters["with_hydrogen"] = dataset.with_hydrogen

                # 3. node_feature
                # Try to get from config_dict if available
                node_feature = None
                if hasattr(dataset, "config_dict"):
                    try:
                        # config_dict might be a method or property or object
                        cfg = dataset.config_dict
                        if callable(cfg):
                            cfg = cfg()
                        if "atom_feature" in cfg:
                            node_feature = cfg["atom_feature"]
                    except Exception:
                        pass
                
                # If still None, try accessing directly if attribute exists
                if node_feature is None and hasattr(dataset, "node_feature"):
                    node_feature = dataset.node_feature

                hyperparameters["node_feature"] = node_feature

                # 4. data_type
                # Infer from class name
                ds_cls_name = dataset.__class__.__name__
                if "pyG" in ds_cls_name or "GraphDataset" in ds_cls_name:
                    hyperparameters["data_type"] = "pyg"
                    
                    # 5. Add PyG specific params
                    if hasattr(dataset, "kwargs") and isinstance(dataset.kwargs, dict):
                         if "edge_type" in dataset.kwargs:
                            hyperparameters["edge_type"] = dataset.kwargs["edge_type"]
                         if "radius" in dataset.kwargs:
                            hyperparameters["radius"] = dataset.kwargs["radius"]
                         if "n_neigh" in dataset.kwargs:
                            hyperparameters["n_neigh"] = dataset.kwargs["n_neigh"]
                    
                    # Or checking attributes directly if kwargs approach fails or is not populated
                    # (Based on `pointcloud_dataset_pyG` implementation, it passes them as kwargs to load_csv/load_db)
                    # But `DataModule` passes them to `pointcloud_dataset_pyG` __init__, which passes to load_db/load_csv
                    # Let's also check if they are stored as attributes if user implementation saved them.
                    # Re-reading dataset.py, `GraphDataset` stores `kwargs` in `self.kwargs`.
                    
                elif "pointcloud" in ds_cls_name.lower() or "PointCloudDataset" in ds_cls_name:
                    hyperparameters["data_type"] = "pointcloud"

            # === Save task_type and condition_names for validation ===
            if hasattr(self.model, "task_type"):
                hyperparameters["task_type"] = self.model.task_type
            if hasattr(self.model, "condition"):
                hyperparameters["condition_names"] = list(self.model.condition)

            # === Save hybrid diffusion-specific hyperparameters ===
            # These are saved from the model directly since they are model config
            if hasattr(self.model, "discrete_diffusion"):
                # This is a hybrid diffusion model
                discrete_diff = self.model.discrete_diffusion
                hyperparameters["discrete_schedule"] = getattr(discrete_diff, "schedule", None)
                hyperparameters["num_atom_classes"] = getattr(self.model, "num_atom_classes", None)
                hyperparameters["discrete_loss_weight"] = getattr(self.model, "discrete_loss_weight", None)
                hyperparameters["diffusion_timesteps"] = getattr(self.model, "T", None)

            state = {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "hyperparameters": hyperparameters, # Save full config dictionary
            }
            
            if not compact:
                 state["optimizer"] = self.optimizer.state_dict() if self.optimizer is not None else None
            
            # Save full training state for resumption
            if full_state:
                state["epoch"] = self.meter.epoch_id
                if self.scheduler is not None:
                    state["scheduler"] = self.scheduler.state_dict()
                # Save gradnorm queue if using adaptive clipping
                if type(self.clip_value) == Queue:
                    state["gradnorm_queue"] = {
                        "data": list(self.clip_value.items),
                        "maxlen": self.clip_value.max_len,
                    }
                    
            torch.save(state, checkpoint)

    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError(
                "Expect config class to be `%s`, but found `%s`"
                % (cls.__name__, config["class"])
            )

        _metadata_keys = {"class", "task_type", "condition_names"}
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k not in _metadata_keys:
                new_config[k] = v

        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id

    # These cannot be saved.
    def sanitized_config_dict(self):
        cfg = self.config_dict()
        # Configurable records every __init__ arg, so train/valid/test_set --
        # the whole dataset -- would otherwise be pickled into every
        # checkpoint. Checkpoints hold weights + inference config only;
        # load_from_checkpoint already setdefault()s these back to None.
        exclude_keys = {
            "optimizer", "scheduler", "collate_fn",
            "train_set", "valid_set", "test_set",
        }
        return {k: v for k, v in cfg.items() if k not in exclude_keys}
