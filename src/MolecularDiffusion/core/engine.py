import copy
import logging
import os
import sys
from itertools import islice
from tqdm import tqdm

import torch
from torch import distributed as dist
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils import data as torch_data

from MolecularDiffusion import utils, core, data
from MolecularDiffusion.utils import comm, pretty, recursive_module_to_device
from MolecularDiffusion.callbacks import EMA, Queue, gradient_clipping

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
        debug=False,
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

        self.debug = debug

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
            result = task.preprocess(train_set)
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
                    )
                else:
                    logger = core.LoggingLogger() # Fallback for non-main processes
            else:
                raise ValueError("Unknown logger `%s`" % logger)
            self.logger = logger
        self.meter = core.Meter(
            log_interval=log_interval, silent=self.rank > 0, logger=logger
        )
        # self.meter.log_config(self.config_dict())

    def train(self, num_epoch=1, batch_per_epoch=None, use_amp=False, precision="bf16"):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
            use_amp(bool, optional): whether to use automatic mixed precision (AMP) during training.
            precision (str, optional): precision to use for AMP, either "bfloat16" or "float16".
        Returns:
            dict: metrics
        """
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        sampler = torch_data.DistributedSampler(
            self.train_set, self.world_size, self.rank
        )

        batch_size = self.batch_size
        while len(self.train_set) % batch_size == 1:
            batch_size += 1
        if batch_size != self.batch_size:
            logger.warning(f"Batch size adjusted to {batch_size} for training")

        dataloader = data.dataloader.DataLoader(
            self.train_set,
            batch_size,
            sampler=sampler,
            num_workers=self.num_worker,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            shuffle=False
        )
        batch_per_epoch = batch_per_epoch or len(dataloader)
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
            start_id = 0
            batch_loss = 0
            # the last gradient update may contain less than gradient_interval
            # batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            progress_bar = tqdm(
                enumerate(islice(dataloader, batch_per_epoch)),
                desc=f"Training Epoch [{epoch + 1}]",
                leave=True,
                dynamic_ncols=True,
                total=batch_per_epoch,
                disable=self.rank != 0
            )

            for batch_id, batch in progress_bar:
                if len(batch) == 0 or batch is None:
                    continue
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                # Forward pass with autocast
                with autocast(enabled=use_amp, dtype=amp_dtype, device_type=self.device.type):
                    loss, metric = model(batch)
                    if not loss.requires_grad:
                        raise RuntimeError(
                            "Loss doesn't require grad. Did you define any loss in the task?"
                        )
                    batch_loss += loss.item()
                    loss = loss / gradient_interval

                # Backward pass with AMP scaling
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                        if self.debug:
                            module.logger.info(
                                f"Gradient - {name}: {param.grad.norm().item()}"
                            )

                metrics.append(metric)
                if torch.isnan(torch.tensor(grad_norms)).any() and self.debug:
                    module.logger.info(
                        "NaN gradients detected in batch {}. Skipping this batch.".format(
                            batch_id
                        )
                    )
                    self.optimizer.zero_grad()
                    continue

                if type(self.clip_value) == float:
                    if self.clipping_gradient == "norm":
                        if use_amp:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=self.clip_value
                        )
                    if self.clipping_gradient == "value":
                        if use_amp:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), clip_value=self.clip_value
                        )
                elif type(self.clip_value) == Queue:
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    grad_norms = self.clipper(model, self.clip_value)
                    if self.debug:
                        module.logger.info(f"Gradient norm: {grad_norms}")

                if batch_id - start_id + 1 == gradient_interval:
                    if use_amp:
                        # Check for scaler overflow
                        scaler_result = scaler.step(self.optimizer)
                        scaler.update()
                        if scaler_result is not None and self.debug:
                            module.logger.warning(f"Scaler overflow detected in batch {batch_id}")
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    self.meter.update(metric)

                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(
                        batch_per_epoch - start_id, self.gradient_interval
                    )

                if self.EMA:
                    self.EMA.update_model_average(self.ema_model, self.model)
                progress_bar.set_postfix({
                    "batch": batch_id + 1,
                })
                for metric_name in metric.keys():
                    progress_bar.set_postfix({metric_name: metric[metric_name].item() if isinstance(metric[metric_name], torch.Tensor) else metric[metric_name]})

            if self.scheduler:
                if type(self.scheduler).__name__ == "ReduceLROnPlateau":
                    try:
                        self.scheduler.step(
                            batch_loss / len(dataloader)
                        )  # mean loss over an epoch
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
        state = torch.load(checkpoint_path, map_location="cpu")  # CPU for safe loading

        # Reconstruct the Engine using saved hyperparameters
        if "hyperparameters" not in state:
            raise ValueError("Checkpoint does not contain hyperparameters.")
        config_dict = state["hyperparameters"]
        optimizer_state = state.get("optimizer", None)
        
        if interference_mode:
            config_dict["train_set"] = None
            config_dict["valid_set"] = None
            config_dict["test_set"] = None
            optimizer_state = None

        engine = cls.load_config_dict(config_dict)  # Uses class method to build Engine

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
        state = torch.load(checkpoint, map_location=self.device)

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

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "hyperparameters": self.sanitized_config_dict(), # Save full config dictionary
            }
            # if self.scheduler is not None:
            #     state["scheduler"] = self.scheduler.state_dict()
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

        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
                
        return cls(**new_config)

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id

    # These cannot be saved.
    def sanitized_config_dict(self):
        cfg = self.config_dict()
        exclude_keys = {"optimizer", "scheduler", "collate_fn",}
        return {k: v for k, v in cfg.items() if k not in exclude_keys}
