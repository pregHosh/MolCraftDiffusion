import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    LambdaLR
)
from MolecularDiffusion.callbacks import Queue

class OptimSchedulerFactory:
    """
    Factory to build optimizer, scheduler, and gradient-norm queue for training.

    Supported optimizers: adam, amsgrad, adamw, radam
    Supported schedulers: steplr, multisteplr, exponentiarlr, cosineannealing, caws, onecyclelr, reducelronplateau, lambdalr

    Usage:
        factory = OptimSchedulerFactory(
            parameters=model.parameters(),
            optimizer_choice="adam",
            lr=1e-3,
            eps=1e-8,
            weight_decay=0,
            betas=(0.9,0.999),
            foreach=False,
            scheduler="reducelronplateau",
            scheduler_kwargs={"mode":"min","factor":0.1,"patience":10},
            num_epochs=100,
            train_set=train_ds,
            batch_size=32,
            queue_size=3000,
            init_grad_norm=3000,
        )
        optimizer = factory.get_optimizer()
        scheduler = factory.get_scheduler(optimizer)
        gradnorm_queue = factory.gradnorm_queue
    """
    def __init__(
        self,
        parameters,
        optimizer_choice: str = "adam",
        lr: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0,
        betas: tuple = (0.9, 0.999),
        foreach: bool = False,
        scheduler: str = None,
        scheduler_kwargs: dict = None,
        num_epochs: int = None,
        train_set=None,
        batch_size: int = None,
        queue_size: int = 100,
        init_grad_norm: float = None,
        ema_decay: float = 0.9999,
        gradient_clip_mode: str = "value" # value or norm
    ):
        self.parameters = parameters
        self.optimizer_choice = optimizer_choice.lower()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.betas = betas
        self.foreach = foreach
        self.scheduler_choice = scheduler.lower() if scheduler else None
        self.scheduler_choice_kwargs = scheduler_kwargs or {}
        self.num_epochs = num_epochs
        self.train_set = train_set
        self.batch_size = batch_size
        self.gradnorm_queue = Queue(max_len=queue_size)
        if init_grad_norm is not None:
            self.gradnorm_queue.add(init_grad_norm)
        self.ema_decay = ema_decay
        self.gradient_clip_mode = gradient_clip_mode



    def get_optimizer(self):
        """Return optimizer based on choice and hyperparameters."""
        if self.optimizer_choice == "adam":
            self.optimizer = optim.Adam(
                self.parameters,
                lr=self.lr,
                eps=self.eps,
                weight_decay=self.weight_decay,
                betas=self.betas,
                foreach=self.foreach,
            )
        elif self.optimizer_choice == "amsgrad":
            self.optimizer =  optim.Adam(
                self.parameters,
                lr=self.lr,
                eps=self.eps,
                weight_decay=self.weight_decay,
                betas=self.betas,
                amsgrad=True,
                foreach=self.foreach,
            )
        elif self.optimizer_choice == "adamw":
            self.optimizer = optim.AdamW(
                self.parameters,
                lr=self.lr,
                eps=self.eps,
                weight_decay=self.weight_decay,
                betas=self.betas,
                foreach=self.foreach,
            )
        elif self.optimizer_choice == "radam":
            self.optimizer = optim.RAdam(
                self.parameters,
                lr=self.lr,
                eps=self.eps,
                weight_decay=self.weight_decay,
                betas=self.betas,
                foreach=self.foreach,
            )
        else:
            raise ValueError(
                f"Invalid optimizer {self.optimizer_choice} (allowed: adam, amsgrad, adamw, radam)"
            )

    def get_scheduler(self, optimizer, **kw):
        """Return scheduler based on type and provided kwargs."""
        if not self.scheduler_choice:
            self.scheduler = None
        sch = self.scheduler_choice
        kw = self.scheduler_choice_kwargs
        if sch == "steplr":
            self.scheduler = StepLR(optimizer, **kw)
        elif sch == "multisteplr":
            self.scheduler = MultiStepLR(optimizer, **kw)
        elif sch == "exponentiarlr":
            self.scheduler = ExponentialLR(optimizer, **kw)
        elif sch == "cosineannealing":
            self.scheduler = CosineAnnealingLR(optimizer, **kw)
        elif sch == "caws":
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, **kw)
        elif sch == "onecyclelr":
            if self.train_set is None or self.batch_size is None or self.num_epochs is None:
                raise ValueError("Train set, batch size, and num_epochs must be set for OneCycleLR.")
            steps_per_epoch = len(self.train_set) // self.batch_size
            self.scheduler = OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=steps_per_epoch, epochs=self.num_epochs, **kw)
        elif sch == "reducelronplateau":
            self.scheduler = ReduceLROnPlateau(optimizer, **kw)
        elif sch == "lambdalr":
            self.scheduler = LambdaLR(optimizer, **kw)
        else:
            raise ValueError(
                f"Invalid scheduler {self.scheduler_choice} (allowed: steplr, multisteplr, exponentiarlr, cosineannealing, caws, onecyclelr, reducelronplateau, lambdalr)"
            )
