class Logger:
    def __init__(
            self,
            logger,
            log_interval,
            name_wandb,
            project_wandb,
            dir_wandb
            ):
        self.logger = logger
        self.log_interval = log_interval
        self.name_wandb = name_wandb
        self.project_wandb = project_wandb
        self.dir_wandb = dir_wandb

        
