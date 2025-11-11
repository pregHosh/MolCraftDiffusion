"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from typing import Any, Dict,  Optional, Tuple

import pickle
import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #
from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.train import evaluate, DataModule, Logger, OptimSchedulerFactory
from MolecularDiffusion.utils import (
    RankedLogger,
    task_wrapper,
    seed_everything,
)
import os

log = RankedLogger(__name__, rank_zero_only=True)


def is_rank_zero():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def engine_wrapper(task_module, data_module, trainer_module, logger_module, **kwargs):
    
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
            )
    use_amp = False
    if trainer_module.precision in ["bf16", 16]:
        use_amp = True

    best_checkpoints = []
    if task_module.task_type == "diffusion" and kwargs.get("generative_analysis"):
        best_metrics = -torch.inf
        models_to_save = {"node": task_module.task.node_dist_model}

        if len(task_module.condition_names) > 0:
            models_to_save["prop"] = task_module.task.prop_dist_model

        if is_rank_zero():
            with open(os.path.join(trainer_module.output_path, "edm_stat.pkl"), "wb") as f:
                pickle.dump(models_to_save, f)       
    else:
        best_metrics = torch.inf
    for i in range(trainer_module.num_epochs):
        solver.train(num_epoch=1, use_amp=use_amp, precision=trainer_module.precision)
        if i % trainer_module.validation_interval == 0 or i == trainer_module.num_epochs - 1:
            if  task_module.task_type == "diffusion":
                output_generated_dir = os.path.join(
                    trainer_module.output_path, "generated_molecules"
                )
                if not os.path.exists(output_generated_dir):
                    os.makedirs(output_generated_dir, exist_ok=True)
                best_metrics, best_checkpoints = evaluate(
                    task_module.task_type, 
                    solver, 
                    i, 
                    best_metrics, 
                    best_checkpoints, 
                    logger_module.logger, 
                    output_generated_dir=output_generated_dir,
                    generative_analysis=kwargs.get("generative_analysis", False),
                    n_samples=kwargs.get("n_samples", 100),
                    metric=kwargs.get("metric", "Validity Relax and connected"),
                    output_path=trainer_module.output_path,
                    use_amp=use_amp,
                    precision=trainer_module.precision,
                    use_posebuster=kwargs.get("use_posebuster", False),
                    batch_size=kwargs.get("batch_size", 1),
                    )
            else:
                best_metrics, best_checkpoints = evaluate(
                    task_module.task_type,
                    solver,
                    i,
                    best_metrics,
                    best_checkpoints,
                    logger_module.logger,
                    output_path=trainer_module.output_path
                    )
    return best_metrics, solver
    
#TODO to safely retrieve metric value for hydra-based hyperparameter optimization
@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the autoencoder model for reconstruction.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    
    # Create the output directory
    output_path = cfg.trainer.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(
        cfg.data, task_type=cfg.tasks.task_type
    )
    data_module.load()
    log.info(f"Instantiating task <{cfg.tasks._target_}>")

    data_point_chk = data_module.train_set[0]
    n_dim = data_point_chk["node_feature"].shape[1]
    n_dim_extra_data = n_dim - len(set(data_module.train_set.atom_types()))
    n_dim_extra_model = len(cfg.tasks.extra_norm_values)
    if n_dim_extra_data != n_dim_extra_model:
        raise ValueError(f"The number of extra node feature dimensions in the data ({n_dim_extra_data}) does not match the model configuration ({n_dim_extra_model}).")
    
    factory_cfg = cfg.tasks
    
    # The EGT factory requires the train_set, which is not in the config.
    # We add it to the instantiation call as an override.
    overrides = {}
    
    if "tasks_egt" in factory_cfg._target_:
        overrides["train_set"] = data_module.train_set
        # The EGT factory uses 'task_names', but for consistency in configs we might have 'condition_names'
        # This allows using 'condition_names' in the yaml and it will be passed as 'task_names'
        if "condition_names" in factory_cfg:
            overrides["task_names"] = factory_cfg.condition_names
        
    if cfg.data.get("allow_unknown", False):
        overrides["atom_vocab"].append("Suisei") # add an extra token for unknown atoms
    
    if cfg.tasks.get("metrics", None) == "valid_posebuster":
        overrides["use_posebuster"] = True
        try :
            import posebusters
        except ImportError:
            log.warning("PoseBuster is not installed. Please install PoseBuster to use this metric.")
            log.warning("Falling back to 'Validity Relax and connected' metric.")
            overrides["use_posebuster"] = False
            overrides["metrics"] = ["Validity Relax and connected"]
        
    task_module = hydra.utils.instantiate(factory_cfg, **overrides)
    task_module.build()
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(cfg.trainer, parameters=task_module.task.parameters())

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

    # This does not work atm
    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    if task_module.task_type == "diffusion":
        metrics = engine_wrapper(
            task_module, 
            data_module,
            trainer_module,
            logger_module,
            generative_analysis=cfg.tasks.generative_analysis,
            n_samples=cfg.tasks.n_samples,
            metric=cfg.tasks.metrics,
            use_posebuster=cfg.tasks.use_posebuster,    
            batch_size=cfg.tasks.batch_size,  
            )
    else:
        metrics = engine_wrapper(task_module, data_module, trainer_module, logger_module)


    return metrics, object_dict

def log_hyperparameters(object_dict: dict):
    if not is_rank_zero():
        return  # Skip logging for non-zero ranks

    def extract_hparams(obj):
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return str(obj)

    log.info("\n========== Logging Hyperparameters ==========\n")

    for name, obj in object_dict.items():
        log.info(f"{'=' * 20} {name.upper()} {'=' * 20}")
        if name == "cfg":
            if isinstance(obj, dict):
                log.info("\n" + OmegaConf.to_yaml(OmegaConf.create(obj)))
            else:
                log.info("\n" + OmegaConf.to_yaml(obj))
        else:
            hparams = extract_hparams(obj)
            if isinstance(hparams, dict):
                for k, v in hparams.items():
                    log.info(f"{k}: {v}")
            else:
                log.info(hparams)
        log.info(f"{'=' * (44 + len(name))}\n")

    # Log model parameter counts
    if "task" in object_dict and hasattr(object_dict["task"], "task"):
        model = object_dict["task"].task
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        log.info(f"{'=' * 20} MODEL PARAMS {'=' * 20}")
        log.info(f"model/params/total: {total}")
        log.info(f"model/params/trainable: {trainable}")
        log.info(f"model/params/non_trainable: {non_trainable}")
        log.info("=" * 54 + "\n")
        
    log.info("========== End of Hyperparameters ==========\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """


    # train the model
    metric, _ = train(cfg)

    
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    return metric


if __name__ == "__main__":
    main()

