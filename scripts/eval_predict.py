"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from turtle import pd
from typing import Any, Dict,  Optional, Tuple
import numpy as np
import pandas as pd
import hydra
import rootutils
import torch
from torch.utils.data import ConcatDataset
from omegaconf import DictConfig, OmegaConf
import os
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
from MolecularDiffusion.runmodes.train import  DataModule,  ModelTaskFactory_EGCL, OptimSchedulerFactory
from MolecularDiffusion.utils import (
    RankedLogger,
    seed_everything,
)
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution, 
    plot_histogram_distribution, 
    plot_kde_distribution_multiple,
    plot_correlation_with_histograms
)


log = RankedLogger(__name__, rank_zero_only=True)


def engine_wrapper(task_module, data_module, trainer_module):
    
    trainer_module.get_optimizer()
    trainer_module.get_scheduler()

    pred_dataset = ConcatDataset([data_module.valid_set, data_module.test_set])
    solver = Engine(
                task_module.task,
                None,
                None,
                pred_dataset,
                batch_size=data_module.batch_size,
                collate_fn=data_module.collate_fn,
                # optimizer=trainer_module.optimizer,
                # ema_decay=trainer_module.ema_decay,
                # scheduler=trainer_module.scheduler,
                # clipping_gradient=trainer_module.gradient_clip_mode,
                # clip_value=trainer_module.gradnorm_queue,
                logger="logging",
            )


    # _, preds, targets = solver.evaluate("valid")
    _, preds_test, targets_test = solver.evaluate("test")
    y_preds = torch.cat(preds_test, dim=0)
    y_trues = torch.cat(targets_test, dim=0)
    return y_preds, y_trues

    
def filenames_in_values_order(df: pd.DataFrame, task_names, values_array: np.ndarray):
    # values_array: shape (n_samples, n_tasks), column order matches task_names
    # Build a DataFrame for the values with a row index to preserve order
    vals = pd.DataFrame(values_array, columns=task_names)
    vals["__row__"] = np.arange(len(vals))

    # Merge on task columns to fetch the filename for each row of values_array
    merged = vals.merge(df[["filename"] + task_names], on=task_names, how="left")

    # Report issues
    missing = merged["filename"].isna().sum()
    if missing:
        print(f"⚠️ {missing} rows in values_array did not match any row in df.")

    # If multiple df rows share identical task values, merge will duplicate rows.
    # Group back by original order and pick the first filename per values row.
    filenames_ordered = (
        merged.sort_values("__row__")
              .groupby("__row__", as_index=False)["filename"]
              .first()["filename"]
              .to_numpy()
    )

    return filenames_ordered

def predict(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the autoencoder model for reconstruction.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(
        cfg.data, task_type=cfg.tasks.task_type, train_ratio=0
    )
    data_module.load()
    log.info(f"Instantiating task <{cfg.tasks._target_}>")
    
    act_fn = hydra.utils.instantiate(cfg.tasks.act_fn)
    task_module: ModelTaskFactory_EGCL = hydra.utils.instantiate(cfg.tasks, act_fn=act_fn)
    task_module.build()
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(cfg.trainer, parameters=task_module.task.parameters())

    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "task": task_module,
        "trainer": trainer_module,
    }

    # This does not work atm
    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    

    y_preds, y_trues = engine_wrapper(task_module,
                             data_module,
                             trainer_module)
    df = pd.read_csv(cfg.data.filename)
    task_matrix = df[cfg.tasks.task_learn].to_numpy()
    filenames = df["filename"].to_numpy()
    filenames_aligned = []
    for row in y_trues.cpu().numpy():
        # Compare each row of task_matrix to the row from ext_array
        # axis=1: per-row comparison, then .all across columns
        mask = np.all(np.isclose(task_matrix, row, atol=1e-4), axis=1)

        idx = np.flatnonzero(mask)
       
        if idx.size == 0:
            raise ValueError(f"No match for row {row}")
        if idx.size > 1:
            raise ValueError(f"Multiple matches for row {row}: {filenames[idx].tolist()}")

        filenames_aligned.append(filenames[idx[0]])
    df_compiled = pd.DataFrame(
        {
            "filename": filenames_aligned,
            "y_true": y_trues.cpu().numpy().tolist(),
            "y_pred": y_preds.cpu().numpy().tolist(),
        }
    )
    
    os.makedirs(cfg.output_directory, exist_ok=True)
    df_compiled.to_csv(f"{cfg.output_directory}/predictions.csv", index=False)
    log.info("Prediction statistics:")
    for task_name in cfg.tasks.task_learn:
        log.info(f"--- {task_name} ---")
        log.info(f"Mean: {df[task_name].mean():.4f}")
        log.info(f"Std: {df[task_name].std():.4f}")
        log.info(f"Min: {df[task_name].min():.4f}")
        log.info(f"Max: {df[task_name].max():.4f}")

    log.info("Plotting distributions...")
    props = []
    for i, prop in enumerate(cfg.tasks.task_learn):
        plot_kde_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_kde.png")
        plot_histogram_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_hist.png")
        plot_correlation_with_histograms(
            y_trues[:, i].cpu().numpy(), 
            y_preds[:, i].cpu().numpy(), 
            prop,
            "",
            f"{cfg.output_directory}/{prop}_correlation.png")
        props.append(df[prop].values)
    
    props = np.array(props).T
    plot_kde_distribution_multiple(props, cfg.tasks.task_learn, f"{cfg.output_directory}/kde_all.png")
    


def log_hyperparameters(object_dict: dict):


    def is_rank_zero():
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

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


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_predict.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    predict(cfg) 


if __name__ == "__main__":
    main()

