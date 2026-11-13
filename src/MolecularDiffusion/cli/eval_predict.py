"""Eval-Predict command for MolCraft CLI.

Adapted from scripts/eval_predict.py for package-level execution.
"""

import os

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.train import DataModule, ModelTaskFactory_EGCL, OptimSchedulerFactory
from MolecularDiffusion.utils import RankedLogger, seed_everything
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution,
    plot_histogram_distribution,
    plot_kde_distribution_multiple,
    plot_correlation_with_histograms,
)

from MolecularDiffusion.cli._common import (  # noqa: F401  (is_rank_zero stays importable from here)
    is_rank_zero,
    log_hyperparameters as _log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


def load_checkpoint_weights(task, chkpt_path):
    """Load weights from checkpoint with support for Engine and Lightning formats."""
    log.info(f"Loading weights from: {chkpt_path}")
    
    checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    
    # Check if it's a Lightning checkpoint
    if "state_dict" in checkpoint:
        log.info("Detected Lightning checkpoint.")
        state_dict = checkpoint["state_dict"]
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("task."):
                cleaned_state_dict[k[5:]] = v
            else:
                cleaned_state_dict[k] = v
        
        load_result = task.load_state_dict(cleaned_state_dict, strict=False)
        log.info(f"Loaded {len(cleaned_state_dict)} parameters from state_dict")
        if load_result.missing_keys:
            log.warning(f"Missing keys: {load_result.missing_keys}")
        
        # Recover statistics
        for key in ["mean", "std", "weight"]:
            val = None
            if key in checkpoint:
                 val = checkpoint[key]
            elif f"task.{key}" in state_dict:
                 val = state_dict[f"task.{key}"]
            elif key in state_dict:
                 val = state_dict[key]
            
            if val is not None:
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(val, dtype=torch.float32)
                
                # Register as buffer to ensure it moves with the model to the correct device
                if key in task._buffers:
                    task._buffers[key].copy_(val)
                else:
                    task.register_buffer(key, val)
    elif "model" in checkpoint:
        log.info("Detected original Engine checkpoint.")
        task.load_state_dict(checkpoint["model"], strict=False)
        # Recover statistics
        for key in ["mean", "std", "weight"]:
            if key in checkpoint["model"]:
                val = checkpoint["model"][key]
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(val, dtype=torch.float32)
                
                # Register as buffer to ensure it moves with the model to the correct device
                if key in task._buffers:
                    task._buffers[key].copy_(val)
                else:
                    task.register_buffer(key, val)
    else:
        # Fallback for unexpected formats
        log.warning("Unknown checkpoint format. Attempting direct load.")
        task.load_state_dict(checkpoint, strict=False)

    # Ensure task has a device attribute for initial loading, 
    # but don't hardcode it if it's about to be moved by Engine
    if not hasattr(task, 'device'):
        task.device = next(task.parameters()).device if list(task.parameters()) else torch.device('cpu')


def _extract_model_config_from_checkpoint(chkpt_path):
    """Load model_config from a Lightning checkpoint hyper_parameters block."""
    try:
        checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    except Exception as ex:
        log.warning("Could not load checkpoint for model_config extraction: %s", ex)
        return None

    if not isinstance(checkpoint, dict):
        return None

    model_cfg = checkpoint.get("hyper_parameters", {}).get("model_config", None)
    if model_cfg is None:
        return None

    if isinstance(model_cfg, DictConfig):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
    if not isinstance(model_cfg, dict):
        return None
    return model_cfg


def engine_wrapper(task_module, data_module, trainer_module):
    """Run evaluation with Engine."""
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
        logger="logging",
    )
    # Ensure task.device is updated to the actual device solver is using
    task_module.task.device = solver.device

    _, preds_test, targets_test = solver.evaluate("test")
    y_preds = torch.cat(preds_test, dim=0)
    y_trues = torch.cat(targets_test, dim=0)
    return y_preds, y_trues


def predict(cfg: DictConfig) -> None:
    """Evaluate predictions on validation/test sets."""
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(
        cfg.data, task_type=cfg.tasks.task_type, train_ratio=0
    )
    data_module.load()
    
    log.info(f"Instantiating task <{cfg.tasks._target_}>")
    act_fn_cfg = cfg.tasks.get("act_fn")
    act_fn = hydra.utils.instantiate(act_fn_cfg) if act_fn_cfg is not None else None
    
    # Store checkpoint path and temporarily disable it for task_module.build()
    # to avoid the factory's internal (legacy) loading.
    chkpt_path = cfg.tasks.get("chkpt_path")
    
    # Prefer model architecture/task settings saved with the checkpoint.
    # This avoids config drift (e.g., node feature dims, backbone options).
    cfg_tasks = OmegaConf.to_container(cfg.tasks, resolve=True)
    ckpt_model_cfg = _extract_model_config_from_checkpoint(chkpt_path) if chkpt_path else None
    tasks_cfg_dict = dict(ckpt_model_cfg) if ckpt_model_cfg else dict(cfg_tasks)

    # Keep explicit runtime overrides from eval config for target selection/metrics if provided.
    for key in ("task_learn", "criterion", "metric"):
        if key in cfg_tasks and cfg_tasks[key] is not None:
            tasks_cfg_dict[key] = cfg_tasks[key]

    tasks_cfg_dict["chkpt_path"] = None
    tasks_cfg = OmegaConf.create(tasks_cfg_dict)

    # Mirror train-time overrides so eval uses identical feature dimensionality.
    instantiate_kwargs = {"act_fn": act_fn} if act_fn is not None else {}
    data_atom_vocab = None
    if "atom_vocab" in cfg.data:
        data_atom_vocab = list(cfg.data.atom_vocab)
    if cfg.data.get("allow_unknown", False):
        if data_atom_vocab is None:
            data_atom_vocab = []
        data_atom_vocab.append("Suisei")
    if data_atom_vocab is not None:
        instantiate_kwargs["atom_vocab"] = data_atom_vocab

    node_feature_choice = cfg.data.get("node_feature_choice", None)
    if node_feature_choice is not None:
        instantiate_kwargs["node_feature"] = node_feature_choice
        instantiate_kwargs["node_feature_choice"] = node_feature_choice

    feature_dim = None
    for subset in (data_module.valid_set, data_module.test_set, data_module.train_set):
        if subset is None or len(subset) == 0:
            continue
        sample = subset[0]
        if isinstance(sample, dict):
            node_feature_0 = sample.get("node_feature", None)
            if node_feature_0 is None and "graph" in sample:
                node_feature_0 = getattr(sample["graph"], "x", None)
        else:
            node_feature_0 = getattr(sample, "node_feature", None)
            if node_feature_0 is None:
                node_feature_0 = getattr(sample, "x", None)
        if node_feature_0 is not None:
            feature_dim = int(node_feature_0.shape[1])
            break

    if feature_dim is not None:
        base_feature_dim = len(data_atom_vocab or list(tasks_cfg.get("atom_vocab", [])))
        if not cfg.data.get("use_ohe_feature", True):
            base_feature_dim = 0
        inferred_extra_dim = max(0, feature_dim - base_feature_dim)
        instantiate_kwargs["node_feature_dim"] = inferred_extra_dim
    elif chkpt_path:
        # Fallback to checkpoint metadata when dataset wrappers hide raw features.
        try:
            checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
            model_cfg = checkpoint.get("hyper_parameters", {}).get("model_config", {})
            if isinstance(model_cfg, DictConfig):
                model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
            if isinstance(model_cfg, dict) and model_cfg.get("node_feature_dim") is not None:
                instantiate_kwargs["node_feature_dim"] = int(model_cfg["node_feature_dim"])
        except Exception as ex:
            log.warning("Could not infer node_feature_dim from checkpoint metadata: %s", ex)

    task_module: ModelTaskFactory_EGCL = hydra.utils.instantiate(tasks_cfg, **instantiate_kwargs)
    task_module.build()
    
    # Manually load weights using our robust loader
    if chkpt_path:
        load_checkpoint_weights(task_module.task, chkpt_path)
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_module: OptimSchedulerFactory = hydra.utils.instantiate(
        cfg.trainer, parameters=task_module.task.parameters()
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "task": task_module,
        "trainer": trainer_module,
    }

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)

    y_preds, y_trues = engine_wrapper(task_module, data_module, trainer_module)
    
    data_source = cfg.data.get("filename")
    if not data_source or str(data_source) in {"path_to_csv.csv", "None"}:
        data_source = cfg.data.get("ase_db_path")
    elif str(data_source).endswith(".csv") and not os.path.exists(str(data_source)):
        # Prefer ASE DB when CSV is a placeholder or missing in eval configs.
        data_source = cfg.data.get("ase_db_path", data_source)

    if data_source is not None and str(data_source).endswith('.db'):
        from ase.db import connect
        db = connect(data_source)
        records = []
        for i, row in enumerate(db.select()):
            record = {"filename": f"db_entry_{i}"}
            for t in cfg.tasks.task_learn:
                val = row.data.get(t, -1) if hasattr(row, 'data') and row.data is not None else row.get(t, -1)
                
                # dataset.py sometimes has literal_eval
                if val == "":
                    val = -1
                import math
                from MolecularDiffusion.utils import literal_eval
                try:
                    val = literal_eval(str(val))
                except (ValueError, SyntaxError):
                    val = val
                if val == "":
                    val = math.nan
                record[t] = float(val)
            records.append(record)
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(data_source)
    
    task_matrix = df[cfg.tasks.task_learn].to_numpy()
    filenames = df["filename"].to_numpy()
    filenames_aligned = []
    used_match_indices = set()
    
    for row in y_trues.cpu().numpy():
        mask = np.all(np.isclose(task_matrix, row, atol=1e-4), axis=1)
        idx = np.flatnonzero(mask)
        
        if idx.size == 0:
            raise ValueError(f"No match for row {row}")
        if idx.size > 1:
            # Multiple records can share identical target values; pick a stable
            # unmatched candidate to keep 1:1 alignment with predictions.
            chosen = None
            for j in idx.tolist():
                if j not in used_match_indices:
                    chosen = j
                    break
            if chosen is None:
                chosen = int(idx[0])
            used_match_indices.add(chosen)
            filenames_aligned.append(filenames[chosen])
        else:
            chosen = int(idx[0])
            used_match_indices.add(chosen)
            filenames_aligned.append(filenames[chosen])
    
    df_compiled = pd.DataFrame({
        "filename": filenames_aligned,
        "y_true": y_trues.cpu().numpy().tolist(),
        "y_pred": y_preds.cpu().numpy().tolist(),
    })

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
            f"{cfg.output_directory}/{prop}_correlation.png",
        )
        props.append(df[prop].values)

    props = np.array(props).T
    plot_kde_distribution_multiple(props, cfg.tasks.task_learn, f"{cfg.output_directory}/kde_all.png")


def log_hyperparameters(object_dict: dict):
    """Log hyperparameters for debugging."""
    _log_hyperparameters(object_dict, log)


def eval_predict_main(cfg: DictConfig):
    """Entry point for CLI eval-predict command."""
    predict(cfg)
