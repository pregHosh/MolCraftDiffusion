import glob
import os
import shutil
from typing import Any, Dict, Literal

import pandas as pd
import torch
import torch.distributed
from tqdm import tqdm
import wandb
import numpy as np
from ase.data import atomic_numbers

from MolecularDiffusion.core import Engine
from MolecularDiffusion.utils.chem.geom_analyzer import (
    create_pyg_graph,
    correct_edges,
)
from MolecularDiffusion.utils.metrics.geom_metrics import check_validity_v0, load_molecules_from_xyz, run_postbuster
from MolecularDiffusion.utils.geometry.geom_utils import (
    read_xyz_file, save_xyz_file, save_xyz_file_atomic_numbers)

from dataclasses import dataclass, replace as dataclass_replace
from typing import Optional

@dataclass
class TaskEvalConfig:
    """Declarative evaluation config for a task type."""
    higher_is_better: bool
    metric_key: str                    # key extracted from metric_dict or performances dict
    needs_generative: bool             # True → analyze_and_save path; False → loss/MAE path
    split_reset_hook: Optional[str]    # model method to call before each solver.evaluate() split
    mae_eval: bool = False             # True → regression MAE path (preds/targets concatenation)


# ---------------------------------------------------------------------------
# Task registry — add one entry per task type family.
# Prefix matching is used for families like "vae_transformer", "vae_equiformer".
# ---------------------------------------------------------------------------
TASK_REGISTRY: dict[str, TaskEvalConfig] = {
    # --- generative diffusion -------------------------------------------------
    "diffusion": TaskEvalConfig(
        higher_is_better=True,
        metric_key="Validity Relax and connected",
        needs_generative=True,
        split_reset_hook=None,
    ),
    "diffusion_hybrid": TaskEvalConfig(
        higher_is_better=True,
        metric_key="Validity Relax and connected",
        needs_generative=True,
        split_reset_hook=None,
    ),
    "diffusion_pyg": TaskEvalConfig(
        higher_is_better=True,
        metric_key="Validity Relax and connected",
        needs_generative=True,
        split_reset_hook=None,
    ),
    "diffusion_tabasco": TaskEvalConfig(
        higher_is_better=True,
        metric_key="Validity Relax and connected",
        needs_generative=True,
        split_reset_hook=None,
    ),
    # --- regression / guidance ------------------------------------------------
    "regression": TaskEvalConfig(
        higher_is_better=False,
        metric_key="mae",
        needs_generative=False,
        split_reset_hook=None,
        mae_eval=True,
    ),
    "guidance": TaskEvalConfig(
        higher_is_better=False,
        metric_key="mae",
        needs_generative=False,
        split_reset_hook=None,
        mae_eval=True,
    ),
    # --- VAE family (prefix-matched: vae_transformer, vae_equiformer, …) -----
    "vae": TaskEvalConfig(
        higher_is_better=True,
        metric_key="match_rate",
        needs_generative=False,
        split_reset_hook="on_validation_epoch_start",
    ),
    # --- LDM ------------------------------------------------------------------
    "diffusion_adit": TaskEvalConfig(
        higher_is_better=False,
        metric_key="valid_posebuster",
        needs_generative=True,
        split_reset_hook=None,
    ),
    # --- SSL3D family (prefix-matched: ssl3d_egcl, ssl3d_equiformer, …) ------
    "ssl3d": TaskEvalConfig(
        higher_is_better=False,
        metric_key="ssl/total_loss",
        needs_generative=False,
        split_reset_hook=None,
    ),
}


def _resolve_task_config(task_type: str) -> TaskEvalConfig:
    """Exact match first, then prefix match (e.g. 'vae_transformer' → 'vae')."""
    if task_type in TASK_REGISTRY:
        return TASK_REGISTRY[task_type]
    for key in TASK_REGISTRY:
        if task_type.startswith(key):
            return TASK_REGISTRY[key]
    raise ValueError(
        f"Unknown task_type '{task_type}'. "
        f"Register it in TASK_REGISTRY in eval.py. "
        f"Known prefixes: {list(TASK_REGISTRY.keys())}"
    )


def resolve_higher_is_better(
    cfg: TaskEvalConfig,
    generative_analysis: bool = False,
    override: Optional[bool] = None,
) -> bool:
    """Direction in which the selection metric improves, for one run.

    ``engine.eval_higher_is_better`` (``override``) always wins. A generative
    task trained *without* ``generative_analysis`` is selected on its test
    loss, so lower is better there; the registry's ``higher_is_better=True``
    describes the validity metric of the generative path and used to keep the
    HIGHEST loss as ``best`` for these runs.
    """
    if override is not None:
        return bool(override)
    if cfg.needs_generative and not generative_analysis:
        return False
    return cfg.higher_is_better


def initial_best_metric(
    task_type: str,
    generative_analysis: bool = False,
    override: Optional[bool] = None,
) -> float:
    """Starting value of the running best, matching :func:`resolve_higher_is_better`."""
    cfg = _resolve_task_config(task_type)
    return -torch.inf if resolve_higher_is_better(cfg, generative_analysis, override) else torch.inf


DIST_THRESHOLD = 3
DIST_RELAX_BOND = 0.25
ANGLE_RELAX = 20
SCALE_FACTOR = 1.2

# Note: The following constant represents the default timeout (in seconds) for
# torch.distributed operations. This value is configured during the initialization
# of the process group (e.g., in the main training script), not here. It is
# included for informational purposes.
DISTRIBUTED_DEFAULT_TIMEOUT_SEC = 30 * 60


import logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, ERROR, or CRITICAL as needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_versioned_output_path(base_output_path: str) -> str:
    """
    Get next available version folder (engine_logs/version_X).
    
    This mimics Lightning's behavior of creating version_0, version_1, etc.
    
    Parameters:
        base_output_path (str): The base output directory (e.g., training_outputs/my_model)
    
    Returns:
        str: Path to the versioned checkpoint folder (e.g., training_outputs/my_model/engine_logs/version_0)
    """
    logs_dir = os.path.join(base_output_path, "engine_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Find highest existing version
    existing = [d for d in os.listdir(logs_dir) if d.startswith("version_") and os.path.isdir(os.path.join(logs_dir, d))]
    if existing:
        versions = []
        for d in existing:
            try:
                versions.append(int(d.split("_")[1]))
            except (ValueError, IndexError):
                continue
        next_version = max(versions) + 1 if versions else 0
    else:
        next_version = 0
    
    version_path = os.path.join(logs_dir, f"version_{next_version}")
    os.makedirs(version_path, exist_ok=True)
    logging.info(f"Checkpoint version folder: {version_path}")
    return version_path

def _manage_best_checkpoints(
    metric_value: float,
    epoch: int,
    solver: Engine,
    output_path: str,
    best_checkpoints: list,
    task_name: str,
    top_k: int = 3,
    higher_is_better: bool = False,
) -> list:
    """Manages saving top-k checkpoints and removing older, less performant ones."""
    
    is_top_k = False
    if len(best_checkpoints) < top_k:
        is_top_k = True
    else:
        best_checkpoints.sort(key=lambda x: x[0], reverse=higher_is_better)
        worst_best_metric = best_checkpoints[-1][0]
        if higher_is_better:
            if metric_value > worst_best_metric:
                is_top_k = True
        else:
            if metric_value < worst_best_metric:
                is_top_k = True

    if is_top_k:
        checkpoint_name = f"{task_name}-epoch={epoch}-metric={metric_value:.4f}.pkl"
        new_checkpoint_path = os.path.join(output_path, checkpoint_name)
        solver.save(new_checkpoint_path, compact=False, full_state=True)
        print(f"\033[92m🚀 Saved new top-k checkpoint: {new_checkpoint_path}\033[0m")

        best_checkpoints.append((metric_value, new_checkpoint_path))

        if len(best_checkpoints) > top_k:
            best_checkpoints.sort(key=lambda x: x[0], reverse=higher_is_better)
            worst_checkpoint_to_remove = best_checkpoints.pop()
            worst_checkpoint_path = worst_checkpoint_to_remove[1]
            try:
                os.remove(worst_checkpoint_path)
                logging.info(f"Removed old top-k checkpoint: {worst_checkpoint_path}")
            except OSError as e:
                logging.warning(f"Error removing old checkpoint {worst_checkpoint_path}: {e}")

    return best_checkpoints

def _call_split_reset(model, hook_name: Optional[str]):
    """Call a reset hook on the model if it exists."""
    if hook_name and hasattr(model, hook_name):
        getattr(model, hook_name)()


def _log_metrics(val_metric_dict: dict, test_metric_dict: dict, logger: str):
    """Log all keys from val/test metric dicts to wandb or logging."""
    is_main_process = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if not is_main_process:
        return
    if logger == "wandb":
        log_dict = {
            f"valid/{k}": (v.item() if hasattr(v, "item") else v)
            for k, v in val_metric_dict.items()
        }
        log_dict.update({
            f"test/{k}": (v.item() if hasattr(v, "item") else v)
            for k, v in test_metric_dict.items()
        })
        wandb.log(log_dict)
    else:
        for k, v in val_metric_dict.items():
            v_s = v.item() if hasattr(v, "item") else v
            logging.info(f"  valid/{k}: {v_s:.4f}")


def _run_eval(
    cfg: TaskEvalConfig,
    task: str,
    solver: Engine,
    epoch: int,
    logger: str,
    output_path: str,
    use_amp: bool,
    precision: str,
    **kwargs,
) -> float:
    """
    Run evaluation for the given task config and return the scalar metric value.

    Generative path  → analyze_and_save → validity metric
    MAE path         → solver.evaluate valid+test → mean absolute error
    Loss path (VAE)  → solver.evaluate valid+test → val_loss from metric_dict
    """
    model = solver.ema_model if solver.ema_decay > 0 else solver.model

    # --- generative diffusion -------------------------------------------------
    if cfg.needs_generative and kwargs.get("generative_analysis", False):
        output_generated_dir = kwargs.get("output_generated_dir", "generated_molecules")
        path = os.path.join(output_generated_dir, f"gen_xyz_{epoch}")
        _, val_loss_raw, _ = solver.evaluate("valid", use_amp=use_amp, precision=precision)
        _, test_loss_raw, _ = solver.evaluate("test", use_amp=use_amp, precision=precision)
        logging.info(
            f"Diffusion — val_loss: {torch.tensor(val_loss_raw).mean().item():.4f}  "
            f"test_loss: {torch.tensor(test_loss_raw).mean().item():.4f}"
        )
        performances = analyze_and_save(
            model, epoch,
            n_samples=kwargs.get("n_samples", 100),
            batch_size=kwargs.get("batch_size", 1),
            logger=logger,
            path_save=path,
            use_posebuster=kwargs.get("use_posebuster", False),
            postbuster_timeout=kwargs.get("postbuster_timeout", 120),
        )
        return performances[kwargs.get("metric", cfg.metric_key)]

    # --- diffusion without generative analysis (loss only) --------------------
    if cfg.needs_generative and not kwargs.get("generative_analysis", False):
        _, val_loss_raw, _ = solver.evaluate("valid", use_amp=use_amp, precision=precision)
        _, test_loss_raw, _ = solver.evaluate("test", use_amp=use_amp, precision=precision)
        val_loss = torch.tensor(val_loss_raw).mean().item()
        test_loss = torch.tensor(test_loss_raw).mean().item()
        logging.info(f"Diffusion (loss only) — val_loss: {val_loss:.4f}  test_loss: {test_loss:.4f}")
        return test_loss

    # --- regression / guidance (MAE) -----------------------------------------
    if cfg.mae_eval:
        _, preds, targets = solver.evaluate("valid", use_amp=use_amp, precision=precision)
        _, preds_test, targets_test = solver.evaluate("test", use_amp=use_amp, precision=precision)
        preds_t = torch.cat(preds, dim=0)
        targets_t = torch.cat(targets, dim=0)
        metric = torch.mean(torch.abs(preds_t - targets_t)).item()
        # stash for saving on improvement
        kwargs["_preds_test"] = torch.cat(preds_test, dim=0)
        kwargs["_trues_test"] = torch.cat(targets_test, dim=0)
        return metric

    # --- loss-based (VAE, LDM, etc.) -----------------------------------------
    _call_split_reset(model, cfg.split_reset_hook)
    val_metric_dict, _, _ = solver.evaluate("valid", use_amp=use_amp, precision=precision)
    _call_split_reset(model, cfg.split_reset_hook)
    test_metric_dict, _, _ = solver.evaluate("test", use_amp=use_amp, precision=precision)

    val_loss = val_metric_dict[cfg.metric_key].item()
    test_loss = test_metric_dict.get(cfg.metric_key, torch.tensor(float("nan"))).item()
    logging.info(f"{task} eval — val_loss: {val_loss:.4f}  test_loss: {test_loss:.4f}")
    _log_metrics(val_metric_dict, test_metric_dict, logger)
    return val_loss


def evaluate(
    task: str,
    solver: Engine,
    epoch: int = 0,
    current_best_metric: float = torch.inf,
    best_checkpoints: list = None,
    logger: Literal["wandb", "logging"] = "logging",
    output_path: str = None,
    use_amp: bool = False,
    precision: str = "bf16",
    **kwargs,
):
    """
    Unified evaluation entry point for all task types.

    Resolves task → TaskEvalConfig via TASK_REGISTRY, runs _run_eval(),
    then executes one shared compare-and-checkpoint block.

    Returns:
        Tuple[float, list]: (current_best_metric, best_checkpoints)
    """
    is_main_process = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    if best_checkpoints is None:
        best_checkpoints = []

    cfg = _resolve_task_config(task)
    # Apply engine-level overrides (from cfg.engine.eval_metric_key / eval_higher_is_better)
    if kwargs.get("eval_metric_key") is not None:
        cfg = dataclass_replace(cfg, metric_key=kwargs["eval_metric_key"])
    cfg = dataclass_replace(
        cfg,
        higher_is_better=resolve_higher_is_better(
            cfg, kwargs.get("generative_analysis", False), kwargs.get("eval_higher_is_better")
        ),
    )
    save_top_k = kwargs.get("save_top_k", 3)
    save_every_val_epoch = kwargs.get("save_every_val_epoch", False)

    if output_path:
        last_path = os.path.join(output_path, "last.pkl")
        solver.save(last_path, compact=False, full_state=True)
        if is_main_process:
            logging.info(f"Saved last checkpoint → {last_path}")

    metric = _run_eval(
        cfg=cfg, task=task, solver=solver, epoch=epoch,
        logger=logger, output_path=output_path,
        use_amp=use_amp, precision=precision, **kwargs,
    )

    improved = (
        metric > current_best_metric if cfg.higher_is_better
        else metric < current_best_metric
    )

    if save_every_val_epoch and output_path and is_main_process:
        ckpt_name = f"{task}-epoch={epoch}-metric={metric:.4f}.pkl"
        solver.save(os.path.join(output_path, ckpt_name), compact=False, full_state=True)
        logging.info(f"Saved per-epoch checkpoint: {ckpt_name}")

    if improved:
        if is_main_process:
            print(
                f"\033[92m🚀 New best at epoch {epoch}: {metric:.4f} "
                f"(was: {current_best_metric:.4f})\033[0m"
            )
        current_best_metric = metric
        best_checkpoints = _manage_best_checkpoints(
            metric_value=metric, epoch=epoch, solver=solver,
            output_path=output_path, best_checkpoints=best_checkpoints,
            task_name=task, top_k=save_top_k, higher_is_better=cfg.higher_is_better,
        )
        if cfg.mae_eval and output_path and is_main_process:
            y_preds = kwargs.get("_preds_test")
            y_trues = kwargs.get("_trues_test")
            if y_preds is not None:
                np.save(os.path.join(output_path, f"y_preds_{epoch}.npy"), y_preds.detach().cpu().numpy())
                np.save(os.path.join(output_path, f"y_trues_{epoch}.npy"), y_trues.detach().cpu().numpy())
    else:
        if is_main_process:
            print(
                f"\033[93m🤷 No improvement at epoch {epoch}: {metric:.4f} "
                f"(best: {current_best_metric:.4f})\033[0m"
            )

    return current_best_metric, best_checkpoints
    

def analyze_and_save(
    model,
    epoch: int,
    n_samples: int = 1000,
    batch_size: int = 100,
    logger: Literal["wandb", "logging"] = "logging",
    path_save: str = "samples",
    use_posebuster: bool = False,
    postbuster_timeout: int = 60,
) -> Dict[str, Any]:
    """
    Samples molecules from a generative model, saves them as XYZ files,
    and computes structural validity statistics.

    Args:
        model: The generative model used for sampling.
        epoch (int): The current training epoch (for logging purposes).
        n_samples (int): Total number of molecules to sample.
        batch_size (int): Number of molecules sampled per batch.
        logger (str): Logging backend, either "wandb" or "logging".
        path_save (str): Directory to save the sampled XYZ files and CSV.

    Returns:
        Dict[str, Any]: Dictionary summarizing validity and connectivity statistics.
    """

    logging.warning(f"Analyzing molecule stability at epoch {epoch}...")

 
    model.max_n_nodes = 150
    molecules = {"one_hot": [], "x": [], "node_mask": []}

    n_batches = n_samples // batch_size
    if n_samples % batch_size != 0:
        n_batches += 1
    current_batch_size = batch_size
    os.makedirs(path_save, exist_ok=True)

    fail_count = 0
    progress_bar = tqdm(range(n_batches), desc="Sampling molecules", leave=True)
    for i in progress_bar:
        # Unwrap EngineLightning or other wrappers
        if hasattr(model, "task"):
            model = model.task

        # Check for Tabasco model first to avoid node_dist_model access
        if hasattr(model, "task_type") and model.task_type == "diffusion_tabasco":
             nodesxsample = None # Not used for Tabasco here
        else:
             nodesxsample = model.node_dist_model.sample(batch_size)

        if getattr(model, "prop_dist_model", None):
            size = nodesxsample[0].item()
            target_value = model.prop_dist_model.sample(size)
            conditions = getattr(model, "condition", getattr(model, "condition_names", []))
            for cond_i, cond in enumerate(conditions):
                if cond == "distortion_d":
                    target_value[..., cond_i] = 0
                elif cond == "num_graph":
                    target_value[..., cond_i] = 1
        try:
            if hasattr(model, "task_type") and model.task_type == "diffusion_tabasco":
                # TABASCO specific sampling - now returns tuple (one_hot, charges, coords, node_mask)
                one_hot, charges, x, node_mask = model.sample(batch_size=batch_size)
                x = x.detach().cpu()
                charges = charges.detach().cpu()
                one_hot = one_hot.detach().cpu()
                node_mask = node_mask.detach().cpu()
                
                # Map internal indices to actual atomic numbers
                if hasattr(model, "atom_vocab") and model.atom_vocab is not None:
                    # Create mapping tensor (e.g., [1, 6, 7, 8, 9] for QM9)
                    mapping = torch.tensor(
                        [atomic_numbers.get(s, 0) for s in model.atom_vocab],
                        dtype=torch.long
                    )
                    # charges are currently indices 0..N-1
                    # we map them to atomic numbers
                    charges_indices = charges.long()
                    charges = mapping[charges_indices]

                
            elif model.prop_dist_model:
                if model.model.context_mask_rate > 0:
                    one_hot, charges, x, node_mask = model.sample_guidance_conitional(
                                                                nodesxsample=nodesxsample,
                                                                target_value=target_value,
                                                                cfg_scale=1,
                                                                target_function=None,
                                                                guidance_ver="cfg") 
                else:
                    one_hot, charges, x, node_mask = model.sample_conditonal(nodesxsample=nodesxsample,
                                                                target_value=target_value,)
            else:
                one_hot, charges, x, node_mask = model.sample(nodesxsample=nodesxsample)
                # keep = (charges > 0).squeeze()
                # one_hot = one_hot[ keep, :]
                # x = x[ keep, :]

            molecules["one_hot"].append(one_hot.squeeze(0) if one_hot.ndim > 2 else one_hot)
            molecules["x"].append(x.squeeze(0) if x.ndim > 2 else x)
            molecules["node_mask"].append(node_mask.squeeze(0) if node_mask.ndim > 2 else node_mask)

            if torch.all(one_hot == 0) or getattr(model, "atom_vocab", None) is None:
                charges_2d = charges.squeeze(-1) if charges.ndim == 3 else charges
                save_xyz_file_atomic_numbers(path_save, x, charges_2d)
            else:
                # Pass atomic_numbers (charges) and use_unknown_fallback for proper unknown atom handling
                use_fallback = getattr(model.model, 'use_unknown_fallback', False) if hasattr(model, 'model') else False
                save_xyz_file(
                    path_save, one_hot, x, 
                    atom_decoder=model.atom_vocab,
                    atomic_numbers=charges.squeeze(-1) if charges is not None else None,
                    use_unknown_fallback=use_fallback,
                )

            for j in range(current_batch_size):
                path_xyz = os.path.join(path_save, f"molecule_{str(j).zfill(3)}.xyz")
                idx = i * batch_size + j
                shutil.move(
                    path_xyz,
                    os.path.join(path_save, f"molecule_{str(idx).zfill(4)}.xyz"),
                )
    
        except Exception as e:
            fail_count += 1
            tqdm.write(f"[Batch {i}] Sampling failed: {e}")

        progress_bar.set_postfix({
            "completed": i + 1,
            "failed": fail_count,
            "success": (i + 1 - fail_count),
            "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
        })

    return _validate_xyzs(path_save, logger, use_posebuster=use_posebuster, postbuster_timeout=postbuster_timeout)

def _validate_xyzs(path_save: str, logger: str, use_posebuster: bool = False, postbuster_timeout: int = 60) -> Dict[str, float]:
    """
    Validates the molecular structures saved as XYZ files by checking geometric and
    connectivity criteria, then logs and returns summary statistics.

    Args:
        path_save (str): Directory containing the XYZ files.
        logger (str): Logging backend, either "wandb" or "logging".
        use_posebuster (bool): Whether to run posebuster analysis.

    Returns:
        Dict[str, float]: Dictionary summarizing average metrics:
            - Validity Strict
            - Validity Relax
            - Fully-connected
            - Percent Atom Valid
    """

    xyzs = sorted(glob.glob(f"{path_save}/*.xyz"))
    n = len(xyzs)

    metrics = {
        "Validity Strict": torch.zeros(n, dtype=torch.float16),
        "Validity Relax": torch.zeros(n, dtype=torch.float16),
        "Fully-connected": torch.zeros(n, dtype=torch.float16),
        "Percent Atom Valid": torch.zeros(n, dtype=torch.float16),
        "Validity Relax and connected": torch.zeros(n, dtype=torch.float16),
        "Validity Strict and connected": torch.zeros(n, dtype=torch.float16),
    }

    for idx, xyz in enumerate(tqdm(xyzs, desc="Processing XYZ files", total=n)):
        try:
            coords, atomic_numbers = read_xyz_file(xyz)
            data = create_pyg_graph(coords, atomic_numbers, r=DIST_THRESHOLD)
            data = correct_edges(data, scale_factor=SCALE_FACTOR)

            is_valid, percent_atom_valid, num_components, _, to_recheck = check_validity_v0(
                data, angle_relax=ANGLE_RELAX, verbose=False
            )

            metrics["Validity Strict"][idx] = float(is_valid)
            metrics["Validity Relax"][idx] = float(is_valid or to_recheck)
            metrics["Fully-connected"][idx] = float(num_components == 1)
            metrics["Percent Atom Valid"][idx] = percent_atom_valid
            metrics["Validity Relax and connected"][idx] = float(is_valid or to_recheck and num_components == 1)
            metrics["Validity Strict and connected"][idx] = float(is_valid and num_components == 1)

        except Exception as e:
            logging.debug(f"[Error] Failed to process {xyz}: {e}")

    df = pd.DataFrame({
        "Filename": xyzs,
        **{k: v.numpy() for k, v in metrics.items()},
    })
    df.to_csv(f"{path_save}/validity.csv", index=False)

    summary = {k: v.mean().item() for k, v in metrics.items()}

    if use_posebuster:
        postbuster_results = None
        try:
            mols, _ = load_molecules_from_xyz(path_save)
            if mols:
                postbuster_results = run_postbuster(mols, timeout=postbuster_timeout)
        except Exception as e:
            logging.warning(f"PoseBuster execution failed or timed out: {e}")

        postbuster_output_path = os.path.join(path_save, "postbuster_metrics.csv")
        if postbuster_results is not None and not postbuster_results.empty:
            postbuster_results.to_csv(postbuster_output_path, index=False)

            check_cols = [
                col
                for col in postbuster_results.columns
                if pd.api.types.is_numeric_dtype(postbuster_results[col])
                or pd.api.types.is_bool_dtype(postbuster_results[col])
            ]
            if check_cols:
                summary["valid_posebuster"] = postbuster_results[check_cols].all(axis=1).mean()
            else:
                summary["valid_posebuster"] = 0.0

            summary.update({
                f"posebuster_{col}_mean": postbuster_results[col].mean()
                for col in postbuster_results.columns
                if pd.api.types.is_numeric_dtype(postbuster_results[col])
            })
        else:
            logging.warning("PoseBuster returned no results or failed. Setting posebuster metrics to 0.")
            summary["valid_posebuster"] = 0.0
            pd.DataFrame().to_csv(postbuster_output_path, index=False)

    if logger == "wandb":
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            wandb.log(summary)
        else:
            logging.info("Skipping wandb logging on non-main process.")
    else:
        max_key_len = max(len(k) for k in summary)
        for key, value in summary.items():
            logging.info(f"{key:<{max_key_len}} : {value:.4f}")

    return summary
