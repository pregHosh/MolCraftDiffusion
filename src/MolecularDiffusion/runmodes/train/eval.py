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

from MolecularDiffusion.core import Engine
from MolecularDiffusion.utils.geom_analyzer import (
    create_pyg_graph,
    correct_edges,
)
from MolecularDiffusion.utils.geom_metrics import check_validity_v0, load_molecules_from_xyz, run_postbuster
from MolecularDiffusion.utils.geom_utils import (
    read_xyz_file, save_xyz_file)

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
        solver.save(new_checkpoint_path)
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
    Evaluates the performance of a trained model based on the specified task.

    For 'diffusion' tasks, it evaluates generative performance by sampling molecules,
    saving them, and analyzing their structural validity and connectivity.
    For 'property' or 'guidance' tasks, it evaluates predictive performance
    by calculating Mean Absolute Error (MAE).

    Args:
        task (str): The type of task being evaluated ("diffusion", "property", or "guidance").
        solver (Engine): The training engine containing the model and evaluation methods.
        epoch (int, optional): The current training epoch, used for naming generated files. Defaults to 0.
        best_checkpoints (list, optional): A list of tuples containing the metric and path of the best checkpoints.
        logger (Literal["wandb", "logging"], optional): The logging backend to use. Defaults to "logging".
        **kwargs: Additional keyword arguments specific to the task, such as:
            - output_generated_dir (str): Directory to save generated molecules (for diffusion).
            - generative_analysis (bool): Whether to perform generative analysis (for diffusion).
            - n_samples (int): Number of samples to generate (for diffusion).
            - metric (str): The metric to return from generative analysis (for diffusion).

    Returns:
        Tuple[float, list]: A tuple containing the best performance metric and the list of best checkpoints.
    """
    is_main_process = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    if best_checkpoints is None:
        best_checkpoints = []

    if output_path:
        last_path = os.path.join(output_path, "last.pkl")
        solver.save(last_path)
        if is_main_process:
            logging.info(f"Saved last model checkpoint to {last_path}")

    save_top_k = kwargs.get("save_top_k", 3)
    save_every_val_epoch = kwargs.get("save_every_val_epoch", False)

    if task == "diffusion":
        output_generated_dir = kwargs.get("output_generated_dir", None)
        if output_generated_dir is None:
            output_generated_dir = "generated_molecules"

        _, val_loss, _ = solver.evaluate("valid", use_amp=use_amp, precision=precision)
        _, test_loss, _ = solver.evaluate("test", use_amp=use_amp, precision=precision)
        val_loss = torch.tensor(val_loss).mean().item()
        test_loss = torch.tensor(test_loss).mean().item()

        if kwargs.get("generative_analysis", False):
            metrics = 0.0  # Default value
            # if is_main_process:
            path = os.path.join(output_generated_dir, f"gen_xyz_{epoch}")
            model_to_eval = (
                solver.ema_model if solver.ema_decay > 0 else solver.model
            )
            performances = analyze_and_save(
                model_to_eval,
                epoch,
                n_samples=kwargs.get("n_samples", 100),
                batch_size=kwargs.get("batch_size", 1),
                logger=logger,
                path_save=path,
                use_posebuster=kwargs.get("use_posebuster", False),
                postbuster_timeout=kwargs.get("postbuster_timeout", 120),
            )

            metrics = performances[
                kwargs.get("metric", "Validity Relax and connected")
            ]
            if save_every_val_epoch and is_main_process:
                checkpoint_name = f"edm-gen-epoch={epoch}-metric={metrics:.4f}.pkl"
                checkpoint_path = os.path.join(output_path, checkpoint_name)
                solver.save(checkpoint_path)
                logging.info(f"Saved checkpoint for epoch {epoch} with metric {metrics:.4f} at {checkpoint_path}")
            if metrics > current_best_metric:
                if is_main_process:
                    print(
                        f"\033[92m🚀 New best metric at epoch {epoch}: {metrics:.4f} (previously: {current_best_metric:.4f})\033[0m"
                    )
                current_best_metric = metrics
                # if is_main_process:
                best_checkpoints = _manage_best_checkpoints(
                    metric_value=metrics,
                    epoch=epoch,
                    solver=solver,
                    output_path=output_path,
                    best_checkpoints=best_checkpoints,
                    task_name="edm-gen",
                    top_k=save_top_k,
                    higher_is_better=True,
                )
            else:
                if is_main_process:
                    print(
                        f"\033[93m🤷 No improvement at epoch {epoch}: {metrics:.4f} (best: {current_best_metric:.4f})\033[0m"
                    )
            # # if torch.distributed.is_initialized():
            #     objects_to_broadcast = [best_checkpoints] if is_main_process else [None]
            #     torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
            #     best_checkpoints = objects_to_broadcast[0]

        else:
            metrics = test_loss

            if metrics < current_best_metric:
                if is_main_process:
                    print(
                        f"\033[92m🚀 New best metric at epoch {epoch}: {metrics:.4f} (previously: {current_best_metric:.4f})\033[0m"
                    )
                current_best_metric = metrics
                # if is_main_process:
                best_checkpoints = _manage_best_checkpoints(
                    metric_value=metrics,
                    epoch=epoch,
                    solver=solver,
                    output_path=output_path,
                    best_checkpoints=best_checkpoints,
                    task_name="edm-loss",
                    top_k=save_top_k,
                    higher_is_better=False,
                )
            else:
                if is_main_process:
                    print(
                        f"\033[93m🤷 No improvement at epoch {epoch}: {metrics:.4f} (best: {current_best_metric:.4f})\033[0m"
                    )

            # if torch.distributed.is_initialized():
            #     objects_to_broadcast = [best_checkpoints] if is_main_process else [None]
            #     torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
            #     best_checkpoints = objects_to_broadcast[0]

    elif task in ("regression", "guidance"):
        _, preds, targets = solver.evaluate("valid")
        _, preds_test, targets_test = solver.evaluate("test")
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        mae_per_property = torch.mean(torch.abs(preds - targets), dim=0)
        y_preds = torch.cat(preds_test, dim=0)
        y_trues = torch.cat(targets_test, dim=0)
        metrics = torch.mean(mae_per_property)
        if metrics < current_best_metric:
            if is_main_process:
                print(
                    f"\033[92m🚀 New best metric at epoch {epoch}: {metrics:.4f} (previously: {current_best_metric:.4f})\033[0m"
                )
            current_best_metric = metrics
            # if is_main_process:
            best_checkpoints = _manage_best_checkpoints(
                metric_value=metrics,
                epoch=epoch,
                solver=solver,
                output_path=output_path,
                best_checkpoints=best_checkpoints,
                task_name=task,
                top_k=3,
                higher_is_better=False,
            )
            np.save(
                os.path.join(output_path, f"y_preds_{epoch}.npy"),
                y_preds.detach().cpu().numpy(),
            )
            np.save(
                os.path.join(output_path, f"y_trues_{epoch}.npy"),
                y_trues.detach().cpu().numpy(),
            )
        else:
            if is_main_process:
                print(
                    f"\033[93m🤷 No improvement at epoch {epoch}: {metrics:.4f} (best: {current_best_metric:.4f})\033[0m"
                )

        # if torch.distributed.is_initialized():
        #     objects_to_broadcast = [best_checkpoints] if is_main_process else [None]
        #     torch.distributed.broadcast_object_list(objects_to_broadcast, src=0)
        #     best_checkpoints = objects_to_broadcast[0]

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
        nodesxsample = model.node_dist_model.sample(batch_size)
        if model.prop_dist_model:
            size = nodesxsample.item()
            target_value = model.prop_dist_model.sample(size)
            target_value = model.prop_dist_model.sample(size)
            if "distortion_d" in model.condition: # only sample clean molecules during the interference
                target_value[-2] = 0
        try:
            if model.prop_dist_model:
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

            molecules["one_hot"].append(one_hot.detach().cpu().squeeze(0))
            molecules["x"].append(x.detach().cpu().squeeze(0))
            molecules["node_mask"].append(node_mask.detach().cpu().squeeze(0))

            save_xyz_file(path_save, one_hot, x, atom_decoder=model.atom_vocab)

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
