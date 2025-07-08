import glob
import os
import shutil
from typing import Any, Dict, Literal

import pandas as pd
import torch
from tqdm import tqdm
import wandb
import numpy as np

from MolecularDiffusion.core import Engine
from MolecularDiffusion.utils.geom_analyzer import (
    create_pyg_graph,
    correct_edges,
)
from MolecularDiffusion.utils.geom_metrics import check_validity_v0
from MolecularDiffusion.utils.geom_utils import (
    read_xyz_file, save_xyz_file)

DIST_THRESHOLD = 3
DIST_RELAX_BOND = 0.25
ANGLE_RELAX = 20
SCALE_FACTOR = 1.2

import logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG, WARNING, ERROR, or CRITICAL as needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate(
    task: str,
    solver: Engine,
    epoch: int = 0,
    current_best_metric: float = torch.inf,
    logger: Literal["wandb", "logging"] = "logging",
    output_path: str = None,
    **kwargs
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
        logger (Literal["wandb", "logging"], optional): The logging backend to use. Defaults to "logging".
        **kwargs: Additional keyword arguments specific to the task, such as:
            - output_generated_dir (str): Directory to save generated molecules (for diffusion).
            - generative_analysis (bool): Whether to perform generative analysis (for diffusion).
            - n_samples (int): Number of samples to generate (for diffusion).
            - metric (str): The metric to return from generative analysis (for diffusion).

    Returns: float
        The performance metric for the given task.
        - For 'diffusion' with generative analysis: The specified performance metric (e.g., "Validity Relax and connected").
        - For 'diffusion' without generative analysis: The test loss.
        - For 'property' or 'guidance': The average MAE on the validation set.
    """
    
    if task == "diffusion":
        
        output_generated_dir = kwargs.get("output_generated_dir", None)
        if output_generated_dir is None:
            output_generated_dir = "generated_molecules"

        _, val_loss, _ = solver.evaluate("valid")
        _, test_loss, _ = solver.evaluate("test")
        val_loss = torch.tensor(val_loss).mean().item()
        test_loss = torch.tensor(test_loss).mean().item()
        if kwargs.get("generative_analysis", False):
            path = os.path.join(output_generated_dir, f"gen_xyz_{epoch}")
            performances = analyze_and_save(
                                solver.model,
                                epoch,
                                n_samples=kwargs.get("n_samples", 100),
                                batch_size=1,
                                logger=logger,
                                path_save=path,
                            )

            metrics = performances[kwargs.get("metric", "Validity Relax and connected")]
            logging.info("Improvement by {:.4f} at epoch {}".format(
                metrics, epoch))
            if metrics > current_best_metric:
                solver.save(os.path.join(output_path, f"edm_{epoch}.pkl"))

        else:
            metrics = test_loss
            logging.info("Improvement by {:.4f} at epoch {}".format(
                metrics, epoch))
            if metrics < current_best_metric:
                solver.save(os.path.join(output_path, f"edm_{epoch}.pkl"))
        

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
            logging.info("Improvement by {:.4f} at epoch {}".format(
                metrics, epoch))
            solver.save(os.path.join(output_path, f"{task}_{epoch}.pkl"))
            np.save(
                os.path.join(output_path, f"y_preds_{epoch}.npy"),
                y_preds.detach().cpu().numpy(),
            )
            np.save(
                os.path.join(output_path, f"y_trues_{epoch}.npy"),
                y_trues.detach().cpu().numpy(),
            )
            
    return metrics
    
def analyze_and_save(
    model,
    epoch: int,
    n_samples: int = 1000,
    batch_size: int = 100,
    logger: Literal["wandb", "logging"] = "logging",
    path_save: str = "samples",
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

    batch_size = min(batch_size, n_samples)
    model.max_n_nodes = 150
    molecules = {"one_hot": [], "x": [], "node_mask": []}

    n_batches = n_samples // batch_size
    os.makedirs(path_save, exist_ok=True)

    fail_count = 0
    progress_bar = tqdm(range(n_batches), desc="Sampling molecules", leave=True)
    for i in progress_bar:
        nodesxsample = model.node_dist_model.sample(batch_size)
        try:
            one_hot, charges, x, node_mask = model.sample(nodesxsample=nodesxsample)
            keep = (charges > 0).squeeze()

            one_hot = one_hot[:, keep, :]
            x = x[:, keep, :]

            molecules["one_hot"].append(one_hot.detach().cpu().squeeze(0))
            molecules["x"].append(x.detach().cpu().squeeze(0))
            molecules["node_mask"].append(node_mask.detach().cpu().squeeze(0))

            save_xyz_file(path_save, one_hot, x, atom_decoder=model.atom_vocab)
            xyz_path = os.path.join(path_save, "molecule_000.xyz")
            new_name = os.path.join(path_save, f"molecule_{str(i + 1).zfill(4)}.xyz")
            shutil.move(xyz_path, new_name)

        except Exception as e:
            fail_count += 1
            tqdm.write(f"[Batch {i}] Sampling failed: {e}")

        progress_bar.set_postfix({
            "completed": i + 1,
            "failed": fail_count,
            "success": (i + 1 - fail_count),
            "success_rate": f"{100 * (i + 1 - fail_count) / (i + 1):.1f}%",
        })

    return _validate_xyzs(path_save, logger)

# TODO postbuster
def _validate_xyzs(path_save: str, logger: str) -> Dict[str, float]:
    """
    Validates the molecular structures saved as XYZ files by checking geometric and
    connectivity criteria, then logs and returns summary statistics.

    Args:
        path_save (str): Directory containing the XYZ files.
        logger (str): Logging backend, either "wandb" or "logging".

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

    if logger == "wandb":
        wandb.log(summary)
    elif logger == "logging":
        max_key_len = max(len(k) for k in summary)
        for key, value in summary.items():
            logging.info(f"{key:<{max_key_len}} : {value:.4f}")

    return summary
