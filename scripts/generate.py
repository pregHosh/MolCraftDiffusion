
"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import glob
import re
from typing import Any, Dict,  Optional, Tuple

import pickle
import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from MolecularDiffusion.core import Engine
from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory
from MolecularDiffusion.utils import (
    RankedLogger,
    # task_wrapper,
    seed_everything,
    recursive_module_to_device
)
import os



log = RankedLogger(__name__, rank_zero_only=True)


def load_model(chkpt_directory, total_step=0):

    model_path = os.path.join(chkpt_directory, "edm_chem.pkl")
    
    if not os.path.exists(model_path):
        log.info(f"'edm_chem.pkl' not found in {chkpt_directory}. Searching for other checkpoints.")
        
        checkpoint_files = glob.glob(os.path.join(chkpt_directory, '*.pkl'))
        checkpoint_files = [f for f in checkpoint_files if 'edm_stat.pkl' not in os.path.basename(f)]

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {chkpt_directory}")

        best_metric = -1.0
        best_checkpoint = None
        
        for ckpt_file in checkpoint_files:
            match = re.search(r"metric=([\d.]+)\.pkl", os.path.basename(ckpt_file))
            if match:
                metric = float(match.group(1))
                if metric > best_metric:
                    best_metric = metric
                    best_checkpoint = ckpt_file
        
        if best_checkpoint:
            model_path = best_checkpoint
        else:
            log.warning("Could not determine best checkpoint from metrics in filenames. Using the first one found.")
            model_path = checkpoint_files[0]

    log.info(f"Loading model from: {model_path}")
    
    try:
        with open(os.path.join(chkpt_directory, "edm_stat.pkl"), "rb") as file:
            edm_stats = pickle.load(file)   
    except (ImportError, FileNotFoundError):
        edm_stats = {"node": None}
    
    engine = Engine(None, None, None, None, None)
    engine = engine.load_from_checkpoint(model_path, interference_mode=True)
    engine.model.node_dist_model = edm_stats["node"]
    if "prop" in edm_stats:
        engine.model.prop_dist_model = edm_stats["prop"]
    
    if total_step > 0:
        engine.model.model.T = total_step

    engine.model.eval()
    return engine

    
# @task_wrapper
def generate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate mode

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)


    log.info(f"Instantiating diffusion task and loading the model <{cfg.tasks._target_}>")
    solver = load_model(cfg.chkpt_directory, cfg.diffusion_steps)
    if not(hasattr(solver.model, 'atom_vocab')) or solver.model.atom_vocab is None:
        solver.model.atom_vocab = cfg.atom_vocab
    
    if not(hasattr(solver.model, 'device')):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        recursive_module_to_device(solver.model, device)



    log.info(f"Instantiating generator... <{cfg.interference._target_}>")
    generator: GenerativeFactory = hydra.utils.instantiate(cfg.interference, task=solver.model)

    object_dict = {
        "cfg": cfg,
        "solver": solver,
        "generator": generator,
    }

    # This does not work atm
    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    
    generator.run()

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



@hydra.main(version_base="1.3", config_path="../configs", config_name="generate.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    generate(cfg)

if __name__ == "__main__":
    main()
