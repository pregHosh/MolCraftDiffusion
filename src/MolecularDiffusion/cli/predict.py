"""Prediction command for MolCraft CLI.

Adapted from scripts/predict.py for package-level execution.
"""

import os
from glob import glob

import hydra
import numpy as np
import pandas as pd
import torch
from ase.data import atomic_numbers
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from tqdm import tqdm

from MolecularDiffusion.core import Engine
from MolecularDiffusion.data.datasets.pointcloud import PointCloud_Mol
from MolecularDiffusion.data.features.feature import (
    NodeFeaturizer,
)
from MolecularDiffusion.utils import RankedLogger, seed_everything
from MolecularDiffusion.utils.plot_function import (
    plot_kde_distribution,
    plot_histogram_distribution,
    plot_kde_distribution_multiple,
)

from MolecularDiffusion.cli._common import (  # noqa: F401  (is_rank_zero stays importable from here)
    is_rank_zero,
    log_hyperparameters as _log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _to_plain(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _extract_node_feature(metadata):
    """Recover the training node feature choice from common checkpoint layouts."""
    if metadata is None:
        return None
    metadata = _to_plain(metadata)
    if not isinstance(metadata, dict):
        return None

    for key in ("node_feature", "node_feature_choice"):
        value = metadata.get(key)
        if value is not None:
            return value

    for section in ("model_config", "data", "datamodule", "cfg", "hyperparameters"):
        nested = metadata.get(section)
        if nested is None:
            continue
        value = _extract_node_feature(nested)
        if value is not None:
            return value

    return None


def load_model(chkpt_path, task_config=None, atom_vocab=None):
    """Load a pre-trained model from checkpoint with auto-detection."""
    log.info(f"Loading checkpoint from: {chkpt_path}")
    
    # Try loading as Lightning checkpoint first if it has .ckpt extension
    if chkpt_path.endswith('.ckpt'):
        try:
            from MolecularDiffusion.core.engine_lightning import EngineLightning
            wrapper = EngineLightning.load_from_checkpoint(chkpt_path, map_location="cpu")
            log.info("Successfully loaded model using EngineLightning.load_from_checkpoint")
            node_feature = _extract_node_feature(getattr(wrapper, "hparams", {}))
            if node_feature is not None:
                wrapper.task.node_feature = node_feature
                wrapper.task.node_feature_choice = node_feature
            
            # Need to return something that has a .model attribute for backward compatibility
            class SolverWrapper:
                def __init__(self, task):
                    self.model = task
            
            solver = SolverWrapper(wrapper.task)
            solver.model.eval()
            return solver
        except Exception as e:
            log.warning(f"EngineLightning.load_from_checkpoint failed ({type(e).__name__}: {e}). Trying manual fallback.")

    # Manual fallback or original engine (.pkl/no extension)
    checkpoint = torch.load(chkpt_path, map_location="cpu", weights_only=False)
    
    # Check if it's a Lightning checkpoint dictionary
    if "hyper_parameters" in checkpoint:
        log.info("Detected Lightning checkpoint dictionary.")
        hparams = checkpoint.get("hyper_parameters", {})
        node_feature = _extract_node_feature(hparams)
        
        # Try to get model_config from checkpoint
        model_config = hparams.get("model_config", task_config)
        if model_config is None:
             raise ValueError("Lightning checkpoint lacks 'model_config' and no 'task_config' provided.")
             
        # Instantiate task
        if isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        
        # Ensure we have atom_vocab if needed
        if atom_vocab is not None and ('atom_vocab' not in model_config or model_config.atom_vocab is None):
            OmegaConf.set_struct(model_config, False)
            model_config.atom_vocab = atom_vocab

        task_factory = hydra.utils.instantiate(model_config)
        task = task_factory.build()
        if node_feature is not None:
            task.node_feature = node_feature
            task.node_feature_choice = node_feature
        
        # Load weights
        state_dict = checkpoint.get("state_dict", {})
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("task."):
                cleaned_state_dict[k[5:]] = v
            else:
                cleaned_state_dict[k] = v
        
        task.load_state_dict(cleaned_state_dict, strict=False)
        log.info(f"Loaded {len(cleaned_state_dict)} parameters from state_dict")

        # Try to recover mean/std if they are in the checkpoint root or state_dict but not as buffers
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
        
        # Ensure task has a device attribute
        if not hasattr(task, 'device'):
            task.device = next(task.parameters()).device if list(task.parameters()) else torch.device('cpu')

        class SolverWrapper:
            def __init__(self, task):
                self.model = task
        
        solver = SolverWrapper(task)
        solver.model.eval()
        # Ensure task.device is updated to the actual device solver is using
        if hasattr(solver.model, 'device') and solver.model.device != next(solver.model.parameters()).device:
             solver.model.device = next(solver.model.parameters()).device if list(solver.model.parameters()) else torch.device('cpu')
        elif not hasattr(solver.model, 'device'):
             solver.model.device = next(solver.model.parameters()).device if list(solver.model.parameters()) else torch.device('cpu')
        return solver
    else:
        # Original Engine checkpoint
        engine = Engine(None, None, None, None, None)
        solver = engine.load_from_checkpoint(chkpt_path, interference_mode=True)
        solver.model.eval()
        # Ensure task.device is updated to the actual device solver is using
        solver.model.device = solver.device
        node_feature = getattr(solver, "node_feature", None)
        if node_feature is not None:
            solver.model.node_feature = node_feature
            solver.model.node_feature_choice = node_feature
        return solver


def xyz2mol(xyz_file, atom_vocab, node_feature, edge_type="fully_connected", 
            radius=4.0, n_neigh=5, device="cpu"):
    """Convert an XYZ file into a PyTorch Geometric Data object."""
    mol_obj = {}
    mol_xyz = PointCloud_Mol.from_xyz(xyz_file, with_hydrogen=True, forbidden_atoms=[])
    coords = mol_xyz.get_coord()
    n_nodes = len(mol_xyz.atoms)

    atom_symbols = [atom.element for atom in mol_xyz.atoms]
    charges = [
        atomic_numbers[atom.element]
        for atom in mol_xyz.atoms
        if atom.element in atomic_numbers
    ]
    charges = torch.as_tensor(charges, dtype=torch.long)

    if len(charges) != n_nodes:
        raise ValueError(f"Could not map all atoms in {xyz_file} to atomic numbers")

    if isinstance(node_feature, str):
        featurizer = NodeFeaturizer(
            atom_vocab=atom_vocab,
            use_ohe=True,
            geom_feature=node_feature,
            allow_unknown=False,
        )
        node_features = featurizer.featurize_all(atom_symbols, charges, coords)
    elif node_feature:
        raise ValueError(
            "Prediction from XYZ supports string geometric node features only. "
            f"Got node_feature={node_feature!r}."
        )
    else:
        featurizer = NodeFeaturizer(
            atom_vocab=atom_vocab,
            use_ohe=True,
            geom_feature=None,
            allow_unknown=False,
        )
        node_features = featurizer.compute_ohe(atom_symbols)

    node_features = node_features.to(torch.float32)
    node_mask = torch.ones(n_nodes, dtype=torch.int8)

    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(1 * n_nodes * n_nodes, 1)
    h = node_features.view(1 * n_nodes, -1).clone()
    
    if edge_type == "distance":
        edge_index = radius_graph(coords, r=radius)
    elif edge_type == "neighbor":
        edge_index = knn_graph(coords, k=n_neigh)
    elif edge_type == "fully_connected":
        num_nodes = coords.size(0)
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index[:, row != col]
    else:
        raise ValueError(f"Unknown edge type {edge_type}")
    
    graph_data = Data(
        x=h,
        pos=coords,
        atomic_numbers=charges,
        natoms=torch.tensor([n_nodes]),
        edge_index=edge_index,
        times=torch.tensor([0]),
        batch=torch.zeros(n_nodes, dtype=torch.long),
    ).to(device)
    
    mol_obj["graph"] = graph_data
    return mol_obj


def count_atoms_from_xyz(path: str) -> int:
    """Fast atom counter for XYZ files."""
    try:
        with open(path, "r") as f:
            first = f.readline().strip()
            return int(first)
    except Exception:
        return 0


def _runner(solver, xyz_paths: list, max_atoms: int = 100) -> torch.Tensor:
    """Runs predictions on a list of XYZ files."""
    device = getattr(solver.model, 'device', next(solver.model.parameters()).device if list(solver.model.parameters()) else torch.device('cpu'))
    task_names = list(solver.model.task.keys())
    num_molecules = len(xyz_paths)
    node_feature = getattr(solver.model, "node_feature", None)
    expected_dim = getattr(getattr(solver.model, "model", None), "in_node_nf", None)
    if expected_dim is not None and node_feature is None:
        expected_x_dim = int(expected_dim) - 1
        atom_vocab_dim = len(getattr(solver.model, "atom_vocab", []))
        if expected_x_dim > atom_vocab_dim:
            raise ValueError(
                "This checkpoint expects extra node feature columns "
                f"({expected_x_dim} x-columns vs {atom_vocab_dim} atom OHE columns), "
                "but no node_feature/node_feature_choice was found in the checkpoint "
                "or prediction config. Set node_feature in configs/predict.yaml to the "
                "training value, e.g. node_feature: atom_geom_compact."
            )

    progress_bar = tqdm(
        enumerate(xyz_paths),
        desc="Predicting molecules",
        leave=True,
        dynamic_ncols=True,
        total=num_molecules,
    )

    predictions = []
    xyz_paths_clear = []
    skipped = 0
    
    for i, xyz_path in progress_bar:
        n_atoms = count_atoms_from_xyz(xyz_path)
        if n_atoms > max_atoms:
            skipped += 1
            progress_bar.set_postfix({"batch": i + 1, "skipped": skipped})
            log.info(f"Skipping {xyz_path} (atoms={n_atoms} > max_atoms={max_atoms})")
            continue

        try:
            mol_obj = xyz2mol(
                xyz_file=xyz_path,
                atom_vocab=solver.model.atom_vocab,
                node_feature=node_feature,
                device=device,
            )
            if expected_dim is not None:
                expected_x_dim = int(expected_dim) - 1
                actual_x_dim = int(mol_obj["graph"].x.shape[1])
                if actual_x_dim != expected_x_dim:
                    raise ValueError(
                        f"Node feature dimension mismatch for {xyz_path}: built x with "
                        f"{actual_x_dim} columns using node_feature={node_feature!r}, "
                        f"but model expects {expected_x_dim} columns before atomic number. "
                        "Check that the checkpoint saved node_feature/node_feature_choice "
                        "or pass node_feature in the prediction config."
                    )
            prediction = solver.model.predict(mol_obj, evaluate=True)[0]
            predictions.append(prediction.detach().cpu().reshape(-1).numpy())
            current_preds_dict = {prop_name: prediction[j].item() for j, prop_name in enumerate(task_names)}
            progress_bar.set_postfix({"batch": i + 1, "skipped": skipped, **current_preds_dict})
            xyz_paths_clear.append(xyz_path)
        except Exception as e:
            skipped += 1
            progress_bar.set_postfix({"batch": i + 1, "skipped": skipped})
            log.warning(f"Skipping {xyz_path}: {type(e).__name__}: {e}")

    if predictions:
        predictions = np.stack(predictions, axis=0)
    else:
        predictions = np.empty((0, len(task_names)))
    
    return predictions, xyz_paths_clear


def runner(cfg: DictConfig) -> None:
    """Property prediction run."""
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating diffusion task and loading the model <{cfg.tasks._target_}>")
    solver = load_model(cfg.chkpt_directory, task_config=cfg.tasks, atom_vocab=cfg.atom_vocab)
   
    task_names = list(solver.model.task.keys())
    
    if not hasattr(solver.model, 'std') or solver.model.std is None:
        chkpt = torch.load(cfg.chkpt_directory, weights_only=False)
        if "model" in chkpt:
            solver.model.std = chkpt["model"].get("std", torch.ones(1)).to(solver.model.device)
            solver.model.weight = chkpt["model"].get("weight", torch.ones(1)).to(solver.model.device)
            solver.model.mean = chkpt["model"].get("mean", torch.zeros(1)).to(solver.model.device)
        elif "state_dict" in chkpt:
            # Fallback for Lightning if not already loaded by load_model
            sd = chkpt["state_dict"]
            solver.model.std = sd.get("task.std", sd.get("std", torch.ones(1))).to(solver.model.device)
            solver.model.weight = sd.get("task.weight", sd.get("weight", torch.ones(1))).to(solver.model.device)
            solver.model.mean = sd.get("task.mean", sd.get("mean", torch.zeros(1))).to(solver.model.device)
        
    if not hasattr(solver.model, 'atom_vocab'):
        solver.model.atom_vocab = cfg.atom_vocab
    if getattr(solver.model, 'node_feature', None) is None:
        solver.model.node_feature = cfg.node_feature
        solver.model.node_feature_choice = cfg.node_feature
    
    object_dict = {"cfg": cfg, "solver": solver}

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)
    
    os.makedirs(cfg.output_directory, exist_ok=True)

    if is_rank_zero():
        config_path = os.path.join(cfg.output_directory, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)
        log.info(f"Configuration saved to {config_path}")
    
    log.info("Running the predictions...")
    xyz_paths = glob(f"{cfg.xyz_directory}/*.xyz")
    xyz_paths = [str(xyz_path) for xyz_path in xyz_paths]
    predictions, xyz_paths_clear = _runner(solver, xyz_paths, max_atoms=cfg.get("max_atoms", 100))

    df_dicts = {}
    for task_name, prediction in zip(task_names, predictions.T):
        df_dicts[task_name] = prediction
    df_dicts["xyz_path"] = xyz_paths_clear

    df = pd.DataFrame(df_dicts)
    df = df.sort_values(by="xyz_path")
    df.to_csv(f"{cfg.output_directory}/predictions.csv", index=False)

    log.info("Prediction statistics:")
    for task_name in task_names:
        log.info(f"--- {task_name} ---")
        log.info(f"Mean: {df[task_name].mean():.4f}")
        log.info(f"Std: {df[task_name].std():.4f}")
        log.info(f"Min: {df[task_name].min():.4f}")
        log.info(f"Max: {df[task_name].max():.4f}")

    log.info("Plotting distributions...")
    props = []
    for prop in task_names:
        plot_kde_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_kde.png")
        plot_histogram_distribution(df[prop], prop, f"{cfg.output_directory}/{prop}_hist.png")
        props.append(df[prop].values)
    
    props = np.array(props).T
    plot_kde_distribution_multiple(props, task_names, f"{cfg.output_directory}/kde_all.png")


def log_hyperparameters(object_dict: dict):
    """Log hyperparameters for debugging."""
    _log_hyperparameters(object_dict, log)


def predict_main(cfg: DictConfig):
    """Entry point for CLI predict command."""
    runner(cfg)
