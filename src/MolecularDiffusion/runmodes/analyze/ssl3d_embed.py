"""SSL3D embedding helpers: checkpoint loading and graph construction."""

from __future__ import annotations

from pathlib import Path

import torch
from MolecularDiffusion.core.checkpoint import engine_weights
from MolecularDiffusion.device import get_device
import torch.nn as nn


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _read_atom_vocab(ckpt: dict) -> list[str]:
    """Extract atom_vocab from either checkpoint format."""
    if "hyper_parameters" in ckpt:
        mc = ckpt["hyper_parameters"].get("model_config", {})
        vocab = getattr(mc, "atom_vocab", None)
        if vocab is None:
            vocab = mc.get("atom_vocab", None) if isinstance(mc, dict) else None
        if vocab is not None:
            from omegaconf import OmegaConf
            return list(OmegaConf.to_container(vocab)) if hasattr(vocab, "_metadata") else list(vocab)
    if "hyperparameters" in ckpt:
        vocab = ckpt["hyperparameters"].get("atom_vocab")
        if vocab is not None:
            return list(vocab)
    return ["H", "C", "N", "O", "F"]


def _load_lightning_task(ckpt: dict, path: Path, device: str) -> nn.Module:
    # Try EngineLightning.load_from_checkpoint first (cleanest path)
    try:
        from MolecularDiffusion.core.engine_lightning import EngineLightning
        wrapper = EngineLightning.load_from_checkpoint(str(path), map_location=device)
        wrapper.task.eval()
        return wrapper.task
    except Exception:
        pass

    # Fallback: hydra instantiate factory from stored model_config
    import hydra
    from omegaconf import OmegaConf

    hp = ckpt["hyper_parameters"]
    model_config = hp["model_config"]
    OmegaConf.set_struct(model_config, False)

    atom_vocab = _read_atom_vocab(ckpt)
    factory = hydra.utils.instantiate(model_config, atom_vocab=atom_vocab)
    task = factory.build()

    sd = ckpt.get("state_dict", {})
    cleaned = {(k[5:] if k.startswith("task.") else k): v for k, v in sd.items()}
    task.load_state_dict(cleaned, strict=False)
    return task.to(device).eval()


def _load_engine_task(ckpt: dict, device: str) -> nn.Module:
    from MolecularDiffusion.modules.tasks import (
        SSL3D, CoordDenoiseObjective, MaskedAtomTypeObjective, PairwiseDistObjective,
    )
    from MolecularDiffusion.modules.models import EGNN
    from MolecularDiffusion.modules.layers.common import SinusoidsEmbeddingNew

    hp = ckpt["hyperparameters"]
    task_cfg = hp["task"]
    model_cfg = task_cfg["model"]

    act_fn = model_cfg.get("act_fn", None)
    if act_fn is None or isinstance(act_fn, str):
        act_fn = nn.SiLU()

    model = EGNN(
        in_node_nf=model_cfg["in_node_nf"],
        hidden_nf=model_cfg["hidden_nf"],
        act_fn=act_fn,
        n_layers=model_cfg.get("n_layers", 6),
        attention=model_cfg.get("attention", True),
        tanh=model_cfg.get("tanh", True),
        norm_constant=model_cfg.get("norm_constant", 1.0),
        inv_sublayers=model_cfg.get("inv_sublayers", 5),
        sin_embedding=model_cfg.get("sin_embedding", False),
        normalization_factor=model_cfg.get("normalization_factor", 1.0),
        aggregation_method=model_cfg.get("aggregation_method", "sum"),
        dropout=model_cfg.get("dropout", 0.0),
        normalization=model_cfg.get("normalization", False),
        include_cosine=model_cfg.get("include_cosine", True),
    )

    objectives = []
    for obj_cfg in task_cfg.get("objectives", []):
        cls = obj_cfg["class"]
        if cls == "CoordDenoiseObjective":
            objectives.append(CoordDenoiseObjective(
                weight=obj_cfg.get("weight", 1.0),
                sigma_min=obj_cfg.get("sigma_min", 0.01),
                sigma_max=obj_cfg.get("sigma_max", 1.0),
                sigma_schedule=obj_cfg.get("sigma_schedule", "uniform"),
            ))
        elif cls == "MaskedAtomTypeObjective":
            objectives.append(MaskedAtomTypeObjective(
                weight=obj_cfg.get("weight", 0.5),
                mask_rate=obj_cfg.get("mask_rate", 0.15),
                atom_vocab_size=obj_cfg.get("atom_vocab_size", 5),
            ))
        elif cls == "PairwiseDistObjective":
            objectives.append(PairwiseDistObjective(
                weight=obj_cfg.get("weight", 0.0),
                k_pairs=obj_cfg.get("k_pairs", 16),
            ))

    task = SSL3D(
        model,
        objectives,
        include_charge=task_cfg.get("include_charge", True),
        t_embedding=task_cfg.get("t_embedding", "sinusoidal"),
    )

    sd = engine_weights(ckpt)
    task.load_state_dict(sd, strict=False)
    return task.to(device).eval()


def load_ssl3d_task(checkpoint_path: str | Path, device: str | None = None):
    """Load a trained SSL3D task from a Lightning .ckpt or Engine .pkl checkpoint.

    Returns:
        (task, atom_vocab) — SSL3D module in eval mode, list of atom symbols used
                             during training.
    """
    path = Path(checkpoint_path)
    if device is None:
        device = str(get_device())

    ckpt = torch.load(path, map_location=device, weights_only=False)
    atom_vocab = _read_atom_vocab(ckpt)

    if "hyper_parameters" in ckpt:
        task = _load_lightning_task(ckpt, path, device)
    elif "hyperparameters" in ckpt:
        task = _load_engine_task(ckpt, device)
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {path}. "
            "Expected keys 'hyper_parameters' (Lightning) or 'hyperparameters' (Engine)."
        )

    return task, atom_vocab


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def atoms_to_batch(
    atoms_list,
    atom_vocab: list[str],
    edge_radius: float = 5.0,
    device: str = "cpu",
):
    """Convert a list of ASE Atoms to a PyG Batch ready for SSL3D._forward_backbone.

    Returns:
        dict {"graph": Batch} with fields x, pos, edge_index, atomic_numbers,
        natoms, batch.
    """
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import radius_graph
    from ase.data import atomic_numbers as ASE_Z

    vocab_index = {sym: i for i, sym in enumerate(atom_vocab)}
    graphs = []

    for atoms in atoms_list:
        symbols = atoms.get_chemical_symbols()
        n = len(symbols)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        x = torch.zeros(n, len(atom_vocab))
        for i, sym in enumerate(symbols):
            idx = vocab_index.get(sym)
            if idx is not None:
                x[i, idx] = 1.0

        atomic_nums = torch.tensor(
            [ASE_Z.get(s, 0) for s in symbols], dtype=torch.float32
        )

        edge_index = radius_graph(pos, r=edge_radius)

        graphs.append(Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            atomic_numbers=atomic_nums,
            natoms=n,
        ))

    batch = Batch.from_data_list(graphs)
    batch.natoms = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long)
    return {"graph": batch.to(device)}


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_nodes(
    h: torch.Tensor,
    batch_idx: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Reduce per-atom embeddings to per-molecule embeddings."""
    from torch_geometric.nn import global_mean_pool, global_add_pool

    if pooling == "mean":
        return global_mean_pool(h, batch_idx)
    elif pooling == "sum":
        return global_add_pool(h, batch_idx)
    else:
        raise ValueError(f"pooling must be 'mean' or 'sum', got {pooling!r}")
