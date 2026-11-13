"""
Pharmacophore Model Dynamics - Wrapper for ShEPhERD model.

Builds input_dict from HeteroData batch exactly matching
shepherd/src/shepherd/lightning_module.py get_training_input_dict(),
then calls shepherd_arch.Model.forward().
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from MolecularDiffusion.device import get_device
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_noise_schedules(params: dict) -> dict:
    """
    Build the noise schedule arrays that ShEPhERD inference expects in params['noise_schedules'].

    These are computed deterministically from T (number of diffusion timesteps) using
    the same schedule as shepherd/training/parameters/params_x1x3x4_diffusion_mosesaq_20240824.py.
    They are never stored in the checkpoint — they must be recomputed at load time.
    """
    T = params.get('noise_schedules_T', None)
    if T is None:
        # Infer T from dataset params if present, else default to 400
        T = params.get('dataset', {}).get('T', 400)
        if T is None or T == 0:
            T = 400

    ts = np.arange(1, T + 1)

    beta_min = 0.001 / (T // 100)
    beta_max = 0.35 / (T // 100)
    beta_ts_linear = beta_min + ts / T * (beta_max - beta_min)

    ts_ = np.arange(0, T + 1)
    s = 0.008
    f_ts = np.cos(np.pi / 2.1 * ((ts_ / (T + 1)) + s) / (1.0 + s)) ** 2.0
    f_ts = f_ts / f_ts[0]
    f_ts = np.clip(f_ts, 0.0001, 0.9999)
    beta_ts_cosine = 1 - f_ts[1:] / f_ts[:-1]
    beta_ts_cosine = np.clip(beta_ts_cosine, 0.0001, 0.9999)

    beta_ts = 0.65 * beta_ts_cosine + 0.35 * beta_ts_linear
    sigma_ts = beta_ts ** 0.5
    alpha_ts = (1.0 - sigma_ts ** 2.0) ** 0.5
    alpha_dash_ts = np.cumprod(alpha_ts)
    var_dash_ts = 1.0 - alpha_dash_ts ** 2.0
    sigma_dash_ts = var_dash_ts ** 0.5

    sched = {
        'T': T,
        'ts': ts,
        'alpha_ts': alpha_ts,
        'sigma_ts': sigma_ts,
        'alpha_dash_ts': alpha_dash_ts,
        'var_dash_ts': var_dash_ts,
        'sigma_dash_ts': sigma_dash_ts,
    }
    return {'x1': sched, 'x2': sched, 'x3': sched, 'x4': sched}


class PharmacophoreModelWrapper(nn.Module):
    """
    Wrapper around ShEPhERD Model.

    Converts a batched HeteroData into the input_dict format that
    shepherd_arch.Model.forward() expects, runs the forward pass,
    and returns (input_dict, output_dict).
    """

    def __init__(self, shepherd_model, params=None):
        super().__init__()
        self.model = shepherd_model
        self.params = params or {}

        if 'noise_schedules' not in self.params:
            self.params['noise_schedules'] = _build_noise_schedules(self.params)

        if self.model is None:
            self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch, device=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build input_dict from HeteroData batch and run model.

        Args:
            batch: HeteroData batch from pharmacophore_collate_fn
            device: target device

        Returns:
            (input_dict, output_dict)
        """
        if device is None:
            if self.model is not None:
                try:
                    device = next(self.model.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
            else:
                device = self.dummy_param.device

        # During inference the sampler passes a pre-built input_dict (plain dict).
        # During training a HeteroData batch is passed and must be converted.
        if isinstance(batch, dict):
            input_dict = batch
        else:
            input_dict = self._build_input_dict(batch, device)

        if self.model is None:
            output_dict = {
                'x1': {'decoder': {'denoiser': {}}},
                'x2': {'decoder': {'denoiser': {}}},
                'x3': {'decoder': {'denoiser': {}}},
                'x4': {'decoder': {'denoiser': {}}},
            }
        else:
            _, output_dict = self.model.forward(input_dict)

        return input_dict, output_dict

    def _build_input_dict(self, batch, device):
        """
        Build input_dict matching shepherd/lightning_module.py get_training_input_dict().

        The batch is a HeteroData with keys like batch['x1'].pos_forward_noised, etc.
        """
        input_dict = {}

        # x1
        if 'x1' in batch.node_types:
            x1 = batch['x1']
            try:
                edge_store = batch['x1', 'bond', 'x1']
                bond_edge_mask = edge_store.mask
                bond_edge_index = edge_store.edge_index
                bond_edge_x = edge_store.x_forward_noised
                bond_edge_x_noise = edge_store.x_noise
            except (AttributeError, KeyError):
                bond_edge_mask = getattr(x1, 'bond_edge_mask', torch.zeros(0, dtype=torch.bool))
                bond_edge_index = getattr(x1, 'bond_edge_index', torch.zeros(2, 0, dtype=torch.long))
                bond_edge_x = getattr(x1, 'bond_edge_x_forward_noised', torch.zeros(0))
                bond_edge_x_noise = getattr(x1, 'bond_edge_x_noise', torch.zeros(0))

            input_dict['x1'] = {
                'decoder': {
                    'pos': x1.pos_forward_noised.to(device),
                    'x': x1.x_forward_noised.to(device),
                    'batch': x1.batch.to(device),
                    'bond_edge_mask': bond_edge_mask.to(device),
                    'bond_edge_index': bond_edge_index.to(device),
                    'bond_edge_x': bond_edge_x.to(device),
                    'timestep': x1.timestep.to(device),
                    'alpha_t': x1.alpha_t.to(device),
                    'sigma_t': x1.sigma_t.to(device),
                    'alpha_dash_t': x1.alpha_dash_t.to(device),
                    'sigma_dash_t': x1.sigma_dash_t.to(device),
                    'virtual_node_mask': x1.virtual_node_mask.to(device),
                    'pos_noise': x1.pos_noise.to(device),
                    'x_noise': x1.x_noise.to(device),
                    'bond_edge_x_noise': bond_edge_x_noise.to(device),
                },
            }

        # x2
        if 'x2' in batch.node_types:
            x2 = batch['x2']
            input_dict['x2'] = {
                'decoder': {
                    'pos': x2.pos_forward_noised.to(device),
                    'x': x2.x_forward_noised.to(device),
                    'batch': x2.batch.to(device),
                    'timestep': x2.timestep.to(device),
                    'alpha_t': x2.alpha_t.to(device),
                    'sigma_t': x2.sigma_t.to(device),
                    'alpha_dash_t': x2.alpha_dash_t.to(device),
                    'sigma_dash_t': x2.sigma_dash_t.to(device),
                    'virtual_node_mask': x2.virtual_node_mask.to(device),
                    'pos_noise': x2.pos_noise.to(device),
                },
            }

        # x3
        if 'x3' in batch.node_types:
            x3 = batch['x3']
            input_dict['x3'] = {
                'decoder': {
                    'pos': x3.pos_forward_noised.to(device),
                    'x': x3.x_forward_noised.to(device),
                    'batch': x3.batch.to(device),
                    'timestep': x3.timestep.to(device),
                    'alpha_t': x3.alpha_t.to(device),
                    'sigma_t': x3.sigma_t.to(device),
                    'alpha_dash_t': x3.alpha_dash_t.to(device),
                    'sigma_dash_t': x3.sigma_dash_t.to(device),
                    'virtual_node_mask': x3.virtual_node_mask.to(device),
                    'pos_noise': x3.pos_noise.to(device),
                    'x_noise': x3.x_noise.to(device),
                },
            }

        # x4
        if 'x4' in batch.node_types:
            x4 = batch['x4']
            input_dict['x4'] = {
                'decoder': {
                    'x': x4.x_forward_noised.to(device),
                    'pos': x4.pos_forward_noised.to(device),
                    'direction': x4.direction_forward_noised.to(device),
                    'batch': x4.batch.to(device),
                    'timestep': x4.timestep.to(device),
                    'alpha_t': x4.alpha_t.to(device),
                    'sigma_t': x4.sigma_t.to(device),
                    'alpha_dash_t': x4.alpha_dash_t.to(device),
                    'sigma_dash_t': x4.sigma_dash_t.to(device),
                    'virtual_node_mask': x4.virtual_node_mask.to(device),
                    'direction_noise': x4.direction_noise.to(device),
                    'pos_noise': x4.pos_noise.to(device),
                    'x_noise': x4.x_noise.to(device),
                },
            }

        input_dict['device'] = device
        input_dict['dtype'] = torch.float32

        return input_dict



def _convert_omegaconf(obj):
    """Convert OmegaConf DictConfig/ListConfig to plain Python dict/list."""
    try:
        from omegaconf import DictConfig, ListConfig
        if isinstance(obj, DictConfig):
            return {k: _convert_omegaconf(v) for k, v in obj.items()}
        elif isinstance(obj, ListConfig):
            return [_convert_omegaconf(v) for v in obj]
    except ImportError:
        pass
    return obj


class PharmacophoreEnVariationalDiff(nn.Module):
    """
    Wrapper that makes ShEPhERD compatible with MolecularDiffusion's
    EnVariationalDiffusion interface.
    """

    def __init__(
        self,
        model_params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = get_device()

        if model_params is None:
            raise ValueError(
                "model_params must be provided. "
                "Set model_params in tasks config (pharmacophore.yaml)."
            )
        from MolecularDiffusion.modules.models.shepherd_arch.model import Model
        params = _convert_omegaconf(model_params)
        model = Model(params)
        model.to(device)
        model.device = device
        self.dynamics = PharmacophoreModelWrapper(model, params=params)
        logger.info(f"Created fresh ShEPhERD model ({sum(p.numel() for p in model.parameters())} params)")

        self.device = device

    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters())
        return f"PharmacophoreEnVariationalDiff(params={n_params})"

    def forward(self, batch, device=None):
        if device is None:
            device = self.device
        return self.dynamics(batch, device=device)
