"""
Pharmacophore Dataset for ShEPhERD integration.

Faithful port of shepherd/src/shepherd/datasets.py HeteroDataset.
Handles loading, processing, and forward-noising of 4 molecular modalities:
- x1: Molecular structure (atoms + bonds)
- x2: Shape (surface point cloud)
- x3: Electrostatics (surface point cloud + ESP)
- x4: Pharmacophores (points + directions)

Forward-noising is applied per-sample in __getitem__ using the noise schedule
from the ShEPhERD checkpoint hyperparameters.
"""

import logging
import os
import pickle
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    Chem = None
    AllChem = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

from MolecularDiffusion.utils.geometry.shepherd_utils import (
    get_atomic_vdw_radii,
    get_molecular_surface,
    get_electrostatics_given_point_charges,
    get_pharmacophores,
    update_mol_coordinates
)

logger = logging.getLogger(__name__)


# ============================================================
# Noise schedule utilities
# ============================================================

def build_default_noise_schedule(T=400):
    """
    Build the default ShEPhERD cosine+linear noise schedule.
    Matches shepherd/training/parameters/params_x1x3x4_diffusion_mosesaq_20240824.py
    """
    ts = np.arange(1, T + 1)

    # Linear schedule
    beta_min = 0.001 / (T // 100)
    beta_max = 0.35 / (T // 100)
    beta_ts_linear = beta_min + ts / T * (beta_max - beta_min)

    # Cosine schedule
    ts_ = np.arange(0, T + 1)
    s = 0.008
    f_ts = np.cos(np.pi / 2.1 * ((ts_ / (T + 1)) + s) / (1.0 + s)) ** 2.0
    f_ts = f_ts / f_ts[0]
    f_ts = np.clip(f_ts, 0.0001, 0.9999)
    beta_ts_cosine = 1 - f_ts[1:] / f_ts[0:-1]
    beta_ts_cosine = np.clip(beta_ts_cosine, 0.0001, 0.9999)

    # Blend
    beta_ts = 0.65 * beta_ts_cosine + 0.35 * beta_ts_linear

    sigma_ts = beta_ts ** 0.5
    alpha_ts = (1.0 - sigma_ts ** 2.0) ** 0.5
    alpha_dash_ts = np.cumprod(alpha_ts)
    var_dash_ts = 1.0 - alpha_dash_ts ** 2.0
    sigma_dash_ts = var_dash_ts ** 0.5

    return {
        'T': T,
        'ts': ts,
        'alpha_ts': alpha_ts,
        'sigma_ts': sigma_ts,
        'alpha_dash_ts': alpha_dash_ts,
        'var_dash_ts': var_dash_ts,
        'sigma_dash_ts': sigma_dash_ts,
    }


def sample_timestep_biased(ts):
    """
    Biased timestep sampling matching ShEPhERD:
    - 7.5% from high-noise end (t=1..50)
    - 75% from middle (t=50..250)
    - 17.5% from low-noise start (t=250..400)
    """
    T = ts.shape[0]
    ts_end = ts[0:int(T * 0.125)]
    ts_middle = ts[int(T * 0.125):int(T * 0.625)]
    ts_start = ts[int(T * 0.625):]
    p = np.random.uniform(0, 1)
    if p < 0.075:
        return np.random.choice(ts_end)
    elif p < 0.825:
        return np.random.choice(ts_middle)
    else:
        return np.random.choice(ts_start)


# ============================================================
# Dataset
# ============================================================

class PharmacophoreDataset(torch.utils.data.Dataset):
    """
    Dataset for ShEPhERD pharmacophore diffusion.

    Each __getitem__ returns a torch_geometric.data.HeteroData with
    forward-noised fields for x1, x2, x3, x4 — exactly matching
    shepherd/src/shepherd/datasets.py HeteroDataset.
    """

    def __init__(
        self,
        molblocks_and_charges: List,
        noise_schedule_dict: Dict[str, Dict],
        # which modalities to compute
        compute_x1: bool = True,
        compute_x2: bool = False,
        compute_x3: bool = True,
        compute_x4: bool = True,
        # x1 params
        recenter_x1: bool = True,
        add_virtual_node_x1: bool = True,
        remove_noise_COM_x1: bool = True,
        atom_types_x1: List = None,
        charge_types_x1: List = None,
        bond_types_x1: List = None,
        scale_atom_features_x1: float = 0.25,
        scale_bond_features_x1: float = 1.0,
        formal_charge_diffusion: bool = True,
        # x2 params
        independent_timesteps_x2: bool = False,
        recenter_x2: bool = False,
        add_virtual_node_x2: bool = True,
        remove_noise_COM_x2: bool = False,
        num_points_x2: int = 75,
        # x3 params
        independent_timesteps_x3: bool = False,
        recenter_x3: bool = False,
        add_virtual_node_x3: bool = True,
        remove_noise_COM_x3: bool = False,
        num_points_x3: int = 75,
        scale_node_features_x3: float = 2.0,
        # x4 params
        independent_timesteps_x4: bool = False,
        recenter_x4: bool = False,
        add_virtual_node_x4: bool = True,
        remove_noise_COM_x4: bool = False,
        max_node_types_x4: int = 10,
        scale_node_features_x4: float = 2.0,
        scale_vector_features_x4: float = 2.0,
        multivectors: bool = False,
        check_accessibility: bool = False,
        # surface
        probe_radius: float = 0.6,
    ):
        super().__init__()
        self.molblocks_and_charges = molblocks_and_charges
        self.noise_schedule_dict = noise_schedule_dict

        self.compute_x1 = compute_x1
        self.compute_x2 = compute_x2
        self.compute_x3 = compute_x3
        self.compute_x4 = compute_x4

        self.recenter_x1 = recenter_x1
        self.add_virtual_node_x1 = add_virtual_node_x1
        self.remove_noise_COM_x1 = remove_noise_COM_x1
        self.atom_types_x1 = atom_types_x1 or [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
        self.charge_types_x1 = charge_types_x1 or [0, 1, 2, -1, -2]
        self.bond_types_x1 = bond_types_x1 or [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        self.scale_atom_features_x1 = scale_atom_features_x1
        self.scale_bond_features_x1 = scale_bond_features_x1
        self.formal_charge_diffusion = formal_charge_diffusion

        self.independent_timesteps_x2 = independent_timesteps_x2
        self.recenter_x2 = recenter_x2
        self.add_virtual_node_x2 = add_virtual_node_x2
        self.remove_noise_COM_x2 = remove_noise_COM_x2
        self.num_points_x2 = num_points_x2

        self.independent_timesteps_x3 = independent_timesteps_x3
        self.recenter_x3 = recenter_x3
        self.add_virtual_node_x3 = add_virtual_node_x3
        self.remove_noise_COM_x3 = remove_noise_COM_x3
        self.num_points_x3 = num_points_x3
        self.scale_node_features_x3 = scale_node_features_x3

        self.independent_timesteps_x4 = independent_timesteps_x4
        self.recenter_x4 = recenter_x4
        self.add_virtual_node_x4 = add_virtual_node_x4
        self.remove_noise_COM_x4 = remove_noise_COM_x4
        self.max_node_types_x4 = max_node_types_x4
        self.scale_node_features_x4 = scale_node_features_x4
        self.scale_vector_features_x4 = scale_vector_features_x4
        self.multivectors = multivectors
        self.check_accessibility = check_accessibility
        self.probe_radius = probe_radius

    def __len__(self):
        return len(self.molblocks_and_charges)

    # ----------------------------------------------------------
    # x1: Molecular structure
    # ----------------------------------------------------------
    def get_x1_data(self, mol, t, alpha_dash_t, sigma_dash_t):
        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)

        atom_types = [self.atom_types_x1.index(a.GetSymbol()) if a.GetSymbol() in self.atom_types_x1 else 0
                      for a in mol.GetAtoms()]

        pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        num_atoms = len(pos)

        # Bond edge index: fully connected upper triangle (directed)
        bond_adj = 1 - np.diag(np.ones(num_atoms, dtype=int))
        bond_adj = np.triu(bond_adj)
        bond_edge_index = np.stack(bond_adj.nonzero(), axis=0)

        bond_types_dict = {b: self.bond_types_x1.index(b) for b in self.bond_types_x1}
        max_bond_types = len(bond_types_dict)

        bond_types = []
        for b in range(bond_edge_index.shape[1]):
            i, j = int(bond_edge_index[0, b]), int(bond_edge_index[1, b])
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is None:
                bond_types.append(bond_types_dict[None])
            else:
                bond_types.append(bond_types_dict.get(str(bond.GetBondType()), 0))

        data['bond_edge_mask'] = torch.from_numpy((np.array(bond_types) != 0).copy()).bool()

        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if self.recenter_x1:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(self.add_virtual_node_x1))
        if self.add_virtual_node_x1:
            assert self.atom_types_x1[0] is None
            atom_types.insert(0, 0)
            bond_edge_index = bond_edge_index + 1
            virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis=0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis=0)
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1
        num_nodes = num_atoms + int(self.add_virtual_node_x1)

        data['bond_edge_index'] = torch.from_numpy(bond_edge_index.copy()).long()
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()

        # One-hot atom type features
        x = np.zeros((num_nodes, len(self.atom_types_x1)))
        x[np.arange(num_nodes), atom_types] = 1
        x = x * self.scale_atom_features_x1

        if self.formal_charge_diffusion:
            formal_charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]
            charge_map = {c: self.charge_types_x1.index(c) for c in self.charge_types_x1}
            fc_mapped = [charge_map.get(f, 0) for f in formal_charges]
            x_fc = np.zeros((len(fc_mapped), len(self.charge_types_x1)))
            x_fc[np.arange(len(fc_mapped)), fc_mapped] = 1
            x_fc = x_fc * self.scale_atom_features_x1
            if self.add_virtual_node_x1:
                x_fc = np.concatenate([np.zeros((1, len(self.charge_types_x1)), dtype=x_fc.dtype), x_fc], axis=0)
            x = np.concatenate([x, x_fc], axis=1)

        data['x'] = torch.from_numpy(x.copy()).float()

        # Bond features
        bond_edge_x = np.zeros((bond_edge_index.shape[1], max_bond_types))
        bond_edge_x[np.arange(len(bond_types)), bond_types] = 1
        bond_edge_x = bond_edge_x * self.scale_bond_features_x1
        data['bond_edge_x'] = torch.from_numpy(bond_edge_x.copy()).float()

        # Forward noising
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if self.remove_noise_COM_x1:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis=0)
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()

        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0
        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()

        bond_edge_x_noise = np.random.randn(*bond_edge_x.shape)
        data['bond_edge_x_noise'] = torch.from_numpy(bond_edge_x_noise.copy()).float()

        pos_forward_noised = alpha_dash_t * pos + sigma_dash_t * pos_noise
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

        x_forward_noised = alpha_dash_t * x + sigma_dash_t * x_noise
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()

        bond_edge_x_forward_noised = alpha_dash_t * bond_edge_x + sigma_dash_t * bond_edge_x_noise
        data['bond_edge_x_forward_noised'] = torch.from_numpy(bond_edge_x_forward_noised.copy()).float()

        return data, pos, virtual_node_mask

    # ----------------------------------------------------------
    # x2: Shape (surface point cloud)
    # ----------------------------------------------------------
    def get_x2_data(self, radii, atom_centers, num_points, recenter, add_virtual_node,
                    remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos=None):
        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)

        pos = get_molecular_surface(atom_centers, radii, num_points=num_points,
                                    probe_radius=self.probe_radius, num_samples_per_atom=20)

        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node:
            if virtual_node_pos is None or recenter:
                virtual_node_pos = COM
            pos = np.concatenate([virtual_node_pos, pos], axis=0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis=0)
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1

        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()

        # x2 features: one-hot real vs virtual (NOT noised)
        x = np.zeros((pos.shape[0], 2))
        x[~virtual_node_mask, 0] = 1
        x[virtual_node_mask, 1] = 1
        data['x'] = torch.from_numpy(x.copy()).float()
        data['x_forward_noised'] = data['x']

        # Forward noising (positions only)
        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if remove_noise_COM:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis=0)
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()

        pos_forward_noised = alpha_dash_t * pos + sigma_dash_t * pos_noise
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

        return data, pos, virtual_node_mask

    # ----------------------------------------------------------
    # x3: Electrostatics
    # ----------------------------------------------------------
    def get_x3_data_electrostatics_only(self, charges, charge_centers, data, pos, virtual_node_mask,
                                         t, alpha_dash_t, sigma_dash_t):
        x = get_electrostatics_given_point_charges(charges, charge_centers, pos)

        x[virtual_node_mask] = 0.0
        x = x * self.scale_node_features_x3
        data['x'] = torch.from_numpy(x.copy()).float()

        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0
        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()

        x_forward_noised = alpha_dash_t * x + sigma_dash_t * x_noise
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()

        return data

    # ----------------------------------------------------------
    # x4: Pharmacophores
    # ----------------------------------------------------------
    def get_x4_data(self, mol, recenter, add_virtual_node, remove_noise_COM,
                    t, alpha_dash_t, sigma_dash_t, virtual_node_pos=None):
        assert add_virtual_node

        data = {}
        data['timestep'] = torch.tensor([t], dtype=torch.long)

        pharm_types, pos, direction = get_pharmacophores(
            mol, multi_vector=self.multivectors, check_access=self.check_accessibility)

        pharm_types = pharm_types + 1  # offset for virtual node at index 0
        pos = pos + np.random.randn(*pos.shape) * 0.05  # small jitter

        # Edge case: no pharmacophores
        if pharm_types.shape[0] == 0:
            pharm_types = np.array([0])
            x = np.zeros((1, self.max_node_types_x4))
            x[0, 0] = 1
            x = x * self.scale_node_features_x4
            data['x'] = torch.from_numpy(x.copy()).float()

            if virtual_node_pos is None or recenter:
                virtual_node_pos = np.zeros(3)[None, :]
            data['com_before_centering'] = torch.from_numpy(virtual_node_pos.copy()).float()
            data['com'] = torch.from_numpy(virtual_node_pos.copy()).float()

            virtual_node_mask = np.array([True])
            pos = virtual_node_pos
            direction = np.zeros((1, 3))
            direction = direction * self.scale_vector_features_x4

            data['pos'] = torch.from_numpy(pos.copy()).float()
            data['pos_recentered'] = torch.from_numpy((pos * 0.0).copy()).float()
            data['direction'] = torch.from_numpy(direction.copy()).float()
            data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()

            # No noising for virtual-only
            data['x_noise'] = torch.zeros_like(data['x'])
            data['x_forward_noised'] = data['x'].clone()
            data['pos_noise'] = torch.zeros_like(data['pos'])
            data['pos_forward_noised'] = data['pos'].clone()
            data['direction_noise'] = torch.zeros_like(data['direction'])
            data['direction_forward_noised'] = data['direction'].clone()
            return data

        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)
        if recenter:
            pos = pos_recentered
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()

        virtual_node_mask = np.zeros(pos.shape[0] + int(add_virtual_node))
        if add_virtual_node:
            if virtual_node_pos is None or recenter:
                virtual_node_pos = COM
            pharm_types = np.concatenate([np.array([0]), pharm_types], axis=0)
            pos = np.concatenate([virtual_node_pos, pos], axis=0)
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis=0)
            direction = np.concatenate([np.zeros((1, 3)), direction], axis=0)
            virtual_node_mask[0] = 1
        virtual_node_mask = virtual_node_mask == 1

        x = np.zeros((pharm_types.size, self.max_node_types_x4))
        x[np.arange(pharm_types.size), pharm_types] = 1
        x = x * self.scale_node_features_x4
        data['x'] = torch.from_numpy(x.copy()).float()

        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        direction = direction * self.scale_vector_features_x4
        data['direction'] = torch.from_numpy(direction.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()

        # Forward noising
        x_noise = np.random.randn(*x.shape)
        x_noise[virtual_node_mask] = 0.0
        data['x_noise'] = torch.from_numpy(x_noise.copy()).float()
        x_forward_noised = alpha_dash_t * x + sigma_dash_t * x_noise
        x_forward_noised[virtual_node_mask] = x[virtual_node_mask]
        data['x_forward_noised'] = torch.from_numpy(x_forward_noised.copy()).float()

        pos_noise = np.random.randn(*pos.shape)
        pos_noise[virtual_node_mask] = 0.0
        if remove_noise_COM:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis=0)
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()
        pos_forward_noised = alpha_dash_t * pos + sigma_dash_t * pos_noise
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()

        direction_noise = np.random.randn(*direction.shape)
        direction_noise[virtual_node_mask] = 0.0
        data['direction_noise'] = torch.from_numpy(direction_noise.copy()).float()
        direction_forward_noised = alpha_dash_t * direction + sigma_dash_t * direction_noise
        direction_forward_noised[virtual_node_mask] = direction[virtual_node_mask]
        data['direction_forward_noised'] = torch.from_numpy(direction_forward_noised.copy()).float()

        return data

    # ----------------------------------------------------------
    # __getitem__: matches shepherd/src/shepherd/datasets.py
    # ----------------------------------------------------------
    def __getitem__(self, k):
        mol_block = self.molblocks_and_charges[k][0]
        charges = np.array(self.molblocks_and_charges[k][1], dtype=np.float64)

        mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
        if mol is None:
            raise ValueError(f"Failed to parse molblock at index {k}")

        # Center molecule coordinates
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)

        radii = get_atomic_vdw_radii(mol)

        data_dict = {}

        # Sample timestep for x1
        if self.compute_x1:
            ns = self.noise_schedule_dict['x1']
            ts = ns['ts']
            t = sample_timestep_biased(ts)
            t_x1, ts_x1 = t, ts
            t_idx = np.where(ts == t)[0][0]
            alpha_t = ns['alpha_ts'][t_idx]
            sigma_t = ns['sigma_ts'][t_idx]
            alpha_dash_t = ns['alpha_dash_ts'][t_idx]
            sigma_dash_t = ns['sigma_dash_ts'][t_idx]

            x1_data, x1_pos, x1_virtual_node_mask = self.get_x1_data(mol, t, alpha_dash_t, sigma_dash_t)
            x1_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x1_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x1_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x1_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x1'] = x1_data

        # x2
        if self.compute_x2:
            if self.independent_timesteps_x2:
                ns = self.noise_schedule_dict['x2']
                ts = ns['ts']
                t = sample_timestep_biased(ts)
            else:
                assert self.compute_x1
                ts, t = ts_x1, t_x1
            ns2 = self.noise_schedule_dict['x2']
            t_idx = np.where(ns2['ts'] == t)[0][0]
            alpha_t = ns2['alpha_ts'][t_idx]
            sigma_t = ns2['sigma_ts'][t_idx]
            alpha_dash_t = ns2['alpha_dash_ts'][t_idx]
            sigma_dash_t = ns2['sigma_dash_ts'][t_idx]

            if self.compute_x1:
                atom_centers = x1_pos[~x1_virtual_node_mask, :]
                virtual_node_pos = atom_centers.mean(0)[None, :] if (self.add_virtual_node_x2 and not self.recenter_x2) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None

            x2_data, x2_pos, x2_virtual_node_mask = self.get_x2_data(
                radii, atom_centers, self.num_points_x2, self.recenter_x2,
                self.add_virtual_node_x2, self.remove_noise_COM_x2,
                t, alpha_dash_t, sigma_dash_t, virtual_node_pos=virtual_node_pos)
            x2_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x2_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x2_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x2_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x2'] = x2_data

        # x3
        if self.compute_x3:
            if self.independent_timesteps_x3:
                ns = self.noise_schedule_dict['x3']
                ts = ns['ts']
                t = sample_timestep_biased(ts)
            else:
                assert self.compute_x1
                ts, t = ts_x1, t_x1
            ns3 = self.noise_schedule_dict['x3']
            t_idx = np.where(ns3['ts'] == t)[0][0]
            alpha_t = ns3['alpha_ts'][t_idx]
            sigma_t = ns3['sigma_ts'][t_idx]
            alpha_dash_t = ns3['alpha_dash_ts'][t_idx]
            sigma_dash_t = ns3['sigma_dash_ts'][t_idx]

            if self.compute_x1:
                atom_centers = x1_pos[~x1_virtual_node_mask, :]
                virtual_node_pos = atom_centers.mean(0)[None, :] if (self.add_virtual_node_x3 and not self.recenter_x3) else None
            else:
                atom_centers = mol_coordinates
                virtual_node_pos = None

            # x3 uses same surface generation as x2
            x3_data, x3_pos, x3_virtual_node_mask = self.get_x2_data(
                radii, atom_centers, self.num_points_x3, self.recenter_x3,
                self.add_virtual_node_x3, self.remove_noise_COM_x3,
                t, alpha_dash_t, sigma_dash_t, virtual_node_pos=virtual_node_pos)

            x3_COM_displacement = (x3_data['com'] - x3_data['com_before_centering']).detach().cpu().numpy()
            charge_centers = atom_centers + x3_COM_displacement

            x3_data = self.get_x3_data_electrostatics_only(
                charges, charge_centers, x3_data, x3_pos, x3_virtual_node_mask,
                t, alpha_dash_t, sigma_dash_t)
            x3_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x3_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x3_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x3_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x3'] = x3_data

        # x4
        if self.compute_x4:
            if self.independent_timesteps_x4:
                ns = self.noise_schedule_dict['x4']
                ts = ns['ts']
                t = sample_timestep_biased(ts)
            else:
                assert self.compute_x1
                ts, t = ts_x1, t_x1
            ns4 = self.noise_schedule_dict['x4']
            t_idx = np.where(ns4['ts'] == t)[0][0]
            alpha_t = ns4['alpha_ts'][t_idx]
            sigma_t = ns4['sigma_ts'][t_idx]
            alpha_dash_t = ns4['alpha_dash_ts'][t_idx]
            sigma_dash_t = ns4['sigma_dash_ts'][t_idx]

            if self.compute_x1:
                atom_centers = x1_pos[~x1_virtual_node_mask, :]
                virtual_node_pos = atom_centers.mean(0)[None, :] if (self.add_virtual_node_x4 and not self.recenter_x4) else None
            else:
                virtual_node_pos = None

            x4_data = self.get_x4_data(
                mol, self.recenter_x4, self.add_virtual_node_x4, self.remove_noise_COM_x4,
                t, alpha_dash_t, sigma_dash_t, virtual_node_pos)
            x4_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x4_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x4_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x4_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x4'] = x4_data

        # Pack into HeteroData (matches shepherd/datasets.py lines 744-776)
        hetero_data = HeteroData()
        hetero_data.molecule_id = torch.tensor([k], dtype=torch.long)

        if 'x1' in data_dict:
            x1_d = data_dict['x1']
            x1_node_dict = {k: v for k, v in x1_d.items() if 'bond' not in k}
            x1_edge_dict = {
                'edge_index': x1_d['bond_edge_index'],
                'mask': x1_d['bond_edge_mask'],
                'x': x1_d['bond_edge_x'],
                'x_noise': x1_d['bond_edge_x_noise'],
                'x_forward_noised': x1_d['bond_edge_x_forward_noised'],
            }
            for key, value in x1_node_dict.items():
                hetero_data['x1'][key] = value
            for key, value in x1_edge_dict.items():
                hetero_data['x1', 'bond', 'x1'][key] = value

        for modal in ['x2', 'x3', 'x4']:
            if modal in data_dict:
                for key, value in data_dict[modal].items():
                    hetero_data[modal][key] = value

        return hetero_data


# ============================================================
# DataModule
# ============================================================

class PharmacophoreDataModule:
    """
    DataModule wrapper for PharmacophoreDataset.

    Loads molblocks+charges from pkl files, builds noise schedule
    (from checkpoint or default), and creates train/val/test splits.
    """

    def __init__(
        self,
        pkl_files: List[str],
        root: str = "data/pharmacophore_cache",
        dataset_name: str = "pharmacophore",
        num_samples: Optional[int] = None,
        data_fraction: float = 1.0,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 0.8,
        task_type: str = "diffusion_pharmacophore",
        # Checkpoint path to extract noise schedule + dataset params
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        self.pkl_files = pkl_files
        self.root = root
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.data_fraction = data_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.task_type = task_type
        self.checkpoint_path = checkpoint_path
        self.kwargs = kwargs

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.collate_fn = pharmacophore_collate_fn

    def load(self):
        """Load pkl data and create dataset with proper noise schedule."""
        # Load molblocks + charges
        raw_entries = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            raw_entries.extend(data)

        # Apply data_fraction and num_samples before parsing to avoid unnecessary work
        if self.data_fraction < 1.0:
            raw_entries = raw_entries[:int(len(raw_entries) * self.data_fraction)]
        if self.num_samples is not None:
            raw_entries = raw_entries[:self.num_samples]

        molblocks_and_charges = []
        n_invalid = 0
        for entry in tqdm(raw_entries, desc="Parsing molecules"):
            mol = Chem.MolFromMolBlock(entry[0], removeHs=False)
            if mol is None:
                n_invalid += 1
                continue
            molblocks_and_charges.append(entry)
        if n_invalid:
            logger.warning(f"Skipped {n_invalid} unparseable molblocks")
        logger.info(f"Using {len(molblocks_and_charges)} valid samples")

        # Get noise schedule + dataset params from checkpoint or defaults
        if self.checkpoint_path:
            logger.info(f"Loading noise schedule from checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
            params = ckpt['hyper_parameters']['params']
            noise_schedule_dict = params['noise_schedules']
            ds_params = params['dataset']
        else:
            logger.info("Using default noise schedule (no checkpoint provided)")
            default_ns = build_default_noise_schedule(T=400)
            noise_schedule_dict = {m: default_ns.copy() for m in ['x1', 'x2', 'x3', 'x4', 'x5']}
            ds_params = None

        # Build dataset kwargs from checkpoint params or defaults
        ds_kwargs = self.kwargs.copy()
        if ds_params:
            ds_kwargs['compute_x1'] = ds_params.get('compute_x1', True)
            ds_kwargs['compute_x2'] = ds_params.get('compute_x2', False)
            ds_kwargs['compute_x3'] = ds_params.get('compute_x3', True)
            ds_kwargs['compute_x4'] = ds_params.get('compute_x4', True)
            ds_kwargs['probe_radius'] = ds_params.get('probe_radius', 0.6)
            ds_kwargs['formal_charge_diffusion'] = True  # x1x3x4 checkpoint uses this

            if 'x1' in ds_params:
                x1p = ds_params['x1']
                ds_kwargs['recenter_x1'] = x1p.get('recenter', True)
                ds_kwargs['add_virtual_node_x1'] = x1p.get('add_virtual_node', True)
                ds_kwargs['remove_noise_COM_x1'] = x1p.get('remove_noise_COM', True)
                ds_kwargs['atom_types_x1'] = x1p.get('atom_types', None)
                ds_kwargs['charge_types_x1'] = x1p.get('charge_types', None)
                ds_kwargs['bond_types_x1'] = x1p.get('bond_types', None)
                ds_kwargs['scale_atom_features_x1'] = x1p.get('scale_atom_features', 0.25)
                ds_kwargs['scale_bond_features_x1'] = x1p.get('scale_bond_features', 1.0)
            if 'x2' in ds_params:
                x2p = ds_params['x2']
                ds_kwargs['independent_timesteps_x2'] = x2p.get('independent_timesteps', False)
                ds_kwargs['recenter_x2'] = x2p.get('recenter', False)
                ds_kwargs['add_virtual_node_x2'] = x2p.get('add_virtual_node', True)
                ds_kwargs['remove_noise_COM_x2'] = x2p.get('remove_noise_COM', False)
                ds_kwargs['num_points_x2'] = x2p.get('num_points', 75)
            if 'x3' in ds_params:
                x3p = ds_params['x3']
                ds_kwargs['independent_timesteps_x3'] = x3p.get('independent_timesteps', False)
                ds_kwargs['recenter_x3'] = x3p.get('recenter', False)
                ds_kwargs['add_virtual_node_x3'] = x3p.get('add_virtual_node', True)
                ds_kwargs['remove_noise_COM_x3'] = x3p.get('remove_noise_COM', False)
                ds_kwargs['num_points_x3'] = x3p.get('num_points', 75)
                ds_kwargs['scale_node_features_x3'] = x3p.get('scale_node_features', 2.0)
            if 'x4' in ds_params:
                x4p = ds_params['x4']
                ds_kwargs['independent_timesteps_x4'] = x4p.get('independent_timesteps', False)
                ds_kwargs['recenter_x4'] = x4p.get('recenter', False)
                ds_kwargs['add_virtual_node_x4'] = x4p.get('add_virtual_node', True)
                ds_kwargs['remove_noise_COM_x4'] = x4p.get('remove_noise_COM', False)
                ds_kwargs['max_node_types_x4'] = x4p.get('max_node_types', 10)
                ds_kwargs['scale_node_features_x4'] = x4p.get('scale_node_features', 2.0)
                ds_kwargs['scale_vector_features_x4'] = x4p.get('scale_vector_features', 2.0)
                ds_kwargs['multivectors'] = x4p.get('multivectors', False)
                ds_kwargs['check_accessibility'] = x4p.get('check_accessibility', False)

        dataset = PharmacophoreDataset(
            molblocks_and_charges=molblocks_and_charges,
            noise_schedule_dict=noise_schedule_dict,
            **ds_kwargs,
        )

        # Split
        n = len(dataset)
        n_train = int(n * self.train_ratio)
        n_val = (n - n_train) // 2
        indices = list(range(n))
        np.random.shuffle(indices)

        self.train_set = torch.utils.data.Subset(dataset, indices[:n_train])
        self.valid_set = torch.utils.data.Subset(dataset, indices[n_train:n_train + n_val])
        self.test_set = torch.utils.data.Subset(dataset, indices[n_train + n_val:])

        logger.info(f"Split: train={len(self.train_set)}, val={len(self.valid_set)}, test={len(self.test_set)}")


# ============================================================
# Collate function: use PyG's default HeteroData collation
# ============================================================

def pharmacophore_collate_fn(batch: List[HeteroData]) -> HeteroData:
    """
    Collate HeteroData samples into a batch.
    Uses torch_geometric's built-in Batch.from_data_list for HeteroData.
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)
