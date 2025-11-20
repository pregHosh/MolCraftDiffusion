import math
from copy import deepcopy

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from ase.data import covalent_radii
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from torch_scatter import scatter_add
from tqdm import tqdm

from MolecularDiffusion.utils import (
    assert_mean_zero_with_mask,
    check_quality,
    correct_edges,
    create_pyg_graph,
    enforce_min_nodes_per_connector,
    ensure_intact,
    find_close_points_torch_and_push_op2,
    remove_mean_with_mask,
    remove_mean_pyG,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask
)
import logging
logger = logging.getLogger(__name__)

class EnVariationalDiffusion(torch.nn.Module):
    """
    E(n) Equivariant Variational Diffusion Model.

    Args:
        dynamics (nn.Module): Neural network that predicts noise or x.
        dynamics_teacher (Optional[nn.Module]): Teacher model for distillation.
        in_node_nf (int): Total number of input node features per atom.
        n_dims (int): Dimensionality of spatial coordinates (typically 3).
        timesteps (int): Total number of diffusion steps (T).
        parametrization (str): Parametrization used, currently only "eps" supported.
        noise_schedule (str): Either "learned" or predefined schedule name.
        noise_precision (float): Precision used in predefined schedule.
        loss_type (str): Loss function type, either "vlb" or "l2".
        norm_values (Tuple[float, float, float]): Normalization scales for (x, h_cat, h_int).
        extra_norm_values (Sequence[float]): Normalization values for additional features.
        norm_biases (Tuple[Optional[float], float, float]): Biases for (x, h_cat, h_int) normalization.
        include_charges (bool): Whether integer features include formal charge.
        context_mask_rate (float): Probability of masking the context for classifier-free guidance.
        mask_value (float): Value used for masked context tokens.
        eval_mode (bool): If True, disables KL loss during evaluation.
    """
    def __init__(
        self,
        dynamics: torch.nn.Module,
        dynamics_teacher: Optional[torch.nn.Module] = None,
        in_node_nf: int = 12,
        n_dims: int = 3,
        timesteps: int = 1000,
        parametrization: str = "eps",
        noise_schedule: str = "learned",
        noise_precision: float = 1e-4,
        loss_type: str = "vlb",
        norm_values: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        extra_norm_values: Sequence[float] = (),
        norm_biases: Tuple[Optional[float], float, float] = (None, 0.0, 0.0),
        include_charges: bool = True,
        context_mask_rate: float = 0.0,
        mask_value: float = 0.0,
        eval_mode: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.call = 0
        self.eval_mode = eval_mode
        self.debug = debug

        # Loss and parametrization settings
        assert loss_type in {"vlb", "l2"}
        assert parametrization == "eps"  # Only supported parametrization
        self.loss_type = loss_type
        self.parametrization = parametrization

        # Feature and normalization settings
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.ndim_extra = len(extra_norm_values)
        self.include_charges = include_charges
        self.num_classes = self.in_node_nf - int(self.include_charges)
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.extra_norm_values = extra_norm_values

        # Diffusion schedule settings
        self.T = timesteps
        if noise_schedule == "learned":
            assert loss_type == "vlb", (
                "A noise schedule can only be learned with a vlb objective."
            )
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(
                noise_schedule, timesteps=timesteps, precision=noise_precision
            )

        # Denoising networks
        self.dynamics = dynamics
        self.dynamics_teacher = dynamics_teacher

        # Context and mask settings
        self.context_mask_rate = context_mask_rate
        self.mask_value = mask_value
        self.condition_tensor = None

        # Register buffer for device compatibility
        self.register_buffer("buffer", torch.zeros(1))

        # Check normalization values if not using learned schedule
        if noise_schedule != "learned":
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)
        return net_out
    
    def phi_pyg(self, mol_graph):
        """
        mol_graph containing
        - x : node_feature [N, nf]
        - pos L position [N, 3]
        - edge_index
        - context: properties conditioning the model [N, n_prop]
        - time: t_float [N,1]
        """
        net_out = self.dynamics._forward_pyG(mol_graph)
        return net_out

    def phi_distillation(self, x, t, node_mask, edge_mask, context):

        if self.dynamics_teacher is None:
            raise ValueError("Teacher model is not defined.")
        else:
            net_out = self.dynamics_teacher._forward(
                t, x, node_mask, edge_mask, context
            )

        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        """
        Normalizes x, categorical h, integer h, and optionally extra h.

        Args:
            x (Tensor): [B, N, 3] coordinates
            h (dict): {'categorical', 'integer', 'extra' (optional)}
            node_mask (Tensor): [B, N, 1] mask

        Returns:
            Tuple of (x_normalized, h_normalized_dict, delta_log_px)
        """
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * torch.log(
            torch.tensor(self.norm_values[0], device=x.device)
        )

        h_cat = ((h["categorical"].float() - self.norm_biases[1]) / self.norm_values[1]) * node_mask
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]
        if self.include_charges:
            h_int *= node_mask

        if self.ndim_extra > 0:
            norm_extra = torch.tensor(self.extra_norm_values, device=x.device).view(1, 1, -1)
            h_extra = (h["extra"].float() / norm_extra) * node_mask
            h_out = {"categorical": h_cat, "integer": h_int, "extra": h_extra}
        else:
            h_out = {"categorical": h_cat, "integer": h_int}

        return x, h_out, delta_log_px

    def normalize_pyG(self, mol_graph):
        """
        Normalizes node features and position of PyG molecular graph.

        Args:
            mol_graph (dict): contains 'graph' (Data), which includes pos, x, atomic_numbers

        Returns:
            Tuple of (normalized mol_graph, delta_log_px)
        """
        g = mol_graph["graph"]
        g.pos = g.pos / self.norm_values[0]

        n_nodes = g.natoms
        delta_log_px = -(n_nodes - 1) * self.n_dims * torch.log(
            torch.tensor(self.norm_values[0], device=g.pos.device)
        )

        g.atomic_numbers = (g.atomic_numbers - self.norm_biases[2]) / self.norm_values[2]

        if self.ndim_extra > 0:
            norm_extra = torch.tensor(self.extra_norm_values, device=g.x.device).view(1, 1, -1)
            g.x[:, :self.ndim_extra] = (g.x[:, :self.ndim_extra] - self.norm_biases[1]) / self.norm_values[1]
            g.x[:, self.ndim_extra:] = g.x[:, self.ndim_extra:] / norm_extra
        else:
            g.x = (g.x - self.norm_biases[1]) / self.norm_values[1]

        return mol_graph, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        """
        Reverts normalization of x, h_cat, and h_int.

        Args:
            x (Tensor): [B, N, 3]
            h_cat (Tensor): [B, N, C+E]
            h_int (Tensor): [B, N, 1]
            node_mask (Tensor): [B, N, 1]

        Returns:
            Tuple of unnormalized (x, h_cat, h_int)
        """
        x = x * self.norm_values[0]

        if self.ndim_extra > 0:
            nextra = self.ndim_extra
            norm_extra = torch.tensor(self.extra_norm_values, device=x.device).view(1, 1, -1)
            h_extra = h_cat[:, :, -nextra:] * norm_extra
            h_cat_core = h_cat[:, :, :-nextra] * self.norm_values[1] + self.norm_biases[1]
            h_cat_core *= node_mask
            h_cat = torch.cat([h_cat_core, h_extra], dim=2)
        else:
            h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
            h_cat *= node_mask

        h_int = h_int * self.norm_values[2] + self.norm_biases[2]
        if self.include_charges:
            h_int *= node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        """
        Unnormalize x, h_cat, and h_int from latent z.

        Args:
            z (Tensor): [B, N, D]
            node_mask (Tensor): [B, N, 1]

        Returns:
            Tensor of [B, N, D] with unnormalized (x, h_cat, h_int)
        """
        x = z[:, :, :self.n_dims]
        h_cat = z[:, :, self.n_dims : self.n_dims + self.num_classes]

        if self.ndim_extra > 0:
            h_cat = h_cat[:, :, :-self.ndim_extra]

        h_int = z[:, :, -1:].contiguous()
        assert h_int.size(2) == self.include_charges

        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        return torch.cat([x, h_cat, h_int], dim=2)
    
    def sigma_and_alpha_t_given_s(
        self,
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor,
        target_tensor: torch.Tensor,
    ):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, : self.n_dims], mu_T[:, :, self.n_dims :]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(
            gamma_T,
            mu_T_x,
        ).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(
            mu_T_x, sigma_T_x, zeros, ones, d=subspace_d
        )

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == "x":
            x_pred = net_out
        elif self.parametrization == "eps":
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1.0 / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        errors = {}
        eps_t = net_out

        # denominator
        if self.training and self.loss_type == "l2":
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
        else:
            denom = 1

        # position
        eps_t_x = eps_t[:, :, :self.n_dims]
        eps_x   = eps[:, :, :self.n_dims]
        errors["pos"] = sum_except_batch((eps_x - eps_t_x) ** 2) / denom

        # compute hint & categorical slices
        n_core = self.in_node_nf - self.ndim_extra - 1
        start  = self.n_dims
        mid    = start + n_core
        hint_i = mid  # index of the integer hint feature

        if self.ndim_extra > 0:
            # extra features
            eps_t_hextra = eps_t[:, :, -self.ndim_extra:]
            eps_hextra   = eps[:, :, -self.ndim_extra:]
            errors["h_extra"] = sum_except_batch((eps_hextra - eps_t_hextra) ** 2) / denom

            # hint & categorical core
            eps_t_hcat = eps_t[:, :, start:mid]
            eps_hcat   = eps[:, :, start:mid]
            eps_t_hint = eps_t[:, :, hint_i:hint_i+1]
            eps_hint   = eps[:, :, hint_i:hint_i+1]
        else:
            # when no extra dims, hcat runs up to last, hint is last
            eps_t_hcat = eps_t[:, :, start:-1]
            eps_hcat   = eps[:, :, start:-1]
            eps_t_hint = eps_t[:, :, -1:]
            eps_hint   = eps[:, :, -1:]

        # integer (hint) and categorical errors
        errors["integer"]     = sum_except_batch((eps_hint - eps_t_hint) ** 2) / denom
        errors["categorical"] = sum_except_batch((eps_hcat - eps_t_hcat) ** 2) / denom

        return errors

    def compute_error_pyG(self, net_out, eps, natom):
        """
        Vectorized per-molecule loss computation without Python loops.

        Args:
            net_out: Tensor of shape (N_total, F)
            eps:     Tensor of shape (N_total, F)
            natom:   LongTensor of shape (batch_size,) summing to N_total
        Returns:
            errors: dict of (batch_size,) tensors: 'pos','integer','categorical', and optionally 'h_extra'
        """
        # compute batch assignment for each node
        batch_idx = torch.arange(natom.size(0), device=natom.device).repeat_interleave(natom)

        # define denom per-batch
        if self.training and self.loss_type == "l2":
            # denom = (self.n_dims + self.in_node_nf) * natom.to(net_out.dtype)
            denom = (self.n_dims + self.in_node_nf) * natom.max()
        else:
            denom = torch.ones_like(natom, dtype=net_out.dtype)

        # split features
        x_dim = self.n_dims
        hint_dim = 1
        extra_dim = self.ndim_extra

        # positional error
        eps_x = eps[:, :x_dim]
        eps_t_x = net_out[:, :x_dim]
        pos_sq = (eps_x - eps_t_x).pow(2).sum(dim=1)
        pos_sum = scatter_add(pos_sq, batch_idx, dim=0, dim_size=natom.size(0))
        pos_err = pos_sum / denom

        # hint and hcat indices
        if extra_dim > 0:
            start = x_dim
            mid = x_dim + (self.in_node_nf - extra_dim - hint_dim)
            hint_idx = mid
            # hcat
            eps_hcat = eps[:, start:mid]
            eps_t_hcat = net_out[:, start:mid]
            cat_sq = (eps_hcat - eps_t_hcat).pow(2).sum(dim=1)
            cat_sum = scatter_add(cat_sq, batch_idx, dim=0)
            cat_err = cat_sum / denom

            # integer hint
            eps_hint = eps[:, hint_idx:hint_idx+hint_dim]
            eps_t_hint = net_out[:, hint_idx:hint_idx+hint_dim]
            int_sq = (eps_hint - eps_t_hint).pow(2).sum(dim=1)
            int_sum = scatter_add(int_sq, batch_idx, dim=0)
            int_err = int_sum / denom

            # extra dims
            eps_hextra = eps[:, -extra_dim:]
            eps_t_hextra = net_out[:, -extra_dim:]
            extra_sq = (eps_hextra - eps_t_hextra).pow(2).sum(dim=1)
            extra_sum = scatter_add(extra_sq, batch_idx, dim=0)
            extra_err = extra_sum / denom
        else:
            # hcat is middle, hint is last
            eps_hcat = eps[:, x_dim:-hint_dim]
            eps_t_hcat = net_out[:, x_dim:-hint_dim]
            cat_sq = (eps_hcat - eps_t_hcat).pow(2).sum(dim=1)
            cat_sum = scatter_add(cat_sq, batch_idx, dim=0)
            cat_err = cat_sum / denom

            eps_hint = eps[:, -hint_dim:]
            eps_t_hint = net_out[:, -hint_dim:]
            int_sq = (eps_hint - eps_t_hint).pow(2).sum(dim=1)
            int_sum = scatter_add(int_sq, batch_idx, dim=0)
            int_err = int_sum / denom

        return {
            'pos': pos_err,
            'integer': int_err,
            'categorical': cat_err,
            **({'h_extra': extra_err} if extra_dim > 0 else {})
        }


    def log_constants_p_x_given_z0_pyG(self, x, batch_size, n_nodes):
        """Computes p(x|z0)."""
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((batch_size, 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))


    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise
        )

        x = xh[:, :, : self.n_dims]

        if len(self.extra_norm_values) > 0:
            h_int = z0[:, :, self.n_dims + self.in_node_nf - self.ndim_extra - 1] if self.include_charges else torch.zeros(0).to(z0.device)
            h_cat = torch.cat([
                z0[:, :, self.n_dims : self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                z0[:, : , -self.ndim_extra:]   
                ],
                dim=2)
            
        else:
            h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
            h_cat = z0[:, :, self.n_dims : -1]

        if h_int.dim() < 3: 
            h_int = h_int.unsqueeze(-1)
        x, h_cat, h_int = self.unnormalize(
            x, h_cat, h_int, node_mask
        )
        
        if len(self.extra_norm_values) > 0:
            h_extra = h_cat[:, :, -len(self.extra_norm_values) :]
            h_cat = h_cat[:, :, : -len(self.extra_norm_values)]
        else:
            h_extra = torch.zeros(0).to(z0.device)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes - len(self.extra_norm_values)) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {"integer": h_int, "categorical": h_cat, "extra": h_extra}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps


    def log_pxh_given_z0_without_constants(
        self,
        x,
        h,
        z_t,
        t_int,
        eps,
        net_out,
        node_mask,
        reference_indices=None,
        epsilon=1e-10
    ):
        
        z_t = z_t[:, :, self.n_dims :]

        if self.ndim_extra > 0:
            z_h_int = (
                z_t[:, :, self.in_node_nf - self.ndim_extra - 1].unsqueeze(-1)
                if self.include_charges
                else torch.zeros(0).to(z_t.device)
            )
            z_h_cat = z_t[:, :, : self.in_node_nf - self.ndim_extra - 1]
            z_h_extra = z_t[:, :, -self.ndim_extra :]
        else:
            z_h_cat = z_t[:, :, :-1]
            z_h_int = (
                z_t[:, :, -1:]
                if self.include_charges
                else torch.zeros(0).to(z_t.device)
            )

        errors = self.compute_error(net_out, t_int / self.T, eps)
        log_p_x_given_z_without_constants = -0.5 * (errors["pos"])

        # Compute sigma_0 and rescale to the integer scale of the data.
        # t_zeros = torch.zeros(node_mask.size(0), 1).long()
        t_zeros = torch.zeros_like(t_int)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)

        sigma_0_cat = sigma_0[:, :, 0] * self.norm_values[1]
        sigma_0_int = sigma_0[:, :, 0] * self.norm_values[2]
        sigma_0_extra = sigma_0[:, :, 0] * torch.tensor(
            self.extra_norm_values, device=sigma_0.device
        ).view(1, 1, -1)

        # Compute delta indicator masks.
        h_integer = torch.round(
            h["integer"] * self.norm_values[2] + self.norm_biases[2]
        ).long()
        onehot = h["categorical"] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]

        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian(
                (h_integer_centered + 0.5) / sigma_0_int.unsqueeze(-1)
            )
            - cdf_standard_gaussian(
                (h_integer_centered - 0.5) / sigma_0_int.unsqueeze(-1)
            )
            + epsilon
        )
        if reference_indices is not None:
            log_ph_integer[:, reference_indices, :] = 0
  
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat.unsqueeze(-1))
        -
            cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat.unsqueeze(-1))
            + epsilon
        )

        # Normalize the distribution over the categories.
        
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        if reference_indices is not None:
            log_probabilities[:, reference_indices, :] = 0
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        if len(self.extra_norm_values) > 0:
            # log_p_x_given_z_without_constants -= 0.5 * (errors["h_extra"])
            sigma_0_extra = sigma_0[:, :, 0] * torch.tensor(
                self.extra_norm_values, device=sigma_0.device
            )
            sigma_0_extra = sigma_0_extra.view(
                -1,
                1,
                self.ndim_extra,
            )
            h_extra = h["extra"] * torch.tensor(
                self.extra_norm_values, device=sigma_0.device
            ).view(1, 1, -1)
            estimated_h_extra = z_h_extra * torch.tensor(
                self.extra_norm_values, device=sigma_0.device
            ).view(1, 1, -1)
            h_extra_centered = h_extra - estimated_h_extra
            # Compute integral from -0.5 to 0.5 of the normal distribution
            # N(mean=z_h_cat, stdev=sigma_0_cat)
            log_ph_extra = torch.log(
                cdf_standard_gaussian((h_extra_centered + 0.5) / sigma_0_extra)
            -
                cdf_standard_gaussian((h_extra_centered - 0.5) / sigma_0_extra)
                + epsilon
            )
            log_ph_extra = sum_except_batch(log_ph_extra * node_mask)
            log_ph_cat += log_ph_extra


        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z


        return log_p_xh_given_z

    def log_pxh_given_z0_without_constants_pyG(
        self,
        x,                   # (N_total, 3)
        h,                   # dict with "integer":(N_total, 1), "categorical":(N_total, C), optionally "extra":(N_total, E)
        z_t,                 # (N_total, F)
        t_int,               # (batch_size,)
        eps,                 # (N_total, F)
        net_out,             # (N_total, F)
        natom,               # (batch_size,) sums to N_total
        reference_indices=None,
        epsilon=1e-10
    ):
        N_total   = z_t.shape[0]
        batch_sz  = natom.size(0)

        # 1) positional term
        errors = self.compute_error_pyG(net_out, eps, natom)
        log_px = -0.5 * errors["pos"]           # (batch_size,)

        # 2) per-node batch indices
        batch_idx = torch.arange(batch_sz, device=natom.device) \
                        .repeat_interleave(natom)  # (N_total,)

        # 3) split off the hidden-z blocks
        z_h = z_t[:, self.n_dims:]        # (N_total, F_hidden)
        if self.ndim_extra > 0:
            hint_i      = self.in_node_nf - self.ndim_extra - 1
            z_h_int     = z_h[:, hint_i:hint_i+1]
            z_h_cat     = z_h[:, :hint_i]
            z_h_extra   = z_h[:, -self.ndim_extra:]
        else:
            z_h_int     = (z_h[:, -1:] if self.include_charges 
                        else torch.zeros((N_total,0), device=z_h.device))
            z_h_cat     = z_h[:, :-1]
            z_h_extra   = None

        # 4) compute sigma0 for every node
        t0       = torch.zeros_like(t_int)
        gamma0   = self.inflate_batch_array(self.gamma(t0), x)  # (batch_sz, N_total, …)
        sigma0   = self.sigma(gamma0, target_tensor=z_t) \
                        .squeeze(0)           # (N_total, 1)
        batch_idx = torch.arange(batch_sz, device=natom.device) \
                        .repeat_interleave(natom)  # (N_total,)

        sigma0 = sigma0.repeat_interleave(natom, dim=0)  # (N_total, 1)
        sigma0_cat   = sigma0[:, 0:1] * self.norm_values[1]
        sigma0_int   = sigma0[:, 0:1] * self.norm_values[2]
        sigma0_extra = None
        if self.ndim_extra > 0:
            sigma0_extra = (
                sigma0[:, 0:1]
                * torch.tensor(self.extra_norm_values, device=sigma0.device)
                    .view(1, -1)
            )

        # 5) integer log‐π per node
        h_int_round = torch.round(h["integer"] * self.norm_values[2]
                                + self.norm_biases[2]).long()
        est_int     = z_h_int * self.norm_values[2] + self.norm_biases[2]
        centered_i  = h_int_round - est_int           # (N_total,1)
        log_pi = torch.log(
            cdf_standard_gaussian((centered_i + 0.5) / sigma0_int)
        -
            cdf_standard_gaussian((centered_i - 0.5) / sigma0_int)
        +
            epsilon
        ).squeeze(-1)                                 # (N_total,)
        #TODO this is wrong
        if reference_indices is not None:
            log_pi[reference_indices] = 0

        # 6) categorical log‐p per node
        onehot      = h["categorical"] * self.norm_values[1] + self.norm_biases[1]
        est_cat     = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        centered_c  = est_cat - 1                     # (N_total, C)
        log_pc_prop = torch.log(
            cdf_standard_gaussian((centered_c + 0.5) / sigma0_cat)
        -
            cdf_standard_gaussian((centered_c - 0.5) / sigma0_cat)
        +
            epsilon
        )                                             # (N_total, C)
        log_Z       = torch.logsumexp(log_pc_prop, dim=1, keepdim=True)
        log_pcat    = log_pc_prop - log_Z             # (N_total, C)
        #TODO this is wrong
        if reference_indices is not None:
            log_pcat[reference_indices] = 0
        log_pc = (log_pcat * onehot).sum(dim=1)       # (N_total,)

        # 7) extra‐feature log‐p per node (if any)
        if self.ndim_extra > 0:
            h_ex       = h["extra"] * torch.tensor(self.extra_norm_values, device=h["extra"].device)
            est_ex     = z_h_extra * torch.tensor(self.extra_norm_values, device=z_h_extra.device)
            centered_e = h_ex - est_ex                  # (N_total, E)
            log_pe     = torch.log(
                cdf_standard_gaussian((centered_e + 0.5) / sigma0_extra)
            -
                cdf_standard_gaussian((centered_e - 0.5) / sigma0_extra)
            +
            epsilon
            ).sum(dim=1)                                # → (N_total,)
        else:
            log_pe = torch.zeros(N_total, device=x.device)

        # 8) scatter‐sum node contributions into per‐graph sums
        sum_pi = scatter_add(log_pi, batch_idx, dim=0, dim_size=batch_sz)
        sum_pc = scatter_add(log_pc, batch_idx, dim=0, dim_size=batch_sz)
        sum_pe = scatter_add(log_pe, batch_idx, dim=0, dim_size=batch_sz)

        log_ph = sum_pi + sum_pc + sum_pe          # (batch_sz,)
        return log_px + log_ph                    # (batch_sz,)


    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))


    def compute_loss(
        self,
        x: torch.Tensor,
        h: dict,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: Optional[torch.Tensor],
        t0_always: bool,
        reference_indices: Optional[Union[list, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Computes the total training loss (VLB or L2) based on a randomly sampled timestep t.

        Args:
            x (Tensor): [B, N, 3] atom positions
            h (dict): dictionary with 'categorical', 'integer', and optionally 'extra' features
            node_mask (Tensor): [B, N, 1] indicating which nodes are valid
            edge_mask (Tensor): [B, N, N] adjacency mask
            context (Tensor or None): [B, N, D] per-node context or None
            t0_always (bool): whether to explicitly include the loss at t = 0
            reference_indices (list or Tensor, optional): atom indices to freeze during loss

        Returns:
            Tuple of (loss: Tensor of shape [B], diagnostics: dict)
        """
        # ---------------------------------------------------------------------
        # 1. TIMESTEP SAMPLING
        # ---------------------------------------------------------------------
        T_range = (1 if t0_always else 0, self.T + 1)
        t_int = torch.randint(*T_range, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()

        # Normalized t, s ∈ [0, 1]
        t = t_int / self.T
        s = s_int / self.T

        # ---------------------------------------------------------------------
        # 2. GAMMA / NOISE SCHEDULE
        # ---------------------------------------------------------------------
        gamma_t = self.inflate_batch_array(self.gamma(t), x)
        gamma_s = self.inflate_batch_array(self.gamma(s), x)

        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample standard Gaussian noise
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Concatenate all features
        xh = torch.cat(
            [x, h["categorical"], h["integer"], h["extra"]] if self.ndim_extra > 0 else [x, h["categorical"], h["integer"]],
            dim=2,
        )

        # Sample z_t ~ q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        if reference_indices is not None:
            # Freeze reference atoms (use clean xh)
            z_t[:, reference_indices] = xh[:, reference_indices]
            eps[:, reference_indices] = 0.0
            z_t = torch.cat([
                remove_mean_with_mask(z_t[:, :, :self.n_dims], node_mask),
                z_t[:, :, self.n_dims:]
            ], dim=2)

        assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # ---------------------------------------------------------------------
        # 3. CONTEXT MASKING
        # ---------------------------------------------------------------------
        if context is not None:
            if self.dynamics.use_adapter_module:
                if self.context_mask_rate > 0:
                    context_masked = None if torch.randint(1, int(1 / self.context_mask_rate), ()).item() == 1 else context.clone() * node_mask
                else:
                    context_masked = context.clone() * node_mask
            else:
                mask = torch.rand(context.size(0), device=context.device) < self.context_mask_rate
                context_masked = context.clone()
                context_masked[mask] = self.mask_value
                context_masked *= node_mask
        else:
            context_masked = None

        # ---------------------------------------------------------------------
        # 4. MODEL PREDICTION
        # ---------------------------------------------------------------------
        net_out = self.phi(z_t, t, node_mask, edge_mask, context_masked)

        # ---------------------------------------------------------------------
        # 5. ERROR COMPUTATION
        # ---------------------------------------------------------------------
        errors = self.compute_error(net_out, gamma_t, eps)

        # SNR-based weighting
        if self.training and self.loss_type == "l2":
            SNR_weight = torch.ones_like(errors["pos"])
        else:
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)

        # Weighted sum of all error components
        loss_t_larger_than_zero = sum(
            0.5 * (SNR_weight * v if self.loss_type == "vlb" else v)
            for v in errors.values()
        )

        # ---------------------------------------------------------------------
        # 6. NEGATIVE LOG CONSTANTS AND PRIOR TERM
        # ---------------------------------------------------------------------
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        kl_prior = (
            self.kl_prior(xh, node_mask) if not self.eval_mode else torch.zeros_like(neg_log_constants)
        )

        # ---------------------------------------------------------------------
        # 7. INCLUDE t = 0 TERM IF SPECIFIED
        # ---------------------------------------------------------------------
        if t0_always:
            num_terms = self.T
            estimator_loss_terms = num_terms * loss_t_larger_than_zero

            # Additional forward pass at t = 0
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
            )
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out_0 = self.phi(z_0, t_zeros, node_mask, edge_mask, context)
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, t_zeros, eps_0, net_out_0, node_mask, reference_indices=None,
            )

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
        else:
            # Soft mask between t=0 and t>0
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, t_int, eps, net_out, node_mask, reference_indices=None,
            )
            t_is_not_zero = 1.0 - t_is_zero.squeeze()

            loss_t = loss_term_0.squeeze() * t_is_zero.squeeze() + t_is_not_zero * loss_t_larger_than_zero
            num_terms = self.T + 1
            estimator_loss_terms = loss_t if self.loss_type == "l2" else num_terms * loss_t

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert loss.ndim == 1, f"Expected [B], got {loss.shape}"

        return loss, {
            "t": t_int.squeeze(),
            "loss_t": loss.squeeze(),
        }


    # TODO outpaint not yet implemented
    def compute_loss_pyG(self, mol_graph, context, t0_always, reference_indices=None):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE).
        
        If reference_indices is specified, their atoms are frozen during the forward pass.
        The loss is computed only for the unfrozen atoms.

        mol_graph: pyGraph, containing the following attributes:
            x: torch.Tensor, shape [N, D] (node features)
            pos: torch.Tensor, shape [N, 3] (positions)
            charges: torch.Tensor, shape [N, 1] (charges)
            natoms: int, number of atoms
            edge_index: torch.Tensor, shape [2, E] (edge indices)
        context: torch.Tensor, shape [B, D]
        t0_always: bool, whether to include loss term 0 always.   
        reference_indices: list of int, indices of reference nodes
        """
        
        mol_graph_t = deepcopy(mol_graph)
        mol_graph_0 = deepcopy(mol_graph)
        h = mol_graph["graph"].x
        charges = mol_graph["graph"].atomic_numbers.unsqueeze(-1)
        x = mol_graph["graph"].pos

        bs = mol_graph["graph"].batch.max().item() + 1
        n_atoms  = mol_graph["graph"].natoms
        n_nodes = h.size(0) 
        node_mask = torch.ones((n_nodes, 1), device=x.device)
        batch_idx = mol_graph["graph"].batch

        if len(self.extra_norm_values) > 0:
            xh = torch.cat([x, 
                            h[:self.ndim_extra], #OHE
                            charges,
                            h[self.ndim_extra:]], #extra node features
                            dim=1)
            h_dicts = {
                "integer": charges,
                "categorical": h[:self.ndim_extra],
                "extra": h[self.ndim_extra:]
                
            }
        else:
            xh = torch.cat([x, h, charges], dim=1)
            h_dicts = {
                "integer": charges,
                "categorical": h,
                
            }
        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(bs, 1), device=x.device
        ).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T
        t_stretched = torch.repeat_interleave(t, n_atoms, dim=0)

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        alpha_per_node = alpha_t[batch_idx]  # (total_num_nodes, 1)
        sigma_per_node = sigma_t[batch_idx]  # (total_num_nodes, 1)

        # Sample random noise phi for each node
        eps_x = torch.randn(x.size(0), x.size(1), device=x.device)  # (total_num_nodes, 3)
        eps_x = remove_mean_pyG(eps_x, batch_idx)
        eps_h = torch.randn(h.size(0), h.size(1) + 1, device=x.device)  # (total_num_nodes, num_node_features + 1(for c))
        eps = torch.cat([
            eps_x,
            eps_h,
        ], dim=1)

        # Apply the noise formula
        z_t = alpha_per_node * xh + sigma_per_node * eps

        mol_graph_t["graph"].pos = z_t[:, :self.n_dims]
        if len(self.extra_norm_values) > 0:
            # OHE, atomic number, extra node features
            mol_graph_t["graph"].x = torch.cat([
                z_t[:, self.n_dims: self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                z_t[:, self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                z_t[:, self.ndim_extra:],], 
                dim=1)
            mol_graph_t["graph"].atomic_numbers = z_t[:, self.n_dims + self.in_node_nf - self.ndim_extra - 1]    
        else:
            mol_graph_t["graph"].x = z_t[:, self.n_dims:]
        mol_graph_t["t"] = t_stretched


        # TODO figure this out later
        # if reference_indices is not None:
        #     # # Freeze the reference atoms
        #     # xh_ref = xh[:, reference_indices, :]
        #     # z_t[:, reference_indices, :] = xh_ref[:, :, :]
        #     # eps[:, reference_indices, :] = 0
        #     # z_t = torch.cat([
        #     #     remove_mean_with_mask(z_t[:, :, : self.n_dims], node_mask),
        #     #     z_t[:, :, self.n_dims :],
        #     # ],dim=2)

        # assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # TODO figure this out later
        # Classifer-free guidance (if context_mask_rate > 0) or conditioning (context_mask_rate = 0).
        if context is not None:
            if self.dynamics.use_adapter_module:
                if self.context_mask_rate > 0:
                    n_rand = torch.randint(1, int(1/self.context_mask_rate), (1,), device=context.device)
                    if 1 in n_rand:
                        context_masked = None
                    else:
                        context_masked = context.clone()*node_mask
                else:
                    context_masked = context.clone()*node_mask
            else:
                mask = torch.rand(context.size(0), device=context.device) < self.context_mask_rate
                context_masked = context.clone()
                context_masked[mask] = self.mask_value
                context_masked = context_masked * node_mask
        else:
            context_masked = None
        # Neural net prediction.
        net_out = self.phi_pyg(mol_graph_t)
        # Compute the error.
        errors = self.compute_error_pyG(net_out, eps, natom=n_atoms)
        if self.training and self.loss_type == "l2":
            SNR_weight = torch.ones_like(errors["pos"])
        else:
            # Assume statics noise scheduler for now
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1)
        loss_t_larger_than_zero = 0
        if self.training and self.loss_type == "l2":
            for i, component in enumerate(errors):
                loss_t_larger_than_zero += 0.5 * errors[component]
        
        else:
            for i, component in enumerate(errors):
                # Assume statics noise scheduler for now
                loss_t_larger_than_zero += 0.5 * SNR_weight * errors[component]
        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0_pyG(x, bs, n_atoms)
 
        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        # kl_prior = self.kl_prior_pyG(xh, batch_size=bs)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)
            t_zeros_stretched = torch.repeat_interleave(t_zeros, n_atoms, dim=0)
            alpha_0_per_node = alpha_0[batch_idx]  # (total_num_nodes, 1)
            sigma_0_per_node = sigma_0[batch_idx]  # (total_num_nodes, 1)
            
            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_x_0 = torch.randn(x.size(0), x.size(1), device=x.device)  # (total_num_nodes, 3)
            eps_x_0 = remove_mean_pyG(eps_x_0, batch_idx)
            eps_h_0 = torch.randn(h.size(0), h.size(1) + 1, device=x.device)  # (total_num_nodes, num_node_features + 1(for c))
            eps_0 = torch.cat([
                eps_x_0,
                eps_h_0,
            ], dim=1)
            z_0 = alpha_0_per_node * xh + sigma_0_per_node * eps_0
            mol_graph_0["graph"].pos = z_0[:, :self.n_dims]
            if len(self.extra_norm_values) > 0:
                # OHE, atomic number, extra node features
                mol_graph_0["graph"].x = torch.cat([
                    z_0[:, self.n_dims: self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                    z_0[:, self.n_dims + self.in_node_nf - self.ndim_extra - 1],
                    z_0[:, self.ndim_extra:],], 
                    dim=1)
                mol_graph_0["graph"].atomic_numbers = z_0[:, self.n_dims + self.in_node_nf - self.ndim_extra - 1]    
            else:
                mol_graph_0["graph"].x = z_0[:, self.n_dims:]
            mol_graph_0["t"] = t_zeros_stretched
            net_out = self.phi_pyg(mol_graph_0)

            loss_term_0 = -self.log_pxh_given_z0_without_constants_pyG(
                x, h_dicts, z_0, 
                t_zeros, eps_0, net_out, 
                reference_indices=None, natom=n_atoms
            )


            loss =  estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants_pyG(
                x, h_dicts, z_t, 
                t_int, eps, net_out, 
                reference_indices=None, natom=n_atoms
            )
            t_is_zero = t_is_zero.squeeze(-1)
            t_is_not_zero = 1 - t_is_zero
            
            loss_t = (loss_term_0 * t_is_zero
                + t_is_not_zero * loss_t_larger_than_zero
            )

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == "l2":
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t
                
            # assert kl_prior.size() == estimator_loss_terms.size()
            # assert kl_prior.size() == neg_log_constants.size()
            loss = estimator_loss_terms + neg_log_constants

        # assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, {
            "t": t_int.squeeze(),
            "loss_t": loss.squeeze(),
            # "error": error.squeeze(),
        }
    
      
    def compute_loss_distillation(self, x, h, node_mask, edge_mask, context, t0_always, masked_context=0):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        if self.dynamics_teacher is None:
            raise ValueError("Teacher model is not defined.")
        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device
        ).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        # Concatenate x, h[integer] and h[categorical].
        if len(self.extra_norm_values) > 0:
            xh = torch.cat([x, h["categorical"], h["extra"], h["integer"]], dim=2)
        else:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)


        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # Neural net prediction.
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        net_out_distilled = self.phi_distillation(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, net_out_distilled)

        if self.training and self.loss_type == "l2":
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
            )
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)
            net_out_distilled_0 = self.phi_distillation(
                z_t, t_zeros, node_mask, edge_mask, context
            )

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, net_out_distilled_0, net_out, node_mask
            )

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, net_out_distilled, net_out, node_mask
            )

            t_is_not_zero = 1 - t_is_zero

            loss_t = (
                loss_term_0 * t_is_zero.squeeze()
                + t_is_not_zero.squeeze() * loss_t_larger_than_zero
            )

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == "l2":
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, {
            "t": t_int.squeeze(),
            "loss_t": loss.squeeze(),
            "error": error.squeeze(),
        }

    def forward(
        self, 
        x, 
        h, 
        node_mask=None,
        edge_mask=None, 
        context=None,
        reference_indices=None,
        mol_graph=None,
        ):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        if len(self.extra_norm_values) > 0:
            h_full = h["categorical"]
            h["categorical"] = h_full[:, :, : -len(self.extra_norm_values)]
            h["extra"] = h_full[:, :, -len(self.extra_norm_values) :]

        if mol_graph:
            mol_graph, delta_log_px = self.normalize_pyG(mol_graph)
        else:
            x, h, delta_log_px = self.normalize(x, h, node_mask)
        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training and reference_indices is None:
            t0_always = False   
        else:
            t0_always = True

        if self.dynamics_teacher is not None:
            if self.training:
                loss, loss_dict = self.compute_loss_distillation(
                    x, h, node_mask, edge_mask, context, t0_always=t0_always
                )
            else:
                loss, loss_dict = self.compute_loss(
                    x, h, node_mask, edge_mask, context, 
                    reference_indices=reference_indices,
                    t0_always=t0_always, 
                )
        else:
            if mol_graph is not None:
                # Only 1 forward pass when t0_always is False.
                loss, loss_dict = self.compute_loss_pyG(
                    mol_graph, context, 
                    reference_indices=reference_indices,
                    t0_always=t0_always, 
                )
            else:
                # Less variance in the estimator, costs two forward passes.
                loss, loss_dict = self.compute_loss(
                    x, h, node_mask, edge_mask, context, 
                    reference_indices=reference_indices,
                    t0_always=t0_always,
                )
        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_p_zs_given_zt(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        fix_noise=False,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        
        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        return zs


    def sample_p_zs_given_zt_guidance_v0(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        target_function,
        scale,
        fix_noise=False,
        max_norm=20,
        n_backward=0,
    ):
        """Samples from zs ~ p(zs | zt) with guidance applied directly to the latent sample zs.

        This method computes the gradient of a target function with respect to the latent
        variable zs and uses it to guide the sampling process.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            target_function (callable): A function that takes zs and s and returns an energy value.
            scale (float): The strength of the guidance.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.
            max_norm (int, optional): Maximum norm for gradient clipping. Defaults to 20.
            n_backward (int, optional): Number of backward steps for refining the gradient. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, dict]: The guided sample zs and a dictionary with optimization info.
        """
        opt_info = {}
        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        with torch.no_grad():
            eps_t = self.phi(zt, t, node_mask, edge_mask, context=context)
            eps_t = eps_t.nan_to_num(0.0)

        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )
        opt_info["mu0_norm"] = mu.norm(dim=[1, 2])
        
        sigma = sigma_t_given_s * sigma_s / sigma_t
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # guidance
        with torch.enable_grad():
            zs = zs.requires_grad_()
            energy = target_function(zs, s).sum()
            grad = autograd.grad(energy, zs)[0]


        t_min = grad.min()
        t_max = grad.max()
        grad = 2*(grad - t_min) / (t_max - t_min) -1 

        grad = torch.cat(
            [
                remove_mean_with_mask(grad[:, :, : self.n_dims], node_mask),
                grad[:, :, self.n_dims :],
            ],
            dim=2,
        )
        

        zs = zs - scale * grad
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        
        

        if torch.isnan(zs).any():
            zs = zs.nan_to_num(0.0)
        
        opt_info["mu1_norm"] = zs.norm(dim=[1, 2])
        opt_info["grad_norms"] = grad.norm(dim=[1, 2])            
        opt_info["energies"] = target_function(zs, t)
        opt_info["z_norm"] = zs.norm(dim=[1, 2])

        if n_backward > 0:  
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.requires_grad_()
                    energy_r = target_function(zs, s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]      
                    grad_r = grad_r.nan_to_num(0.0)
                    t_min = grad_r.min()
                    t_max = grad_r.max()
                    grad_r = 2*(grad_r - t_min) / (t_max - t_min) -1
                    grad_r = torch.cat(
                        [
                            remove_mean_with_mask(grad_r[:, :, : self.n_dims], node_mask),
                            grad_r[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )
                    reverse_grad_zs_norm = grad_r.norm(dim=[1, 2])
                    clip_coef_reverse_zs = max_norm / (reverse_grad_zs_norm + 1e-6)
                    clip_coef_clamped_reverse_zs = torch.clamp(clip_coef_reverse_zs, max=1)
                    grad_r *= clip_coef_clamped_reverse_zs[:, None, None]
                    zs = zs - scale * grad_r 

        return zs, opt_info

    def sample_p_zs_given_zt_guidance_v1(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        target_function,
        scale,
        fix_noise=False,
        max_norm=20,
        n_backward=0,
        h_weight=1,
        x_weight=1,
    ):
        """Samples from zs ~ p(zs | zt) with guidance applied to the mean of the distribution.

        This method computes the gradient of a target function with respect to the predicted
        clean data z0 and uses it to guide the mean of the sampling distribution.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            target_function (callable): A function that takes z0 and t and returns an energy value.
            scale (float): The strength of the guidance.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.
            max_norm (int, optional): Maximum norm for gradient clipping. Defaults to 20.
            n_backward (int, optional): Number of backward steps for refining the gradient. Defaults to 0.
            h_weight (int, optional): Weight for the feature component of the gradient. Defaults to 1.
            x_weight (int, optional): Weight for the position component of the gradient. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, dict]: The guided sample zs and a dictionary with optimization info.
        """
        opt_info = {}
        D = zt.size(2)
        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        with torch.no_grad():
            eps_t = self.phi(zt, t, node_mask, edge_mask, context=context,)
            eps_t = eps_t.nan_to_num(0.0)

        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        opt_info["mu0_norm"] = mu.norm(dim=[1, 2])

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # z0 = (zt - (1 - sigma_t) * eps_t) / (sigma_t ** (1 / 2))
        z0 = (zt - sigma_t * eps_t) / ((1 - sigma_t**2) ** (1 / 2))
        # guidance
        with torch.enable_grad():
            z0 = z0.requires_grad_()
            energy = target_function(z0, t).sum()
            grad = autograd.grad(energy, z0)[0]

        opt_info["grad_norms"] = grad.norm(dim=[1, 2])

        grad_norm_x = grad[:, : , :self.n_dims].norm(dim=[1, 2])
        grad_norm_h = grad[:, : , self.n_dims:].norm(dim=[1, 2])
        grad_x = x_weight* D**(1/2)/t[0].item() * grad[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
        grad_h = h_weight* D**(1/2)/t[0].item() * grad[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

        grad = torch.cat(
            [
                remove_mean_with_mask(grad_x, node_mask),
                grad_h,
            ],
            dim=2,
        )
        
        mu = mu - scale * grad

        
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)


        if torch.isnan(zs).any():
            zs = zs.nan_to_num(0.0)
        
        opt_info["mu1_norm"] = mu.norm(dim=[1, 2])
        opt_info["energies"] = target_function(zs,  t)
        opt_info["z_norm"] = zs.norm(dim=[1, 2])
    
        if n_backward > 0:  
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.requires_grad_()
                    energy_r = target_function(zs,  s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]      
                    grad_r = grad_r.nan_to_num(0.0)
                    
                    
                    grad_norm_x = grad_r[:, : , :self.n_dims].norm(dim=[1, 2])
                    grad_norm_h = grad_r[:, : , self.n_dims:].norm(dim=[1, 2])
                    grad_r_x = x_weight*D**(1/2)/t[0].item() * grad_r[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
                    grad_r_h = h_weight*D**(1/2)/t[0].item() * grad_r[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

                    grad_r = torch.cat(
                        [
                            remove_mean_with_mask(grad_r_x, node_mask),
                            grad_r_h,
                        ],
                        dim=2,
                    )
        
                    reverse_grad_zs_norm = grad_r.norm(dim=[1, 2])
                    clip_coef_reverse_zs = max_norm / (reverse_grad_zs_norm + 1e-6)
                    clip_coef_clamped_reverse_zs = torch.clamp(clip_coef_reverse_zs, max=1)
                    grad_r *= clip_coef_clamped_reverse_zs[:, None, None]
                    zs = zs - scale * grad_r 
        
        return zs, opt_info

    def sample_p_zs_given_zt_guidance_v2(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        target_function,
        scale,
        fix_noise=False,
        max_norm=20,
        n_backward=0,
        h_weight=1,
        x_weight=1,
        structure_guidance=False,
        t_critical=0, # For outpaint
        mask_node_index=[] # For inpaint
    ):
        """Samples from zs ~ p(zs | zt) with guidance inspired by GeoGuide.

        This method applies guidance to the mean of the sampling distribution, similar to v1,
        but with modifications inspired by the GeoGuide paper. It also supports conditional
        generation using a reference tensor.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            target_function (callable): A function that takes z0 and t and returns an energy value.
            scale (float): The strength of the guidance.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.
            max_norm (int, optional): Maximum norm for gradient clipping. Defaults to 20.
            n_backward (int, optional): Number of backward steps for refining the gradient. Defaults to 0.
            h_weight (int, optional): Weight for the feature component of the gradient. Defaults to 1.
            x_weight (int, optional): Weight for the position component of the gradient. Defaults to 1.
            structure_guidance (bool, optional): If inpaint or outpaint, applies structure guidance. Defaults to False.
            t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
            mask_node_index (list, optional): List of node indices to mask during inpaiting. Defaults to [].

        Returns:
            Tuple[torch.Tensor, dict]: The guided sample zs and a dictionary with optimization info.
        """
        opt_info = {}
        
        
        D = zt.size(2)
        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)

        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            natom_ref = self.condition_tensor.size(1)
            mask_bools = [False] * natom_ref + [True] * (
                node_mask.size(1) - natom_ref
            )
            mask_bools = torch.tensor(mask_bools, device=zt.device, dtype=torch.bool)
            

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        with torch.no_grad():
            eps_t = self.phi(zt, t, node_mask, edge_mask, context=context,)
            eps_t = eps_t.nan_to_num(0.0)

        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        opt_info["mu0_norm"] = mu.norm(dim=[1, 2])

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        z0 = (zt - sigma_t * eps_t) / ((1 - sigma_t**2) ** (1 / 2))
        if self.condition_tensor is not None:
            z0 = torch.cat(
                        [self.condition_tensor, z0[:, mask_bools, :]], dim=1
                    )
        # guidance
        with torch.enable_grad():
            z0 = z0.requires_grad_()
            energy = target_function(z0, t).sum()
            grad = autograd.grad(energy, z0)[0]

        opt_info["grad_norms"] = grad.norm(dim=[1, 2])

        grad_norm_x = grad[:, : , :self.n_dims].norm(dim=[1, 2])
        grad_norm_h = grad[:, : , self.n_dims:].norm(dim=[1, 2])
        grad_x = x_weight* D**(1/2)/t[0].item() * grad[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
        grad_h = h_weight* D**(1/2)/t[0].item() * grad[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

        grad = torch.cat(
            [
                remove_mean_with_mask(grad_x, node_mask),
                grad_h,
            ],
            dim=2,
        )
        
        mu = mu - scale * grad

        opt_info["mu1_norm"] = mu.norm(dim=[1, 2])
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            if s > t_critical:
                # Fix the reference part as conditioning
                zs = torch.cat(
                    [self.condition_tensor, zs[:, mask_bools, :]], dim=1
                )
            else:
                # Fix just the atomm types of reference part as conditioning
                zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
        elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            zs = torch.cat(
                [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
            )

        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )

        if torch.isnan(zs).any():
            zs = zs.nan_to_num(0.0)

        opt_info["energies"] = target_function(zs, t)
        opt_info["z_norm"] = zs.norm(dim=[1, 2])
    
        if n_backward > 0:  
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.requires_grad_()
                    energy_r = target_function(zs,  s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]      
                    grad_r = grad_r.nan_to_num(0.0)
                    
                    
                    grad_norm_x = grad_r[:, : , :self.n_dims].norm(dim=[1, 2])
                    grad_norm_h = grad_r[:, : , self.n_dims:].norm(dim=[1, 2])
                    grad_r_x = x_weight*D**(1/2)/t[0].item() * grad_r[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
                    grad_r_h = h_weight*D**(1/2)/t[0].item() * grad_r[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

                    grad_r = torch.cat(
                        [
                            remove_mean_with_mask(grad_r_x, node_mask),
                            grad_r_h,
                        ],
                        dim=2,
                    )
        
                    reverse_grad_zs_norm = grad_r.norm(dim=[1, 2])
                    clip_coef_reverse_zs = max_norm / (reverse_grad_zs_norm + 1e-6)
                    clip_coef_clamped_reverse_zs = torch.clamp(clip_coef_reverse_zs, max=1)
                    grad_r *= clip_coef_clamped_reverse_zs[:, None, None]
                    zs = zs - scale * grad_r 

                    # In the event the reference tensor is provided
                    if self.condition_tensor is not None and "outpaint" in structure_guidance:
                        if s > t_critical:
                            # Fix the reference part as conditioning
                            zs = torch.cat(
                                [self.condition_tensor, zs[:, mask_bools, :]], dim=1
                            )
                        else:
                            # Fix just the atomm types of reference part as conditioning
                            zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
                    elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
                        zs = torch.cat(
                            [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
                        )     

        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        
        if self.condition_tensor  is not None and "outpaint" in structure_guidance:
            self.condition_tensor = zs[:, ~mask_bools, :]
        elif self.condition_tensor  is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            self.condition_tensor = zs[:, ~mask_node_index, :]  
            
        return zs, opt_info

    def sample_p_zs_given_zt_guidance_cfg(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        scale,
        fix_noise=False,
        context_negative=None,
        structure_guidance=False,
        t_critical=0, # For outpaint
        mask_node_index=[] # For inpaint
    ):
        """
        Samples from zs ~ p(zs | zt) using classifier-free guidance (CFG).

        This method adjusts the diffusion sampling process by guiding the noise prediction
        towards a conditional distribution and away from an unconditional (or negative) one.
        It also supports inpainting-style generation where a reference part of the structure
        can be fixed.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            scale (float): The strength of the classifier-free guidance.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.
            context_negative (torch.Tensor, optional): Negative conditional information for guidance.
                If None, unconditional generation is used as the negative target. Defaults to None.
            structure_guidance (bool, optional): If inpaint or outpaint, applies structure guidance. Defaults to False.
            t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
            mask_node_index (list, optional): List of node indices to mask during inpaiting. Defaults to [].

        Returns:
            torch.Tensor: The sampled data `zs` at timestep `s`.
        """

        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)
        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            natom_ref = self.condition_tensor.size(1)
            mask_bools = [False] * natom_ref + [True] * (
                node_mask.size(1) - natom_ref
            )
            mask_bools = torch.tensor(mask_bools, device=zt.device, dtype=torch.bool)
        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)


        # Neural net prediction.
        with torch.no_grad():
            eps_t_cond_positive = self.phi(zt, t, node_mask, edge_mask, context=context,)
            
            if self.dynamics.use_adapter_module:
                context_null = None
            else:
                context_null = torch.zeros_like(context, device=eps_t_cond_positive.device) + self.mask_value   
            eps_t_uncond = self.phi(zt, t, node_mask, edge_mask, context=context_null)

            if torch.any(torch.isnan(eps_t_cond_positive)):
                print("eps_t_cond is nan, setting to 0")  
                eps_t_cond_positive = eps_t_cond_positive.nan_to_num(0.0)
            if torch.any(torch.isnan(eps_t_uncond)):
                print("eps_t_uncond is nan, setting to 0") 
                eps_t_uncond = eps_t_uncond.nan_to_num(0.0)

            if context_negative is not None:
                eps_t_cond_negative = self.phi(zt, t, node_mask, edge_mask, context=context_negative)
                if torch.any(torch.isnan(eps_t_cond_negative)):
                    print("eps_t_cond_negative is nan, setting to 0")
                    eps_t_cond_negative = eps_t_cond_negative.nan_to_num(0.0)
                eps_t = (1 + scale) * eps_t_cond_positive - scale * eps_t_cond_negative
            else:
                eps_t =  (1 + scale) * eps_t_cond_positive - scale * eps_t_uncond
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
    
        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            if s > t_critical:
                # Fix the reference part as conditioning
                zs = torch.cat(
                    [self.condition_tensor, zs[:, mask_bools, :]], dim=1
                )
            else:
                # Fix just the atomm types of reference part as conditioning
                zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
        elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            zs = torch.cat(
                [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
            )
        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        if self.condition_tensor  is not None and "outpaint" in structure_guidance:
            self.condition_tensor = zs[:, ~mask_bools, :]
        elif self.condition_tensor  is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            self.condition_tensor = zs[:, ~mask_node_index, :]  
            
        return zs

    def sample_p_zs_given_zt_guidance_cfg_gg(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        target_function,
        cfg_scale,
        gg_scale,
        max_norm=20,
        n_backward=0,
        h_weight=1,
        x_weight=1,
        fix_noise=False,
        structure_guidance=False,
        t_critical=0, # For outpaint
        mask_node_index=[] # For inpaint
    ):
        """Combines Classifier-Free Guidance (CFG) with Gradient-Based Guidance (GG).

        This method first computes the noise prediction using CFG and then applies a
        gradient-based correction to the mean of the sampling distribution.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            target_function (callable): A function that takes z0 and t and returns an energy value.
            cfg_scale (float): The strength of the classifier-free guidance.
            gg_scale (float): The strength of the gradient-based guidance.
            max_norm (int, optional): Maximum norm for gradient clipping. Defaults to 20.
            n_backward (int, optional): Number of backward steps for refining the gradient. Defaults to 0.
            h_weight (int, optional): Weight for the feature component of the gradient. Defaults to 1.
            x_weight (int, optional): Weight for the position component of the gradient. Defaults to 1.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.
            structure_guidance (bool, optional): If inpaint or outpaint, applies structure guidance. Defaults to False.
            t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
            mask_node_index (list, optional): List of node indices to mask during inpaiting. Defaults to [].

        Returns:
            torch.Tensor: The guided sample zs.
        """
        D = zt.size(2)
        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)
        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            natom_ref = self.condition_tensor.size(1)
            mask_bools = [False] * natom_ref + [True] * (
                node_mask.size(1) - natom_ref
            )
            mask_bools = torch.tensor(mask_bools, device=zt.device, dtype=torch.bool)
            

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)


        # Neural net prediction.
        with torch.no_grad():
            eps_t_cond = self.phi(zt, t, node_mask, edge_mask, context=context,)
            
            if self.dynamics.use_adapter_module:
                context_null = None
            else:
                context_null = torch.zeros_like(context, device=eps_t_cond.device) + self.mask_value   
            eps_t_uncond = self.phi(zt, t, node_mask, edge_mask, context=context_null)

            if torch.any(torch.isnan(eps_t_cond)):
                print("eps_t_cond is nan, setting to 0")  
                eps_t_cond = context_null.nan_to_num(0.0)
            if torch.any(torch.isnan(eps_t_uncond)):
                print("eps_t_uncond is nan, setting to 0") 
                eps_t_uncond = eps_t_uncond.nan_to_num(0.0)

            eps_t =  (1 + cfg_scale) * eps_t_cond - cfg_scale * eps_t_uncond
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )


        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # gradient guidance
        z0 = (zt - sigma_t * eps_t) / ((1 - sigma_t**2) ** (1 / 2))
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            if s > t_critical:
                # Fix the reference part as conditioning
                z0 = torch.cat(
                    [self.condition_tensor, z0[:, mask_bools, :]], dim=1
                )
            else:
                # Fix just the atomm types of reference part as conditioning
                z0[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
        elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            z0 = torch.cat(
                [ self.condition_tensor, z0[:, mask_node_index, :]], dim=1
            )
            
        with torch.enable_grad():
            z0 = z0.requires_grad_()
            energy = target_function(z0, t).sum()
            grad = autograd.grad(energy, z0)[0]
        grad_norm_x = grad[:, : , :self.n_dims].norm(dim=[1, 2])
        grad_norm_h = grad[:, : , self.n_dims:].norm(dim=[1, 2])
        grad_x = x_weight*D**(1/2)/t[0].item() * grad[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
        grad_h = h_weight*D**(1/2)/t[0].item() * grad[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

        grad = torch.cat(
            [
                remove_mean_with_mask(grad_x, node_mask),
                grad_h,
            ],
            dim=2,
        )
        
        mu = mu - gg_scale * grad

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            if s > t_critical:
                # Fix the reference part as conditioning
                zs = torch.cat(
                    [self.condition_tensor, zs[:, mask_bools, :]], dim=1
                )
            else:
                # Fix just the atomm types of reference part as conditioning
                zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
        elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            zs = torch.cat(
                [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
            )

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )

        if n_backward > 0:  
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.requires_grad_()
                    energy_r = target_function(zs,  s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]      
                    grad_r = grad_r.nan_to_num(0.0)
                    
                    
                    grad_norm_x = grad_r[:, : , :self.n_dims].norm(dim=[1, 2])
                    grad_norm_h = grad_r[:, : , self.n_dims:].norm(dim=[1, 2])
                    grad_r_x = x_weight* D**(1/2)/t[0].item() * grad_r[:, : , :self.n_dims] / (grad_norm_x + 1e-6) * sigma_t
                    grad_r_h = h_weight* D**(1/2)/t[0].item() * grad_r[:, : , self.n_dims:] / (grad_norm_h + 1e-6) * sigma_t

                    grad_r = torch.cat(
                        [
                            remove_mean_with_mask(grad_r_x, node_mask),
                            grad_r_h,
                        ],
                        dim=2,
                    )
        
                    reverse_grad_zs_norm = grad_r.norm(dim=[1, 2])
                    clip_coef_reverse_zs = max_norm / (reverse_grad_zs_norm + 1e-6)
                    clip_coef_clamped_reverse_zs = torch.clamp(clip_coef_reverse_zs, max=1)
                    grad_r *= clip_coef_clamped_reverse_zs[:, None, None]
                    zs = zs - gg_scale * grad_r 

                    # In the event the reference tensor is provided
                    if self.condition_tensor is not None and "outpaint" in structure_guidance:
                        if s > t_critical:
                            # Fix the reference part as conditioning
                            zs = torch.cat(
                                [self.condition_tensor, zs[:, mask_bools, :]], dim=1
                            )
                        else:
                            # Fix just the atomm types of reference part as conditioning
                            zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]      
                    elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
                        zs = torch.cat(
                            [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
                        )    
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        
        if self.condition_tensor  is not None and "outpaint" in structure_guidance:
            self.condition_tensor = zs[:, ~mask_bools, :]
        elif self.condition_tensor  is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            self.condition_tensor = zs[:, ~mask_node_index, :]  
        return zs


    def sample_p_zs_given_zt_ssgd(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        condition_tensor,
        condition_component,
        guidance_strength=0.0,
        fix_noise=False,
    ):
        

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_s = self.phi(zt, t, node_mask, edge_mask, context)

        if torch.all(eps_s == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_s[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_s
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # integrate structure signal
        gamma_t = self.inflate_batch_array(self.gamma(t), condition_tensor)
        gamma_s = self.inflate_batch_array(self.gamma(s), condition_tensor)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, condition_tensor)
        sigma_t = self.sigma(gamma_t, condition_tensor)
        
        alpha_s = self.alpha(gamma_s, condition_tensor)
        sigma_s = self.sigma(gamma_s, condition_tensor)

        if condition_component == "xh":
              
            eps_s_ref = self.sample_combined_position_feature_noise(
            n_samples=condition_tensor.size(0), n_nodes=condition_tensor.size(1), node_mask=node_mask)
        elif condition_component == "x":
            eps_s_ref = sample_center_gravity_zero_gaussian_with_mask(
                size=(condition_tensor.size(0), condition_tensor.size(1), self.n_dims),
                device=node_mask.device,
                node_mask=node_mask,
            )
        elif condition_component == "h":
            eps_s_ref = sample_gaussian_with_mask(
                size=(condition_tensor.size(0), condition_tensor.size(1), self.in_node_nf),
                device=node_mask.device,
                node_mask=node_mask,
            )
        
        z_cond = alpha_s * condition_tensor + sigma_s * eps_s_ref
 
        zs_f = zs
    
        
        if t > guidance_strength:
            if condition_component == "xh":
                zs_f = sigma_s * zs + alpha_s * z_cond 
            elif condition_component == "x":
                zs_f[:, :, : self.n_dims] = sigma_s * zs[:, :, : self.n_dims] + alpha_s * (z_cond[:, :, : self.n_dims])
            elif condition_component == "h":
                zs_f[:, :, self.n_dims:] = sigma_s * zs[:, :, self.n_dims:] + alpha_s * z_cond
            
            # Project down to avoid numerical runaway of the center of gravity.
            zs_f = torch.cat(
                [
                    remove_mean_with_mask(zs_f[:, :, : self.n_dims], node_mask),
                    zs_f[:, :, self.n_dims :],
                ],
                dim=2,
            )
        return zs_f
    
    def sample_p_zs_given_zt_ip(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        mask_node_index,
        connector_dicts,
        t_critical_1=0.8,
        t_critical_2=0.3,
        d_threshold_f=1.8,
        w_b=2, 
        all_frozen=False,
        use_covalent_radii=True,
        scale_factor=1.1,
        fix_noise=False,
    ):
        """Performs inpainting on a molecular structure.

        This method fills in a missing part of a molecule, defined by `mask_node_index`,
        while keeping the rest of the structure fixed. It uses geometric constraints to
        ensure the generated part is chemically plausible.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            mask_node_index (torch.Tensor): Indices of the nodes to be inpainted.
            connector_dicts (dict): A dictionary defining the connector atoms and their degrees.
            t_critical_1 (float, optional): Critical timestep for applying the first set of geometric constraints. Defaults to 0.8.
            t_critical_2 (float, optional): Critical timestep for applying the second set of geometric constraints. Defaults to 0.3.
            d_threshold_f (float, optional): Distance threshold for finding close points. Defaults to 1.8.
            w_b (int, optional): Weight for the bond term in the geometric constraints. Defaults to 2.
            all_frozen (bool, optional): If True, all atoms in the reference fragment are frozen. Defaults to False.
            use_covalent_radii (bool, optional): If True, uses covalent radii for distance checks. Defaults to True.
            scale_factor (float, optional): Scale factor for covalent radii. Defaults to 1.1.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.

        Returns:
            torch.Tensor: The inpainted sample zs.
        """
        
    
        connector_indices = torch.tensor(list(connector_dicts.keys()), device=zt.device, dtype=torch.long)
        connector_degrees = torch.tensor([value[0] for value in connector_dicts.values()], device=zt.device, dtype=torch.long)

        
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        # assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        # assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        if len(mask_node_index) > 0:
            zs = torch.cat(
                [ self.condition_tensor, zs[:, mask_node_index, :]], dim=1
            )
            zs_pos = zs[:, :, : self.n_dims]
            zs_pos = zs_pos.squeeze(0)   
    
            zs_new_pos = zs_pos[mask_node_index]
            condition_pos = self.condition_tensor[:, :, : self.n_dims].squeeze(0)

            if use_covalent_radii:
                condition_charge = self.condition_tensor[:, :, -1].squeeze(0)*self.norm_values[2]
                condition_charge = torch.round(condition_charge).long()
                condition_charge[condition_charge >= 118] = 118
                condition_charge[condition_charge <= 1] = 1
                
                zs_charge = zs[:, :, -1].squeeze(0)*self.norm_values[2]
                zs_charge = torch.round(zs_charge).long()
                zs_charge[zs_charge >= 118] = 118
                zs_charge[zs_charge <= 1] = 1
                
                d_threshold_c = self.COV_R[condition_charge][connector_indices]*2*scale_factor
            else:
                condition_charge = None
                d_threshold_c = [1.5]*len(connector_degrees)
                zs_charge = None
                
            if s[0].item() < t_critical_1:

                _, zl_corr = find_close_points_torch_and_push_op2(
                            condition_pos,
                            zs_new_pos, 
                            connector_indices=connector_indices,    
                            d_threshold_f=d_threshold_f,
                            # d_fixed_move=0.1,
                            w_b=w_b,
                            all_frozen=all_frozen,
                            z_ref=condition_charge,
                            z_tgt=zs_charge,
                            scale_factor=scale_factor
                            )   
                
                zs[:, mask_node_index, :self.n_dims] = zl_corr.unsqueeze(0) 

                zs_pos = zs[:, :, : self.n_dims]
                zs_pos = zs_pos.squeeze(0)   
                zs_new_pos = zs_pos[mask_node_index] 
                
                if s[0].item() > t_critical_2:      
                    zl_corr2 = enforce_min_nodes_per_connector(
                        condition_pos,
                        zs_new_pos,
                        connector_indices=connector_indices,
                        N=connector_degrees,
                        d_threshold_c=d_threshold_c,
                        )
                    zs[:, mask_node_index, :self.n_dims] = zl_corr2.unsqueeze(0) 
            
                    zs_pos = zs[:, :, : self.n_dims]
                    zs_pos = zs_pos.squeeze(0)   
                    zs_new_pos = zs_pos[mask_node_index]
                    _, z1_corr2 = ensure_intact(
                        condition_pos,
                        zs_new_pos,
                        connector_indices=connector_indices,    
                        d_threshold=1.7,   
                        # d_fixed_move=0.1
                    )
                    zs[:, mask_node_index, :self.n_dims] = z1_corr2.unsqueeze(0) 
        
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        
        if len(mask_node_index) > 0:
            self.condition_tensor = zs[:, ~mask_node_index, :]   
    

        return zs
    
    def sample_p_zs_given_zt_op(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        mask_bools,
        connector_dicts,
        t_critical_1=0.8,
        t_critical_2=0.4,
        d_threshold_f=1.8,
        w_b=2, 
        all_frozen=False,
        use_covalent_radii=True,
        scale_factor=1.1,
        fix_noise=False,
    ):
        """Performs outpainting on a molecular structure.

        This method extends a given molecular fragment by generating new atoms and connecting
        them to the fragment at specified connector points.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            mask_bools (torch.Tensor): A boolean mask indicating which atoms are part of the generated structure.
            connector_dicts (dict): A dictionary defining the connector atoms and their degrees.
            t_critical_1 (float, optional): Critical timestep for applying the first set of geometric constraints. Defaults to 0.8.
            t_critical_2 (float, optional): Critical timestep for applying the second set of geometric constraints. Defaults to 0.4.
            d_threshold_f (float, optional): Distance threshold for finding close points. Defaults to 1.8.
            w_b (int, optional): Weight for the bond term in the geometric constraints. Defaults to 2.
            all_frozen (bool, optional): If True, all atoms in the reference fragment are frozen. Defaults to False.
            use_covalent_radii (bool, optional): If True, uses covalent radii for distance checks. Defaults to True.
            scale_factor (float, optional): Scale factor for covalent radii. Defaults to 1.1.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.

        Returns:
            torch.Tensor: The outpainted sample zs.
        """
        # natom_ref = condition_tensor.size(1)
        connector_indices = torch.tensor(list(connector_dicts.keys()), device=zt.device, dtype=torch.long)
        connector_degrees = torch.tensor([value[0] for value in connector_dicts.values()], device=zt.device, dtype=torch.long)
        natom_ref = (~mask_bools).sum()
        
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_s = self.phi(zt, t, node_mask, edge_mask, context)
        if torch.all(eps_s == 0):
            raise ValueError("NaN in eps_t, stop sampling.")
        
        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_s[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_s
        )
     

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        zs = torch.cat(
            [self.condition_tensor,
             zs[:, mask_bools, : ]], dim=1
        )

        zs_pos = zs[:, :, : self.n_dims]
        zs_pos = zs_pos.squeeze(0)   
  

        if s[0].item() < t_critical_1:
            zs_new_pos = zs_pos[natom_ref:]
            condition_pos = self.condition_tensor[:, :, : self.n_dims].squeeze(0)
            
        
            if use_covalent_radii:
                condition_charge = self.condition_tensor[:, :, -1].squeeze(0)*self.norm_values[2]
                condition_charge = torch.round(condition_charge).long()
                condition_charge[condition_charge >= 118] = 118
                condition_charge[condition_charge <= 1] = 1
                
                zs_charge = zs[:, :, -1].squeeze(0)*self.norm_values[2]
                zs_charge = torch.round(zs_charge).long()
                zs_charge[zs_charge >= 118] = 118
                zs_charge[zs_charge <= 1] = 1
                
                d_threshold_c = self.COV_R[condition_charge][connector_indices]*2*scale_factor
            else:
                condition_charge = None
                d_threshold_c = [1.5]*len(connector_degrees)
                zs_charge = None

            _, zl_corr = find_close_points_torch_and_push_op2(
                        condition_pos,
                        zs_new_pos, 
                        connector_indices=connector_indices,    
                        d_threshold_f=d_threshold_f,
                        # d_fixed_move=0.1,
                        w_b=w_b,
                        all_frozen=all_frozen,
                        z_ref=condition_charge,
                        z_tgt=zs_charge,
                        scale_factor=scale_factor
                        )   
            
        
            zs[:, mask_bools, :self.n_dims] = zl_corr.unsqueeze(0) 

            zs_pos = zs[:, :, : self.n_dims]
            zs_pos = zs_pos.squeeze(0)   
            zs_new_pos = zs_pos[natom_ref:]   
            
            if s[0].item() > t_critical_2:      
                zl_corr2 = enforce_min_nodes_per_connector(
                    condition_pos,
                    zs_new_pos,
                    connector_indices=connector_indices,
                    N=connector_degrees,
                    d_threshold_c=d_threshold_c,
                    )
                zs[:, mask_bools, :self.n_dims] = zl_corr2.unsqueeze(0) 
        
                zs_pos = zs[:, :, : self.n_dims]
                zs_pos = zs_pos.squeeze(0)   
                zs_new_pos = zs_pos[natom_ref:] 
                _, z1_corr2 = ensure_intact(
                    condition_pos,
                    zs_new_pos,
                    connector_indices=connector_indices,    
                    d_threshold=1.7,   
                    # d_fixed_move=0.1
                )
                zs[:, mask_bools, :self.n_dims] = z1_corr2.unsqueeze(0) 
  
                        
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        
        self.condition_tensor = zs[:, ~mask_bools, :]   
        
        return zs
        

    def sample_p_zs_given_zt_op_ft(
        self,
        s,
        t,
        zt,
        reference_tensor,  # (1, n_nodes_ref, n_features)
        node_mask,
        edge_mask,
        context,
        t_critical=0.05,
        fix_noise=False,
    ):
        """Performs outpainting with fine-tuning to a reference structure.

        This method guides the outpainting process by fine-tuning the generated structure
        to a given reference tensor.

        Args:
            s (torch.Tensor): The current timestep, s.
            t (torch.Tensor): The next timestep, t.
            zt (torch.Tensor): The noisy data at timestep t.
            reference_tensor (torch.Tensor): The reference structure to fine-tune to.
            node_mask (torch.Tensor): Mask for nodes in the graph.
            edge_mask (torch.Tensor): Mask for edges in the graph.
            context (torch.Tensor): The conditional information for guidance.
            t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to 0.05.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.

        Returns:
            torch.Tensor: The outpainted and fine-tuned sample zs.
        """
        natom_ref = reference_tensor.size(1)
        mask_bools = [False] * natom_ref + [True] * (
            node_mask.size(1) - natom_ref
        )
        mask_bools = torch.tensor(mask_bools, device=zt.device, dtype=torch.bool)

        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)
        )

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_s = self.phi(zt, t, node_mask, edge_mask, context)
        if torch.all(eps_s == 0):
            raise ValueError("NaN in eps_t, stop sampling.")

        # Compute mu for p(zs | zt).
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_s[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_s
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        if s > t_critical:
            # Fix the reference part as conditioning
            zs = torch.cat(
                [self.condition_tensor, zs[:, mask_bools, :]], dim=1
            )
        else:
            # Fix just the atomm types of reference part as conditioning
            zs[:, :natom_ref, 3:] = self.condition_tensor[:, :natom_ref, self.n_dims:]       

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )  
        self.condition_tensor = zs[:, ~mask_bools, :]
        return zs
        
        
    def sample_combined_position_feature_noise(
        self,
        n_samples,
        n_nodes,
        node_mask,
        std=1.0,
    ):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_ddim_step(self, zt, s, t, node_mask, edge_mask, context, eta=0.0):
        """
        Deterministic DDIM step: z_s ← z_t - f(eps_theta), no stochasticity when eta=0.
        """
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)
        
        # Predict noise
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        alpha_s = self.alpha(gamma_s, target_tensor=zt)
        alpha_t = self.alpha(gamma_t, target_tensor=zt)

        # Compute predicted z0
        z0_pred = (zt - eps_t * sigma_t) / alpha_t

        # this is sigma_t in the DDIM paper (NOT TO BE CONFUSED WITH sigma_t from noise scheduler)
        sigma_t_ddim = sigma_s/sigma_t*torch.sqrt(1-(alpha_t/alpha_s)**2)
        # DDIM deterministic step
        z_s = alpha_s * z0_pred + (torch.sqrt(sigma_s**2 - sigma_t_ddim **2)) * eps_t

        # Optional noise injection (eta > 0 reintroduces stochasticity)
        if eta > 0.0:
            eps = self.sample_combined_position_feature_noise(
                n_samples=zt.size(0), n_nodes=zt.size(1), node_mask=node_mask
            )
            z_s += sigma_t_ddim * eps * eta

        # Keep center of gravity zero
        z_s = torch.cat(
            [remove_mean_with_mask(z_s[:, :, : self.n_dims], node_mask), z_s[:, :, self.n_dims:]],
            dim=2,
        )
        return z_s
      
    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples, 
        n_nodes, 
        node_mask,
        edge_mask,
        context, 
        fix_noise=False,
        n_steps=None,
        eta=0.0,
        n_frames=0, # since there are fewer steps, we save all frames
    ):
        """
        DDIM sampling: deterministic, optionally fewer steps.
        eta=0.0 corresponds to the original DDIM (no noise),
        eta > 0 adds noise (reverts toward DDPM-like).
        """
        n_steps = n_steps or self.T  # use full steps unless specified
        step_indices = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long, device=node_mask.device)
        # Initialize with Gaussian noise
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )
        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if n_frames > 0:
            
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
         
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            s_saves = torch.linspace(0, self.T, 
                                         steps=n_frames, device=z.device).long() 
            
        write_index = 0
        # Time steps for DDIM
        with tqdm(total=n_steps-1, dynamic_ncols=True, unit="steps") as pbar:
            for _, i in enumerate(reversed(range(1, len(step_indices)))):
                
                t = step_indices[i]
                s = step_indices[i-1]

                t_array = torch.full((n_samples, 1), t / self.T, device=z.device)
                s_array = torch.full((n_samples, 1), s / self.T, device=z.device)

                z = self.sample_ddim_step(z, s_array, t_array, node_mask, edge_mask, context, eta)
                if n_frames > 0:
                    if s in s_saves:
                        chain[write_index] = self.unnormalize_z(z, node_mask)
                        write_index += 1
                    
                pbar.update(1)
        # Final decode
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        # Center correction
        assert_mean_zero_with_mask(x, node_mask)
        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f"[DDIM] Warning cog drift: {max_cog:.3f}, projecting positions.")
            x = remove_mean_with_mask(x, node_mask)
        
        if n_frames > 0:
            # Save the chain of frames
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain[-1] = xh   
            chain_flat = chain.view(n_frames, n_samples, z.size(1), z.size(2)-len(self.extra_norm_values)) 
        else:
            chain_flat = None
        return x, h, chain_flat
 

    @torch.no_grad()
    def sample(
        self, 
        n_samples, 
        n_nodes,
        node_mask, 
        edge_mask, 
        context, 
        condition_tensor=None,
        condition_mode=None,
        mask_node_index=torch.tensor([[]]), # For Inpainting
        denoising_strength=0.0, # For Inpainting
        noise_initial_mask=False, # For Inpainting
        connector_dicts={}, # For outpainting
        t_start=1.0,
        t_critical_1=0.8,
        t_critical_2=0.5,
        d_threshold_f=1.4,
        w_b=2,
        all_frozen=False,
        use_covalent_radii=True,
        scale_factor=1.1,
        fix_noise=False,
        n_frames=0,
        t_retry=180,
        n_retrys=0,

    ):
        
        """
        Draw samples from the generative model.
        """
        t_int_start = self.T * t_start
        
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )
            
        self.COV_R = torch.tensor(covalent_radii, dtype=torch.float32, device=z.device)
               
        # assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        
        if condition_mode:
            condition_alg, condition_component = condition_mode.split("_")
        else:
            condition_alg = None
            condition_component = None
        
        if condition_component == "xh" or condition_component == "x":
            n_node_cond = condition_tensor.size(1)
            unmasked_node_indices = [i for i in range(n_node_cond) if i not in mask_node_index]
            node_mask_cond = torch.ones((n_samples, n_node_cond,1), device=z.device)  
            if condition_alg != "outpaintft":
                condition_tensor[:,:, : self.n_dims] = remove_mean_with_mask(condition_tensor[:,:, : self.n_dims], node_mask_cond)


        if n_frames > 0:
            
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
         
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            if n_retrys > 0:
                n_frames_retry = int(n_frames*t_retry/self.T)
                try:
                    chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                except RuntimeError:
                    chain_retry = [torch.zeros((n_frames_retry+1,) + z_size, device=z.device) for _ in range(n_retrys)]
            if condition_alg == "inpaint":
                s_saves = torch.linspace(0, int(denoising_strength * self.T), 
                                         steps=n_frames, device=z.device).long() 
            else:
                s_saves = torch.linspace(0, self.T, 
                                         steps=n_frames, device=z.device).long() 
        
        if condition_alg == "inpaint":
            n_node_cond = condition_tensor.size(1)
            d = torch.full((n_samples, 1), fill_value=denoising_strength, device=z.device)
            
            gamma_d = self.inflate_batch_array(self.gamma(d), condition_tensor)
        
            # Compute alpha_t and sigma_t from gamma.
            alpha_d = self.alpha(gamma_d, condition_tensor)
            sigma_d = self.sigma(gamma_d, condition_tensor)

            #-----------------------partial inpainting------------------------------------#
            if mask_node_index.size(1) > 0:
                # Reorder condition_tensor such that connector nodes are the first nodes
                connector_mask = torch.zeros(condition_tensor.size(1), dtype=torch.bool, device=z.device)
                connector_mask[mask_node_index] = True  
                non_connector_mask = ~connector_mask          
                reordered_condition_tensor = torch.cat([condition_tensor[:, non_connector_mask, :], 
                                                        condition_tensor[:, connector_mask, :]], dim=1)
                condition_tensor = reordered_condition_tensor

                x_ref = condition_tensor[:, :, : self.n_dims].squeeze(0)
                h_int_ref = condition_tensor[:, :, -1].squeeze(0) * self.norm_values[2]

                mol_graph = create_pyg_graph(x_ref, h_int_ref, r=3)
                mol_graph = correct_edges(mol_graph, scale_factor=1.2)

                connector_idxs = []
                connector_n_bonds = []
                mask_node_index_corr = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), condition_tensor.size(1), device=z.device, dtype=torch.long)
                unmasked_node_indices = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), device=z.device).tolist()
                for node in unmasked_node_indices:
                    adj_nodes = mol_graph.edge_index[1][mol_graph.edge_index[0] == node]
                    adj_nodes = adj_nodes.long().to(z.device)
                    if any(torch.isin(adj_nodes, mask_node_index_corr)):
                        connector_idxs.append(node)
                        connector_n_bonds.append(len(adj_nodes))  

                connector_idxs = list(set(connector_idxs))
                connector_indices = torch.tensor(connector_idxs, device=z.device, dtype=torch.long)
                connector_degrees = torch.tensor(connector_n_bonds, device=z.device, dtype=torch.long)      
                connector_dicts = {j.item(): [connector_degrees[i].item()] for i, j in enumerate(connector_indices)}   
                
                node_mask_m = torch.ones_like(mask_node_index, device=z.device)
                node_mask_m = node_mask_m.unsqueeze(-1)
                
                node_mask_um = torch.ones((n_samples, n_node_cond-mask_node_index.size(1)), device=z.device)  
                node_mask_um = node_mask_um.unsqueeze(-1)
    
                mask_node_bool = [True if i in mask_node_index else False for i in range(n_node_cond)]
                mask_node_bool = torch.tensor(mask_node_bool, device=z.device)
                mask_node_bool_corr = torch.cat([torch.zeros(n_node_cond - mask_node_bool.sum(), device=z.device),
                                                 torch.ones(mask_node_bool.sum(), device=z.device)]).bool()
                
                xh_unmasked = condition_tensor[:,~mask_node_bool_corr, :] # not to be denoise

                # noise the masked nodes
                if noise_initial_mask:
                    eps_s_ref = self.sample_combined_position_feature_noise(
                        n_samples=mask_node_index.size(0), n_nodes=mask_node_index.size(1), node_mask=node_mask_m)
                    z = alpha_d * condition_tensor[:, mask_node_bool_corr, :] + sigma_d * eps_s_ref # to be denoised
                    
                    z = torch.cat([xh_unmasked, z], dim=1)
                    z = torch.cat(  
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask_cond),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )      
       
                    zs_pos = z[:, :, : self.n_dims]
                    zs_pos = zs_pos.squeeze(0)  
                    zs_new_pos = zs_pos[mask_node_bool_corr]
                    condition_pos = xh_unmasked[:, :, : self.n_dims].squeeze(0)
                    condition_charge = None
                    zs_charge = None
  
                    _, zl_corr = find_close_points_torch_and_push_op2(
                                condition_pos,
                                zs_new_pos, 
                                connector_indices=torch.tensor(list(connector_dicts.keys()), device=z.device, dtype=torch.long),    
                                d_threshold_f=d_threshold_f,
                                w_b=w_b,
                                all_frozen=all_frozen,
                                z_ref=condition_charge,
                                z_tgt=zs_charge,
                                scale_factor=scale_factor
                                )

                    z[:, mask_node_bool_corr, :self.n_dims] = zl_corr.unsqueeze(0) 

                
                
                # keep the masked nodes clean
                else:
                    z = torch.cat([xh_unmasked, condition_tensor[:, mask_node_bool_corr, :]], dim=1)

            #-----------------------all inpainting------------------------------------#
            else:
                if condition_component == "xh":
                    eps_s_ref = self.sample_combined_position_feature_noise(
                    n_samples=condition_tensor.size(0), n_nodes=condition_tensor.size(1), node_mask=node_mask[:, :condition_tensor.size(1), :])
                    z = alpha_d * condition_tensor + sigma_d * eps_s_ref
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )
                    xh_unmasked = None
                elif condition_component == "x":
                    eps_s_ref = sample_center_gravity_zero_gaussian_with_mask(
                        size=(condition_tensor.size(0), condition_tensor.size(1), self.n_dims),
                        device=node_mask.device,
                        node_mask=node_mask[:, :condition_tensor.size(1), :],
                    )
                    z = torch.cat([
                        alpha_d * condition_tensor[:, :, : self.n_dims] + sigma_d * eps_s_ref,
                        condition_tensor[:, :, self.n_dims:]
                    ], dim=2)
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,
                    )
                elif condition_component == "h":
                    eps_s_ref = sample_gaussian_with_mask(
                        size=(condition_tensor.size(0), condition_tensor.size(1), self.in_node_nf),
                        device=node_mask.device,
                        node_mask=node_mask[:, :condition_tensor.size(1), :],
                    )

                    z = torch.cat([
                        condition_tensor[:, :, : self.n_dims],
                        alpha_d * condition_tensor[:, :, self.n_dims:] + sigma_d * eps_s_ref
                    ], dim=2)
                    z = torch.cat(
                        [
                            remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                            z[:, :, self.n_dims :],
                        ],
                        dim=2,)
                mask_node_bool_corr = []
                connector_dicts = {}
            #-----------------------extended inpainting-----------------------------------
            n_node_extend = n_nodes - n_node_cond
  
            if n_node_extend > 0:
                node_mask_um = torch.ones((n_samples, z.size(1)-mask_node_index.size(1)), device=z.device)  
                z_extend = self.sample_combined_position_feature_noise(
                    mask_node_index.size(0), n_node_extend, 
                    torch.ones((mask_node_index.size(0), n_node_extend,1), device=z.device)  )
                z = torch.cat([z, z_extend], dim=1) 

                zs_pos = z[:, :, : self.n_dims]
                zs_pos = zs_pos.squeeze(0)  

                zs_extend_pos = zs_pos[-n_node_extend:]
                condition_pos = zs_pos[:-n_node_extend]
                
                if mask_node_index.size(1) > 0:
                    n_connector = mask_node_index.size(1)
                    connector_indices = torch.arange(n_connector , condition_pos.size(0), device=z.device)
                else:
                    n_connector = condition_pos.size(0)
                    connector_indices = torch.arange(0 , n_connector, device=z.device)
                
                if use_covalent_radii:
                    condition_charge = z[:, n_node_extend:, -1].squeeze(0)*self.norm_values[2]
                    condition_charge = torch.round(condition_charge).long()
                    condition_charge[condition_charge >= 118] = 118
                    condition_charge[condition_charge <= 1] = 1
                    condition_charge = condition_charge.unsqueeze(-1)
                    zs_charge = z[:, :n_node_extend, -1].squeeze(0)*self.norm_values[2]
                    zs_charge = torch.round(zs_charge).long()
                    zs_charge[zs_charge >= 118] = 118
                    zs_charge[zs_charge <= 1] = 1
                    zs_charge = zs_charge.unsqueeze(-1)
                    
                else:
                    condition_charge = None
                    zs_charge = None
                _, zl_corr = find_close_points_torch_and_push_op2(
                            condition_pos,
                            zs_extend_pos, 
                            connector_indices=connector_indices,    
                            d_threshold_f=1.4,
                            w_b=2,
                            all_frozen=False,
                            z_ref=condition_charge,
                            z_tgt=zs_charge,
                            scale_factor=scale_factor
                            )  
                z[:, -n_node_extend:, :self.n_dims] = zl_corr.unsqueeze(0)    
                if mask_node_index.size(1) > 0:
                    mask_node_bool_corr = torch.cat([mask_node_bool_corr, 
                                                     torch.ones(n_node_extend, device=z.device)]).bool() 
                    # connector_dicts = {key + n_node_extend: value for key, value in connector_dicts.items()}
            if mask_node_index.size(1) > 0:
                self.condition_tensor = z[:, ~mask_node_bool_corr, :] 
        
        elif condition_alg in ["outpaint", "outpaintft"]:
                        
            natom_ref = condition_tensor.size(1)
            
            if any(idx > natom_ref - 1 for idx in connector_dicts.keys()):
                raise ValueError("connector_indices is out of bound")
            if not connector_dicts and condition_alg == "outpaint":
                raise ValueError("connector_indices is empty")
            
            z = torch.cat([condition_tensor, z], dim=1) 
            n_nodes = z.size(1)
            node_mask = torch.ones((n_samples, n_nodes), device=z.device)
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=z.device).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(z.size(0) * n_nodes * n_nodes, 1).to(self.device)  
            node_mask = node_mask.unsqueeze(-1)
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
            
            if n_frames > 0:
                chain = torch.zeros((n_frames,) + z_size, device=z.device)
                if n_retrys > 0:
                    n_frames_retry = int(n_frames*t_retry/self.T)
                    try:
                        chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                    except RuntimeError:
                        chain_retry = [torch.zeros((n_frames_retry+1,) + z_size, device=z.device) for _ in range(n_retrys)]
                    
            condition_tensor = z[:, :natom_ref, :]   
            mask_node_bool_corr = torch.cat([torch.zeros(natom_ref, device=z.device), 
                                                torch.ones(n_nodes - natom_ref, device=z.device)]).bool()
            self.condition_tensor = z[:, ~mask_node_bool_corr, :]     

        if context is not None:
            context = (
                context[0,0].repeat(1, z.size(1), 1).to(self.device)
            )*node_mask
        
        
        z = torch.cat(  
            [
                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask),
                z[:, :, self.n_dims :],
            ],
            dim=2,
        )       
        if self.condition_tensor is not None:
            self.condition_tensor = z[:, ~mask_node_bool_corr, :]
    

        to_inspect_chain = False        
        if n_retrys > 0:
            snap_shot = {}
        write_index = 0
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        with tqdm(total=self.T, dynamic_ncols=True, unit="steps") as pbar:
            for i, s in enumerate(reversed(range(0, self.T))):
                s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                t_array = s_array + 1
                s_array = s_array / self.T
                t_array = t_array / self.T

                if s > t_int_start:
                    pbar.update(1)
                    if n_frames > 0:
                        if s in s_saves:
                            chain[write_index] = self.unnormalize_z(z, node_mask)
                            write_index += 1
                            if n_retrys > 0:
                                info_dict = {
                                    "z": z,
                                    "condition_tensor": self.condition_tensor,
                                    "mask_node_bool_corr": mask_node_bool_corr
                                }
                                snap_shot[s] = info_dict
                    continue


                try:
                    if condition_alg == "ssgd":
                        z = self.sample_p_zs_given_zt_ssgd(
                        s_array,
                        t_array,
                        z, 
                        node_mask, 
                        edge_mask, 
                        context,
                        condition_tensor,
                        condition_component,
                        fix_noise=fix_noise,
                    )
                    elif condition_alg == "inpaint":
                            
                        if s/self.T < denoising_strength: 
                            z = self.sample_p_zs_given_zt_ip(
                                s_array,
                                t_array, 
                                z,  
                                node_mask, 
                                edge_mask, 
                                context,
                                mask_node_bool_corr,
                                connector_dicts,
                                t_critical_1,
                                t_critical_2,
                                d_threshold_f,
                                w_b,
                                all_frozen,
                                use_covalent_radii,
                                scale_factor,
                                fix_noise=fix_noise
                                ) 
                        else:
                            z = z
                    elif condition_alg == "outpaint":
                    
                        z = self.sample_p_zs_given_zt_op(
                            s_array, t_array, z, 
                            node_mask, edge_mask, 
                            context,
                            mask_node_bool_corr,
                            connector_dicts=connector_dicts,
                            fix_noise=fix_noise,
                            # t_critical=0.4
                            )
                    elif condition_alg == "outpaintft": 
                        z = self.sample_p_zs_given_zt_op_ft(
                            s_array,
                            t_array, 
                            z, 
                            condition_tensor, 
                            node_mask, edge_mask, 
                            context,
                            fix_noise=fix_noise
                            )
                    else:
                        z = self.sample_p_zs_given_zt(
                        s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise
                        )
                except ValueError as e:
                    if s < self.T/100: # arbitary threshold
                        to_inspect_chain = True
                    else:
                        t_retry = int(self.T*0.9)
                        if n_retrys > 0:
                            n_frames_retry = int(n_frames*t_retry/self.T)
                            chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                        # raise e  or just fuck this
                    break
                if n_frames > 0:
                    if s in s_saves:
                        chain[write_index] = self.unnormalize_z(z, node_mask)
                        write_index += 1
                        if n_retrys > 0:
                            info_dict = {
                                "z": z,
                                "condition_tensor": self.condition_tensor,
                                "mask_node_bool_corr": mask_node_bool_corr
                            }
                            snap_shot[s] = info_dict
                    
                pbar.update(1)
    
        N_check = 5
        N_COMPONENT_MAX = 10
        N_DEGREE_MAX = 8

        try:
            # Finally sample p(x, h | z_0).
            x, h = self.sample_p_xh_given_z0(
                    z, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        except ValueError as e:
            to_inspect_chain = True       
        # 2 ---------------------------------------------------------------------
        # Address overshoot cases, there maybe a good xh already before 0    
        #TODO extra nf!
        if to_inspect_chain and n_frames > 0:
            n_chain = chain.size(0)
            x_chain = chain[:, :, :, 0:3]
            x_chain = x_chain.squeeze(1)
            one_hot_chain = chain[:, :, :, 3:-1]
            one_hot_chain = one_hot_chain.squeeze(1)
            one_hot_chain = F.one_hot(
                torch.argmax(one_hot_chain, dim=2), num_classes=self.num_classes - len(self.extra_norm_values)
            )
            charges_chain = torch.round(chain[:, :, :, -1]).long().squeeze(1)
            charges_chain = torch.abs(charges_chain)*self.norm_values[2]

            good_chain = False
            for i in range(N_check):
                try:
                    is_connected, num_components, n_degrees = check_quality(x_chain[-1-i], charges_chain[-1-i])
                except Exception as e:
                    # assume errors with check_quality is associated with bad mols
                    is_connected = False
                    num_components = 1000000
                    n_degrees = [0]*x_chain.size(1)      
                    
                good_chain = (num_components < N_COMPONENT_MAX) and all([n_degree < N_DEGREE_MAX for n_degree in n_degrees])
                if good_chain:
                    h = {}
                    x  = x_chain[-1-i].unsqueeze(0)
                    h["integer"] = charges_chain[-1-i].unsqueeze(0)
                    h["integer"] = h["integer"].unsqueeze(-1)
                    h["categorical"] = one_hot_chain[-1-i].unsqueeze(0)

                    break
            # cannot find a good xh, last resort
            if not(good_chain):
                z = chain[-1-i]
                x, h = self.sample_p_xh_given_z0(
                    z, node_mask, edge_mask, context, fix_noise=fix_noise
                )
        
        # assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean_with_mask(x, node_mask)

        # 3 ---------------------------------------------------------------------
        try:
            is_connected, num_components, n_degrees = check_quality(x.squeeze(0), h["integer"].squeeze(0))
        except:
            # assume errors with check_quality is associated with bad mols
            is_connected = False
            num_components = 1000000
            n_degrees = [0]*x.size(1)
    
        # 3.5 ---------------------------------------------------------------------
        #TODO extra nf!
        # In the event EGNN fails at p(xh | z0), we retry the chain
        bad_xh = (num_components > N_COMPONENT_MAX) or any([n_degree > N_DEGREE_MAX for n_degree in n_degrees])
        if bad_xh and n_retrys > 0:
            x = chain[-2, :, :, 0:3]
            h["integer"] = chain[-2, :, :, -1:].long()  
            h["categorical"] = chain[-2, :, :, 3:-1]
            try:
                is_connected, num_components, n_degrees = check_quality(x.squeeze(0), h["integer"].squeeze(0))
            except:
                # assume errors with check_quality is associated with bad mols
                is_connected = False
                num_components = 1000000
                n_degrees = [0]*x.size(1)
                
        if condition_alg in ["outpaint", "inpaint"]:
            match_n_degree_connector = True
            for key, value in connector_dicts.items():
                if not(n_degrees[key] in value):
                    match_n_degree_connector = False
                    break
            to_retry = not(is_connected and match_n_degree_connector)
        else:
            to_retry = not(is_connected)

        if n_frames > 0: 
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain[-1] = xh     
        else:
            chain_flat = None
            chain = None
        
        
        # 4 ---------------------------------------------------------------------
        if to_retry and n_retrys > 0 and n_frames > 0:
            n_retry = 0
            while (n_retry < n_retrys) and to_retry:
                closest_t = min(snap_shot.keys(), key=lambda x: abs(x - t_retry))
                z = snap_shot[closest_t]["z"]
                self.condition_tensor = snap_shot[closest_t]["condition_tensor"]
                mask_node_bool_corr = snap_shot[closest_t]["mask_node_bool_corr"]

                write_index = 0
                with tqdm(total=t_retry, dynamic_ncols=True, unit="steps") as pbar:
                    for i, s in enumerate(reversed(range(0, t_retry))):
                        s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                        t_array = s_array + 1
                        s_array = s_array / self.T
                        t_array = t_array / self.T                
                        try:
                            if condition_alg == "inpaint":
                                    
                                if s/self.T < denoising_strength: 
                                    z = self.sample_p_zs_given_zt_ip(
                                        s_array,
                                        t_array, 
                                        z,  
                                        node_mask, 
                                        edge_mask, 
                                        context,
                                        mask_node_bool_corr,
                                        connector_dicts,
                                        t_critical_1,
                                        t_critical_2,
                                        d_threshold_f,
                                        w_b,
                                        all_frozen,
                                        use_covalent_radii,
                                        scale_factor,
                                        fix_noise=fix_noise
                                        ) 
                            elif condition_alg == "outpaint":
                            
                                z = self.sample_p_zs_given_zt_op(
                                    s_array, t_array, z, 
                                    node_mask, edge_mask, 
                                    context,
                                    mask_node_bool_corr,
                                    connector_dicts=connector_dicts,
                                    fix_noise=fix_noise,
                                    # t_critical=0.4
                                    )
                            elif condition_alg == "outpaintft": 
                                z = self.sample_p_zs_given_zt_op_ft(
                                    s_array,
                                    t_array, 
                                    z, 
                                    condition_tensor, 
                                    node_mask, edge_mask, 
                                    context,
                                    fix_noise=fix_noise
                                    )
                            else:
                                z = self.sample_p_zs_given_zt(
                                s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise
                                )
                        except ValueError as e:
                            if s < self.T/100: # arbitary threshold
                                to_inspect_chain = True
                            else:
                                t_retry = int(self.T*0.9)
                                if n_retrys > 0:
                                    n_frames_retry = int(n_frames*t_retry/self.T)
                                    chain_retry = [torch.zeros((n_frames_retry,) + z_size, device=z.device) for _ in range(n_retrys)]
                                # raise e  or just fuck this
                            break
                                    
                        if n_frames > 0 and n_retrys > 0:
                            if s in s_saves:
                                # print(f"write_index: {write_index}")
                                
                                if write_index < n_frames_retry:
                                    chain_retry[n_retry][write_index] = self.unnormalize_z(z, node_mask)
                                    write_index += 1
                        pbar.update(1)               
                    # Finally sample p(x, h | z_0).

                #TODO extra nf!
                if to_inspect_chain:
                    n_chain = chain_retry[n_retry].size(0)
                    x_chain = chain_retry[n_retry][:, :, :, 0:3]
                    x_chain = x_chain.squeeze(1)
                    one_hot_chain = chain_retry[n_retry][:, :, :, 3:-1]
                    one_hot_chain = one_hot_chain.squeeze(1)
                    one_hot_chain = F.one_hot(
                        torch.argmax(one_hot_chain, dim=2), num_classes=self.num_classes - len(self.extra_norm_values)
                    )
                    charges_chain = torch.round(chain_retry[n_retry][:, :, :, -1:]).long().squeeze(1)

                    good_chain = False

                
                    for i in range(N_check):
                        try:
                            is_connected, num_components, n_degrees = check_quality(x_chain[-1-i], charges_chain[-1-i])
                        except Exception as e:
                            # assume errors with check_quality is associated with bad mols
                            is_connected = False
                            num_components = 1000000
                            n_degrees = [0]*x_chain.size(1)      
                        good_chain = (num_components < N_COMPONENT_MAX) and all([n_degree < N_DEGREE_MAX for n_degree in n_degrees])

                        if good_chain:
                            h_retry = {}
                            x_retry  = x_chain[-1-i].unsqueeze(0)
                            h_retry["integer"] = charges_chain[-1-i].unsqueeze(0)
                            h_retry["categorical"] = one_hot_chain[-1-i].unsqueeze(0)
                            break
                    # cannot find a good xh, last resort
                    if not(good_chain):
                        z = chain[-1-i]
                        x_retry, h_retry = self.sample_p_xh_given_z0(
                            z, node_mask, edge_mask, context, fix_noise=fix_noise
                        )
                            
                else:
                    # Finally sample p(x, h | z_0).
                    x_retry, h_retry = self.sample_p_xh_given_z0(
                        z, node_mask, edge_mask, context, fix_noise=fix_noise
                    )
                    
                max_cog = torch.sum(x_retry, dim=1, keepdim=True).abs().max().item()
                if max_cog > 5e-2:
                    print(
                        f"Warning cog drift with error {max_cog:.3f}. Projecting "
                        f"the positions down."
                    )
                    x_retry = remove_mean_with_mask(x_retry, node_mask)
                    
                assert_mean_zero_with_mask(x_retry, node_mask)
                try:
                    is_connected, num_components, n_degrees = check_quality(x_retry.squeeze(0), h_retry["integer"].squeeze(0))
                except:
                    # assume errors with check_quality is associated with bad mols
                    is_connected = False
                    num_components = 1000000
                    n_degrees = [0]*x_retry.size(1)
                    print("Error in check_quality")

   
                bad_xh = (num_components > N_COMPONENT_MAX) or any([n_degree > N_DEGREE_MAX for n_degree in n_degrees])
                #TODO extra nf!
                # In the event EGNN fails at p(xh | z0), we retry the chain
                if bad_xh and n_frames > 0:
                    x_retry = chain_retry[n_retry][-2, :, :, 0:3]
                    h_retry["integer"] = chain_retry[n_retry][-2, :, :, -1:].long()  
                    h_retry["categorical"] = chain_retry[n_retry][-2, :, :, 3:-1]
                    try:
                        is_connected, num_components, n_degrees = check_quality(x_retry.squeeze(0), h_retry["integer"].squeeze(0))
                    except:
                        # assume errors with check_quality is associated with bad mols
                        is_connected = False
                        num_components = 1000000
                        n_degrees = [0]*x.size(1)
                
                if condition_alg in ["outpaint", "inpaint"]:
                    match_n_degree_connector = True
                    for key, value in connector_dicts.items():
                        if not(n_degrees[key] in value):
                            match_n_degree_connector = False
                            break
                    to_retry = not(is_connected and match_n_degree_connector)
                else:
                    to_retry = not(is_connected)

                max_cog = torch.sum(x_retry, dim=1, keepdim=True).abs().max().item()
                if max_cog > 5e-2:
                    print(
                        f"Warning cog drift with error {max_cog:.3f}. Projecting "
                        f"the positions down."
                    )
                    x_retry = remove_mean_with_mask(x_retry, node_mask)
                    
                assert_mean_zero_with_mask(x_retry, node_mask)
  
                    
                x = torch.cat([x, x_retry], dim=0)
                h["categorical"] = torch.cat([h["categorical"], h_retry["categorical"]], dim=0) 
                h["integer"] = torch.cat([h["integer"], h_retry["integer"]], dim=0)

                n_retry+=1
                

            if n_frames > 0: 
                xh_retry = torch.cat([x_retry, h_retry["categorical"], h_retry["integer"]], dim=2)
                chain_retry[n_retry][-1] = xh_retry                          
            else:
                chain_flat = None
                chain = None
        else:
            if n_frames:
                chain_flat = chain.view(n_frames, n_samples , z.size(1), z.size(2)-len(self.extra_norm_values)) 
  
        return x, h, chain_flat

    

    @torch.no_grad()
    def sample_chain(
        self,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context
            )

            assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    #TODO we have to preprocess the data for structure-guidnace here
    #TODO for the hybrid, intend to use with fine-tuned adapted model
    #TODO 3D constaints will not be employed
    @torch.no_grad()
    def sample_guidance(
        self,
        n_samples,
        target_function,
        node_mask,
        edge_mask,
        context=None,
        context_negative=None,
        gg_scale=1,
        cfg_scale=1,
        max_norm=10,
        fix_noise=False,
        std=1.0,
        scheduler=None,
        guidance_at=1,
        guidance_stop=0,
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        condition_tensor=None,
        condition_mode=None,
        mask_node_index=torch.tensor([[]]), # For Inpainting
        denoising_strength=0.0, # For Inpainting
        noise_initial_mask=False, # For Inpainting
        t_start=1.0,
        t_critical=0,
        n_frames=0,
        debug=False,
    ):
        """
        Guided sampling from the generative model.

        Parameters:
        - n_samples (int): Number of samples to generate.
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. The higher the value, the better.
        - node_mask (Tensor): Mask indicating valid nodes.
        - edge_mask (Tensor): Mask indicating valid edges.
        - context (Tensor): Conditonal properties. Default is None.   
        - context_negative (Tensor): Conditonal properties. Default is None.
        - gg_scale (float): Scale factor for gradient guidance. Default is 1.0.
        - cfg_scale (float): Scale factor for classifier-free guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - n_frames (int, optional): Number of frames for sampling. Defaults to 0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.
        - condition_tensor (torch.Tensor, optional): Tensor for conditional guidance. Defaults to None.
        - condition_mode (str, optional): Mode for conditional guidance. Defaults to None.
        - mask_node_index (torch.Tensor, optional): Indices of nodes to be inpainted. Defaults to an empty tensor.
        - denoising_strength (float, optional): Strength of denoising for inpainting
        - noise_initial_mask (bool, optional): Whether to noise the initial masked region. Defaults to False.
        - t_start (float, optional): Timestep to start applying guidance. Defaults to 1.0.
        - t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.


        Returns:
        Tuple[Tensor, Tensor]: Sampled positions and features.
        """
        debug = False
        
        n_nodes = node_mask.size(1)
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask, std)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask, std
            )

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        opt_infos = {
            "grad_norms": [],
            "mu0_norm": [],
            "mu1_norm": [],
            "energies": [],
            "z_norm": [],
            "values": [],
        }
        
        guidance_at = int(guidance_at * self.T)
        guidance_stop = int(guidance_stop * self.T)

        scale_deploy = gg_scale
    
        # Structure guidance preprocesssing
        if condition_mode:
            condition_alg, condition_component = condition_mode.split("_")
        else:
            condition_alg = None
            condition_component = None
        
        if condition_component == "xh" or condition_component == "x":
            n_node_cond = condition_tensor.size(1)
            unmasked_node_indices = [i for i in range(n_node_cond) if i not in mask_node_index]
            node_mask_cond = torch.ones((n_samples, n_node_cond,1), device=z.device)  
            if "ft" not in condition_alg:
                condition_tensor[:,:, : self.n_dims] = remove_mean_with_mask(condition_tensor[:,:, : self.n_dims], node_mask_cond)

    
        # for tracking intermediate frames
        if n_frames > 0:
            
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
         
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            if condition_alg == "inpaint":
                s_saves = torch.linspace(0, int(denoising_strength * self.T), 
                                         steps=n_frames, device=z.device).long() 
            else:
                s_saves = torch.linspace(0, self.T, 
                                         steps=n_frames, device=z.device).long() 
        
        
        if condition_alg:
            if  "inpaint" in condition_alg:
                t_int_start = int(self.T * denoising_strength)
                n_node_cond = condition_tensor.size(1)
                d = torch.full((n_samples, 1), fill_value=denoising_strength, device=z.device)
                
                gamma_d = self.inflate_batch_array(self.gamma(d), condition_tensor)
            
                # Compute alpha_t and sigma_t from gamma.
                alpha_d = self.alpha(gamma_d, condition_tensor)
                sigma_d = self.sigma(gamma_d, condition_tensor)

                #-----------------------partial inpainting------------------------------------#
                if mask_node_index.size(1) > 0:
                    #
                    # CONSTANTS
                    scale_factor = 1.1
                    all_frozen = False
                    use_covalent_radii = True
                    w_b = 2
                    d_threshold_f = 1.4
                    # Reorder condition_tensor such that connector nodes are the first nodes
                    connector_mask = torch.zeros(condition_tensor.size(1), dtype=torch.bool, device=z.device)
                    connector_mask[mask_node_index] = True  
                    non_connector_mask = ~connector_mask          
                    reordered_condition_tensor = torch.cat([condition_tensor[:, non_connector_mask, :], 
                                                            condition_tensor[:, connector_mask, :]], dim=1)
                    condition_tensor = reordered_condition_tensor

                    x_ref = condition_tensor[:, :, : self.n_dims].squeeze(0)
                    h_int_ref = condition_tensor[:, :, -1].squeeze(0) * self.norm_values[2]

                    mol_graph = create_pyg_graph(x_ref, h_int_ref, r=3)
                    mol_graph = correct_edges(mol_graph, scale_factor=scale_factor)

                    connector_idxs = []
                    connector_n_bonds = []
                    mask_node_index_corr = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), condition_tensor.size(1), device=z.device, dtype=torch.long)
                    unmasked_node_indices = torch.arange(condition_tensor.size(1) - mask_node_index.size(1), device=z.device).tolist()
                    for node in unmasked_node_indices:
                        adj_nodes = mol_graph.edge_index[1][mol_graph.edge_index[0] == node]
                        adj_nodes = adj_nodes.long().to(z.device)
                        if any(torch.isin(adj_nodes, mask_node_index_corr)):
                            connector_idxs.append(node)
                            connector_n_bonds.append(len(adj_nodes))  

                    connector_idxs = list(set(connector_idxs))
                    connector_indices = torch.tensor(connector_idxs, device=z.device, dtype=torch.long)
                    connector_degrees = torch.tensor(connector_n_bonds, device=z.device, dtype=torch.long)      
                    connector_dicts = {j.item(): [connector_degrees[i].item()] for i, j in enumerate(connector_indices)}   
                    
                    node_mask_m = torch.ones_like(mask_node_index, device=z.device)
                    node_mask_m = node_mask_m.unsqueeze(-1)
                    
                    node_mask_um = torch.ones((n_samples, n_node_cond-mask_node_index.size(1)), device=z.device)  
                    node_mask_um = node_mask_um.unsqueeze(-1)
        
                    mask_node_bool = [True if i in mask_node_index else False for i in range(n_node_cond)]
                    mask_node_bool = torch.tensor(mask_node_bool, device=z.device)
                    mask_node_bool_corr = torch.cat([torch.zeros(n_node_cond - mask_node_bool.sum(), device=z.device),
                                                    torch.ones(mask_node_bool.sum(), device=z.device)]).bool()
                    
                    xh_unmasked = condition_tensor[:,~mask_node_bool_corr, :] # not to be denoise

                    # noise the masked nodes
                    if noise_initial_mask:
                        eps_s_ref = self.sample_combined_position_feature_noise(
                            n_samples=mask_node_index.size(0), n_nodes=mask_node_index.size(1), node_mask=node_mask_m)
                        z = alpha_d * condition_tensor[:, mask_node_bool_corr, :] + sigma_d * eps_s_ref # to be denoised
                        
                        z = torch.cat([xh_unmasked, z], dim=1)
                        z = torch.cat(  
                            [
                                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask_cond),
                                z[:, :, self.n_dims :],
                            ],
                            dim=2,
                        )      
        
                        zs_pos = z[:, :, : self.n_dims]
                        zs_pos = zs_pos.squeeze(0)  
                        zs_new_pos = zs_pos[mask_node_bool_corr]
                        condition_pos = xh_unmasked[:, :, : self.n_dims].squeeze(0)
                        condition_charge = None
                        zs_charge = None
    
                        _, zl_corr = find_close_points_torch_and_push_op2(
                                    condition_pos,
                                    zs_new_pos, 
                                    connector_indices=torch.tensor(list(connector_dicts.keys()), device=z.device, dtype=torch.long),    
                                    d_threshold_f=d_threshold_f,
                                    w_b=w_b,
                                    all_frozen=all_frozen,
                                    z_ref=condition_charge,
                                    z_tgt=zs_charge,
                                    scale_factor=scale_factor
                                    )

                        z[:, mask_node_bool_corr, :self.n_dims] = zl_corr.unsqueeze(0) 

                    # keep the masked nodes clean
                    else:
                        z = torch.cat([xh_unmasked, condition_tensor[:, mask_node_bool_corr, :]], dim=1)

                #-----------------------all inpainting------------------------------------#
                else:
                 
                    if condition_component == "xh":
                        eps_s_ref = self.sample_combined_position_feature_noise(
                        n_samples=condition_tensor.size(0), n_nodes=condition_tensor.size(1), node_mask=node_mask[:, :condition_tensor.size(1), :])
                        if noise_initial_mask:
                            z = alpha_d * condition_tensor + sigma_d * eps_s_ref
                        else:
                            z = condition_tensor

                        z = torch.cat(
                            [
                                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                                z[:, :, self.n_dims :],
                            ],
                            dim=2,
                        )
                        xh_unmasked = None
                    elif condition_component == "x":
                        eps_s_ref = sample_center_gravity_zero_gaussian_with_mask(
                            size=(condition_tensor.size(0), condition_tensor.size(1), self.n_dims),
                            device=node_mask.device,
                            node_mask=node_mask[:, :condition_tensor.size(1), :],
                        )
                        z = torch.cat([
                            alpha_d * condition_tensor[:, :, : self.n_dims] + sigma_d * eps_s_ref,
                            condition_tensor[:, :, self.n_dims:]
                        ], dim=2)
                        z = torch.cat(
                            [
                                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                                z[:, :, self.n_dims :],
                            ],
                            dim=2,
                        )
                    elif condition_component == "h":
                        eps_s_ref = sample_gaussian_with_mask(
                            size=(condition_tensor.size(0), condition_tensor.size(1), self.in_node_nf),
                            device=node_mask.device,
                            node_mask=node_mask[:, :condition_tensor.size(1), :],
                        )

                        z = torch.cat([
                            condition_tensor[:, :, : self.n_dims],
                            alpha_d * condition_tensor[:, :, self.n_dims:] + sigma_d * eps_s_ref
                        ], dim=2)
                        z = torch.cat(
                            [
                                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask[:, :condition_tensor.size(1), :]),
                                z[:, :, self.n_dims :],
                            ],
                            dim=2,)
                    mask_node_bool_corr = []
                    connector_dicts = {}
                #-----------------------extended inpainting-----------------------------------
                n_node_extend = n_nodes - n_node_cond
    
                if n_node_extend > 0:
                    node_mask_um = torch.ones((n_samples, z.size(1)-mask_node_index.size(1)), device=z.device)  
                    z_extend = self.sample_combined_position_feature_noise(
                        mask_node_index.size(0), n_node_extend, 
                        torch.ones((mask_node_index.size(0), n_node_extend,1), device=z.device)  )
                    z = torch.cat([z, z_extend], dim=1) 

                    zs_pos = z[:, :, : self.n_dims]
                    zs_pos = zs_pos.squeeze(0)  

                    zs_extend_pos = zs_pos[-n_node_extend:]
                    condition_pos = zs_pos[:-n_node_extend]
                    
                    if mask_node_index.size(1) > 0:
                        n_connector = mask_node_index.size(1)
                        connector_indices = torch.arange(n_connector , condition_pos.size(0), device=z.device)
                    else:
                        n_connector = condition_pos.size(0)
                        connector_indices = torch.arange(0 , n_connector, device=z.device)
                    
                    condition_charge = z[:, n_node_extend:, -1].squeeze(0)*self.norm_values[2]
                    condition_charge = torch.round(condition_charge).long()
                    condition_charge[condition_charge >= 118] = 118
                    condition_charge[condition_charge <= 1] = 1
                    condition_charge = condition_charge.unsqueeze(-1)
                    zs_charge = z[:, :n_node_extend, -1].squeeze(0)*self.norm_values[2]
                    zs_charge = torch.round(zs_charge).long()
                    zs_charge[zs_charge >= 118] = 118
                    zs_charge[zs_charge <= 1] = 1
                    zs_charge = zs_charge.unsqueeze(-1)
   
                    _, zl_corr = find_close_points_torch_and_push_op2(
                                condition_pos,
                                zs_extend_pos, 
                                connector_indices=connector_indices,    
                                d_threshold_f=1.4,
                                w_b=2,
                                all_frozen=False,
                                z_ref=condition_charge,
                                z_tgt=zs_charge,
                                scale_factor=1.2
                                )  
                    z[:, -n_node_extend:, :self.n_dims] = zl_corr.unsqueeze(0)    
                    if mask_node_index.size(1) > 0:
                        mask_node_bool_corr = torch.cat([mask_node_bool_corr, 
                                                        torch.ones(n_node_extend, device=z.device)]).bool() 
                        # connector_dicts = {key + n_node_extend: value for key, value in connector_dicts.items()}
                if mask_node_index.size(1) > 0:
                    self.condition_tensor = z[:, ~mask_node_bool_corr, :] 
            
            elif "outpaint" in condition_alg:
                t_int_start = self.T * t_start
                natom_ref = condition_tensor.size(1)
                
                if any(idx > natom_ref - 1 for idx in connector_dicts.keys()):
                    raise ValueError("connector_indices is out of bound")
                if not connector_dicts and condition_alg == "outpaint":
                    raise ValueError("connector_indices is empty")
                
                z = torch.cat([condition_tensor, z], dim=1) 
                n_nodes = z.size(1)
                node_mask = torch.ones((n_samples, n_nodes), device=z.device)
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=z.device).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(z.size(0) * n_nodes * n_nodes, 1).to(self.device)  
                node_mask = node_mask.unsqueeze(-1)
                condition_tensor = z[:, :natom_ref, :]   
                mask_node_bool_corr = torch.cat([torch.zeros(natom_ref, device=z.device), 
                                                    torch.ones(n_nodes - natom_ref, device=z.device)]).bool()
                self.condition_tensor = z[:, ~mask_node_bool_corr, :]               
                
        else:
            t_int_start = self.T
            mask_node_bool_corr = None
        z = torch.cat(  
            [
                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask),
                z[:, :, self.n_dims :],
            ],
            dim=2,
        )       
        
        if self.condition_tensor is not None:
            self.condition_tensor = z[:, ~mask_node_bool_corr, :]
            
        write_index = 0
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        with tqdm(total=self.T, dynamic_ncols=True, unit="steps") as pbar:
            for s in reversed(range(0, self.T)):
                
                s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
                t_array = s_array + 1
                s_array = s_array / self.T
                t_array = t_array / self.T
    
                if s > t_int_start:
                    pbar.update(1)
                    if n_frames > 0:
                        if s in s_saves:
                            chain[write_index] = self.unnormalize_z(z, node_mask)
                            write_index += 1
                    continue
                else:
                    
                    if s > guidance_at or s < guidance_stop:
                        if s == guidance_stop:
                            logger.info("Guidance stops at ", s)
                        z = self.sample_p_zs_given_zt(
                            s_array, t_array, z, node_mask, edge_mask, context=context, fix_noise=fix_noise
                        )

                    else:
                        if s == guidance_at:
                            logger.info("Guidance starts at ", s)
                        #NOTE this does not support structure-guidance yet
                        if guidance_ver == 0:
                            z, opt_info = self.sample_p_zs_given_zt_guidance_v0(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                target_function,
                                scale_deploy,
                                fix_noise=fix_noise,
                                max_norm=max_norm,
                                n_backward=n_backwards,
                            )
                        #NOTE this does not support structure-guidance yet
                        elif guidance_ver == 1:
                            z, opt_info = self.sample_p_zs_given_zt_guidance_v1(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                target_function,
                                scale_deploy,
                                fix_noise=fix_noise,
                                max_norm=max_norm,
                                n_backward=n_backwards,
                            )
                        elif guidance_ver == 2:
                            z, opt_info = self.sample_p_zs_given_zt_guidance_v2(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                target_function,
                                scale_deploy,
                                fix_noise=fix_noise,
                                max_norm=max_norm,
                                n_backward=n_backwards,
                                t_critical=t_critical,
                                structure_guidance=condition_alg,
                                mask_node_index=mask_node_bool_corr
                            )
                        elif guidance_ver == "cfg":
                            z = self.sample_p_zs_given_zt_guidance_cfg(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                cfg_scale,
                                fix_noise=fix_noise,
                                context_negative=context_negative,
                                t_critical=t_critical,
                                structure_guidance=condition_alg,
                                mask_node_index=mask_node_bool_corr
                            )
                        elif guidance_ver == "cfg_gg":
                            z = self.sample_p_zs_given_zt_guidance_cfg_gg(
                                s_array,
                                t_array,
                                z,
                                node_mask,
                                edge_mask,
                                context,
                                target_function,
                                cfg_scale,
                                scale_deploy,
                                max_norm=max_norm,
                                n_backward=n_backwards,
                                h_weight=h_weight,
                                x_weight=x_weight,
                                fix_noise=fix_noise,
                                t_critical=t_critical,
                                structure_guidance=condition_alg,
                                mask_node_index=mask_node_bool_corr
                            )
                        else:
                            raise ValueError(f"Unknown guidance version {guidance_ver}.")

                        if scheduler is not None:
                            scale_deploy = scheduler.step(
                                opt_info["energies"], scale_deploy
                            )

                    if debug:
                        opt_infos["grad_norms"].append(
                            opt_info["grad_norms"].detach().cpu().item()
                        )
                        opt_infos["mu0_norm"].append(
                            opt_info["mu0_norm"].detach().cpu().item()
                        )
                        opt_infos["mu1_norm"].append(
                            opt_info["mu1_norm"].detach().cpu().item()
                        )
                        opt_infos["z_norm"].append(opt_info["z_norm"].detach().cpu().item())
                        opt_infos["energies"].append(
                            opt_info["energies"].detach().cpu().item()
                        )
                    if n_frames > 0:
                        if s in s_saves:
                            chain[write_index] = self.unnormalize_z(z, node_mask)
                            write_index += 1
                    pbar.update(1)

        if debug:
            for key, value in opt_infos.items():
                filename = f"{key}.txt"
                np.savetxt(filename, value)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context=context, fix_noise=fix_noise
        )

        x, h = self.check_sanity_xh(
            x, h, node_mask, edge_mask, context, chain
        )
        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean_with_mask(x, node_mask)

        if n_frames > 0: 
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain[-1] = xh     
            chain_flat = chain.view(n_frames, n_samples , z.size(1), z.size(2)-len(self.extra_norm_values)) 
            
        else:
            chain_flat = None
        
        return x, h, chain_flat

    @torch.no_grad()
    def sample_chain(
        self,
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        keep_frames=None,
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context
            )

            assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def check_sanity_xh(
        self,
        x: torch.Tensor,
        h: dict,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor,
        chain: torch.Tensor,
        n_frame_look_back: int = 4,    
    ):
        """
        Performs a sanity check on the generated molecule (x, h) by computing its loss.
        If the loss is infinite, it attempts to find a "clean" molecule from previous frames
        in the sampling chain by iterating backward and re-sampling x and h from z_0.
        This helps to recover from potential numerical instabilities during sampling.

        Args:
            x (torch.Tensor): Current atom positions (B, N, 3).
            h (dict): Dictionary of atom features {'categorical', 'integer', 'extra'}.
            node_mask (torch.Tensor): Mask indicating valid nodes (B, N, 1).
            edge_mask (torch.Tensor): Mask indicating valid edges (B, N, N).
            context (torch.Tensor): Context tensor for the diffusion model.
            chain (torch.Tensor): A tensor containing intermediate states (z) from the sampling process.
                                  Shape: (num_frames, B, N, D_latent).
            n_frame_look_back (int, optional): Number of frames to look back in the chain
                                               if the current molecule is "unclean". Defaults to 4.

        Returns:
            Tuple[torch.Tensor, dict]: A tuple containing the (potentially corrected) atom positions (x)
                                      and atom features (h) that result in a finite loss.
        """
        self.loss_type = "vlb"
        self.context_mask_rate = 0

        loss, _ = self.compute_loss(
            x,
            h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
            reference_indices=None,
            t0_always=True)
                        
        n_core = self.in_node_nf - self.ndim_extra - 1
        start  = self.n_dims
        mid    = start + n_core
        hint_i = mid 

        if torch.isinf(loss).any():
            for i in range(1, n_frame_look_back+1):
                
                x_i = chain[-i-1][:, :, : self.n_dims]
                
                if self.ndim_extra > 0:
                    h_cat_i = chain[-i-1][:, :, start:mid]
                    h_int_i = chain[-i-1][:, :, hint_i:hint_i+1]
                    h_extra_i = chain[-i-1][:, :, -self.ndim_extra:]
                    h_i = {
                        "categorical": h_cat_i,
                        "integer": h_int_i,
                        "extra": h_extra_i,
                    }
                else:
                    h_cat_i = chain[-i-1][:, :, start: -1]
                    h_int_i = chain[-i-1][:, :, -1:]
                    h_i = {
                        "categorical": h_cat_i,
                        "integer": h_int_i,
                        "extra": None,
                    }
                
                zx_i, zh_i, _ = self.normalize(x_i, h_i, node_mask)
                
                # Reconstruct z_i from normalized components
                if self.ndim_extra > 0:
                    zh_i_combined = torch.cat([zh_i["categorical"], zh_i["integer"], zh_i["extra"]], dim=2)
                else:
                    zh_i_combined = torch.cat([zh_i["categorical"], zh_i["integer"]], dim=2)

                z_i = torch.cat([zx_i, zh_i_combined], dim=2)

                x, h = self.sample_p_xh_given_z0(
            z_i, node_mask, edge_mask, context=context, fix_noise=False
        )
                loss, _ = self.compute_loss(       
                        x,
                        h,
                        node_mask=node_mask,
                        edge_mask=edge_mask,
                        context=context,
                        reference_indices=None,
                        t0_always=True)
                if not torch.isinf(loss).any():
                    # if self.debug:
                    #     logger.info(f"Found clean molecule at chain frame {i}")
                    break
        # else:
            # if self.debug:
            #     logger.info("The generated molecule is clean.")

        return x, h


    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {"log_SNR_max": log_SNR_max.item(), "log_SNR_min": log_SNR_min.item()}
        print(info)

        return info


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#%% Noise models
class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if "cosine" in noise_schedule:
            splits = noise_schedule.split("_")
            power = float(splits[1]) if len(splits) > 1 else 1
            alphas2 = cosine_beta_schedule(timesteps, s=precision, raise_to_power=power)
        elif "polynomial" in noise_schedule:
            splits = noise_schedule.split("_")
            power = float(splits[1]) if len(splits) > 1 else 2
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        elif "issnr" in noise_schedule:
            splits = noise_schedule.split("_")
            eta = float(splits[1]) if len(splits) > 1 else 1
            kappa = float(splits[2]) if len(splits) > 2 else 2
            alphas2 = vp_issnr_schedule(
                timesteps,
                eta=eta,
                kappa=kappa,
            )
        elif "smld" in noise_schedule:
            splits = noise_schedule.split("_")
            sigma_min = float(splits[1]) if len(splits) > 1 else 1
            sigma_max = float(splits[2]) if len(splits) > 2 else 50
            alphas2 = vp_smld_schedule(
                timesteps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        else:
            raise ValueError(
                f"Unknown noise schedule {noise_schedule}. Supported: polynomial, issnr, cosine."
            )


        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2


        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""

    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.0]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.0]))
        # self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print("Gamma schedule:")
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0
            )

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def vp_issnr_schedule(timesteps: int, eta: float = 1.0, kappa: float = 2.0,
                      tmin: float = 1e-2, tmax: float = 1 - 1e-2):
    """
    Variance-Preserving Inverse Sigmoid SNR (VP-ISSNR) schedule based on:
    τ²(t) = 1 (i.e., total variance is constant)
    γ²(t) = ((1 / (t * (tmax - tmin) + tmin)) - 1)^(2η) * exp(2κ)

    Args:
        timesteps (int): number of steps (T)
        eta (float): controls steepness of SNR decay
        kappa (float): controls the offset of SNR
        tmin (float): min effective time (0 < tmin < 1)
        tmax (float): max effective time (0 < tmax < 1)

    Returns:
        alphas2s (np.ndarray): shape (timesteps + 1, len(nu_array))
    """
    steps = timesteps + 1
    t = np.linspace(0, 1, steps)

    gamma2 = ((1 / (t * (tmax - tmin) + tmin)) - 1) ** (2 * eta) * np.exp(2 * kappa)

    # From TV/SNR framework:
    # α²(t) = τ²(t) * γ²(t) / (1 + γ²(t))  with τ²(t) = 1 (VP)
    alphas2 = gamma2 / (1 + gamma2)

    # Optional: clipping if needed
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    alphas2 = alphas2**2

    return alphas2

def vp_smld_schedule(timesteps: int, sigma_min: float = 0.01, sigma_max: float = 50.0):
    """
    Variance-Preserving SMLD schedule.
    """
    steps = timesteps + 1
    t = np.linspace(0, 1, steps)

    gamma2 = sigma_min**(-2) * (sigma_min / sigma_max) ** (2 * t)

    alphas2 = gamma2 / (1 + gamma2)
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    return alphas2


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.0):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


#%% Utils

def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)

def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5)


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
            - 0.5
        )
        * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    # assert len(q_sigma.size()) == 1
    # assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )

class DistributionNodes:
    def __init__(self, histogram):
        """

        histogram of number of nodes in the dataset

        for example:
        {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
        15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
        8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1}

        - There are 3393 molecules in the dataset that each have 22 atoms.
        - There are 13025 molecules that each have 17 atoms.
        - and so on.
        """
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        prob = np.array(prob)
        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = []
        for i in batch_n_nodes:
            try:
                idcs.append(self.keys[i.item()])
            except Exception:
                continue
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:

    def __init__(
        self,
        num_atoms,
        props,
        property_names,
        num_bins=1000,
        normalizer=None,
    ):
        """Props and propety_names order must the same"""

        self.num_bins = num_bins
        self.distributions = {}
        self.property_names = property_names

        for i, prop in enumerate(property_names):
            self.distributions[prop] = {}
            self._create_prob_dist(
                num_atoms,
                props[i],
                self.distributions[prop],
            )

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]["mean"]
        mad = self.normalizer[prop]["mad"]
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.property_names:
                
            try:
                dist = self.distributions[prop][n_nodes]
                
            except KeyError:
                logger.info(f"No exact distribution for n_nodes={n_nodes}, property='{prop}'. Interpolating...")
        
                available_nodes = sorted(self.distributions[prop].keys())
                
                if not available_nodes:
                    raise ValueError(f"No distributions available for property '{prop}'")
                if n_nodes < available_nodes[0]:
                    # If requested n_nodes is smaller than any available, use the smallest.
                    closest_node = available_nodes[0]
                    dist = self.distributions[prop][closest_node]
                elif n_nodes > available_nodes[-1]:
                    closest_node = available_nodes[-1]
                    dist = self.distributions[prop][closest_node]
                else:
                    lower_node = max(k for k in available_nodes if k < n_nodes)
                    upper_node = min(k for k in available_nodes if k > n_nodes)
                
                    lower_dist = self.distributions[prop][lower_node]
                    upper_dist = self.distributions[prop][upper_node]
                    
                    lower_probs_tensor = lower_dist['probs'].probs
                    upper_probs_tensor = upper_dist['probs'].probs
                    avg_probs = (lower_probs_tensor + upper_probs_tensor) / 2.0
                    
                    avg_params = [(p1 + p2) / 2.0 for p1, p2 in zip(lower_dist['params'], upper_dist['params'])]
                    
                    dist = {
                        'probs': Categorical(probs=avg_probs),
                        'params': avg_params
                    }

            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            # val = self.normalize_tensor(val, prop) # Disable Normalize the value for now
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


