"""Training objective of ``EnVariationalDiffusion``: loss terms, KL, likelihood of the final denoising step, ``forward``."""

from MolecularDiffusion.utils import assert_mean_zero_with_mask
from MolecularDiffusion.utils import remove_mean_pyG
from MolecularDiffusion.utils import remove_mean_with_mask
from copy import deepcopy
from torch_scatter import scatter_add
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
import torch
from MolecularDiffusion.modules.models.edm._en_diffusion_math import (
    _n_int,
    cdf_standard_gaussian,
    gaussian_KL,
    gaussian_KL_for_dimension,
    split_z,
    sum_except_batch,
)


class EnDiffusionLossMixin:
    """Training objective of ``EnVariationalDiffusion``: loss terms, KL, likelihood of the final denoising step, ``forward``.

    Plain mixin: no parameters or state of its own, so the ``state_dict`` of
    ``EnVariationalDiffusion`` is unchanged. Methods use attributes set in its ``__init__``.
    """

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
            # when no extra dims, hcat runs up to the integer block, which is
            # empty for models without an atomic-number channel
            _, eps_t_hcat, eps_t_hint = split_z(self, eps_t)
            _, eps_hcat, eps_hint = split_z(self, eps)

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
            n_cat = self.num_classes
            z_h_cat = z_t[:, :, :n_cat]
            z_h_int = z_t[:, :, n_cat : n_cat + _n_int(self)]

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
            n_cat       = self.num_classes
            z_h_int     = z_h[:, n_cat : n_cat + _n_int(self)]
            z_h_cat     = z_h[:, :n_cat]
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
        reference_freeze_mode: str = "all",
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
            if reference_freeze_mode == "all":
                z_t[:, reference_indices] = xh[:, reference_indices]
                eps[:, reference_indices] = 0.0
                z_t = torch.cat([
                    remove_mean_with_mask(z_t[:, :, :self.n_dims], node_mask),
                    z_t[:, :, self.n_dims:]
                ], dim=2)
            elif reference_freeze_mode == "features_only":
                z_t[:, reference_indices, self.n_dims:] = xh[
                    :, reference_indices, self.n_dims:
                ]
                eps[:, reference_indices, self.n_dims:] = 0.0
            else:
                raise ValueError(
                    "reference_freeze_mode must be one of {'all', 'features_only'}, "
                    f"got {reference_freeze_mode!r}"
                )

        assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        # ---------------------------------------------------------------------
        # 3. CONTEXT MASKING
        # ---------------------------------------------------------------------
        if context is not None:
            has_adapter = hasattr(self.dynamics, 'n_adapter_context') and self.dynamics.n_adapter_context > 0
            has_concat = hasattr(self.dynamics, 'n_concat_context') and self.dynamics.n_concat_context > 0
            adapter_only = has_adapter and not has_concat

            if adapter_only:
                # Pure adapter: drop entire context with probability
                if self.context_mask_rate > 0:
                    context_masked = None if torch.randint(1, int(1 / self.context_mask_rate), ()).item() == 1 else context.clone() * node_mask
                else:
                    context_masked = context.clone() * node_mask
            else:
                # Concat-only or hybrid: mask concat columns with mask_value
                mask = torch.rand(context.size(0), device=context.device) < self.context_mask_rate
                context_masked = context.clone()
                if has_adapter:
                    # Hybrid: zero adapter columns, set concat columns to mask_value
                    adapter_idx = self.dynamics.adapter_indices
                    concat_idx = self.dynamics.concat_indices
                    context_masked[mask, :][:, adapter_idx] = 0.0
                    context_masked[mask, :][:, concat_idx] = self.mask_value
                else:
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
            if reference_indices is not None:
                if reference_freeze_mode == "all":
                    z_0[:, reference_indices] = xh[:, reference_indices]
                    eps_0[:, reference_indices] = 0.0
                    z_0 = torch.cat([
                        remove_mean_with_mask(z_0[:, :, :self.n_dims], node_mask),
                        z_0[:, :, self.n_dims:]
                    ], dim=2)
                elif reference_freeze_mode == "features_only":
                    z_0[:, reference_indices, self.n_dims:] = xh[
                        :, reference_indices, self.n_dims:
                    ]
                    eps_0[:, reference_indices, self.n_dims:] = 0.0

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
    def compute_loss_pyG(
        self,
        mol_graph,
        context,
        t0_always,
        reference_indices=None,
        reference_freeze_mode="all",
    ):
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

        if context is not None:
            has_adapter = hasattr(self.dynamics, 'n_adapter_context') and self.dynamics.n_adapter_context > 0
            has_concat = hasattr(self.dynamics, 'n_concat_context') and self.dynamics.n_concat_context > 0
            adapter_only = has_adapter and not has_concat

            if adapter_only:
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
                if has_adapter:
                    adapter_idx = self.dynamics.adapter_indices
                    concat_idx = self.dynamics.concat_indices
                    context_masked[mask, :][:, adapter_idx] = 0.0
                    context_masked[mask, :][:, concat_idx] = self.mask_value
                else:
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
            "loss_t": loss_t,
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

    def compute_loss_per_graph(
        self,
        mol_graph,
        context=None,
        reference_indices=None,
        reference_freeze_mode="all",
    ):
        """Returns raw per-graph error for sp_regularizer filtering (PyG path).

        Returns loss_t (the per-graph prediction error before T multiplication
        and neg_log_constants), so the scale is comparable to FM's per-graph MSE.
        Only valid when batch uses PyG format ('graph' key).
        """
        mol_graph, delta_log_px = self.normalize_pyG(mol_graph)
        if self.training and self.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)
        t0_always = not self.training
        _, loss_dict = self.compute_loss_pyG(
            mol_graph,
            context,
            t0_always,
            reference_indices=reference_indices,
            reference_freeze_mode=reference_freeze_mode,
        )
        return loss_dict["loss_t"]

    def forward(
        self,
        x,
        h,
        node_mask=None,
        edge_mask=None,
        context=None,
        reference_indices=None,
        reference_freeze_mode="all",
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

        t0_always = not self.training

        if self.dynamics_teacher is not None:
            if self.training:
                loss, loss_dict = self.compute_loss_distillation(
                    x, h, node_mask, edge_mask, context, t0_always=t0_always
                )
            else:
                loss, loss_dict = self.compute_loss(
                    x, h, node_mask, edge_mask, context, 
                    reference_indices=reference_indices,
                    reference_freeze_mode=reference_freeze_mode,
                    t0_always=t0_always, 
                )
        else:
            if mol_graph is not None:
                # Only 1 forward pass when t0_always is False.
                loss, loss_dict = self.compute_loss_pyG(
                    mol_graph, context, 
                    reference_indices=reference_indices,
                    reference_freeze_mode=reference_freeze_mode,
                    t0_always=t0_always, 
                )
            else:
                # Less variance in the estimator, costs two forward passes.
                loss, loss_dict = self.compute_loss(
                    x, h, node_mask, edge_mask, context, 
                    reference_indices=reference_indices,
                    reference_freeze_mode=reference_freeze_mode,
                    t0_always=t0_always,
                )
        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh
