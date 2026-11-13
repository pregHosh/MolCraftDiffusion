"""E(n) hierarchical VAE + latent diffusion, ported from GeoLDM's
`equivariant_diffusion/en_diffusion.py`
(commit 03ae2031c712a1a6c1678e747bdcdc7a7560e00b), lines 34-1243.

`DistributionNodes`, `gaussian_KL`, `gaussian_KL_for_dimension`,
`PredefinedNoiseSchedule`, `GammaNetwork` are reused from
`MolecularDiffusion.modules.models.en_diffusion` (that fork already carries
the exact same implementations) instead of being re-ported. Mean-zero /
Gaussian sampling utilities come from `MolecularDiffusion.utils`.

Two compatibility shims (not present upstream) are added directly on these
ported classes so `modules.tasks.diffusion.GeomMolecularGenerative`'s
inherited `__init__`/`density_estimation`/`sample_chain` machinery works
without raising, per the integration plan (revision 2):
  - `self.extra_norm_values = ()` / `self.ndim_extra = 0` in
    `EnVariationalDiffusion.__init__` / `EnHierarchicalVAE.__init__`.
  - `forward(...)` on all three classes accepts and ignores
    `reference_indices=None, reference_freeze_mode="all"` (no-op
    passthrough; scaffold-freezing is out of scope for this integration).
"""

import math

import numpy as np
import torch
from torch.nn import functional as F

from MolecularDiffusion.modules.models.edm.en_diffusion import (
    DistributionNodes,
    GammaNetwork,
    PredefinedNoiseSchedule,
    gaussian_KL,
    gaussian_KL_for_dimension,
)
from MolecularDiffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask,
)

from .networks import EGNN_decoder_QM9, EGNN_encoder_QM9

__all__ = ["EnVariationalDiffusion", "EnHierarchicalVAE", "EnLatentDiffusion"]


def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """The E(n) Diffusion Module."""

    def __init__(
        self,
        dynamics,
        in_node_nf: int,
        n_dims: int,
        timesteps: int = 1000,
        parametrization="eps",
        noise_schedule="learned",
        noise_precision=1e-4,
        loss_type="vlb",
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
    ):
        super().__init__()

        assert loss_type in {"vlb", "l2"}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == "learned":
            assert loss_type == "vlb", "A noise schedule can only be learned with a vlb objective."

        assert parametrization == "eps"

        if noise_schedule == "learned":
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        # ponytail: shim for GeomMolecularGenerative's inherited __init__/
        # sample_chain, which read these unconditionally; GeoLDM has no
        # extra per-atom feature channels so these are always empty/zero.
        self.extra_norm_values = ()
        self.ndim_extra = 0

        if noise_schedule != "learned":
            self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

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

    def inflate_batch_array(self, array, target):
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        h_cat = (h["categorical"].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        h_int = (h["integer"].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        h = {"categorical": h_cat, "integer": h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        x, h_cat = z[:, :, 0 : self.n_dims], z[:, :, self.n_dims : self.n_dims + self.num_classes]
        h_int = z[:, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1]
        assert h_int.size(2) == self.include_charges

        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, : self.n_dims], mu_T[:, :, self.n_dims :]

        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
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
        eps_t = net_out
        if self.training and self.loss_type == "l2":
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, : self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims : -1], h_int, node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {"integer": h_int, "categorical": h_cat}
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        z_h_cat = z_t[:, :, self.n_dims : -1] if self.include_charges else z_t[:, :, self.n_dims :]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        eps_x = eps[:, :, : self.n_dims]
        net_x = net_out[:, :, : self.n_dims]

        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)

        h_integer = torch.round(h["integer"] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h["categorical"] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon
        )
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        centered_h_cat = estimated_h_cat - 1

        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon
        )

        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        log_p_h_given_z = log_ph_integer + log_ph_cat

        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        if t0_always:
            lowest_t = 1
        else:
            lowest_t = 0

        t_int = torch.randint(lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()

        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        eps = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        z_t = alpha_t * xh + sigma_t * eps

        assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)

        net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == "l2":
            SNR_weight = torch.ones_like(error)
        else:
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        kl_prior = self.kl_prior(xh, node_mask)

        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T
            estimator_loss_terms = num_terms * loss_t

            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
            )
            z_0 = alpha_0 * xh + sigma_0 * eps_0

            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            loss_term_0 = -self.log_pxh_given_z0_without_constants(x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            if self.training and self.loss_type == "l2":
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, {"t": t_int.squeeze(), "loss_t": loss.squeeze(), "error": error.squeeze()}

    def forward(
        self,
        x,
        h,
        node_mask=None,
        edge_mask=None,
        context=None,
        reference_indices=None,
        reference_freeze_mode="all",
    ):
        """Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.

        `reference_indices`/`reference_freeze_mode` are accepted and ignored
        -- see module docstring; scaffold-freezing is out of scope for this
        integration.
        """
        x, h, delta_log_px = self.normalize(x, h, node_mask)

        if self.training and self.loss_type == "l2":
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False)
        else:
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True)

        neg_log_pxh = loss

        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px

        return neg_log_pxh

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        sigma = sigma_t_given_s * sigma_s / sigma_t

        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        zs = torch.cat(
            [remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask), zs[:, :, self.n_dims :]],
            dim=2,
        )
        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device, node_mask=node_mask
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device, node_mask=node_mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        if fix_noise:
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)

        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f"Warning cog drift with error {max_cog:.3f}. Projecting the positions down.")
            x = remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context)

            assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {"log_SNR_max": log_SNR_max.item(), "log_SNR_min": log_SNR_min.item()}
        print(info)

        return info


class EnHierarchicalVAE(torch.nn.Module):
    """The E(n) Hierarchical VAE Module."""

    def __init__(
        self,
        encoder: EGNN_encoder_QM9,
        decoder: EGNN_decoder_QM9,
        in_node_nf: int,
        n_dims: int,
        latent_node_nf: int,
        kl_weight: float,
        norm_values=(1.0, 1.0, 1.0),
        norm_biases=(None, 0.0, 0.0),
        include_charges=True,
    ):
        super().__init__()

        self.include_charges = include_charges

        self.encoder = encoder
        self.decoder = decoder

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.latent_node_nf = latent_node_nf
        self.num_classes = self.in_node_nf - self.include_charges
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        # ponytail: same shim as EnVariationalDiffusion, see module docstring.
        self.extra_norm_values = ()
        self.ndim_extra = 0

    def subspace_dimensionality(self, node_mask):
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def compute_reconstruction_error(self, xh_rec, xh):
        bs, n_nodes, dims = xh.shape

        x_rec = xh_rec[:, :, : self.n_dims]
        x = xh[:, :, : self.n_dims]
        error_x = sum_except_batch((x_rec - x) ** 2)

        h_cat_rec = xh_rec[:, :, self.n_dims : self.n_dims + self.num_classes]
        h_cat = xh[:, :, self.n_dims : self.n_dims + self.num_classes]
        h_cat_rec = h_cat_rec.reshape(bs * n_nodes, self.num_classes)
        h_cat = h_cat.reshape(bs * n_nodes, self.num_classes)
        error_h_cat = F.cross_entropy(h_cat_rec, h_cat.argmax(dim=1), reduction="none")
        error_h_cat = error_h_cat.reshape(bs, n_nodes, 1)
        error_h_cat = sum_except_batch(error_h_cat)

        if self.include_charges:
            h_int_rec = xh_rec[:, :, -self.include_charges :]
            h_int = xh[:, :, -self.include_charges :]
            error_h_int = sum_except_batch((h_int_rec - h_int) ** 2)
        else:
            error_h_int = 0.0

        error = error_x + error_h_cat + error_h_int

        if self.training:
            denom = (self.n_dims + self.in_node_nf) * xh.shape[1]
            error = error / denom

        return error

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def compute_loss(self, x, h, node_mask, edge_mask, context):
        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encode(x, h, node_mask, edge_mask, context)

        zeros, ones = torch.zeros_like(z_h_mu), torch.ones_like(z_h_sigma)
        loss_kl_h = gaussian_KL(z_h_mu, ones, zeros, ones, node_mask)
        assert z_x_sigma.mean(dim=(1, 2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1, 2)))
        subspace_d = self.subspace_dimensionality(node_mask)
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_h + loss_kl_x

        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = torch.cat([z_x_sigma.expand(-1, -1, 3), z_h_sigma], dim=2)
        z_xh = self.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        assert_correctly_masked(z_xh, node_mask)
        assert_mean_zero_with_mask(z_xh[:, :, : self.n_dims], node_mask)

        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        xh_rec = torch.cat([x_recon, h_recon], dim=2)
        loss_recon = self.compute_reconstruction_error(xh_rec, xh)

        assert loss_recon.size() == loss_kl.size()
        loss = loss_recon + self.kl_weight * loss_kl

        assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."

        return loss, {"loss_t": loss.squeeze(), "rec_error": loss_recon.squeeze()}

    def forward(
        self,
        x,
        h,
        node_mask=None,
        edge_mask=None,
        context=None,
        reference_indices=None,
        reference_freeze_mode="all",
    ):
        """Computes the ELBO. `reference_indices`/`reference_freeze_mode`
        are accepted and ignored -- see module docstring."""
        loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context)
        neg_log_pxh = loss
        return neg_log_pxh

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device, node_mask=node_mask
        )
        z_h = sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.latent_node_nf), device=node_mask.device, node_mask=node_mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def encode(self, x, h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""
        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)

        assert_mean_zero_with_mask(xh[:, :, : self.n_dims], node_mask)

        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder._forward(xh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032
        sigma_0_h = torch.ones(bs, 1, self.latent_node_nf).to(z_h_mu) * 0.0032

        return z_x_mu, sigma_0_x, z_h_mu, sigma_0_h

    def decode(self, z_xh, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""
        x_recon, h_recon = self.decoder._forward(z_xh, node_mask, edge_mask, context)
        assert_mean_zero_with_mask(x_recon, node_mask)

        xh = torch.cat([x_recon, h_recon], dim=2)

        x = xh[:, :, : self.n_dims]
        assert_correctly_masked(x, node_mask)

        h_int = xh[:, :, -1:] if self.include_charges else torch.zeros(0).to(xh)
        h_cat = xh[:, :, self.n_dims : -1]
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask
        h = {"integer": h_int, "categorical": h_cat}

        return x, h

    @torch.no_grad()
    def reconstruct(self, x, h, node_mask=None, edge_mask=None, context=None):
        pass

    def log_info(self):
        info = None
        print(info)
        return info


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class EnLatentDiffusion(EnVariationalDiffusion):
    """The E(n) Latent Diffusion Module."""

    def __init__(self, **kwargs):
        vae = kwargs.pop("vae")
        trainable_ae = kwargs.pop("trainable_ae", False)
        super().__init__(**kwargs)

        self.trainable_ae = trainable_ae
        self.instantiate_first_stage(vae)

    def unnormalize_z(self, z, node_mask):
        x, h_cat = z[:, :, 0 : self.n_dims], z[:, :, self.n_dims : self.n_dims + self.num_classes]
        h_int = z[:, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1]
        assert h_int.size(2) == self.include_charges

        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def log_constants_p_h_given_z0(self, h, node_mask):
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (-log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = xh[:, :, : self.n_dims]

        # Make the data structure compatible with EnVariationalDiffusion's
        # sample() and sample_chain().
        h = {"integer": xh[:, :, self.n_dims :], "categorical": torch.zeros(0).to(xh)}

        return x, h

    def log_pxh_given_z0_without_constants(self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        log_pxh_given_z_without_constants = -0.5 * self.compute_error(net_out, gamma_0, eps)
        log_p_xh_given_z = log_pxh_given_z_without_constants
        return log_p_xh_given_z

    def forward(
        self,
        x,
        h,
        node_mask=None,
        edge_mask=None,
        context=None,
        reference_indices=None,
        reference_freeze_mode="all",
    ):
        """`reference_indices`/`reference_freeze_mode` are accepted and
        ignored -- see module docstring."""
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.vae.encode(x, h, node_mask, edge_mask, context)
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, x)

        z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
        assert_correctly_masked(z_xh_mean, node_mask)
        z_xh_sigma = sigma_0
        z_xh = self.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
        z_xh = z_xh.detach()  # Always keep the encoder fixed.
        assert_correctly_masked(z_xh, node_mask)

        if self.trainable_ae:
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            x_recon, h_recon = self.vae.decoder._forward(z_xh, node_mask, edge_mask, context)
            xh_rec = torch.cat([x_recon, h_recon], dim=2)
            loss_recon = self.vae.compute_reconstruction_error(xh_rec, xh)
        else:
            loss_recon = 0

        z_x = z_xh[:, :, : self.n_dims]
        z_h = z_xh[:, :, self.n_dims :]
        assert_mean_zero_with_mask(z_x, node_mask)
        z_h = {"categorical": torch.zeros(0).to(z_h), "integer": z_h}

        if self.training:
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=False)
        else:
            loss_ld, loss_dict = self.compute_loss(z_x, z_h, node_mask, edge_mask, context, t0_always=True)

        neg_log_constants = -self.log_constants_p_h_given_z0(
            torch.cat([h["categorical"], h["integer"]], dim=2), node_mask
        )
        if self.training and self.loss_type == "l2":
            neg_log_constants = torch.zeros_like(neg_log_constants)

        neg_log_pxh = loss_ld + loss_recon + neg_log_constants

        return neg_log_pxh

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False):
        z_x, z_h = super().sample(n_samples, n_nodes, node_mask, edge_mask, context, fix_noise)

        z_xh = torch.cat([z_x, z_h["categorical"], z_h["integer"]], dim=2)
        assert_correctly_masked(z_xh, node_mask)
        x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)

        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None):
        chain_flat = super().sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames)

        chain = chain_flat.view(keep_frames, n_samples, *chain_flat.size()[1:])
        chain_decoded = torch.zeros(
            size=(*chain.size()[:-1], self.vae.in_node_nf + self.vae.n_dims), device=chain.device
        )

        for i in range(keep_frames):
            z_xh = chain[i]
            assert_mean_zero_with_mask(z_xh[:, :, : self.n_dims], node_mask)

            x, h = self.vae.decode(z_xh, node_mask, edge_mask, context)
            xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
            chain_decoded[i] = xh

        chain_decoded_flat = chain_decoded.view(n_samples * keep_frames, *chain_decoded.size()[2:])

        return chain_decoded_flat

    def instantiate_first_stage(self, vae: EnHierarchicalVAE):
        if not self.trainable_ae:
            self.vae = vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True
