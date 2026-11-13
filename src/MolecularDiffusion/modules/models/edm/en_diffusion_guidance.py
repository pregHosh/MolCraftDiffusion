"""Guided, conditional, inpainting and outpainting sampling for ``EnVariationalDiffusion``."""

from MolecularDiffusion.utils import assert_mean_zero_with_mask
from MolecularDiffusion.utils import build_extra_node_template
from MolecularDiffusion.utils import correct_edges
from MolecularDiffusion.utils import create_pyg_graph
from MolecularDiffusion.utils import enforce_min_nodes_per_connector
from MolecularDiffusion.utils import ensure_intact
from MolecularDiffusion.utils import find_close_points_torch_and_push_op2
from MolecularDiffusion.utils import remove_mean_with_mask
from MolecularDiffusion.utils import resolve_connectors
from MolecularDiffusion.utils import sample_center_gravity_zero_gaussian_with_mask
from MolecularDiffusion.utils import sample_gaussian_with_mask
from torch import autograd
from torch.nn import functional as F
from tqdm import tqdm
import logging
import numpy as np
import torch
from MolecularDiffusion.modules.models.edm._en_diffusion_math import (
    _resolve_jitter_scale,
)

# same logger name as before the split
logger = logging.getLogger("MolecularDiffusion.modules.models.edm.en_diffusion")


class EnDiffusionGuidanceMixin:
    """Guided, conditional, inpainting and outpainting sampling for ``EnVariationalDiffusion``.

    Plain mixin: no parameters or state of its own, so the ``state_dict`` of
    ``EnVariationalDiffusion`` is unchanged. Methods use attributes set in its ``__init__``.
    """

    def _mask_padded_nodes(self, z, node_mask):
        """Zero all channels for padded nodes before masked centering/assertions."""
        return torch.where(node_mask > 0, z, torch.zeros_like(z))

    def _normalize_guidance_grad(
        self, grad, D, t_scale, sigma_t, x_weight, h_weight, node_mask
    ):
        """Normalize a raw guidance gradient into the equivariant subspace.

        Splits grad into position ([:n_dims]) and feature ([n_dims:]) parts,
        scales each part by its per-sample L2 norm and the GeoGuide factor
        D^0.5 / t_scale * sigma_t, masks padded nodes, and recenters the
        position part to zero CoM.
        """
        min_t = torch.as_tensor(
            1.0 / float(self.T), device=t_scale.device, dtype=t_scale.dtype
        )
        t_scale = torch.clamp(t_scale, min=min_t)
        grad = grad.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        sigma_t = sigma_t.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        n = self.n_dims
        nx = grad[:, :, :n].norm(dim=[1, 2], keepdim=True)
        nh = grad[:, :, n:].norm(dim=[1, 2], keepdim=True)
        gx = (
            x_weight * D**0.5 / t_scale
            * grad[:, :, :n] / (nx + 1e-6)
            * sigma_t
        )
        gh = (
            h_weight * D**0.5 / t_scale
            * grad[:, :, n:] / (nh + 1e-6)
            * sigma_t
        )
        gx = gx * node_mask
        gh = gh * node_mask
        grad = torch.cat([remove_mean_with_mask(gx, node_mask), gh], dim=2)
        return grad.nan_to_num(0.0, posinf=0.0, neginf=0.0)

    def _apply_structure_fix(
        self,
        zs,
        s,
        structure_guidance,
        condition_to_use,
        mask_bools,
        natom_ref,
        mask_node_index,
        t_critical,
    ):
        """Apply outpaint or inpaint structure constraint to zs.

        For outpaint, hard-fixes the reference block when s > t_critical,
        otherwise only fixes the feature channels. For inpaint, replaces the
        reference block entirely. No-ops when condition_tensor is None.
        """
        if self.condition_tensor is None:
            return zs
        if "outpaint" in structure_guidance:
            if s[0].item() > t_critical:
                zs = torch.cat(
                    [condition_to_use, zs[:, mask_bools, :]], dim=1
                )
            else:
                zs[:, :natom_ref, self.n_dims:] = (
                    condition_to_use[:, :natom_ref, self.n_dims:]
                )
        elif "inpaint" in structure_guidance and len(mask_node_index) > 0:
            zs = torch.cat(
                [condition_to_use, zs[:, mask_node_index, :]], dim=1
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

        grad = grad.nan_to_num(0.0)
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

        grad = grad.nan_to_num(0.0)
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

        # Defaults; overwritten below when outpaint structure guidance is active.
        mask_bools = None
        natom_ref = 0

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
            z0 = self._mask_padded_nodes(z0, node_mask)
        # guidance
        with torch.enable_grad():
            z0 = z0.requires_grad_()
            energy = target_function(z0, t).sum()
            grad = autograd.grad(energy, z0)[0]

        grad = grad.nan_to_num(0.0)
        opt_info["grad_norms"] = grad.norm(dim=[1, 2])

        t_scale = t.view(-1, 1, 1) if t.numel() > 1 else t[0].view(1, 1, 1)
        grad = self._normalize_guidance_grad(
            grad, D, t_scale, sigma_t, x_weight, h_weight, node_mask
        )

        # Clip before applying to mu (consistent with backward refinement steps).
        clip_coef = torch.clamp(
            max_norm / (grad.norm(dim=[1, 2], keepdim=True) + 1e-6), max=1
        )
        grad = grad * clip_coef

        mu = mu - scale * grad

        opt_info["mu1_norm"] = mu.norm(dim=[1, 2])
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        condition_to_use = self._select_conditioning_tensor(t)

        zs = self._apply_structure_fix(
            zs, s, structure_guidance, condition_to_use,
            mask_bools, natom_ref, mask_node_index, t_critical,
        )

        zs = self._mask_padded_nodes(zs, node_mask)
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
    
        if n_backward > 0 and torch.all(s > 0):
            # s_scale / sigma_s: correct timestep for refining zs at step s.
            s_scale = s.view(-1, 1, 1) if s.numel() > 1 else s[0].view(1, 1, 1)
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.detach().requires_grad_(True)
                    energy_r = target_function(zs, s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]
                    grad_r = grad_r.nan_to_num(0.0)

                    grad_r = self._normalize_guidance_grad(
                        grad_r, D, s_scale, sigma_s, x_weight, h_weight, node_mask
                    )

                    # Per-sample gradient clipping.
                    clip_coef_r = torch.clamp(
                        max_norm / (grad_r.norm(dim=[1, 2], keepdim=True) + 1e-6),
                        max=1,
                    )
                    grad_r = grad_r * clip_coef_r
                    zs = zs - scale * grad_r

                    zs = self._apply_structure_fix(
                        zs, s, structure_guidance, condition_to_use,
                        mask_bools, natom_ref, mask_node_index, t_critical,
                    )

                    # Re-project after every step so energy sees valid geometry.
                    zs = self._mask_padded_nodes(zs, node_mask)
                    zs = torch.cat(
                        [
                            remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                            zs[:, :, self.n_dims:],
                        ],
                        dim=2,
                    )
                    zs = zs.detach()

        zs = self._mask_padded_nodes(zs, node_mask)
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
        mask_node_index=[], # For inpaint
        scale_schedule_type=None
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
            scale_schedule_type (str, optional): Type of scheduler for CFG scale. [Available: 'linear', 'cosine']. Defaults to None.

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

        if scale_schedule_type is not None:
            scale = self.scale_schedule(t, 0, scale, schedule_type=scale_schedule_type)
            scale = scale.view(-1, 1, 1) # Reshape for broadcasting
        
        # Neural net prediction.
        with torch.no_grad():
            eps_t_cond_positive = self.phi(zt, t, node_mask, edge_mask, context=context,)

            if torch.any(torch.isnan(eps_t_cond_positive)):
                logger.warning("eps_t_cond is nan, setting to 0")  
                eps_t_cond_positive = eps_t_cond_positive.nan_to_num(0.0)            
            if context_negative is not None:
                eps_t_cond_negative = self.phi(zt, t, node_mask, edge_mask, context=context_negative)
                if torch.any(torch.isnan(eps_t_cond_negative)):
                    logger.warning("eps_t_cond_negative is nan, setting to 0")
                    eps_t_cond_negative = eps_t_cond_negative.nan_to_num(0.0)
                eps_t = (1 + scale) * eps_t_cond_positive - scale * eps_t_cond_negative      
            else:      
                has_adapter = hasattr(self.dynamics, 'n_adapter_context') and self.dynamics.n_adapter_context > 0
                has_concat = hasattr(self.dynamics, 'n_concat_context') and self.dynamics.n_concat_context > 0
                adapter_only = has_adapter and not has_concat

                if adapter_only:
                    context_null = None
                elif has_adapter:
                    # Hybrid: zero adapter columns, mask_value for concat columns
                    context_null = torch.zeros_like(context, device=eps_t_cond_positive.device)
                    context_null[:, :, self.dynamics.concat_indices] = self.mask_value
                else:
                    context_null = torch.zeros_like(context, device=eps_t_cond_positive.device) + self.mask_value   
                eps_t_uncond = self.phi(zt, t, node_mask, edge_mask, context=context_null)
                if torch.any(torch.isnan(eps_t_uncond)):
                    print("eps_t_uncond is nan, setting to 0") 
                    eps_t_uncond = eps_t_uncond.nan_to_num(0.0)
                eps_t =  (1 + scale) * eps_t_cond_positive - scale * eps_t_uncond
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t
        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)    

        condition_to_use = self._select_conditioning_tensor(t)
    
        # In the event the reference tensor is provided
        if self.condition_tensor is not None and "outpaint" in structure_guidance:  
            if s[0].item() > t_critical:
                # Fix the reference part as conditioning
                zs = torch.cat(
                    [condition_to_use, zs[:, mask_bools, :]], dim=1
                )
            else:
                # Fix just the atomm types of reference part as conditioning
                zs[:, :natom_ref, 3:] = condition_to_use[:, :natom_ref, self.n_dims:]      
        elif  self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            zs = torch.cat(
                [ condition_to_use, zs[:, mask_node_index, :]], dim=1
            )
        # Project down to avoid numerical runaway of the center of gravity.
        zs = self._mask_padded_nodes(zs, node_mask)
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
        context_negative=None,
        structure_guidance=False,
        t_critical=0, # For outpaint
        mask_node_index=[], # For inpaint
        scale_schedule_type=None
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
            context_negative (torch.Tensor, optional): Negative conditional information for guidance.
                If None, unconditional generation is used as the negative target. Defaults to None.
            structure_guidance (bool, optional): If inpaint or outpaint, applies structure guidance. Defaults to False.
            t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
            mask_node_index (list, optional): List of node indices to mask during inpaiting. Defaults to [].
            scale_schedule_type (str, optional): Type of scheduler for CFG scale. [Available: 'linear', 'cosine']. Defaults to None.

        Returns:
            Tuple[torch.Tensor, dict]: The guided sample zs and an opt_info dict
            with keys mu0_norm, mu1_norm, grad_norms, energies, z_norm.
        """
        opt_info = {}
        D = zt.size(2)
        gamma_s = self.gamma(s).to(s.device)
        gamma_t = self.gamma(t).to(s.device)

        # Defaults; overwritten below when outpaint structure guidance is active.
        mask_bools = None
        natom_ref = 0

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


        if scale_schedule_type is not None:
            cfg_scale = self.scale_schedule(t, 0, cfg_scale, schedule_type=scale_schedule_type)
            cfg_scale = cfg_scale.view(-1, 1, 1)  # Reshape for broadcasting

        # Neural net prediction.
        with torch.no_grad():
            eps_t_cond = self.phi(zt, t, node_mask, edge_mask, context=context,)

            if torch.any(torch.isnan(eps_t_cond)):
                logger.warning("eps_t_cond is nan, setting to 0")
                eps_t_cond = eps_t_cond.nan_to_num(0.0)

            if context_negative is not None:
                eps_t_uncond = self.phi(zt, t, node_mask, edge_mask, context=context_negative)
            else:
                has_adapter = hasattr(self.dynamics, 'n_adapter_context') and self.dynamics.n_adapter_context > 0
                has_concat = hasattr(self.dynamics, 'n_concat_context') and self.dynamics.n_concat_context > 0
                adapter_only = has_adapter and not has_concat

                if adapter_only:
                    context_null = None
                elif has_adapter:
                    context_null = torch.zeros_like(context, device=eps_t_cond.device)
                    context_null[:, :, self.dynamics.concat_indices] = self.mask_value
                else:
                    context_null = torch.zeros_like(context, device=eps_t_cond.device) + self.mask_value
                eps_t_uncond = self.phi(zt, t, node_mask, edge_mask, context=context_null)

            if torch.any(torch.isnan(eps_t_uncond)):
                logger.warning("eps_t_uncond is nan, setting to 0")
                eps_t_uncond = eps_t_uncond.nan_to_num(0.0)

            eps_t = (1 + cfg_scale) * eps_t_cond - cfg_scale * eps_t_uncond
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )


        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # gradient guidance
        z0 = (zt - sigma_t * eps_t) / ((1 - sigma_t**2) ** (1 / 2))
        
        condition_to_use = self._select_conditioning_tensor(t)

        z0 = self._apply_structure_fix(
            z0, s, structure_guidance, condition_to_use,
            mask_bools, natom_ref, mask_node_index, t_critical,
        )
        z0 = self._mask_padded_nodes(z0, node_mask)

        with torch.enable_grad():
            z0 = z0.requires_grad_()
            energy = target_function(z0, t).sum()
            grad = autograd.grad(energy, z0)[0]

        grad = grad.nan_to_num(0.0)
        opt_info["grad_norms"] = grad.norm(dim=[1, 2])

        t_scale = t.view(-1, 1, 1) if t.numel() > 1 else t[0].view(1, 1, 1)
        grad = self._normalize_guidance_grad(
            grad, D, t_scale, sigma_t, x_weight, h_weight, node_mask
        )

        # Clip before applying to mu (consistent with backward refinement steps).
        clip_coef = torch.clamp(
            max_norm / (grad.norm(dim=[1, 2], keepdim=True) + 1e-6), max=1
        )
        grad = grad * clip_coef

        opt_info["mu0_norm"] = mu.norm(dim=[1, 2])
        mu = mu - gg_scale * grad
        opt_info["mu1_norm"] = mu.norm(dim=[1, 2])

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        zs = self._apply_structure_fix(
            zs, s, structure_guidance, condition_to_use,
            mask_bools, natom_ref, mask_node_index, t_critical,
        )

        # Project to keep CoM at zero and zero padded nodes.
        zs = self._mask_padded_nodes(zs, node_mask)
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                zs[:, :, self.n_dims:],
            ],
            dim=2,
        )

        opt_info["energies"] = target_function(zs, t)
        opt_info["z_norm"] = zs.norm(dim=[1, 2])

        if n_backward > 0 and torch.all(s > 0):
            # s_scale / sigma_s: correct timestep for refining zs at step s.
            s_scale = s.view(-1, 1, 1) if s.numel() > 1 else s[0].view(1, 1, 1)
            with torch.enable_grad():
                for i in range(n_backward):
                    zs = zs.detach().requires_grad_(True)
                    energy_r = target_function(zs, s).sum()
                    grad_r = autograd.grad(energy_r, zs)[0]
                    grad_r = grad_r.nan_to_num(0.0)

                    grad_r = self._normalize_guidance_grad(
                        grad_r, D, s_scale, sigma_s, x_weight, h_weight, node_mask
                    )

                    # Per-sample gradient clipping.
                    clip_coef_r = torch.clamp(
                        max_norm / (grad_r.norm(dim=[1, 2], keepdim=True) + 1e-6),
                        max=1,
                    )
                    grad_r = grad_r * clip_coef_r
                    zs = zs - gg_scale * grad_r

                    zs = self._apply_structure_fix(
                        zs, s, structure_guidance, condition_to_use,
                        mask_bools, natom_ref, mask_node_index, t_critical,
                    )

                    # Re-project after every step so energy sees valid geometry.
                    zs = self._mask_padded_nodes(zs, node_mask)
                    zs = torch.cat(
                        [
                            remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                            zs[:, :, self.n_dims:],
                        ],
                        dim=2,
                    )
                    zs = zs.detach()

        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, :self.n_dims], node_mask),
                zs[:, :, self.n_dims:],
            ],
            dim=2,
        )

        if self.condition_tensor is not None and "outpaint" in structure_guidance:
            self.condition_tensor = zs[:, ~mask_bools, :]
        elif self.condition_tensor is not None and "inpaint" in structure_guidance and len(mask_node_index) > 0:
            self.condition_tensor = zs[:, ~mask_node_index, :]
        return zs, opt_info

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
        constraint_strength=0.8,
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
            constraint_strength (float, optional): Controls when geometric constraints are
                applied. Constraints activate for diffusion steps s < constraint_strength.
                Higher values apply constraints over a wider window. Defaults to 0.8.
            scale_factor (float, optional): Scale factor applied to covalent radii when
                computing bond-distance tolerances. Increase to allow more spacing between
                generated and scaffold atoms. Defaults to 1.1.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.

        Returns:
            torch.Tensor: The inpainted sample zs.
        """
        t_critical_1 = constraint_strength
        # Bonding constraints (enforce_min_nodes + ensure_intact) are intentionally
        # disabled for inpainting: connector topology is derived from the existing
        # molecule graph so proximity-pull logic is redundant and can disrupt structure.
        t_critical_2 = constraint_strength + 1.0
        d_threshold_f = 1.8
        w_b = 2
        all_frozen = False

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
        if not torch.isfinite(zt).all():
            n_bad = (~torch.isfinite(zt)).sum().item()
            print(f"[ip] zt has {n_bad} non-finite values before phi — sanitising")
            zt = zt.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)
        eps_t = eps_t.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        if torch.all(eps_t == 0):
            raise ValueError("NaN in eps_t, stop sampling.")

        # Compute mu for p(zs | zt).
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)
        zs = zs.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        condition_to_use = self._select_conditioning_tensor(t)

        if len(mask_node_index) > 0 and len(connector_dicts) > 0:
            zs = torch.cat(
                [ condition_to_use, zs[:, mask_node_index, :]], dim=1
            )

            apply_geom = s[0].item() < t_critical_1
            apply_enforce = apply_geom and s[0].item() > t_critical_2

            for b in range(zs.size(0)):
                condition_pos_b = condition_to_use[b, :, : self.n_dims]

                # NOTE: scaffold conditioning reads atomic numbers from the
                # latent's trailing charge column and so requires
                # include_charges=True. Left as-is deliberately: this path
                # is unreachable without a condition_tensor.
                condition_charge_b = condition_to_use[b, :, -1] * self.norm_values[2]
                condition_charge_b = torch.round(condition_charge_b).long().clamp(1, 118)

                zs_charge_b = zs[b, :, -1] * self.norm_values[2]
                zs_charge_b = torch.round(zs_charge_b).long().clamp(1, 118)

                d_threshold_c_b = self.COV_R[condition_charge_b][connector_indices] * 2 * scale_factor

                if apply_geom:
                    zs_new_pos_b = zs[b, mask_node_index, : self.n_dims]
                    _, zl_b = find_close_points_torch_and_push_op2(
                        condition_pos_b,
                        zs_new_pos_b,
                        connector_indices=connector_indices,
                        d_threshold_f=d_threshold_f,
                        w_b=w_b,
                        all_frozen=all_frozen,
                        z_ref=condition_charge_b,
                        z_tgt=zs_charge_b,
                        scale_factor=scale_factor,
                    )
                    zs[b, mask_node_index, : self.n_dims] = zl_b

                    if apply_enforce:
                        zs_new_pos_b = zs[b, mask_node_index, : self.n_dims]
                        zl2_b = enforce_min_nodes_per_connector(
                            condition_pos_b,
                            zs_new_pos_b,
                            connector_indices=connector_indices,
                            N=connector_degrees,
                            d_threshold_c=d_threshold_c_b,
                        )
                        zs[b, mask_node_index, : self.n_dims] = zl2_b

                        zs_new_pos_b = zs[b, mask_node_index, : self.n_dims]
                        d_intact_b = (
                            float(d_threshold_c_b.min().item())
                            if torch.is_tensor(d_threshold_c_b)
                            else min(d_threshold_c_b)
                        )
                        _, z1_b = ensure_intact(
                            condition_pos_b,
                            zs_new_pos_b,
                            connector_indices=connector_indices,
                            d_threshold=d_intact_b,
                        )
                        zs[b, mask_node_index, : self.n_dims] = z1_b
        
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
        constraint_strength=0.8,
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
            constraint_strength (float, optional): Controls when geometric constraints are
                applied. Constraints activate for diffusion steps s < constraint_strength.
                Bonding constraints (enforce + ensure_intact) are additionally restricted to
                the upper half of the window (s > constraint_strength / 2). Defaults to 0.8.
            scale_factor (float, optional): Scale factor applied to covalent radii when
                computing bond-distance tolerances. Increase to allow more spacing between
                generated and scaffold atoms. Defaults to 1.1.
            fix_noise (bool, optional): If True, uses fixed noise for sampling. Defaults to False.

        Returns:
            torch.Tensor: The outpainted sample zs.
        """
        t_critical_1 = constraint_strength
        t_critical_2 = constraint_strength / 2.0
        d_threshold_f = 1.8
        w_b = 2
        all_frozen = False

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

        condition_to_use = self._select_conditioning_tensor(t)

        zs = torch.cat(
            [condition_to_use,
             zs[:, mask_bools, : ]], dim=1
        )

        apply_geom = s[0].item() < t_critical_1
        apply_enforce = apply_geom and s[0].item() > t_critical_2

        if apply_geom:
            for b in range(zs.size(0)):
                condition_pos_b = condition_to_use[b, :, : self.n_dims]

                # NOTE: scaffold conditioning reads atomic numbers from the
                # latent's trailing charge column and so requires
                # include_charges=True. Left as-is deliberately: this path
                # is unreachable without a condition_tensor.
                condition_charge_b = condition_to_use[b, :, -1] * self.norm_values[2]
                condition_charge_b = torch.round(condition_charge_b).long().clamp(1, 118)

                zs_charge_b = zs[b, :, -1] * self.norm_values[2]
                zs_charge_b = torch.round(zs_charge_b).long().clamp(1, 118)

                d_threshold_c_b = self.COV_R[condition_charge_b][connector_indices] * 2 * scale_factor

                zs_new_pos_b = zs[b, natom_ref:, : self.n_dims]
                _, zl_b = find_close_points_torch_and_push_op2(
                    condition_pos_b,
                    zs_new_pos_b,
                    connector_indices=connector_indices,
                    d_threshold_f=d_threshold_f,
                    w_b=w_b,
                    all_frozen=all_frozen,
                    z_ref=condition_charge_b,
                    z_tgt=zs_charge_b,
                    scale_factor=scale_factor,
                )
                zs[b, mask_bools, : self.n_dims] = zl_b

                if apply_enforce:
                    zs_new_pos_b = zs[b, natom_ref:, : self.n_dims]
                    zl2_b = enforce_min_nodes_per_connector(
                        condition_pos_b,
                        zs_new_pos_b,
                        connector_indices=connector_indices,
                        N=connector_degrees,
                        d_threshold_c=d_threshold_c_b,
                    )
                    zs[b, mask_bools, : self.n_dims] = zl2_b

                    zs_new_pos_b = zs[b, natom_ref:, : self.n_dims]
                    d_intact_b = (
                        float(d_threshold_c_b.min().item())
                        if torch.is_tensor(d_threshold_c_b)
                        else min(d_threshold_c_b)
                    )
                    _, z1_b = ensure_intact(
                        condition_pos_b,
                        zs_new_pos_b,
                        connector_indices=connector_indices,
                        d_threshold=d_intact_b,
                    )
                    zs[b, mask_bools, : self.n_dims] = z1_b
  
                        
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

        condition_to_use = self._select_conditioning_tensor(t)

        if s[0].item() > t_critical:
            # Fix the reference part as conditioning
            zs = torch.cat(
                [condition_to_use, zs[:, mask_bools, :]], dim=1
            )
        else:
            # Fix just the atomm types of reference part as conditioning
            zs[:, :natom_ref, 3:] = condition_to_use[:, :natom_ref, self.n_dims:]       

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

    def sample_p_zs_given_zt_silvr(
        self,
        s,
        t,
        zt,
        node_mask,
        edge_mask,
        context,
        xh_ref,
        silvr_rate,
        ref_node_mask,
        total_shift,
        condition_component,
        fix_noise=False,
    ):
        """One SILVR reverse step (Runcie & Mey, JCIM 2023, 63, 5996-6005).

        Reproduces the published sampler's per-step block verbatim
        (docs/model_integrations/silvr/reference_source/
        silvr_fork_en_diffusion.py L893-976):

        1. re-centre zt on its own coordinate mean (over node_mask.sum());
        2. rebuild + re-centre the padded reference, accumulating the removed
           mean into ``total_shift``;
        3. take the ordinary, unmodified EDM reverse step;
        4. blend the reference's noised latent into the reference rows.

        ``alpha_t``/``sigma_t`` come from ``gamma(t)`` -- the higher-noise
        timestep, one level noisier than the ``z`` step 3 just produced. That
        is what the source does; it is not a typo for ``gamma(s)``.

        Returns:
            tuple: ``(zs, xh_ref, total_shift)`` -- the latter two are carried
            into the next step by the caller.
        """
        n_pad = zt.size(2) - self.n_dims
        gamma_t = self.inflate_batch_array(self.gamma(t), zt)
        alpha_t = self.alpha(gamma_t, zt)
        sigma_t = self.sigma(gamma_t, zt)

        n_live = node_mask.sum(1, keepdim=True)

        # 1. re-centre z (sample_p_zs_given_zt asserts a zero-mean input).
        mean = zt[:, :, : self.n_dims].sum(1, keepdim=True) / n_live
        zt = zt - F.pad(mean, (0, n_pad)) * node_mask

        # 2. rebuild + re-centre the reference.
        #
        # REPRODUCED FROM SOURCE, AND SUSPICIOUS -- NOT A TYPO OF OURS:
        # (ref_node_mask - node_mask) == -1 over the dummy region, so the dummy
        # rows of xh_ref end up holding the *negated* live latent (fork L936;
        # the adjacent commented-out attempt at L1074-1075 uses +z instead).
        # It cannot corrupt the reference rows -- next step's `xh_ref *
        # ref_node_mask` discards the dummy region -- but it does enter `mean`
        # below, hence total_shift, hence the final coordinates. Changing it
        # changes the published output geometry, and there is no reference
        # output to validate a "corrected" variant against.
        xh_ref = xh_ref * ref_node_mask + zt * (ref_node_mask - node_mask)
        mean = xh_ref[:, :, : self.n_dims].sum(1, keepdim=True) / n_live
        mean = F.pad(mean, (0, n_pad))
        xh_ref = xh_ref - mean * ref_node_mask
        total_shift = total_shift + mean

        # 3. the ordinary EDM reverse step.
        zs = self.sample_p_zs_given_zt(
            s, t, zt, node_mask, edge_mask, context, fix_noise=fix_noise
        )

        # 4. SILVR (fork L971-976).
        eps = self.sample_combined_position_feature_noise(
            zt.size(0), zt.size(1), node_mask
        )
        z_t = alpha_t * xh_ref + sigma_t * eps
        update = (
            -(zs * alpha_t * ref_node_mask) * silvr_rate
            + (z_t * ref_node_mask) * silvr_rate
        )
        if condition_component == "x":
            update[:, :, self.n_dims :] = 0
        return zs + update, xh_ref, total_shift

    def _build_outpaint_extras(
        self, condition_tensor, connector_indices, natom_extra,
        init_method, skeleton_type, seed_dist, min_dist, spread, n_bq_atom,
        bond_len=1.5,
    ):
        """Build clean extra-node coords for dense outpaint init.

        Supports init_method in {'skeleton', 'seed'} via build_extra_node_template
        (the legacy seed blob is method='seed'). 'data'/'fragment' are PyG-only
        for now and raise a clear error here.
        """
        if init_method in ("skeleton", "fragment", "seed"):
            return build_extra_node_template(
                condition_tensor, connector_indices, natom_extra,
                method=init_method, skeleton_type=skeleton_type,
                seed_dist=seed_dist, min_dist=min_dist, spread=spread,
                n_bq_atom=n_bq_atom, bond_len=bond_len,
            )
        raise NotImplementedError(
            f"outpaint init_method={init_method!r} is not available on the dense "
            "model (use 'skeleton', 'fragment', or 'seed'); 'data' is wired on "
            "the PyG model."
        )

    def _outpaint_forward_noise(
        self, z, condition_tensor, natom_ref, node_mask, t_int_start,
        reference_freeze_mode="all", forward_noise="jitter", jitter_scale=0.75,
    ):
        """Inject noise into a clean outpaint template before denoising.

        Modes (``forward_noise``):
          * "schedule" — full training-marginal noise ``alpha_t*xh0 + sigma_t*eps``
            (SDEdit). At typical t_start this injects large sigma and can swamp the
            placeholder — historically the model was used with clean starts.
          * "jitter" — small fixed scatter (~jitter_scale) on positions only,
            matching the legacy seed-blob noise regime around a structured
            placeholder. Default.
          * "off" — use the clean placeholder directly.
        In all modes the scaffold prefix is pinned to clean values and recentered.
        """
        mode = str(forward_noise).strip().lower()
        if mode in ("schedule", "true", "True") and t_int_start is not None:
            t_norm = min((int(t_int_start) + 1) / self.T, 1.0)
            t_tensor = torch.full((z.size(0), 1), t_norm, device=z.device)
            gamma_t = self.inflate_batch_array(self.gamma(t_tensor), z)
            alpha_t = self.alpha(gamma_t, z)
            sigma_t = self.sigma(gamma_t, z)
            eps = self.sample_combined_position_feature_noise(
                z.size(0), z.size(1), node_mask
            )
            z = alpha_t * z + sigma_t * eps
        elif mode == "jitter":
            eps = self.sample_combined_position_feature_noise(
                z.size(0), z.size(1), node_mask
            )
            z = torch.cat(
                [z[:, :, :self.n_dims] + float(jitter_scale) * eps[:, :, :self.n_dims],
                 z[:, :, self.n_dims:]],
                dim=2,
            )
        # "off": leave z as the clean placeholder.
        if reference_freeze_mode == "features_only":
            z[:, :natom_ref, self.n_dims:] = condition_tensor[:, :natom_ref, self.n_dims:]
        else:
            z[:, :natom_ref] = condition_tensor[:, :natom_ref]
        z = torch.cat(
            [
                remove_mean_with_mask(z[:, :, :self.n_dims], node_mask),
                z[:, :, self.n_dims:],
            ],
            dim=2,
        )
        return z

    def _init_inpaint_extend_nodes(
        self, z, connector_indices, n_extend,
        init_method="skeleton", skeleton_type="random_walk", bond_len=1.5,
        seed_dist=2.0, min_dist=1.0, spread=1.0, n_bq_atom=0,
        forward_noise="jitter", jitter_scale=None, alpha_d=None, sigma_d=None,
    ):
        """Build + noise the extra atoms for extended inpainting (same as outpaint).

        Places a clean skeleton/fragment for the ``n_extend`` new atoms, anchored
        at the connectors in the current latent ``z``, then injects noise per
        ``forward_noise``: "schedule" uses the inpaint denoising_strength
        alpha_d/sigma_d, "jitter" a fixed scatter set by jitter_scale,
        "off"/"seed" none. ``init_method="seed"`` is the outpaint seed-blob (no
        noise). Returns ``(B, n_extend, D)``.
        """
        new = build_extra_node_template(
            z, connector_indices, n_extend, method=init_method,
            skeleton_type=skeleton_type, seed_dist=seed_dist, min_dist=min_dist,
            spread=spread, n_bq_atom=n_bq_atom, bond_len=bond_len,
        )
        mode = "off" if init_method == "seed" else str(forward_noise).strip().lower()
        if mode == "jitter" and jitter_scale is None:
            raise ValueError("jitter_scale must be set explicitly for jitter noise")
        jit = 0.75 if jitter_scale is None else float(jitter_scale)
        nm = torch.ones((new.size(0), n_extend, 1), device=new.device)
        if mode in ("schedule", "true", "True") and alpha_d is not None:
            eps = self.sample_combined_position_feature_noise(new.size(0), n_extend, nm)
            new = alpha_d[:, :1, :] * new + sigma_d[:, :1, :] * eps
        elif mode == "jitter":
            eps = self.sample_combined_position_feature_noise(new.size(0), n_extend, nm)
            new = torch.cat(
                [new[:, :, :self.n_dims] + jit * eps[:, :, :self.n_dims],
                 new[:, :, self.n_dims:]],
                dim=2,
            )
        return new

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
        cfg_scale_schedule=None,
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
        inpaint_cfgs={},
        outpaint_cfgs={},
        use_noised_conditioning=False,
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
        - cfg_scale_schedule (str|optional): The schedule for cfg_scale. Can be "linear", "exponential", or "cosine". Default is None.
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
        - inpaint_cfgs (dict, optional): Configuration for inpainting. 
            The dictionary must contains:        
                - mask_node_index (torch.Tensor, optional): Indices of nodes to be inpainted. Defaults to an empty tensor.
                - denoising_strength (float, optional): Strength of denoising for inpainting
                - noise_initial_mask (bool, optional): Whether to noise the initial masked region. Defaults to False.
        - outpaint_cfgs (dict, optional): Configuration for outpainting. 
            The dictionary must contains:
                - t_start (float, optional): Timestep to start the generation. Defaults to 1.0.
                - t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
`               - connector_index (torch.Tensor, optional): Indices of connector nodes for outpainting. Defaults to an empty tensor.
                - seed_dist (float, optional): Distance of the seed from the connector atom (used if n_bq_atom == 0)..
                - min_dist (float, optional): Minimum distance from any existing atom in xh_cond (except the connector itself). Defaults to 1.
                - spread (float, optional): Random-walk angular dispersion, or
                    legacy seed-cloud position spread. Defaults to 1.
                - jitter_scale (float): Explicit positional noise magnitude
                    when forward_noise is "jitter".
                - n_bq_atom (int, optional): Number of dummy atoms. Defaults is 0.

        Returns:
        Tuple[Tensor, Tensor]: Sampled positions and features.
        """
        debug = False
        if condition_tensor is not None:
            condition_tensor = condition_tensor.to(device=node_mask.device)
            if condition_tensor.size(0) == 1 and n_samples > 1:
                condition_tensor = condition_tensor.expand(n_samples, -1, -1).contiguous()
            elif condition_tensor.size(0) != n_samples:
                raise ValueError(
                    "condition_tensor batch dimension must be 1 or match n_samples "
                    f"({condition_tensor.size(0)} != {n_samples})"
                )
            if condition_tensor.size(1) > node_mask.size(1):
                raise ValueError(
                    "condition_tensor has more reference nodes than the sampling "
                    f"node count ({condition_tensor.size(1)} > {node_mask.size(1)})"
                )
        self.clean_condition_tensor = condition_tensor
        self.use_noised_conditioning = use_noised_conditioning
        
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
            node_mask_cond = torch.ones((n_samples, n_node_cond,1), device=z.device)  
            if "ft" not in condition_alg:
                condition_tensor[:,:, : self.n_dims] = remove_mean_with_mask(condition_tensor[:,:, : self.n_dims], node_mask_cond)

    
        # for tracking intermediate frames
        if n_frames > 0:
            z_size = (z.size(0), z.size(1), z.size(2)-len(self.extra_norm_values))
            chain = torch.zeros((n_frames,) + z_size, device=z.device)
            s_saves = torch.linspace(0, self.T, 
                                         steps=n_frames, device=z.device).long() 
        

        if condition_alg:
            if  "inpaint" in condition_alg:
                
                denoising_strength = inpaint_cfgs.get("denoising_strength", None)
                if denoising_strength is None:
                    raise ValueError("denoising_strength must be specified in to use inpainting")
                noise_initial_mask = inpaint_cfgs.get("noise_initial_mask", False)
                # Extended-inpainting init knobs (mirror outpaint).
                ext_init_method = str(inpaint_cfgs.get("init_method", "skeleton"))
                ext_skeleton_type = str(inpaint_cfgs.get("skeleton_type", "random_walk"))
                ext_bond_len = float(inpaint_cfgs.get("bond_len", 1.5))
                ext_forward_noise = str(inpaint_cfgs.get("forward_noise", "jitter"))
                ext_seed_dist = float(inpaint_cfgs.get("seed_dist", 2.0))
                ext_min_dist = float(inpaint_cfgs.get("min_dist", 1.0))
                ext_spread = float(inpaint_cfgs.get("spread", 1.0))
                ext_n_bq = int(inpaint_cfgs.get("n_bq_atom", 0))
                ext_jitter_scale = _resolve_jitter_scale(
                    inpaint_cfgs, ext_init_method, ext_forward_noise
                )
                mask_node_index = inpaint_cfgs.get("mask_node_index", torch.tensor([], device=z.device, dtype=torch.long))
                t_critical = 0.0
                
                mask_node_index = torch.tensor([mask_node_index], device=z.device, dtype=torch.long)
                unmasked_node_indices = [i for i in range(n_node_cond) if i not in mask_node_index]
                s_saves = torch.linspace(0, int(denoising_strength * self.T), 
                                         steps=n_frames, device=z.device).long() 
                t_int_start = int(self.T * denoising_strength)
                n_node_cond = condition_tensor.size(1)
                d = torch.full((n_samples, 1), fill_value=denoising_strength, device=z.device)
                
                gamma_d = self.inflate_batch_array(self.gamma(d), condition_tensor)
            
                # Compute alpha_t and sigma_t from gamma.
                alpha_d = self.alpha(gamma_d, condition_tensor)
                sigma_d = self.sigma(gamma_d, condition_tensor)

                #-----------------------partial inpainting------------------------------------#
                if mask_node_index.size(1) > 0:
                    scale_factor = inpaint_cfgs.get("scale_factor", 1.1)
                    # Reorder condition_tensor such that connector nodes are the first nodes
                    connector_mask = torch.zeros(condition_tensor.size(1), dtype=torch.bool, device=z.device)
                    connector_mask[mask_node_index] = True  
                    non_connector_mask = ~connector_mask          
                    reordered_condition_tensor = torch.cat([condition_tensor[:, non_connector_mask, :], 
                                                            condition_tensor[:, connector_mask, :]], dim=1)
                    condition_tensor = reordered_condition_tensor

                    x_ref = condition_tensor[:, :, : self.n_dims].squeeze(0)
                    # NOTE: scaffold conditioning reads atomic numbers from the
                    # latent's trailing charge column and so requires
                    # include_charges=True. Left as-is deliberately: this path
                    # is unreachable without a condition_tensor.
                    h_int_ref = condition_tensor[:, :, -1].squeeze(0) * self.norm_values[2]
                    
                    if x_ref.dim() > 2:
                        x_ref = x_ref[0]
                    if h_int_ref.dim() > 2:
                        h_int_ref = h_int_ref[0]
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
                    
                    node_mask_m = torch.ones((n_samples, mask_node_index.size(1), 1), device=z.device)
                    
                    node_mask_um = torch.ones((n_samples, n_node_cond-mask_node_index.size(1)), device=z.device)  
                    node_mask_um = node_mask_um.unsqueeze(-1)
        
                    n_masked_nodes = mask_node_index.size(1)
                    mask_node_bool_corr = torch.cat([
                        torch.zeros(n_node_cond - n_masked_nodes, device=z.device),
                        torch.ones(n_masked_nodes, device=z.device),
                    ]).bool()
                    
                    xh_unmasked = condition_tensor[:,~mask_node_bool_corr, :] # not to be denoise

                    # noise the masked nodes
                    if noise_initial_mask:
                        eps_s_ref = self.sample_combined_position_feature_noise(
                            n_samples=n_samples, n_nodes=mask_node_index.size(1), node_mask=node_mask_m)
                        z = alpha_d * condition_tensor[:, mask_node_bool_corr, :] + sigma_d * eps_s_ref # to be denoised
                        
                        z = torch.cat([xh_unmasked, z], dim=1)
                        z = torch.cat(  
                            [
                                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask_cond),
                                z[:, :, self.n_dims :],
                            ],
                            dim=2,
                        )      
        
                        ci_keys = torch.tensor(
                            list(connector_dicts.keys()), device=z.device, dtype=torch.long
                        )
                        for b in range(n_samples):
                            condition_pos_b = xh_unmasked[b, :, : self.n_dims]
                            zs_new_pos_b = z[b, mask_node_bool_corr, : self.n_dims]
                            _, zl_b = find_close_points_torch_and_push_op2(
                                condition_pos_b,
                                zs_new_pos_b,
                                connector_indices=ci_keys,
                                d_threshold_f=1.8,
                                w_b=2,
                                all_frozen=False,
                                z_ref=None,
                                z_tgt=None,
                                scale_factor=scale_factor,
                            )
                            z[b, mask_node_bool_corr, : self.n_dims] = zl_b

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
                    if mask_node_index.size(1) > 0:
                        n_connector = mask_node_index.size(1)
                        connector_indices = torch.arange(0 , n_connector, device=z.device) + condition_tensor[:, ~mask_node_bool_corr, :].shape[1]
                    else:
                        n_connector = condition_tensor.size(1)
                        connector_indices = torch.arange(0 , n_connector, device=z.device)
                    
                    new_nodes = self._init_inpaint_extend_nodes(
                        z, connector_indices, n_node_extend,
                        init_method=ext_init_method, skeleton_type=ext_skeleton_type,
                        bond_len=ext_bond_len, seed_dist=ext_seed_dist,
                        min_dist=ext_min_dist, spread=ext_spread, n_bq_atom=ext_n_bq,
                        forward_noise=ext_forward_noise, jitter_scale=ext_jitter_scale,
                        alpha_d=alpha_d, sigma_d=sigma_d,
                    )
                    z = torch.cat([z, new_nodes], dim=1)
                    if mask_node_index.size(1) > 0:
                        mask_node_bool_corr = torch.cat([mask_node_bool_corr,
                                                        torch.ones(n_node_extend, device=z.device)]).bool()
                        # connector_dicts = {key + n_node_extend: value for key, value in connector_dicts.items()}
                        
                if mask_node_index.size(1) > 0:
                    self.condition_tensor = z[:, ~mask_node_bool_corr, :] 
                
            elif "outpaint" in condition_alg:
                
                t_start = outpaint_cfgs.get("t_start", 1.0)
                t_critical = outpaint_cfgs.get("t_critical", 0.0)
                # This path applies no per-connector bonding constraint, so only the keys
                # of `connectors` are used; the degrees are ignored.
                connector_indices = list(resolve_connectors(outpaint_cfgs).keys())
                connector_indices = torch.tensor(connector_indices, device=z.device, dtype=torch.long)
                min_dist = outpaint_cfgs.get("min_dist", 1.0)
                seed_dist = outpaint_cfgs.get("seed_dist", 2.0)
                spread = outpaint_cfgs.get("spread", 1.0)
                n_bq_atom = outpaint_cfgs.get("n_bq_atom", 0)
                init_method = str(outpaint_cfgs.get("init_method", "skeleton"))
                skeleton_type = str(outpaint_cfgs.get("skeleton_type", "random_walk"))
                bond_len = float(outpaint_cfgs.get("bond_len", 1.5))
                forward_noise = str(outpaint_cfgs.get("forward_noise", "jitter"))
                jitter_scale = _resolve_jitter_scale(
                    outpaint_cfgs, init_method, forward_noise
                )

                s_saves = torch.linspace(0, self.T * t_start,
                                         steps=n_frames, device=z.device).long()
                t_int_start = self.T * t_start
                self.condition_tensor = condition_tensor

                natom_ref = condition_tensor.size(1)
                natom_extra = n_nodes - natom_ref
                new_nodes = self._build_outpaint_extras(
                    condition_tensor, connector_indices, natom_extra,
                    init_method, skeleton_type, seed_dist, min_dist, spread,
                    n_bq_atom, bond_len,
                )
                if any(idx > natom_ref - 1 for idx in connector_indices):
                    raise ValueError("connectors: atom index out of bounds for the reference structure")
                if connector_indices.size(0) == 0 and condition_alg == "outpaint":
                    raise ValueError("connectors is empty")
                z = torch.cat([condition_tensor, new_nodes], dim=1)
                z = self._mask_padded_nodes(z, node_mask)
                # Training-consistent init: forward-noise clean template to t_start.
                if init_method != "seed":
                    z = self._outpaint_forward_noise(
                        z, condition_tensor, natom_ref, node_mask,
                        int(t_int_start), reference_freeze_mode="all",
                        forward_noise=forward_noise, jitter_scale=jitter_scale,
                    )
                    self.condition_tensor = z[:, :natom_ref, :]
                if n_frames > 0:
                    chain = torch.zeros((n_frames,) + z_size, device=z.device)
                mask_node_bool_corr = None
            
        else:
            t_critical = 0
            t_int_start = self.T
            mask_node_bool_corr = None
        z = self._mask_padded_nodes(z, node_mask)
        z = torch.cat(  
            [
                remove_mean_with_mask(z[:, :, : self.n_dims], node_mask),
                z[:, :, self.n_dims :],
            ],
            dim=2,
        )      
        
        if self.condition_tensor is not None:
            if condition_alg == "inpaint":
                self.condition_tensor = z[:, ~mask_node_bool_corr, :]
            elif condition_alg == "outpaint":
                self.condition_tensor = z[:, :natom_ref, :]
        
            
            
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
                                mask_node_index=mask_node_bool_corr,
                                scale_schedule_type=cfg_scale_schedule
                            )
                        elif guidance_ver == "cfg_gg":
                            z, opt_info = self.sample_p_zs_given_zt_guidance_cfg_gg(
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
                                context_negative=context_negative,
                                t_critical=t_critical,
                                structure_guidance=condition_alg,
                                mask_node_index=mask_node_bool_corr,
                                scale_schedule_type=cfg_scale_schedule
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

        if n_frames > 0:
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
