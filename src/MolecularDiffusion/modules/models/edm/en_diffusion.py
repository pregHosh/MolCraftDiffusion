import logging
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
from torch_geometric.data import Data, Batch

from MolecularDiffusion.utils import (
    assert_mean_zero_with_mask,
    check_quality,
    correct_edges,
    create_pyg_graph,
    enforce_min_nodes_per_connector,
    ensure_intact,
    resolve_connectors,
    build_extra_node_template,
    find_close_points_torch_and_push_op2,
    remove_mean_with_mask,
    remove_mean_pyG,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask
)
from MolecularDiffusion.utils.chem.geom_analyzer import symbol2num


from MolecularDiffusion.modules.models.edm._en_diffusion_math import (  # noqa: F401  (re-exported: other modules import these from here)
    _resolve_jitter_scale,
    _n_int,
    split_feature_axis,
    split_z,
    split_frame,
    atomic_numbers,
    cdf_standard_gaussian,
    expm1,
    softplus,
    sum_except_batch,
    gaussian_entropy,
    gaussian_KL,
    gaussian_KL_for_dimension,
)
from MolecularDiffusion.modules.models.edm.en_diffusion_loss import EnDiffusionLossMixin
from MolecularDiffusion.modules.models.edm.en_diffusion_sampling import EnDiffusionSamplingMixin
from MolecularDiffusion.modules.models.edm.en_diffusion_guidance import EnDiffusionGuidanceMixin

logger = logging.getLogger(__name__)


class EnVariationalDiffusion(
    EnDiffusionLossMixin, EnDiffusionSamplingMixin, EnDiffusionGuidanceMixin, torch.nn.Module
):
    """
    E(n) Equivariant Variational Diffusion Model.

    Args:
        dynamics (nn.Module): Neural network that predicts noise or x
            (calling convention: ``models._protocols.DenseDynamics``).
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
        atom_vocab: Optional[Sequence[str]] = None,
        use_noised_conditioning: bool = False,
    ):
        super().__init__()
        self.call = 0
        self.eval_mode = eval_mode
        self.debug = debug
        self.atom_vocab = list(atom_vocab) if atom_vocab else None

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
        self.use_noised_conditioning = use_noised_conditioning
        self.clean_condition_tensor = None

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

    def _get_noised_condition_tensor(self, t, clean_tensor=None):
        """
        Generate noised version of condition tensor at timestep t.
        
        Args:
            t: diffusion timestep tensor
            clean_tensor: clean scaffold tensor (default: self.condition_tensor)
        
        Returns:
            noised_tensor: forward-noised version at timestep t
        """
        if clean_tensor is None:
            clean_tensor = getattr(self, "clean_condition_tensor", None)
            if clean_tensor is None:
                clean_tensor = self.condition_tensor
        
        if clean_tensor is None:
            return None
        
        # Get noise schedule parameters at timestep t
        gamma_t = self.gamma(t)
        sigma_t = self.sigma(gamma_t, target_tensor=clean_tensor)
        alpha_t = torch.sqrt(1 - sigma_t**2)
        
        # Sample noise with zero-mean positions
        pos_mask = torch.ones((clean_tensor.size(0), clean_tensor.size(1), 1), device=clean_tensor.device)
        eps_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(clean_tensor.size(0), clean_tensor.size(1), self.n_dims),
            device=clean_tensor.device,
            node_mask=pos_mask
        )
        eps_h = torch.randn(
            (clean_tensor.size(0), clean_tensor.size(1), clean_tensor.size(2) - self.n_dims),
            device=clean_tensor.device
        )
        epsilon = torch.cat([eps_x, eps_h], dim=2)
        
        # Forward diffusion: z_t = alpha_t * x_0 + sigma_t * epsilon
        noised = alpha_t * clean_tensor + sigma_t * epsilon
        
        # Ensure center of mass is zero for the position part
        noised[:, :, :self.n_dims] = remove_mean_with_mask(noised[:, :, :self.n_dims], pos_mask)
        
        return noised

    def _select_conditioning_tensor(self, t, clean_tensor=None):
        """
        Return appropriate conditioning tensor based on mode.
        
        Returns: clean_tensor if use_noised_conditioning=False
                 noised_tensor if use_noised_conditioning=True
        """
        if not getattr(self, "use_noised_conditioning", False):
            return clean_tensor if clean_tensor is not None else self.condition_tensor
        else:
            return self._get_noised_condition_tensor(t, clean_tensor)

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
        x, h_cat, h_int = split_z(self, z)
        h_int = h_int.contiguous()
        assert h_int.size(2) == self.include_charges

        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)

        # Strip extra features after unnormalization — chain frames only
        # need core OHE features, not extras
        if self.ndim_extra > 0:
            h_cat = h_cat[:, :, :-self.ndim_extra]

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

        x_refined = []
        h_refined = []
        for idx, loss_i in enumerate(loss):
            if torch.isinf(loss_i).any():
                for i in range(1, n_frame_look_back+1):
                    x_i = chain[-i-1][idx, :, : self.n_dims].unsqueeze(0)

                    if self.ndim_extra > 0:
                        h_cat_i = chain[-i-1][idx, :, start:mid].unsqueeze(0)
                        h_int_i = chain[-i-1][idx, :, hint_i:hint_i+1].unsqueeze(0)
                        h_extra_i = chain[-i-1][idx, :, -self.ndim_extra:].unsqueeze(0)
                        h_i = {
                            "categorical": h_cat_i,
                            "integer": h_int_i,
                            "extra": h_extra_i,
                        }
                    else:
                        _, h_cat_i, h_int_i = split_frame(self,
                            chain[-i-1][idx]
                        )
                        h_cat_i = h_cat_i.unsqueeze(0)
                        h_int_i = h_int_i.unsqueeze(0)
                        h_i = {
                            "categorical": h_cat_i,
                            "integer": h_int_i,
                            "extra": None,
                        }
                    node_mask_i = node_mask[idx].unsqueeze(0)

                    zx_i, zh_i, _ = self.normalize(x_i, h_i, node_mask_i)
                    
                    # Reconstruct z_i from normalized components
                    if self.ndim_extra > 0:
                        zh_i_combined = torch.cat([zh_i["categorical"], zh_i["integer"], zh_i["extra"]], dim=2)
                    else:
                        zh_i_combined = torch.cat([zh_i["categorical"], zh_i["integer"]], dim=2)

                    z_i = torch.cat([zx_i, zh_i_combined], dim=2)
                    edge_mask_i = node_mask_i.unsqueeze(1) * node_mask_i.unsqueeze(2)
                    context_i = context[idx].unsqueeze(0)
                    x_i, h_i = self.sample_p_xh_given_z0(
                z_i, node_mask_i, edge_mask_i, context=context_i, fix_noise=False
            )
                    loss, _ = self.compute_loss(       
                            x_i,
                            h_i,
                            node_mask=node_mask_i,
                            edge_mask=edge_mask_i,
                            context=context_i,
                            reference_indices=None,
                            t0_always=True)
                    if not torch.isinf(loss).any():
                        x_refined.append(x_i)
                        h_refined.append(h_i)
                        if self.debug:
                            logger.info(f"Found clean molecule at chain frame {i}")
                        break
                    else:
                        # Cannot find a clean molecule in the look-back frames
                        if i == n_frame_look_back:
                            x_refined.append(x[idx].unsqueeze(0))
                            h_refined.append(
                                {"categorical" : h["categorical"][idx].unsqueeze(0),
                                 "integer"   : h["integer"][idx].unsqueeze(0),
                                 "extra"     : h["extra"][idx].unsqueeze(0) if self.ndim_extra > 0 else None        }
                            )
            else:
                x_refined.append(x[idx].unsqueeze(0))
                h_refined.append(
                    {"categorical" : h["categorical"][idx].unsqueeze(0),
                     "integer"   : h["integer"][idx].unsqueeze(0),
                     "extra"     : h["extra"][idx].unsqueeze(0) if self.ndim_extra > 0 else None        }
                )        
        
        h_final = {}
        x_refined = torch.cat(x_refined, dim=0)
        h_final["categorical"] = torch.cat([h_i["categorical"] for h_i in h_refined], dim=0)
        h_final["integer"] = torch.cat([h_i["integer"] for h_i in h_refined], dim=0)
        if self.ndim_extra > 0:
            h_final["extra"] = torch.cat([h_i["extra"] for h_i in h_refined], dim=0)
        else:
            h_final["extra"] = None

        return x_refined, h_final


    def scale_schedule(self, t, initial_scale=10.0, final_scale=1.0, schedule_type="linear"):
        """
        Calculates a dynamic guidance scale.
        Example: Linear decay from an initial scale to a final scale.
        """
        # t is a tensor of shape [B, 1], normalized between 0 and 1
        t_normalized = t.squeeze()
        x = 1 - t_normalized
        schedule_type = schedule_type.lower()
        if schedule_type == "linear":
            scale = initial_scale + (final_scale - initial_scale) * x
        elif schedule_type == "exponential":
            # Using a power of 2 for a gentle curve
            scale = initial_scale + (final_scale - initial_scale) * (x**2)
        elif schedule_type == "cosine":
            scale = initial_scale + (final_scale - initial_scale) * (
                1 - torch.cos(x * math.pi / 2)
            )
        else:
            raise ValueError(f"Unknown scale schedule: {schedule_type}")
        return scale


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
