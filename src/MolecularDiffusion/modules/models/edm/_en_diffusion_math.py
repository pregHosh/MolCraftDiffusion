"""Pure helper functions of the EDM diffusion model (moved out of ``en_diffusion``).

They are re-exported from ``en_diffusion``, so ``from ...en_diffusion import gaussian_KL`` still works.
"""

from MolecularDiffusion.utils.chem.geom_analyzer import symbol2num
from torch.nn import functional as F
import math
import numpy as np
import torch


def _resolve_jitter_scale(config, init_method, forward_noise):
    """Return an explicit jitter scale without falling back to ``spread``."""
    jitter_scale = config.get("jitter_scale")
    uses_jitter = (
        init_method != "seed" and str(forward_noise).strip().lower() == "jitter"
    )
    if uses_jitter and jitter_scale is None:
        raise ValueError(
            "jitter_scale must be set explicitly when forward_noise='jitter'; "
            "it no longer falls back to spread."
        )
    return 0.75 if jitter_scale is None else float(jitter_scale)


def _n_int(model: torch.nn.Module) -> int:
    """Width of a model's trailing integer (atomic-number) block.

    ``getattr`` with a ``True`` default so any object that borrows these
    methods without declaring ``include_charges`` keeps the historical
    one-column layout.
    """
    return int(getattr(model, "include_charges", True))


def split_feature_axis(
    model: torch.nn.Module, t: torch.Tensor, n_cat: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[..., n_dims | n_cat | n_int]`` along the last axis.

    The integer block is empty for models built with
    ``include_charges=False``, which keeps every downstream ``cat`` / mask /
    ``sum_except_batch`` shape-correct without per-site branching. With
    ``include_charges=True`` these slices are exactly the ``[..., :-1]`` /
    ``[..., -1:]`` arithmetic they replace.
    """
    a = model.n_dims
    b = a + n_cat
    return t[..., :a], t[..., a:b], t[..., b : b + _n_int(model)]


def split_z(
    model: torch.nn.Module, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a normalized latent into ``(x, h_cat (+extra), h_int)``."""
    return split_feature_axis(model, z, model.num_classes)


def split_frame(
    model: torch.nn.Module, xh: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an unnormalized chain frame into ``(x, h_cat, h_int)``.

    ``unnormalize_z`` strips the extra-feature block, so frames carry only
    the core categorical features.
    """
    return split_feature_axis(
        model, xh, model.num_classes - model.ndim_extra
    )


def atomic_numbers(
    model: torch.nn.Module, h_int: torch.Tensor, h_cat: torch.Tensor
) -> torch.Tensor:
    """Per-atom atomic numbers for the geometry/quality checks.

    Returns the model's integer channel when it has one -- the same tensor
    object callers used before. Models built with ``include_charges=False``
    carry no such channel, so the numbers are decoded from the categorical
    one-hot through ``atom_vocab`` instead, falling back to the empty
    integer block when no vocabulary was configured.
    """
    if getattr(model, "include_charges", True) or not getattr(
        model, "atom_vocab", None
    ):
        return h_int
    z_table = torch.tensor(
        [symbol2num.get(sym, 0) for sym in model.atom_vocab],
        device=h_cat.device,
        dtype=torch.long,
    )
    return z_table[torch.argmax(h_cat, dim=-1)].unsqueeze(-1)


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
