"""SO(3) irreps: real spherical harmonics and Clebsch-Gordan coefficients.

A direct PyTorch reimplementation of the `e3x <https://github.com/google-research/e3x>`_
surface DiTMC calls (Apache-2.0). **Not** e3nn: e3x's ordering, normalization
and parity layout are all different, and matching them is what makes the
published DiTMC checkpoints convertible.

Conventions, all pinned here and never left implicit (``e3x/config.py`` sets
``cartesian_order=True``, ``normalization='racah'``, ``use_fused_tensor=False``):

* **Racah / Schmidt semi-normalization** -- the normalization constant is
  literally ``1``, i.e. ``∫ Y_lm Y_l'm' dΩ = 4π/(2l+1)·δδ``.
* **No Condon-Shortley phase.**
* **Cartesian order** within a degree: ``m = +l, -l, +(l-1), -(l-1), ..., 0``.
  So degree 1 evaluates to ``(x, y, z)`` -- *not* the ``(y, z, x)`` an
  m-ascending convention (e3nn's) would give.

The coefficient tables in ``_tables.npz`` are produced by
``docs/model_integrations/ditmc/scripts/generate_e3x_tables.py``, which runs
e3x's own SymPy generator verbatim.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch

_TABLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tables.npz")


@functools.lru_cache(maxsize=1)
def _tables() -> dict:
    with np.load(_TABLES_PATH) as data:
        return {
            "max_degree": int(data["max_degree"]),
            "cm": data["cm"],
            "ls": data["ls"],
            "cg": data["cg"],
        }


def _check_degree(degree: int) -> None:
    if degree < 0:
        msg = f"degree must be positive or zero, received {degree}"
        raise ValueError(msg)
    if degree > _tables()["max_degree"]:
        msg = (
            f"degree {degree} exceeds the generated table max_degree "
            f"{_tables()['max_degree']}; regenerate _tables.npz with "
            f"scripts/generate_e3x_tables.py --max-degree {degree}"
        )
        raise ValueError(msg)


def cartesian_permutation_for_degree(l: int) -> np.ndarray:  # noqa: E741
    """Permutation from m-ascending to Cartesian order, for one degree.

    Verbatim port of ``e3x/so3/_common._cartesian_permutation_for_degree``.
    For ``l=2`` this is ``[4, 0, 3, 1, 2]``.
    """
    _check_degree(l)
    p = np.empty(2 * l + 1, dtype=np.int64)
    i = 0
    for m in range(l):
        p[i] = 2 * l + 1 - (m + 1)
        i += 1
        p[i] = m
        i += 1
    p[i] = (2 * l + 1) // 2
    return p


def cartesian_permutation(max_degree: int) -> np.ndarray:
    """Permutation to Cartesian order for all degrees ``0..max_degree``."""
    _check_degree(max_degree)
    p = np.empty((max_degree + 1) ** 2, dtype=np.int64)
    for l in range(max_degree + 1):  # noqa: E741
        p[l**2 : (l + 1) ** 2] = cartesian_permutation_for_degree(l) + l**2
    return p


def _num_monomials(max_degree: int) -> int:
    return sum(((l + 1) * (l + 2)) // 2 for l in range(max_degree + 1))  # noqa: E741


@functools.lru_cache(maxsize=8)
def _sh_coefficients(max_degree: int) -> tuple:
    """(cm, ls) for ``max_degree``, Racah-normalized and Cartesian-ordered."""
    _check_degree(max_degree)
    t = _tables()
    num_car = _num_monomials(max_degree)
    num_sph = (max_degree + 1) ** 2
    cm = t["cm"][:num_car, :num_sph].copy()
    ls = t["ls"][:num_car].copy()
    # Racah normalization constant is 1 for every degree, so there is nothing
    # to multiply here -- kept explicit so a future normalization is one line.
    cm = cm[:, cartesian_permutation(max_degree)]
    return cm, ls


@functools.lru_cache(maxsize=8)
def _cg_table(max_degree1: int, max_degree2: int, max_degree3: int) -> np.ndarray:
    for d in (max_degree1, max_degree2, max_degree3):
        _check_degree(d)
    cg = _tables()["cg"][
        : (max_degree1 + 1) ** 2,
        : (max_degree2 + 1) ** 2,
        : (max_degree3 + 1) ** 2,
    ]
    p1 = cartesian_permutation(max_degree1)
    p2 = cartesian_permutation(max_degree2)
    p3 = cartesian_permutation(max_degree3)
    return cg[p1, :, :][:, p2, :][:, :, p3].copy()


def clebsch_gordan(
    max_degree1: int,
    max_degree2: int,
    max_degree3: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Real-SH Clebsch-Gordan coefficients, Cartesian order.

    Shape ``((L1+1)**2, (L2+1)**2, (L3+1)**2)``. ``cg[0, 0, 0] == 1`` exactly.
    """
    cg = _cg_table(max_degree1, max_degree2, max_degree3)
    return torch.as_tensor(cg, dtype=dtype, device=device)


def _integer_powers(x: torch.Tensor, max_degree: int) -> torch.Tensor:
    """All integer powers ``0..max_degree`` of ``x`` along axis -2.

    Via ``cumprod`` rather than ``pow`` on purpose: ``pow`` is not NaN-safe in
    its gradient at 0, and the input here is a unit vector that can have zero
    components. Same reasoning as e3x's own comment.
    """
    ones = torch.ones_like(x)
    if max_degree == 0:
        return ones
    rep = x.expand(*x.shape[:-2], max_degree, x.shape[-1])
    return torch.cumprod(torch.cat((ones, rep), dim=-2), dim=-2)


def spherical_harmonics(
    r: torch.Tensor,
    max_degree: int,
    *,
    r_is_normalized: bool = True,
) -> torch.Tensor:
    """Real spherical harmonics, Racah-normalized, Cartesian order.

    Args:
        r: ``(..., 3)`` Cartesian vectors.
        max_degree: maximum degree ``L``.
        r_is_normalized: if ``False``, ``r`` is normalized first.

    Returns:
        ``(..., (L+1)**2)``. Degree ``l`` occupies ``[l**2, (l+1)**2)``.
        For a unit vector ``(x, y, z)``: index 0 is ``1``; indices 1..3 are
        ``x, y, z``; indices 4..8 are
        ``√3/2(x²−y²), √3xy, √3xz, √3yz, (3z²−1)/2``.
    """
    if r.shape[-1] != 3:
        msg = f"r must have shape (..., 3), received {tuple(r.shape)}"
        raise ValueError(msg)
    _check_degree(max_degree)

    cm_np, ls_np = _sh_coefficients(max_degree)
    cm = torch.as_tensor(cm_np, dtype=r.dtype, device=r.device)
    ls = torch.as_tensor(ls_np, dtype=torch.long, device=r.device)

    if not r_is_normalized:
        norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        r = r / torch.where(norm > 0, norm, torch.ones_like(norm))

    rp = _integer_powers(r.unsqueeze(-2), max_degree)  # (..., L+1, 3)
    monomials = (
        rp[..., 0].index_select(-1, ls[:, 0])
        * rp[..., 1].index_select(-1, ls[:, 1])
        * rp[..., 2].index_select(-1, ls[:, 2])
    )
    return monomials @ cm


def random_rotation(
    num: int = 1,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Haar-uniform ``SO(3)`` rotation matrices, ``(num, 3, 3)``.

    Shoemake's uniform-quaternion construction, same as
    ``e3x.so3.random_rotation`` at ``perturbation=1.0`` (the only value DiTMC
    uses). The RNG stream cannot match JAX's, so this is a distributional
    match, not a bit-level one -- which is all rotation augmentation needs.
    """
    u = torch.rand(num, 3, generator=generator, device=device, dtype=dtype)
    twopi = 2 * torch.pi
    sqrt1 = torch.sqrt(1 - u[:, 0])
    sqrt2 = torch.sqrt(u[:, 0])
    a1 = twopi * u[:, 1]
    a2 = twopi * u[:, 2]
    r, i, j, k = (
        sqrt1 * torch.sin(a1),
        sqrt1 * torch.cos(a1),
        sqrt2 * torch.sin(a2),
        sqrt2 * torch.cos(a2),
    )
    i2, j2, k2 = i * i, j * j, k * k
    ij, ik, jk, ir, jr, kr = i * j, i * k, j * k, i * r, j * r, k * r
    row1 = torch.stack((1 - 2 * (j2 + k2), 2 * (ij - kr), 2 * (ik + jr)), dim=-1)
    row2 = torch.stack((2 * (ij + kr), 1 - 2 * (i2 + k2), 2 * (jk - ir)), dim=-1)
    row3 = torch.stack((2 * (ik - jr), 2 * (jk + ir), 1 - 2 * (i2 + j2)), dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)
