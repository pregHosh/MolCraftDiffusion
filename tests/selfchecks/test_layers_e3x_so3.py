"""Checks that used to run via ``python -m MolecularDiffusion.modules.layers.e3x.so3``; moved here unchanged."""

import numpy as np
import torch
from MolecularDiffusion.modules.layers.e3x.so3 import (
    cartesian_permutation_for_degree,
    clebsch_gordan,
    random_rotation,
    spherical_harmonics,
)


def _self_check() -> None:  # pragma: no cover - run via ``python -m``
    """One runnable check that fails if the tables or the ordering break."""
    torch.manual_seed(0)
    r = torch.randn(64, 3, dtype=torch.float64)
    u = r / r.norm(dim=-1, keepdim=True)
    y = spherical_harmonics(u, 2)
    x, yy, z = u[:, 0], u[:, 1], u[:, 2]
    expect = torch.stack(
        [
            torch.ones_like(x),
            x,
            yy,
            z,
            3**0.5 / 2 * (x**2 - yy**2),
            3**0.5 * x * yy,
            3**0.5 * x * z,
            3**0.5 * yy * z,
            (3 * z**2 - 1) / 2,
        ],
        dim=-1,
    )
    assert torch.allclose(y, expect, atol=1e-12), (y - expect).abs().max()
    assert (cartesian_permutation_for_degree(2) == np.array([4, 0, 3, 1, 2])).all()
    cg = clebsch_gordan(1, 1, 1, dtype=torch.float64)
    assert abs(cg[0, 0, 0].item() - 1.0) < 1e-12
    rot = random_rotation(8, dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
    assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(rot), torch.ones(8, dtype=torch.float64))
    print("e3x.so3 self-check OK")


def test_layers_e3x_so3_self_check():
    _self_check()
