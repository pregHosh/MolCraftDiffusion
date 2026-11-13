"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.equifm.eot``; moved here unchanged."""

import torch
from MolecularDiffusion.modules.models.equifm.eot import (
    solve_eot,
)


def _demo() -> None:
    """Self-check: EOT must exactly recover a known permutation + rotation."""
    torch.manual_seed(0)
    n = 12
    x0 = torch.randn(1, n, 3)
    x0 = x0 - x0.mean(dim=1, keepdim=True)
    # Build noise that IS a rotated+permuted copy of the data: EOT should undo
    # both and land on x0 itself.
    angle = torch.tensor(0.7)
    rot = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    perm = torch.randperm(n)
    eps = (x0[0] @ rot.T)[perm].unsqueeze(0)
    mask = torch.ones(1, n, 1)

    out = solve_eot(eps, x0, mask)
    err = (out - x0).abs().max().item()
    assert err < 1e-4, f"EOT failed to recover the alignment: max err {err}"

    # Padded rows must stay exactly zero and must not influence the alignment.
    eps_pad = torch.cat([eps, torch.zeros(1, 4, 3)], dim=1)
    x0_pad = torch.cat([x0, torch.zeros(1, 4, 3)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(1, 4, 1)], dim=1)
    out_pad = solve_eot(eps_pad, x0_pad, mask_pad)
    assert out_pad[:, n:].abs().max().item() == 0.0
    assert (out_pad[:, :n] - out).abs().max().item() < 1e-6
    print("eot._demo: OK")


def test_models_equifm_eot_self_check():
    _demo()
