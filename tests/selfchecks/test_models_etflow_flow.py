"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.etflow.flow``; moved here unchanged."""

import torch
from MolecularDiffusion.modules.models.etflow.flow import (
    HarmonicSampler,
    extend_bond_index,
    find_rigid_alignment,
    switch_parity_of_pos,
)


def _demo() -> None:
    """Smallest runnable check of the two pieces that can silently be wrong."""
    torch.manual_seed(0)

    # Kabsch really removes a rotation + translation.
    a = torch.randn(12, 3)
    angle = torch.tensor(0.7)
    rot = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    b = a @ rot.T + torch.tensor([3.0, -1.0, 2.0])
    r, t = find_rigid_alignment(a, b)
    assert torch.allclose((r @ a.T).T + t, b, atol=1e-4), "Kabsch is wrong"
    assert torch.det(r) > 0, "Kabsch returned a reflection"

    # Harmonic prior of a 4-atom chain is finite and COM-free per molecule.
    bond = torch.tensor([[0, 1, 2], [1, 2, 3]])
    both = torch.cat([bond, bond.flip(0)], dim=1)
    batch = torch.zeros(4, dtype=torch.long)
    x0 = HarmonicSampler().sample((4, 3), both, batch, keys=["chain4"])
    assert torch.isfinite(x0).all(), "harmonic prior produced NaN"
    assert x0.mean(0).abs().max() < 1e-4, "harmonic prior is not COM-free"

    # A disconnected graph is the documented failure, not a silent bad sample.
    lone = torch.tensor([[0, 1], [1, 0]])
    bad = HarmonicSampler().sample((4, 3), lone, batch)
    assert torch.isnan(bad).any(), "disconnected graph should blow up"

    # extend_bond_index refuses a non-mirrored bond list.
    pos = torch.randn(4, 3)
    edge_index, edge_type = extend_bond_index(pos, both, batch)
    assert int((edge_type > 0).sum()) == both.shape[1]
    assert edge_index.shape[1] >= both.shape[1]
    try:
        extend_bond_index(pos, torch.cat([both, both], dim=1), batch)
    except ValueError:
        pass
    else:  # pragma: no cover
        msg = "extend_bond_index accepted duplicated bond edges"
        raise AssertionError(msg)

    # Parity switching is a no-op when there is nothing chiral.
    empty_i = torch.zeros(1, 0, dtype=torch.long)
    out = switch_parity_of_pos(
        pos, empty_i, empty_i, torch.zeros(0), batch
    )
    assert torch.equal(out, pos), "parity switch mangled an achiral molecule"

    print("etflow.flow demo OK")


def test_models_etflow_flow_self_check():
    _demo()
