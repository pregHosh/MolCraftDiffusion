"""The Wigner ``Jd.pt`` table ships once; every loader reads that copy."""

from pathlib import Path

import torch

LAYERS = (
    Path(__file__).resolve().parents[1]
    / "src/MolecularDiffusion/modules/layers"
)


def test_one_jd_file_under_layers() -> None:
    assert [p.relative_to(LAYERS).as_posix() for p in LAYERS.rglob("Jd.pt")] == [
        "equiformer_v2/Jd.pt"
    ]


def test_equiformer_loaders_share_the_table() -> None:
    from MolecularDiffusion.modules.layers.equiformer_v2 import wigner as a
    from MolecularDiffusion.modules.layers.equiformer_v2_s import wigner as b

    assert len(a._Jd) == len(b._Jd) > 0
    assert all(torch.equal(x, y) for x, y in zip(a._Jd, b._Jd))


def test_esen_backbone_loads_the_shared_table() -> None:
    from MolecularDiffusion.modules.models.esen import eSEN_Backbone

    net = eSEN_Backbone(
        sphere_channels=8, hidden_channels=8, edge_channels=8,
        num_layers=1, num_distance_basis=8, lmax=1, mmax=1,
    )
    assert net.Jd_0.numel() > 0 and net.Jd_1.numel() > 0
