"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.apo2mol.residue_ops``; moved here unchanged."""

import torch
from MolecularDiffusion.modules.models.apo2mol.residue_ops import (
    axis_angle_to_quaternion,
    axis_angle_to_rotation_matrix,
    quaternion_product,
    quaternion_to_rotation_matrix,
    slerp_identity_to_q,
)


def _self_check() -> None:
    """Pin the inlined kornia conversions. See tests/test_apo2mol_residue_ops.py."""
    # 90 degrees about +z, (w, x, y, z) = (cos45, 0, 0, sin45)
    s = 2.0 ** -0.5
    q = torch.tensor([[s, 0.0, 0.0, s]])
    expect = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    got = quaternion_to_rotation_matrix(q)
    assert torch.allclose(got, expect, atol=1e-6), got
    # x-hat -> y-hat under R @ v
    v = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(got[0] @ v, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

    aa = torch.tensor([[0.0, 0.0, torch.pi / 2]])
    assert torch.allclose(axis_angle_to_rotation_matrix(aa), expect, atol=1e-6)
    assert torch.allclose(axis_angle_to_quaternion(aa), q, atol=1e-6)

    # zero rotation must not produce NaN in either branch
    z = torch.zeros(1, 3)
    assert torch.allclose(
        axis_angle_to_rotation_matrix(z), torch.eye(3)[None], atol=1e-6
    )
    assert torch.allclose(
        axis_angle_to_quaternion(z), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )

    # round trip on random rotations
    torch.manual_seed(0)
    aa = torch.randn(64, 3)
    q2 = axis_angle_to_quaternion(aa)
    assert torch.allclose(
        quaternion_to_rotation_matrix(q2),
        axis_angle_to_rotation_matrix(aa),
        atol=1e-5,
    )
    # rotation matrices are orthonormal with det +1
    r = quaternion_to_rotation_matrix(q2)
    eye = torch.eye(3).expand_as(r)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(64), atol=1e-5)

    # slerp endpoints
    assert torch.allclose(
        slerp_identity_to_q(q, torch.ones(1, 1)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(slerp_identity_to_q(q, torch.zeros(1, 1)), q, atol=1e-6)

    # Hamilton product: two 90-degree z rotations compose to 180 degrees
    assert torch.allclose(
        quaternion_product(q, q), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), atol=1e-6
    )
    print("apo2mol residue_ops self-check OK")


def test_models_apo2mol_residue_ops_self_check():
    _self_check()
