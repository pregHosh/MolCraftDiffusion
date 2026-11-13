"""Per-residue rigid + side-chain transforms, and the quaternion math for them.

This is the half of Apo2Mol that no other pocket model in-tree has: the pocket
is not a fixed condition, it is *co-generated*. A residue's holo pose is
represented as ``(rigid rotation quaternion, translation, 5 chi angles)``
relative to its apo pose, and :func:`apply_transforms_tensor_batch` rebuilds
Cartesian pocket coordinates from that representation. Ported from
``others/Apo2Mol/utils/data.py`` (``CHI_BOND_DICTS`` at :303-388,
``apply_transforms_tensor`` at :470-534, ``apply_transforms_tensor_batch`` at
:535-596).

## Why the three ``kornia`` conversions are inlined here

Upstream imports ``kornia`` for exactly three functions. ``kornia`` is not a
dependency of this repo and is not installed in ``moleculardiffusion_dev``;
per the approved integration plan we inline them instead of taking the
dependency. That is ~40 lines, but it is 40 lines the released checkpoint was
trained under, so a sign or ordering slip would be **silent** -- the model
would still emit plausible pockets, just wrong ones.

Conventions, matched to kornia exactly:

* quaternions are ``(w, x, y, z)`` (kornia's ``QuaternionCoeffOrder.WXYZ``,
  the default since kornia 0.6);
* :func:`quaternion_to_rotation_matrix` does **not** normalise its input,
  matching ``kornia.geometry.conversions.quaternion_to_rotation_matrix``;
* rotation matrices act as ``R @ v`` on column vectors.

``tests/test_apo2mol_residue_ops.py`` pins all three against hand-computed
values plus round-trip identities. Run ``python -m
MolecularDiffusion.modules.models.apo2mol.residue_ops`` for the same checks
as a standalone self-check.

Not ported: ``apply_transforms`` (the numpy twin, no call sites) and
``compute_residue_transforms`` (converter-only -- it lives in
``docs/model_integrations/apo2mol/scripts/convert_dataset.py``, where scipy
replaces ``numpy-quaternion``).
"""

from __future__ import annotations

from typing import List

import torch

__all__ = [
    "CHI_BOND_DICTS",
    "CHI_ORDER",
    "MAX_CHI",
    "apply_transforms_tensor",
    "apply_transforms_tensor_batch",
    "axis_angle_to_quaternion",
    "axis_angle_to_rotation_matrix",
    "quaternion_product",
    "quaternion_to_rotation_matrix",
    "slerp_identity_to_q",
]

CHI_ORDER = ["chi1", "chi2", "chi3", "chi4", "chi5"]
MAX_CHI = 5

# --------------------------------------------------------------------------- #
# kornia replacements (see the module docstring for why these are inlined)
# --------------------------------------------------------------------------- #

_EPS = 1e-6


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """``(*, 4)`` ``(w,x,y,z)`` -> ``(*, 3, 3)``.

    Replaces ``kornia.geometry.conversions.quaternion_to_rotation_matrix``
    (``utils/data.py:490``, ``models/molopt_score_model.py:654,760``).
    Deliberately does NOT normalise the input, exactly as kornia does not:
    callers here always pass unit quaternions, and silently normalising would
    hide a caller bug.
    """
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"expected a (*, 4) quaternion, got shape {tuple(quaternion.shape)}"
        )
    w, x, y, z = torch.unbind(quaternion, dim=-1)

    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    one = torch.ones_like(w)

    matrix = torch.stack(
        [
            one - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, one - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, one - (txx + tyy),
        ],
        dim=-1,
    )
    return matrix.view(*quaternion.shape[:-1], 3, 3)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """``(*, 3)`` rotation vector -> ``(*, 3, 3)`` (Rodrigues).

    Replaces ``kornia.geometry.conversions.axis_angle_to_rotation_matrix``
    (``utils/data.py:524``). kornia branches to a first-order Taylor form
    below ``theta^2 = 1e-6``; ``I + [w]_x`` is exactly that first-order form,
    so the two agree to float precision on both sides of the branch.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"expected a (*, 3) axis-angle, got shape {tuple(axis_angle.shape)}"
        )
    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    # kornia divides by (theta + eps); reproduced so the sub-eps branch lines up
    unit = axis_angle / (theta + _EPS)
    wx, wy, wz = torch.unbind(unit, dim=-1)
    cos_t = torch.cos(theta).squeeze(-1)
    sin_t = torch.sin(theta).squeeze(-1)
    one_minus = 1.0 - cos_t
    one = torch.ones_like(cos_t)

    full = torch.stack(
        [
            cos_t + wx * wx * one_minus,
            wx * wy * one_minus - wz * sin_t,
            wy * sin_t + wx * wz * one_minus,
            wz * sin_t + wx * wy * one_minus,
            cos_t + wy * wy * one_minus,
            wy * wz * one_minus - wx * sin_t,
            -wy * sin_t + wx * wz * one_minus,
            wx * sin_t + wy * wz * one_minus,
            cos_t + wz * wz * one_minus,
        ],
        dim=-1,
    )

    rx, ry, rz = torch.unbind(axis_angle, dim=-1)
    taylor = torch.stack(
        [one, -rz, ry, rz, one, -rx, -ry, rx, one], dim=-1
    )

    use_full = (theta2.squeeze(-1) > _EPS).unsqueeze(-1)
    matrix = torch.where(use_full, full, taylor)
    return matrix.view(*axis_angle.shape[:-1], 3, 3)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """``(*, 3)`` rotation vector -> ``(*, 4)`` ``(w,x,y,z)`` unit quaternion.

    Replaces ``kornia.geometry.conversions.axis_angle_to_quaternion``
    (``models/molopt_score_model.py:606``). kornia uses the Taylor expansion
    ``sin(theta/2)/theta -> 0.5 - theta^2/48`` below ``theta^2 = 1e-6``; the
    same branch is reproduced so behaviour matches at theta = 0 (where the
    naive form is 0/0).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"expected a (*, 3) axis-angle, got shape {tuple(axis_angle.shape)}"
        )
    theta2 = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    half = 0.5 * theta

    # theta is 0 in the taylor branch, so guard the division before selecting.
    safe_theta = torch.where(theta2 > _EPS, theta, torch.ones_like(theta))
    exact = torch.sin(half) / safe_theta
    taylor = 0.5 - theta2 / 48.0
    scale = torch.where(theta2 > _EPS, exact, taylor)

    w = torch.where(theta2 > _EPS, torch.cos(half), 1.0 - theta2 / 8.0)
    return torch.cat([w, axis_angle * scale], dim=-1)


def quaternion_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, ``(w, x, y, z)`` layout.

    Ported from ``models/molopt_score_model.py:968-978``.
    """
    w1, w2 = q1[..., :1], q2[..., :1]
    v1, v2 = q1[..., 1:], q2[..., 1:]
    w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)
    v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
    return torch.cat([w, v], dim=-1)


def slerp_identity_to_q(q: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation from the identity rotation towards ``q``.

    ``q``: ``(B, 4)``; ``lambdas``: ``(B, 1)`` -- the weight on the IDENTITY
    end, so ``lambdas = 1`` returns identity and ``lambdas = 0`` returns ``q``.
    Ported from ``models/molopt_score_model.py:564-588`` (this is the pocket
    channel's noising interpolant, not a DDPM step).
    """
    q0 = torch.zeros_like(q)
    q0[:, 0] = 1.0  # identity quaternion, w = 1
    q1 = q / q.norm(dim=-1, keepdim=True)

    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)  # shortest arc
    dot = dot.abs().clamp(-1, 1)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    mask = sin_theta > 1e-6
    s0 = torch.where(mask, torch.sin(lambdas * theta) / sin_theta, lambdas)
    s1 = torch.where(
        mask, torch.sin((1 - lambdas) * theta) / sin_theta, 1 - lambdas
    )

    out = s0 * q0 + s1 * q1
    return out / out.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# chi-angle bond tables (utils/data.py:303-388)
#
# For each residue and each chi slot: (atom1, atom2, [atoms rotated about the
# atom1->atom2 bond]). `None` means the residue has no such chi angle.
# --------------------------------------------------------------------------- #

_CHI1 = {
    "ALA": None,
    "ARG": ("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN": ("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP": ("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS": ("CA", "CB", ["SG"]),
    "GLN": ("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU": ("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY": None,
    "HIS": ("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE": ("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU": ("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS": ("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET": ("CA", "CB", ["CG", "SD", "CE"]),
    "PHE": ("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO": ("CA", "CB", ["CG", "CD"]),
    "SER": ("CA", "CB", ["OG"]),
    "THR": ("CA", "CB", ["CG2", "OG1"]),
    "TRP": (
        "CA", "CB",
        ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"],
    ),
    "TYR": ("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL": ("CA", "CB", ["CG1", "CG2"]),
}

_CHI2 = {
    "ALA": None,
    "ARG": ("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN": ("CB", "CG", ["ND2", "OD1"]),
    "ASP": ("CB", "CG", ["OD1", "OD2"]),
    "CYS": None,
    "GLN": ("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU": ("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY": None,
    "HIS": ("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE": ("CB", "CG1", ["CD1"]),
    "LEU": ("CB", "CG", ["CD1", "CD2"]),
    "LYS": ("CB", "CG", ["CD", "CE", "NZ"]),
    "MET": ("CB", "CG", ["SD", "CE"]),
    "PHE": ("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO": ("CB", "CG", ["CD"]),
    "SER": None,
    "THR": None,
    "TRP": (
        "CB", "CG",
        ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"],
    ),
    "TYR": ("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL": None,
}

_CHI3 = {
    "ALA": None,
    "ARG": ("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN": None,
    "ASP": None,
    "CYS": None,
    "GLN": ("CG", "CD", ["NE2", "OE1"]),
    "GLU": ("CG", "CD", ["OE1", "OE2"]),
    "GLY": None,
    "HIS": None,
    "ILE": None,
    "LEU": None,
    "LYS": ("CG", "CD", ["CE", "NZ"]),
    "MET": ("CG", "SD", ["CE"]),
    "PHE": None,
    "PRO": None,
    "SER": None,
    "THR": None,
    "TRP": None,
    "TYR": None,
    "VAL": None,
}

_CHI4 = {
    "ARG": ("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS": ("CD", "CE", ["NZ"]),
}

_CHI5 = {
    "ARG": ("NE", "CZ", ["NH1", "NH2"]),
}

CHI_BOND_DICTS = [_CHI1, _CHI2, _CHI3, _CHI4, _CHI5]


# --------------------------------------------------------------------------- #
# coordinate reconstruction
# --------------------------------------------------------------------------- #


def apply_transforms_tensor(
    protein_pos: torch.Tensor,
    protein_atom_name: List[str],
    protein_atom_to_aa_name: List[str],
    protein_atom_to_aa_group: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    chi_update: torch.Tensor,
    chi_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply one protein's per-residue rigid transform, then the chi deltas.

    Ported from ``utils/data.py:470-534``.

    Args:
        protein_pos: ``(N, 3)`` this protein's atom coordinates.
        protein_atom_name: length ``N`` PDB atom names.
        protein_atom_to_aa_name: length ``N`` residue names.
        protein_atom_to_aa_group: ``(N,)`` atom -> LOCAL residue id.
        rotations: ``(M, 4)`` per-residue quaternions, row ``r`` matching the
            ``r``-th distinct residue id in ascending order.
        translations: ``(M, 3)``.
        chi_update: ``(M, 5)`` delta-chi in radians.
        chi_mask: ``(M, 5)`` 0/1.

    Returns:
        ``(N, 3)`` transformed coordinates, in the input's atom order.

    # ponytail: the chi pass is a Python loop over residues x 5 slots, as
    # upstream. It runs on 5 of 1000 reverse steps (protein_update_steps), so
    # it is not the bottleneck; vectorise per chi-slot if that ever changes.
    """
    if protein_pos.ndim != 2 or protein_pos.size(1) != 3:
        raise ValueError(
            f"protein_pos must be (N, 3), got {tuple(protein_pos.shape)}"
        )

    # 1. rigid part, vectorised over atoms
    rot_mats = quaternion_to_rotation_matrix(rotations)  # (M, 3, 3)
    row_idx = protein_atom_to_aa_group
    new_pos = (
        torch.einsum(
            "ni,nij->nj", protein_pos, rot_mats[row_idx].transpose(-1, -2)
        )
        + translations[row_idx]
    )

    # 2. side-chain torsions, per residue
    gid_unique = torch.unique(row_idx).tolist()
    for row, gid in enumerate(gid_unique):
        idxs = (row_idx == gid).nonzero(as_tuple=True)[0]
        resname = protein_atom_to_aa_name[idxs[0].item()]
        name2idx = {protein_atom_name[i.item()]: i.item() for i in idxs}

        for chi_slot in range(MAX_CHI):
            if chi_mask[row, chi_slot] == 0:
                continue
            bond_dict = CHI_BOND_DICTS[chi_slot]
            if resname not in bond_dict or bond_dict[resname] is None:
                continue
            atom1, atom2, rot_atoms = bond_dict[resname]
            if atom1 not in name2idx or atom2 not in name2idx:
                continue

            p1 = new_pos[name2idx[atom1]]
            p2 = new_pos[name2idx[atom2]]
            axis = p2 - p1
            norm = torch.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis_unit = axis / norm
            theta = chi_update[row, chi_slot]
            rot_mat = axis_angle_to_rotation_matrix(
                (axis_unit * theta).unsqueeze(0)
            )[0]

            for at in rot_atoms:
                if at not in name2idx:
                    continue
                k = name2idx[at]
                v = new_pos[k] - p1
                new_pos[k] = p1 + v @ rot_mat.T

    return new_pos


def apply_transforms_tensor_batch(
    protein_pos: torch.Tensor,
    protein_atom_name: List[List[str]],
    protein_atom_to_aa_name: List[List[str]],
    protein_atom_to_aa_group: torch.Tensor,
    protein_element_batch: torch.Tensor,
    rotations: torch.Tensor,
    translations: torch.Tensor,
    chi_update: torch.Tensor,
    chi_mask: torch.Tensor,
    protein_translations_batch: torch.Tensor,
) -> torch.Tensor:
    """Batched :func:`apply_transforms_tensor` (``utils/data.py:535-596``).

    ``protein_atom_name`` / ``protein_atom_to_aa_name`` are **nested per
    complex** (``list[list[str]]``), not flat, and
    ``protein_atom_to_aa_group`` restarts at 0 in every complex. The collate
    in ``data/component/apo2mol_data.py`` preserves both properties; flatten
    either one and this returns wrong coordinates without erroring.
    """
    new_pos = protein_pos.clone()

    num_proteins = len(protein_atom_name)
    if num_proteins != len(protein_atom_to_aa_name):
        raise ValueError(
            "protein_atom_name and protein_atom_to_aa_name must have one "
            f"entry per complex; got {num_proteins} and "
            f"{len(protein_atom_to_aa_name)}."
        )

    for p_idx in range(num_proteins):
        atom_mask = protein_element_batch.squeeze(-1) == p_idx
        if atom_mask.sum() == 0:
            continue
        resid_mask = protein_translations_batch.squeeze(-1) == p_idx

        new_pos[atom_mask] = apply_transforms_tensor(
            protein_pos[atom_mask],
            protein_atom_name[p_idx],
            protein_atom_to_aa_name[p_idx],
            protein_atom_to_aa_group[atom_mask],
            rotations[resid_mask],
            translations[resid_mask],
            chi_update[resid_mask],
            chi_mask[resid_mask],
        )

    return new_pos
