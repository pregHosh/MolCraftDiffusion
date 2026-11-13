"""Self-contained substituent-fragment placement for outpaint skeleton init.

Uses the bundled idealized fragment geometries in ``substituent_geometries.py``
(attachment atom = row 0, bond pointing along local +x) to build clean,
connector-anchored placeholder coordinates for the to-be-generated atoms. Only
coordinates are used (a geometry prior); node features are randomized and
denoised. No external dependency.
"""

import torch

from MolecularDiffusion.utils.geometry.substituent_geometries import SUBSTITUENT_GEOMETRIES


def select_fragment(tags_all, n_extra):
    """Pick the bundled fragment matching all ``tags_all`` closest to n_extra.

    Among fragments whose ``tags`` include every tag in ``tags_all`` (empty
    ``tags_all`` matches any), prefer the one whose atom count is >= n_extra and
    closest to it (so we subset rather than extend); otherwise the largest
    available. Returns ``(name, coords_tensor (M, 3))`` or ``(None, None)``.
    """
    tags_all = set(tags_all or [])
    cands = [
        (name, d)
        for name, d in SUBSTITUENT_GEOMETRIES.items()
        if tags_all.issubset(set(d["tags"]))
    ]
    if not cands:
        return None, None

    def _key(item):
        m = len(item[1]["coords"])
        # prefer m >= n_extra (flag 0) over m < n_extra (flag 1), then closeness.
        return (0 if m >= n_extra else 1, abs(m - n_extra))

    name, d = min(cands, key=_key)
    coords = torch.tensor(d["coords"], dtype=torch.float32)
    return name, coords


def _rotation_x_to(direction):
    """Rotation matrix mapping the local +x axis onto unit ``direction`` (3,)."""
    a = torch.tensor([1.0, 0.0, 0.0], device=direction.device)
    b = direction / (torch.norm(direction) + 1e-8)
    v = torch.cross(a, b, dim=0)
    c = torch.dot(a, b)
    if c > 0.9999:
        return torch.eye(3, device=direction.device)
    if c < -0.9999:
        # 180° about z maps +x -> -x
        return torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=direction.device))
    vx = torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        device=direction.device,
    )
    return torch.eye(3, device=direction.device) + vx + vx @ vx * (1.0 / (1.0 + c))


def _roll_about(axis, angle):
    """Rotation by ``angle`` (rad) about unit ``axis`` (3,) — Rodrigues."""
    k = axis / (torch.norm(axis) + 1e-8)
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        device=axis.device,
    )
    return (
        torch.eye(3, device=axis.device)
        + torch.sin(angle) * K
        + (1.0 - torch.cos(angle)) * (K @ K)
    )


def place_fragment(frag_coords, connector_pos, direction, roll_angle=None):
    """Align a local fragment so its bond points along ``direction`` at the connector.

    The fragment's attachment atom (row 0, local ~(bond, 0, 0)) lands at
    ``connector_pos + bond * direction``; an optional roll about ``direction``
    adds orientational diversity. Returns placed coords ``(M, 3)``.
    """
    device = connector_pos.device
    frag = frag_coords.to(device)
    R = _rotation_x_to(direction)
    if roll_angle is not None:
        R = _roll_about(direction, roll_angle) @ R
    return frag @ R.T + connector_pos
