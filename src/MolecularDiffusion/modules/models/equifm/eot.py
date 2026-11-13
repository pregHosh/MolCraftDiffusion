"""Equivariant Optimal Transport coupling -- EquiFM paper Algorithm 3, p. 18.

    1: initialise R = I, tau = inf
    2: repeat
    3:   Pi = argmin_Pi ||Pi (R z)^T - y^T||_2      {Jonker-Volgenant}
    4:   R  = argmin_R  ||R (Pi z)^T - y^T||_2      {Kabsch}
    5:   tau = ||Pi (R z)^T - y^T||_2
    6: until tau converges

``Pi`` is a permutation of the noise nodes, ``R`` a rotation. Jonker-Volgenant
is exactly ``scipy.optimize.linear_sum_assignment`` (scipy is already a declared
dependency, ``pyproject.toml``, and is already used this way at
``runmodes/data/preparation.py:231-261``). Kabsch is an SVD of the 3x3
cross-covariance. Neither needs a new dependency.

Table 4 of the paper reports 4.67 iterations / 1.10 ms per QM9 sample, so the
per-molecule Python loop below is affordable at the paper's batch size of 64.

ponytail: plain per-molecule loop. The assignment problem is inherently
per-molecule (different node counts) and scipy's LSA has no batched form; batch
it with a C-level solver only if profiling shows this dominates a training step.
"""

import torch
from scipy.optimize import linear_sum_assignment


def _kabsch(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Rotation ``R`` (3x3, det = +1) minimising ``||z R^T - y||_F``.

    Both inputs are ``(n, 3)`` and assumed already centred (EquiFM works in the
    zero-CoM subspace throughout, so no re-centring here).
    """
    cov = z.transpose(0, 1) @ y  # (3, 3)
    if cov.device.type == "mps":  # MPS has no float64: do the 3x3 SVD on CPU
        cov = cov.cpu()
    u, _, vt = torch.linalg.svd(cov.double())
    d = torch.sign(torch.linalg.det(vt.transpose(0, 1) @ u.transpose(0, 1)))
    correction = torch.diag(
        torch.tensor([1.0, 1.0, 1.0], dtype=u.dtype, device=u.device)
    ).clone()
    correction[2, 2] = d
    rot = vt.transpose(0, 1) @ correction @ u.transpose(0, 1)
    return rot.to(dtype=z.dtype, device=z.device)


@torch.no_grad()
def solve_eot(
    eps_x: torch.Tensor,
    x_0: torch.Tensor,
    node_mask: torch.Tensor,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Align prior noise to data under the EOT plan (paper Eq. 7 / Algorithm 3).

    Args:
        eps_x: ``(B, N, 3)`` zero-CoM Gaussian noise, padded rows zeroed.
        x_0: ``(B, N, 3)`` zero-CoM data coordinates, padded rows zeroed.
        node_mask: ``(B, N, 1)`` 1 for real atoms.
        max_iters: cap on the alternating loop (paper needs ~5).
        tol: relative change in ``tau`` below which the loop stops.

    Returns:
        ``(B, N, 3)`` -- ``pi*(R* eps_x)``, i.e. ``eps_x`` permuted and rotated
        onto ``x_0``. Same distribution as ``eps_x`` (a permutation and rotation
        of an isotropic zero-CoM Gaussian is that same Gaussian), which is what
        makes Eq. 8 a valid conditional path; only the *coupling* changes.
    """
    aligned = torch.zeros_like(eps_x)
    counts = node_mask.squeeze(-1).sum(dim=1).long()

    for b in range(eps_x.size(0)):
        n = int(counts[b].item())
        if n == 0:
            continue
        z = eps_x[b, :n]
        y = x_0[b, :n]

        rot = torch.eye(3, dtype=z.dtype, device=z.device)
        prev_tau = None
        perm = torch.arange(n, device=z.device)
        for _ in range(max_iters):
            rz = z @ rot.transpose(0, 1)
            # Line 3: optimal permutation of the (rotated) noise onto the data.
            cost = torch.cdist(rz, y).pow(2)
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            # row is 0..n-1 sorted; noise node row[k] pairs with data node col[k].
            # We want perm such that z[perm][i] is the noise assigned to y[i].
            perm = torch.as_tensor(row[col.argsort()], device=z.device)
            pz = z[perm]
            # Line 4: optimal rotation given that permutation.
            rot = _kabsch(pz, y)
            # Line 5: residual.
            tau = torch.linalg.norm(pz @ rot.transpose(0, 1) - y).item()
            if prev_tau is not None and abs(prev_tau - tau) <= tol * max(prev_tau, 1e-8):
                break
            prev_tau = tau

        aligned[b, :n] = z[perm] @ rot.transpose(0, 1)

    return aligned * node_mask
