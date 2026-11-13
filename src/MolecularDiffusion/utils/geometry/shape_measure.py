"""Continuous shape measures for coordination geometries.

A continuous shape measure (CShM) scores how far a set of coordinating atoms
is from an ideal polyhedron.  For a problem structure ``Q`` (the central atom
plus its ``N`` neighbours) and an ideal reference ``P``::

    S(Q, P) = 100 * min  sum_i |q_i - s R p_perm(i)|^2 / sum_i |q_i - q_mean|^2

minimised over permutations of the vertices, rotations ``R`` and a uniform
scale ``s``.  ``S = 0`` means a perfect match; larger is more distorted.
Values are on the conventional 0-100 scale, so they are directly comparable
with published shape-measure tables (e.g. an ideal tetrahedron scores 33.33
against the square-planar reference, and an ideal trigonal prism scores
16.74 against the octahedron).

Reference polyhedra follow the usual convention for shapes that include a
central atom: vertices sit on a unit sphere whose centre is the central atom,
and for the fully-symmetric shapes the vertices also average to that centre.
"Vacant" shapes keep their parent polyhedron's vertices and simply omit one
or more of them, which is why their vertices do *not* average to the centre.

Only geometry is needed -- no external shape package.
"""

from __future__ import annotations

from functools import cache
from itertools import permutations

import numpy as np

__all__ = ["available_shapes", "reference_shape", "shape_measure"]


# ---------------------------------------------------------------------------
# Reference polyhedra
# ---------------------------------------------------------------------------


def _ring(n: int, z: float = 0.0, phase: float = 0.0) -> list[list[float]]:
    """``n`` points evenly spaced on a circle at height ``z`` (unit sphere)."""
    r = float(np.sqrt(max(0.0, 1.0 - z * z)))
    return [
        [
            r * np.cos(2 * np.pi * k / n + phase),
            r * np.sin(2 * np.pi * k / n + phase),
            z,
        ]
        for k in range(n)
    ]


def _pyramid(n: int) -> list[list[float]]:
    """Apex plus an ``n``-ring, whose vertices average to the centre."""
    # 1 + n*cos(theta) = 0  =>  the base sits at z = -1/n
    return [[0.0, 0.0, 1.0]] + _ring(n, z=-1.0 / n)


def _bipyramid(n: int) -> list[list[float]]:
    """Two axial vertices plus an equatorial ``n``-ring."""
    return [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] + _ring(n, z=0.0)


_TETRA = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]
_TETRA = [(np.array(v) / np.sqrt(3.0)).tolist() for v in _TETRA]

_OCTA = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
]

# trigonal bipyramid: axial +/- z, three equatorial
_TBPY = _bipyramid(3)


def _equal_edge_bipyramid() -> list[list[float]]:
    """Triangular bipyramid with all edges equal (Johnson J12).

    Equatorial edge ``r*sqrt(3)`` must equal the axial edge ``sqrt(r^2+h^2)``,
    giving ``h = r*sqrt(2)``.
    """
    r, h = 1.0, np.sqrt(2.0)
    return [[0.0, 0.0, h], [0.0, 0.0, -h]] + [
        [r * np.cos(2 * np.pi * k / 3), r * np.sin(2 * np.pi * k / 3), 0.0]
        for k in range(3)
    ]


def _equal_edge_prism() -> list[list[float]]:
    """Trigonal prism whose triangle edge equals its height (square sides)."""
    r = 1.0
    h = r * np.sqrt(3.0)  # triangle edge = r*sqrt(3)
    top = [
        [r * np.cos(2 * np.pi * k / 3), r * np.sin(2 * np.pi * k / 3), h / 2]
        for k in range(3)
    ]
    bot = [[x, y, -h / 2] for x, y, _ in top]
    return top + bot


def _square_antiprism() -> list[list[float]]:
    """Square antiprism with all edges equal."""
    # squares of edge a=sqrt(2)*r twisted by 45 deg; equal edges fix the height
    r = 1.0
    a = np.sqrt(2.0) * r
    # inter-square edge^2 = h^2 + (r^2 + r^2 - 2 r^2 cos45) = a^2
    h = np.sqrt(a**2 - 2 * r**2 * (1 - np.cos(np.pi / 4)))
    top = _ring(4, z=0.0)
    bot = _ring(4, z=0.0, phase=np.pi / 4)
    return [[x, y, h / 2] for x, y, _ in top] + [
        [x, y, -h / 2] for x, y, _ in bot
    ]


#: Ideal vertex sets, keyed by shape label.  The central atom is implicit at
#: the origin and is added by :func:`reference_shape`.
_SHAPES: dict[str, list[list[float]]] = {
    # 2 vertices
    "L-2": [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    "vT-2": _TETRA[:2],
    "vOC-2": [_OCTA[0], _OCTA[2]],
    # 3 vertices
    "TP-3": _ring(3),
    "vT-3": _TETRA[:3],
    "fvOC-3": [_OCTA[0], _OCTA[2], _OCTA[4]],
    "mvOC-3": [_OCTA[0], _OCTA[1], _OCTA[4]],
    # 4 vertices
    "SP-4": _ring(4),
    "T-4": _TETRA,
    "SS-4": [_TBPY[0], _TBPY[1], _TBPY[2], _TBPY[3]],
    "vTBPY-4": [_TBPY[0], _TBPY[2], _TBPY[3], _TBPY[4]],
    # 5 vertices
    "PP-5": _ring(5),
    "vOC-5": _OCTA[:5],
    "TBPY-5": _TBPY,
    "SPY-5": _pyramid(4),
    "JTBPY-5": _equal_edge_bipyramid(),
    # 6 vertices
    "HP-6": _ring(6),
    "PPY-6": _pyramid(5),
    "OC-6": _OCTA,
    "TPR-6": _equal_edge_prism(),
    # 7 vertices
    "HPY-7": _pyramid(6),
    "PBPY-7": _bipyramid(5),
    # 8 vertices
    "HPY-8": _pyramid(7),
    "HBPY-8": _bipyramid(6),
    "SAPR-8": _square_antiprism(),
}


def available_shapes(n_vertices: int | None = None) -> list[str]:
    """Shape labels this module knows, optionally filtered by vertex count."""
    if n_vertices is None:
        return sorted(_SHAPES)
    return sorted(k for k, v in _SHAPES.items() if len(v) == n_vertices)


@cache
def reference_shape(label: str) -> np.ndarray:
    """Ideal coordinates for ``label``: central atom first, then vertices."""
    try:
        vertices = _SHAPES[label]
    except KeyError:
        known = ", ".join(available_shapes())
        raise KeyError(
            f"Unknown shape {label!r}. Known shapes: {known}"
        ) from None
    return np.vstack([np.zeros(3), np.asarray(vertices, dtype=float)])


@cache
def _permutation_index(n: int) -> np.ndarray:
    """All vertex permutations, with the central atom pinned at position 0."""
    perms = np.array(list(permutations(range(1, n + 1))), dtype=int)
    central = np.zeros((len(perms), 1), dtype=int)
    return np.hstack([central, perms])


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------


def shape_measure(positions, label: str, central_atom: int = 1) -> float:
    """Continuous shape measure of ``positions`` against the ideal ``label``.

    Parameters
    ----------
    positions:
        ``(N+1, 3)`` coordinates of the central atom and its neighbours, in
        any order.
    label:
        Reference shape, e.g. ``"OC-6"`` (see :func:`available_shapes`).
    central_atom:
        **1-based** index of the central atom within ``positions``.

    Returns:
    -------
    float
        The measure on the 0-100 scale; ``0.0`` is a perfect match.

    Examples:
    --------
    >>> import numpy as np
    >>> tetra = np.array([[0., 0., 0.], [1., 1., 1.], [1., -1., -1.],
    ...                   [-1., 1., -1.], [-1., -1., 1.]])
    >>> round(shape_measure(tetra, "T-4"), 6)
    0.0
    >>> round(shape_measure(tetra, "SP-4"), 2)
    33.33
    """
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"positions must be (N+1, 3), got {q.shape}")

    idx = central_atom - 1
    if not 0 <= idx < len(q):
        raise ValueError(
            f"central_atom {central_atom} is out of range for {len(q)} atoms"
        )
    # move the central atom to row 0 without disturbing the rest
    order = [idx] + [i for i in range(len(q)) if i != idx]
    q = q[order]

    p = reference_shape(label)
    if len(p) != len(q):
        raise ValueError(
            f"{label} expects {len(p) - 1} neighbours, got {len(q) - 1}"
        )

    n = len(q) - 1
    q = q - q.mean(axis=0)
    denom = float((q**2).sum())
    if denom <= 0.0:
        return 0.0

    # every permutation of the reference, each centred on its own centroid
    perms = _permutation_index(n)
    p_all = p[perms]  # (n_perm, N+1, 3)
    p_all = p_all - p_all.mean(axis=1, keepdims=True)

    # Kabsch: the best rotation makes <Q, R P> the sum of singular values of
    # P^T Q, with the smallest one flipped if the rotation would be improper.
    cov = np.einsum("pij,ik->pjk", p_all, q)  # (n_perm, 3, 3)
    sv = np.linalg.svd(cov, compute_uv=False)  # (n_perm, 3)
    det = np.linalg.det(cov)
    sv[det < 0, -1] *= -1.0
    trace = sv.sum(axis=1)

    p_norm = np.einsum("pij,pij->p", p_all, p_all)
    # residual after the optimal uniform scale s = trace / |P|^2
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = denom - np.where(p_norm > 0, trace**2 / p_norm, 0.0)

    return float(max(0.0, residual.min() / denom) * 100.0)
