"""ET-Flow's flow-matching pieces: harmonic prior, Kabsch alignment, loss.

Ported from ET-Flow's ``etflow/models/{utils,loss}.py`` and
``etflow/commons/utils.py`` (MIT, (c) 2024 Majdi Hassan, Nikhil Shenoy,
Jungyoon Lee). The harmonic prior itself is adapted upstream from
FlowSite/HarmonicFlow (``models/utils.py:91-94``).

None of this carries parameters, so nothing here appears in a checkpoint.

The one deliberate change from upstream is the prior's cache key.
:class:`HarmonicSampler` upstream keys its eigendecomposition cache by SMILES
(``models/utils.py:128-141``), which is only safe if a SMILES pins the atom
ORDER too -- upstream's dataset guarantees that, ours does not (the platform
stores a canonical SMILES beside whatever atom order the source had, and
conformer-pool items carry ``smiles=None`` entirely). Here the caller supplies
an opaque per-molecule key derived from the bond graph itself, so a hit is
always the same graph in the same order. The cache is also bounded; upstream's
grows without limit for the life of the process.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import get_laplacian, scatter, to_dense_adj

#: Edge-type value radius-graph edges get. Bonds get 1. There is no bond-order
#: channel: ``edge_attr_dim: 1`` in every shipped config is a binary
#: bonded/not-bonded indicator (``commons/utils.py:172``).
UNSPECIFIED_EDGE_TYPE = 0


def center_of_mass(x: Tensor, dim: int = 0, batch: Optional[Tensor] = None) -> Tensor:
    """Subtract each graph's mean. ``batch=None`` treats ``x`` as one graph."""
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return x - scatter(x, batch, dim=dim, reduce="mean")[batch]


def unsqueeze_like(x: Tensor, target: Tensor) -> Tensor:
    """Reshape ``x`` to ``(x.size(0), 1, 1, ...)`` matching ``target.dim()``."""
    return x.view(x.size(0), *([1] * (target.dim() - 1)))


def signed_volume(local_coords: Tensor) -> Tensor:
    """Sign of the tetrahedron volume at each chiral centre (from GeoMol).

    ``local_coords``: ``(n_centers, 4, n_confs, 3)``.
    """
    v1 = local_coords[:, 0] - local_coords[:, 3]
    v2 = local_coords[:, 1] - local_coords[:, 3]
    v3 = local_coords[:, 2] - local_coords[:, 3]
    return torch.sign(torch.sum(v1 * v2.cross(v3, dim=-1), dim=-1))


def batchwise_l2_loss(
    prediction: Tensor, target: Tensor, batch: Optional[Tensor] = None
) -> Tensor:
    """Per-atom L2 NORM (not squared), meaned per molecule then over the batch.

    Upstream's objective (``models/loss.py:61-78``). Swapping it for MSE
    changes the gradient scale and is not a cosmetic difference.
    """
    if batch is None:
        batch = torch.zeros(
            prediction.size(0), dtype=torch.long, device=prediction.device
        )
    return scatter(
        torch.norm(prediction - target, p=2, dim=-1), index=batch, reduce="mean"
    ).mean(dim=0)


# --- edge construction -----------------------------------------------------


def extend_bond_index(  # noqa: PLR0913
    pos: Tensor,
    bond_index: Tensor,
    batch: Tensor,
    cutoff: float = 10.0,
    max_num_neighbors: int = 32,
) -> tuple[Tensor, Tensor]:
    """Bond edges union a radius graph over the CURRENT coordinates.

    The edge set is therefore rebuilt at every integration step, not fixed.
    Bond edges keep type 1, radius edges type 0; the two sparse adjacencies are
    composed by ``coalesce``, which adds values -- so a radius edge coinciding
    with a bond contributes 0 and the bond survives as 1.

    The assertion is upstream's (``models/utils.py:72-74``). What it catches is
    a DUPLICATED directed edge -- the realistic mistake here, since the platform
    stores only the upper triangle and the adapter mirrors it: mirroring an
    already-bidirectional list makes each edge coalesce to type 2, so the count
    of positive types falls below the number of edges that went in.
    """
    n = pos.size(0)
    bond_type = torch.ones(
        bond_index.shape[1], dtype=torch.long, device=pos.device
    )

    from torch_cluster import radius_graph

    bgraph = torch.sparse_coo_tensor(
        bond_index, bond_type, torch.Size([n, n])
    )
    rgraph_index = radius_graph(
        pos, r=cutoff, batch=batch, max_num_neighbors=max_num_neighbors
    )
    rgraph = torch.sparse_coo_tensor(
        rgraph_index,
        torch.full(
            (rgraph_index.size(1),),
            UNSPECIFIED_EDGE_TYPE,
            dtype=torch.long,
            device=pos.device,
        ),
        torch.Size([n, n]),
    )

    composed = (bgraph + rgraph).coalesce()
    edge_index = composed.indices()
    edge_type = composed.values().long()

    n_bond_edges = int((edge_type > 0).sum())
    if bond_index.shape[1] != n_bond_edges:
        msg = (
            f"{bond_index.shape[1]} bond edges went in but {n_bond_edges} came "
            "out of the sparse coalesce. Duplicated directed edges are the "
            "usual cause -- the adapter must emit each directed edge exactly "
            "once (mirror the stored upper triangle, never a full list)."
        )
        raise ValueError(msg)
    return edge_index, edge_type


# --- harmonic prior --------------------------------------------------------


class HarmonicSampler:
    """Gaussian prior whose covariance is the bond-graph Laplacian's inverse.

    A sample already looks roughly like a bonded molecule, which is why the
    flow has so little work to do. A DISCONNECTED graph has extra zero
    eigenvalues, ``1/sqrt(D)`` blows up, and the sample is NaN -- salts and
    multi-fragment inputs genuinely do not work with this model.
    """

    def __init__(self, alpha: float = 1.0, cache_size: int = 20_000) -> None:
        self.alpha = alpha
        self.cache_size = cache_size
        self.eig_cache: dict = {}

    def _cache_put(self, key, value) -> None:
        # ponytail: clear-on-full instead of LRU eviction. The cache exists to
        # skip a ~1 ms eigh; a rare full flush costs nothing. Swap in an
        # OrderedDict LRU only if profiling ever says otherwise.
        if len(self.eig_cache) >= self.cache_size:
            self.eig_cache.clear()
        self.eig_cache[key] = value

    def diagonalize(
        self,
        n_nodes: int,
        edges: Tensor,
        batch: Optional[Tensor] = None,
        keys: Optional[list] = None,
    ) -> tuple[Tensor, Tensor]:
        """Eigendecompose the batched bond Laplacian, block by block."""
        a = self.alpha * torch.ones((edges.shape[0],), device=edges.device)
        edge_index, edge_weight = get_laplacian(edges.T, a, num_nodes=n_nodes)
        lap = to_dense_adj(
            edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n_nodes
        ).squeeze()

        if batch is None:
            return torch.linalg.eigh(lap)

        evals, evecs = [], []
        for i in range(int(batch.max()) + 1):
            idx = torch.where(batch == i)[0]
            start, end = int(idx.min()), int(idx.max()) + 1
            key = keys[i] if keys is not None else None

            cached = self.eig_cache.get(key) if key is not None else None
            if cached is not None:
                d, p = (t.to(edge_index.device) for t in cached)
            else:
                d, p = torch.linalg.eigh(lap[start:end, start:end])
                if key is not None:
                    self._cache_put(key, (d.detach().cpu(), p.detach().cpu()))

            evals.append(d)
            evecs.append(p)

        return torch.cat(evals), torch.block_diag(*evecs)

    def sample(
        self,
        size: torch.Size,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        keys: Optional[list] = None,
    ) -> Tensor:
        """Draw one prior sample of shape ``size`` = ``(n_total_atoms, 3)``."""
        if edge_index.size(0) == 2:
            edge_index = edge_index.T

        n_nodes = size[0]
        d, p = self.diagonalize(n_nodes, edge_index, batch, keys)

        # Zero the first (translational) mode of every molecule -- this is what
        # removes the centre of mass from the prior sample.
        start_index = 0
        if batch is not None:
            _, counts = torch.unique(batch, return_counts=True)
            zero = torch.zeros(1, device=d.device)
            start_index = torch.cat((zero, counts.cumsum(0)[:-1])).long()

        std = 1.0 / torch.sqrt(d)
        std[start_index] = 0.0

        noise = std[:, None] * torch.randn(size, device=d.device)
        noise[noise.isnan()] = 0.0
        return p @ noise


# --- Kabsch alignment ------------------------------------------------------


def find_rigid_alignment(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Kabsch: rotation+translation taking point cloud ``a`` onto ``b``.

    Reflections are excluded (``det(R) < 0`` flips the last singular vector),
    which is why aligning the prior to the data cannot launder a mirror image.
    """
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    h = (a - a_mean).T.mm(b - b_mean)
    u, _, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    r = v.mm(u.T)
    if torch.det(r) < 0:
        v = v.clone()
        v[:, -1] = -v[:, -1]
        r = v.mm(u.T)
    t = b_mean[None, :] - r.mm(a_mean[None, :].T).T
    return r, t.T.squeeze()


def rmsd_align(pos: Tensor, ref_pos: Tensor, batch: Tensor) -> Tensor:
    """Per-molecule Kabsch alignment of ``pos`` onto ``ref_pos``.

    This is the "equivariant" in ET-Flow: aligning the prior sample to the data
    conformer removes the global rotation from the regression target.
    """
    aligned = []
    for i in range(int(batch.max()) + 1):
        index = torch.where(batch == i)[0]
        pos_i, ref_i = pos[index], ref_pos[index]
        r, t = find_rigid_alignment(pos_i, ref_i)
        aligned.append((r @ pos_i.T).T + t)
    return torch.cat(aligned, dim=0)


def switch_parity_of_pos(
    pos: Tensor,
    chiral_index: Tensor,
    chiral_nbr_index: Tensor,
    chiral_tag: Tensor,
    batch: Tensor,
) -> Tensor:
    """Post-hoc parity correction: mirror any molecule whose centres inverted.

    Compares the signed volume at every tetrahedral centre with the input's
    chiral tag and reflects the WHOLE molecule if any centre came out wrong.
    A no-op for achiral inputs, where ``chiral_index`` is ``(1, 0)``.
    """
    num_graphs = int(batch.max()) + 1
    n_centers = chiral_index.shape[1]
    sv = signed_volume(
        pos[chiral_nbr_index.reshape(n_centers, 4)].unsqueeze(2)
    ).reshape(-1)
    z_flip = sv * chiral_tag

    graph_diag = torch.ones(num_graphs, device=pos.device)
    wrong = batch[chiral_index][:, z_flip == -1.0].reshape(-1)
    graph_diag[wrong] = -1.0
    return pos * graph_diag[batch].unsqueeze(1)
