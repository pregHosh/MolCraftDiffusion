"""Dense <-> flat adapter binding the ported PaiNN/OM-Diff ``EquivNet``
backbone to the ``dynamics._forward(t, xh, node_mask, edge_mask, context)``
contract that ``EnVariationalDiffusion.phi``
(``modules/models/en_diffusion.py:163``) calls.

Mirrors ``gfmdiff/dynamics.py::GFMDiffDynamics`` in role, but has real
work to do: ``EnVariationalDiffusion`` speaks dense padded batches
``(B, N, ...)`` while ``EquivNet`` speaks a flat concatenation of graphs
``(n, ...)`` with an ``(E, 2)`` edge list, so this wrapper packs and
unpacks around it.

It also owns the pieces OM-Diff kept outside the backbone in their
``AtomisticModel`` input/output modules: the linear one-hot embedding,
the Fourier time features, and the scalar readout MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from MolecularDiffusion.modules.models.dynamics.painn_backbone import (
    EnvelopLayer,
    EquivNet,
    EquivNetHParams,
    GaussianLinearRBFLayer,
)
from MolecularDiffusion.utils.geom_utils import remove_mean_with_mask_v2


class FourierTimeFeatures(nn.Module):
    """Random Fourier expansion of the diffusion time (om-diff
    ``layers/features.py::FourierFeatures``). Output is ``2 * n_features``
    wide; the projection is a fixed buffer unless ``trainable``.
    """

    def __init__(
        self,
        n_features: int = 16,
        std: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        weight = torch.normal(mean=torch.zeros(n_features, 1), std=std)
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = nn.functional.linear(t, self.weight)
        return torch.cat(
            (torch.cos(2 * torch.pi * x), torch.sin(2 * torch.pi * x)),
            dim=-1,
        )


class PaiNNDynamics(nn.Module):
    """Denoising network: PaiNN scalar+vector backbone, EDM interface.

    Args:
        in_node_nf: Node feature channels the diffusion model expects
            back (atom-type one-hot + atomic number [+ extra values]),
            excluding time and context, which are added internally.
        context_node_nf: Conditioning channels, concatenated to the node
            features before embedding.
        n_dims: Spatial dimensions (3).
        num_interactions: Interaction/update blocks.
        node_size, edge_size, embedding_dim: backbone widths. OM-Diff's
            defaults are 256 / 64 / 256.
        rbf_features, rbf_max_distance: Gaussian radial basis size and
            range, in angstrom.
        time_features: Fourier time features; contributes ``2 x`` this to
            the backbone input width.
        cutoff: Optional edge cutoff in angstrom. ``None`` (default)
            keeps the platform's dense fully-connected ``edge_mask``;
            set e.g. ``7.5`` to reproduce OM-Diff's radius graph, which
            they rebuild from the *noisy* coordinates at every step —
            this does the same, since it is applied per forward pass.
        envelop_p: If ``cutoff`` is set, order of the polynomial cutoff
            envelope applied to the radial basis. ``None`` disables it.
    """

    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int = 0,
        n_dims: int = 3,
        num_interactions: int = 5,
        node_size: int = 256,
        edge_size: int = 64,
        embedding_dim: int = 256,
        rbf_features: int = 64,
        rbf_max_distance: float = 5.0,
        time_features: int = 16,
        cutoff: float | None = None,
        envelop_p: int | None = 6,
    ) -> None:
        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self.cutoff = cutoff

        # om-diff OneHotEmbedding: a bias-free linear map, not nn.Embedding
        self.embedding = nn.Linear(
            in_node_nf + context_node_nf, embedding_dim, bias=False
        )
        self.time_embedding = FourierTimeFeatures(time_features)

        self.equivnet = EquivNet(
            hparams=EquivNetHParams(
                num_interactions=num_interactions,
                input_size=embedding_dim + 2 * time_features,
                node_size=node_size,
                edge_size=edge_size,
                update_node_positions=True,
            ),
            rbf_layer=GaussianLinearRBFLayer(
                n_features=rbf_features, max_distance=rbf_max_distance
            ),
            envelop_layer=(
                EnvelopLayer(p=envelop_p, xc=cutoff)
                if cutoff is not None and envelop_p is not None
                else None
            ),
        )
        # om-diff's output_modules.readout
        self.readout = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, in_node_nf),
        )

    # -- EnVariationalDiffusion dynamics interface --------------------- #

    def forward(self, t, xh, node_mask, edge_mask, context=None):  # noqa: ANN001, ANN201, D102
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):  # noqa: ANN001, ANN201
        """Bind the masks so an ODE/SDE solver can call ``fwd(t, x)``."""

        def fwd(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):  # noqa: ANN201
        """Return the unbound forward."""
        return self._forward

    # -- packing helpers ----------------------------------------------- #

    def _edge_index(
        self,
        edge_mask: torch.Tensor,
        node_mask_flat: torch.Tensor,
        compact: torch.Tensor,
        positions: torch.Tensor,
        b: int,
        n: int,
    ) -> torch.Tensor:
        """Build an ``(E, 2)`` edge list over *compact* node indices.

        ``edge_mask`` reaches this wrapper as either ``(B*N*N, 1)`` or
        ``(B, N*N)`` depending on the caller
        (``modules/tasks/diffusion.py:475`` reshapes it, the sampler at
        ``:724`` does not), so it is reshaped defensively here.
        """
        adjacency = edge_mask.reshape(b, n, n).bool()
        # Intersect with the node mask and drop self-loops rather than
        # trusting the caller to have done both.
        valid = node_mask_flat.reshape(b, n).bool()
        adjacency = adjacency & valid[:, :, None] & valid[:, None, :]
        adjacency = adjacency & ~torch.eye(
            n, dtype=torch.bool, device=adjacency.device
        )

        graph, src, dst = torch.nonzero(adjacency, as_tuple=True)
        edges = torch.stack(
            [compact[graph * n + src], compact[graph * n + dst]], dim=1
        )

        if self.cutoff is not None:
            d = torch.linalg.vector_norm(
                positions[edges[:, 1]] - positions[edges[:, 0]], dim=-1
            )
            edges = edges[d < self.cutoff]
        return edges

    def _forward(  # noqa: C901
        self,
        t: torch.Tensor,
        xh: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict eps for a dense padded batch.

        Args:
            t: scalar or ``(B, 1)`` diffusion time in ``[0, 1]``.
            xh: ``(B, N, 3 + in_node_nf)`` noisy positions ++ features.
            node_mask: ``(B, N, 1)``.
            edge_mask: ``(B*N*N, 1)`` or ``(B, N*N)``.
            context: ``(B, N, context_node_nf)`` or ``None``.

        Returns:
            ``(B, N, 3 + in_node_nf)``, zero on padded rows, with the
            position channels projected to the zero-CoM subspace that
            ``EnVariationalDiffusion`` assumes everywhere.
        """
        b, n, _ = xh.shape
        device = xh.device

        node_mask_flat = node_mask.reshape(b * n, 1)
        keep = node_mask_flat.squeeze(1).bool()
        index = torch.nonzero(keep, as_tuple=True)[0]

        # map padded (b*n) row -> compact row, -1 where padded
        compact = torch.full((b * n,), -1, dtype=torch.long, device=device)
        compact[index] = torch.arange(index.numel(), device=device)

        positions = xh[:, :, : self.n_dims].reshape(b * n, self.n_dims)[index]
        features = xh[:, :, self.n_dims :].reshape(b * n, -1)[index]

        if context is not None and self.context_node_nf > 0:
            ctx = context.reshape(b * n, -1)[index]
            features = torch.cat([features, ctx], dim=1)

        if torch.numel(t) == 1:
            t_flat = t.expand(index.numel(), 1)
        else:
            t_flat = t.reshape(b, 1).expand(b, n).reshape(b * n, 1)[index]

        node_states = torch.cat(
            [self.embedding(features), self.time_embedding(t_flat)], dim=1
        )

        num_nodes = node_mask.reshape(b, n).sum(dim=1).long()
        edge_index = self._edge_index(
            edge_mask, node_mask_flat, compact, positions, b, n
        )

        delta_pos, states_s = self.equivnet(
            node_positions=positions,
            node_states=node_states,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        h_out = self.readout(states_s)

        out = torch.zeros(
            b * n, self.n_dims + self.in_node_nf, device=device, dtype=xh.dtype
        )
        out[index] = torch.cat([delta_pos, h_out], dim=1).to(xh.dtype)
        out = out.reshape(b, n, self.n_dims + self.in_node_nf) * node_mask

        # OM-Diff pins translation with a fixed metal centre at the origin
        # instead (center_keys: []); that mechanism is out of scope, so
        # project to the zero-CoM subspace EnVariationalDiffusion needs.
        vel = remove_mean_with_mask_v2(
            out[:, :, : self.n_dims], node_mask
        ) * node_mask

        if torch.any(torch.isnan(vel)):
            vel = torch.zeros_like(vel)
            out = torch.zeros_like(out)

        return torch.cat([vel, out[:, :, self.n_dims :]], dim=2)

    def _forward_pyG(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201, N802
        """Alias kept for API parity with the other dynamics wrappers."""
        return self._forward(*args, **kwargs)
