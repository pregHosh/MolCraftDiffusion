"""PaiNN-style scalar+vector equivariant backbone, ported from OM-Diff.

Source: ``om-diff`` @ ``976084eaf407cc84a5f169f4204f72d4a1fbdbbc``
(https://github.com/Aalto-QuML/om-diff — MIT), files
``src/models/backbones/equivnet.py``, ``src/models/layers/{rbf,norm,
readout,mlp,features}.py`` and ``src/models/ops.py``.

Only the network is ported. OM-Diff's diffusion objective, metal-centre
masking, conditional size prior and sampler are deliberately not part of
this file — the platform's ``EnVariationalDiffusion`` supplies all of
that (see ``docs/model_integrations/omdiff/INTEGRATION_PLAN.md``).

Two deliberate deviations from upstream:

* ``EquivNet.forward`` takes plain tensors instead of OM-Diff's ``Batch``
  dataclass, so no part of their data model has to come along.
* ``EdgeLayer`` / ``with_edge_interactions`` is not ported — it is off in
  every shipped OM-Diff config and nothing else here would use it.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

__all__ = [
    "BesselRBFLayer",
    "EnvelopLayer",
    "EquivNet",
    "EquivNetHParams",
    "GaussianLinearRBFLayer",
    "GraphLayerNorm",
    "pairwise_distances",
]


# --------------------------------------------------------------------- #
# segment reductions (om-diff src/models/ops.py)
# --------------------------------------------------------------------- #


def sum_index(
    values: torch.Tensor,
    index: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Scatter-add ``values`` into ``out`` at rows ``index``.

    ``out`` is mandatory here (upstream made it optional): the shape must
    not be inferred from ``index.max()``, or a graph whose last nodes are
    isolated silently loses rows.
    """
    return out.index_put_((index,), values, accumulate=True)


def sum_splits(values: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
    """Sum ``values`` per contiguous split (``splits`` = nodes per graph)."""
    index = torch.repeat_interleave(
        torch.arange(splits.shape[0], device=values.device), splits
    )
    out = values.new_zeros((splits.shape[0], *values.shape[1:]))
    return sum_index(values, index, out)


def mean_splits(values: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` per contiguous split."""
    return sum_splits(values, splits) / splits.unsqueeze(1)


def center_splits(values: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
    """Subtract the per-split mean from ``values``."""
    means = torch.repeat_interleave(mean_splits(values, splits), splits, dim=0)
    return values - means


def pairwise_distances(
    positions: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge distances ``(E, 1)`` and unit vectors ``(E, 3)``, i -> j.

    ``edges`` is ``(E, 2)`` (OM-Diff's layout, not PyG's ``(2, E)``).
    """
    vectors = positions[edges[:, 1], :] - positions[edges[:, 0], :]
    distances = torch.sqrt(
        torch.sum(vectors**2, dim=-1, keepdim=True) + 1e-6
    )
    return distances, vectors / distances


# --------------------------------------------------------------------- #
# radial basis / envelope (om-diff src/models/layers/rbf.py)
# --------------------------------------------------------------------- #


class EnvelopLayer(nn.Module):
    """Polynomial cutoff envelope, smoothly zero beyond ``xc``."""

    def __init__(self, p: int = 6, xc: float = 5.0) -> None:
        super().__init__()
        self.register_buffer("p", torch.LongTensor([p]))
        self.register_buffer("xc", torch.FloatTensor([xc]))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        p, xc = self.p, self.xc
        x_norm = distances / xc
        x_p = x_norm**p
        x_p1 = x_p * x_norm
        x_p2 = x_p1 * x_norm
        return torch.where(
            distances < xc,
            1.0
            - x_p * (p + 1) * (p + 2) / 2.0
            + p * (p + 2) * x_p1
            - p * (p + 1) * x_p2 / 2.0,
            torch.zeros_like(distances),
        )


class RBFLayer(nn.Module):
    """Base class carrying the expanded-feature count."""

    n_features: int

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.n_features = n_features


class GaussianLinearRBFLayer(RBFLayer):
    """Evenly spaced Gaussians — OM-Diff's configured default."""

    def __init__(
        self,
        n_features: int = 64,
        max_distance: float = 5.0,
        min_distance: float = 0.0,
    ) -> None:
        super().__init__(n_features)
        self.register_buffer(
            "delta",
            torch.tensor((max_distance - min_distance) / n_features),
        )
        self.register_buffer(
            "offsets",
            torch.linspace(min_distance, max_distance, steps=n_features),
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((distances - self.offsets[None, :]) ** 2) / self.delta
        )


class BesselRBFLayer(RBFLayer):
    """Bessel radial basis (upstream alternative to the Gaussian one)."""

    def __init__(
        self,
        n_features: int = 20,
        max_distance: float = 5.0,
        trainable: bool = True,
    ) -> None:
        super().__init__(n_features)
        self.r_max = float(max_distance)
        self.prefactor = 2.0 / self.r_max
        weights = (
            torch.linspace(1.0, n_features, steps=n_features) * torch.pi
        )
        if trainable:
            self.bessel_weights = nn.Parameter(weights)
        else:
            self.register_buffer("bessel_weights", weights)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(
            self.bessel_weights[None, :] * distances / self.r_max
        )
        return self.prefactor * (numerator / distances)


# --------------------------------------------------------------------- #
# misc layers (om-diff src/models/layers/{norm,mlp}.py)
# --------------------------------------------------------------------- #


class GraphLayerNorm(nn.Module):
    """Layer norm whose statistics are pooled per graph, not per node."""

    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        x = center_splits(x - x.mean(-1, keepdim=True), splits=splits)
        var_x = torch.mean(
            mean_splits(x * x, splits=splits), dim=-1, keepdim=True
        )
        var_x = torch.repeat_interleave(var_x, splits, dim=0)
        out = x / torch.sqrt(var_x + self.eps)
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out


class MLP(nn.Module):
    """Plain SiLU MLP (om-diff ``layers/mlp.py``)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        num_hidden_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_hidden_layers == 0:
            self.nn: nn.Module = nn.Linear(input_dim, output_dim)
        else:
            assert hidden_dim is not None
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                *[
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
                    )
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)


# --------------------------------------------------------------------- #
# message passing (om-diff src/models/backbones/equivnet.py)
# --------------------------------------------------------------------- #


class InteractionLayer(nn.Module):
    """PaiNN message step with a sigmoid edge-inference gate."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.W = nn.Linear(edge_dim, 3 * node_dim)
        self.msg_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )
        self.edge_inference_nn = nn.Sequential(
            nn.Linear(node_dim, 1), nn.Sigmoid()
        )
        self.ln_node = GraphLayerNorm(node_dim, affine=True)

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        edges: torch.Tensor,
        splits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_states_s = self.ln_node(node_states_s, splits)

        w = self.W(edge_states)
        phi = self.msg_nn(node_states_s)
        w_phi = w * phi[edges[:, 0]]
        phi_s, phi_vv, phi_vs = torch.split(w_phi, self.node_dim, dim=1)
        edge = self.edge_inference_nn(phi_s)

        messages_scalar = phi_s * edge
        messages_vector = (
            node_states_v[edges[:, 0]] * phi_vv[:, None, :]
            + phi_vs[:, None, :] * unit_vectors[..., None]
        ) * edge[..., None]

        reduced_s = sum_index(
            messages_scalar, edges[:, 1], out=torch.zeros_like(node_states_s)
        )
        reduced_v = sum_index(
            messages_vector, edges[:, 1], out=torch.zeros_like(node_states_v)
        )
        return node_states_s + reduced_s, node_states_v + reduced_v


class UpdateLayer(nn.Module):
    """PaiNN gated U/V node update."""

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.UV = nn.Linear(node_dim, 2 * node_dim, bias=False)
        self.UV_nn = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uv = self.UV(node_states_v)  # (n, 3, 2F)
        u_v, v_v = torch.split(uv, self.node_dim, -1)
        v_norm = torch.sqrt(torch.sum(v_v**2, dim=1) + 1e-6)

        a = self.UV_nn(torch.cat((v_norm, node_states_s), dim=1))
        a_vv, a_sv, a_ss = torch.split(a, self.node_dim, dim=1)

        delta_s = a_ss + a_sv * torch.sum(u_v * v_v, dim=1)
        delta_v = a_vv[:, None, :] * u_v
        return node_states_s + delta_s, node_states_v + delta_v


class GatedEquivariantBlock(nn.Module):
    """Gated scalar/vector block (om-diff ``layers/readout.py``)."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False
        )
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(
        self,
        states_s: torch.Tensor,
        states_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vec1 = torch.norm(self.vec1_proj(states_v), dim=-2)
        vec2 = self.vec2_proj(states_v)

        x = torch.cat([states_s, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        return nn.functional.silu(x), v.unsqueeze(1) * vec2


class EquivariantReadout(nn.Module):
    """Two gated blocks reducing vector states to a per-node ``(n, 3)``."""

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels, hidden_channels // 2
                ),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )

    def forward(
        self,
        states_s: torch.Tensor,
        states_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.output_network:
            states_s, states_v = layer(states_s, states_v)
        # Upstream used a bare .squeeze(), which also collapses the node
        # axis for a single-node graph. Squeeze the channel axis only.
        return states_s, states_v.squeeze(-1)


@dataclasses.dataclass
class EquivNetHParams:
    """Hyperparameters of :class:`EquivNet`."""

    num_interactions: int = 5
    input_size: int = 288
    node_size: int = 256
    edge_size: int = 64
    update_node_positions: bool = True


class EquivNet(nn.Module):
    """OM-Diff's scalar+vector message-passing network.

    Unlike upstream this consumes/returns tensors rather than their
    ``Batch`` dataclass, so it can be driven straight from the dense
    diffusion batch by :class:`~MolecularDiffusion.modules.models.
    painn_dynamics.PaiNNDynamics`.
    """

    def __init__(
        self,
        hparams: EquivNetHParams,
        rbf_layer: RBFLayer | None = None,
        envelop_layer: EnvelopLayer | None = None,
    ) -> None:
        super().__init__()
        self.hp = hparams

        self.project_layer = MLP(
            input_dim=hparams.input_size,
            hidden_dim=hparams.input_size,
            output_dim=hparams.node_size,
        )
        self.rbf_layer = rbf_layer
        self.envelop_layer = envelop_layer

        edge_in = 1 if rbf_layer is None else rbf_layer.n_features
        self.edge_featurizers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(edge_in, hparams.edge_size), nn.SiLU()
                )
                for _ in range(hparams.num_interactions)
            ]
        )
        self.interactions = nn.ModuleList(
            [
                InteractionLayer(hparams.node_size, hparams.edge_size)
                for _ in range(hparams.num_interactions)
            ]
        )
        self.updates = nn.ModuleList(
            [
                UpdateLayer(hparams.node_size)
                for _ in range(hparams.num_interactions)
            ]
        )
        if hparams.update_node_positions:
            self.equivariant_readout = EquivariantReadout(
                hidden_channels=hparams.node_size
            )

    def forward(
        self,
        node_positions: torch.Tensor,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network over one flat (concatenated) batch of graphs.

        Args:
            node_positions: ``(n, 3)`` coordinates.
            node_states: ``(n, input_size)`` embedded node features.
            edge_index: ``(E, 2)`` directed edges, ``[:, 0] -> [:, 1]``.
            num_nodes: ``(B,)`` nodes per graph; must sum to ``n``.

        Returns:
            ``(delta_node_positions (n, 3), node_states (n, node_size))``.
        """
        states_v = node_positions.new_zeros(
            (*node_positions.shape, self.hp.node_size)
        )
        states_s = self.project_layer(node_states)

        distances, unit_vectors = pairwise_distances(
            node_positions, edge_index
        )
        edge_embedding = (
            distances if self.rbf_layer is None else self.rbf_layer(distances)
        )
        if self.envelop_layer is not None:
            # Upstream applied the envelope to the *RBF features* rather
            # than the distances they came from; the cutoff is evaluated
            # on distances here, which is what the envelope is for. Moot
            # for the shipped configs, which leave envelop_layer None.
            edge_embedding = self.envelop_layer(distances) * edge_embedding

        for featurizer, interaction, update in zip(
            self.edge_featurizers,
            self.interactions,
            self.updates,
            strict=True,
        ):
            edge_states = featurizer(edge_embedding)
            states_s, states_v = interaction(
                states_s,
                states_v,
                edge_states,
                unit_vectors,
                edge_index,
                num_nodes,
            )
            states_s, states_v = update(states_s, states_v)

        if self.hp.update_node_positions:
            _, delta_positions = self.equivariant_readout(states_s, states_v)
        else:
            delta_positions = torch.zeros_like(node_positions)

        return delta_positions, states_s
