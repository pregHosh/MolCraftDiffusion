"""EGNN backbone for TABASCO's flow-matching model.

Wraps ``MolecularDiffusion.modules.models.egcl.EGNN`` (this platform's
existing ``GCL`` / ``EquivariantUpdate`` / ``EquivariantBlock`` stack,
``modules/layers/conv.py:13-255``) so it can be substituted 1-for-1 for
TABASCO's ``TransformerModule``
(``modules/layers/tabasco/transformer_module.py``) with no change to
``FlowMatchingModel``, the interpolants, the Euler-Maruyama sampler, or the
pointcloud<->TensorDict adapters.

Backbone contract (fixed by ``FlowMatchingModel._call_net``,
``modules/models/tabasco/flow_model.py:84-97``):

    forward(coords, atomics, padding_mask, t) -> (coords, atom_logits)

``coords`` returned is the direct endpoint prediction ``x_1`` -- both
``CenteredMetricInterpolant.compute_loss`` and ``SDEMetricInterpolant.step``
(``modules/models/tabasco/flow/interpolate.py``) treat ``pred["coords"]`` as
an x_1 estimate, not a velocity/residual. EGNN's own coordinate output
already has this shape (each ``EquivariantBlock`` displaces the running
coordinate estimate), so it is returned as-is -- no ``x_final - x`` residual
is taken here (that residual pattern in ``EGNN_dynamics._forward``,
``egcl.py:440-442``, is specific to EDM's noise-prediction contract, not
TABASCO's endpoint-prediction contract).

Mask convention: ``padding_mask`` follows TABASCO's own inverted convention
(``1 = padded``, see ``PointCloudToTensorDictAdapter``,
``modules/tasks/diffusion_tabasco.py:60-113``) -- the opposite of ``EGNN``'s
``node_mask``/``edge_mask`` (``1 = real``, ``modules/models/egcl.py``). This
class inverts once at the boundary.
"""

from typing import Tuple

import torch
import torch.nn as nn

from MolecularDiffusion.modules.models.egcl import EGNN

_ACTIVATIONS = {"SiLU": nn.SiLU, "ReLU": nn.ReLU}


class EGNNBackbone(nn.Module):
    """TABASCO-compatible net: ``(coords, atomics, padding_mask, t) -> (coords, atom_logits)``."""

    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 9,
        activation: str = "SiLU",
        attention: bool = True,
        tanh: bool = True,
        inv_sublayers: int = 1,
        sin_embedding: bool = False,
        include_cosine: bool = True,
        norm_constant: float = 1.0,
        normalization_factor: float = 1.0,
        aggregation_method: str = "sum",
        coords_range: float = 15.0,
        norm_diff: bool = True,
        dropout: float = 0.0,
        normalization: bool = False,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}; expected one of "
                f"{sorted(_ACTIVATIONS)}"
            )
        self.atom_dim = atom_dim

        # +1 input feature slot for scalar time-conditioning, concatenated
        # into `h` before the EGNN stack -- the same pattern
        # `EGNN_dynamics._forward` uses for the en_diffusion family
        # (egcl.py:415-423).
        self.egnn = EGNN(
            in_node_nf=atom_dim + 1,
            hidden_nf=hidden_dim,
            act_fn=_ACTIVATIONS[activation](),
            n_layers=num_layers,
            attention=attention,
            norm_diff=norm_diff,
            out_node_nf=atom_dim,
            tanh=tanh,
            coords_range=coords_range,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            include_cosine=include_cosine,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            dropout=dropout,
            normalization=normalization,
        )

        # Cache of the fully-connected-with-self-loops edge index per
        # (n_nodes, batch_size), same idiom as `EGNN_dynamics.get_adj_matrix`
        # (egcl.py:549-570) -- rewritten locally rather than composed by
        # instantiating `EGNN_dynamics`, whose constructor bundles an
        # unrelated context/adapter config surface this backbone doesn't
        # need (ledger's Derivation Rung: "an implementation detail, not a
        # design decision to relitigate").
        self._edge_cache: dict = {}

    def _fully_connected_edges(
        self, n_nodes: int, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        by_batch = self._edge_cache.setdefault(n_nodes, {})
        if batch_size in by_batch:
            rows, cols = by_batch[batch_size]
            return rows.to(device), cols.to(device)

        rows, cols = [], []
        for b in range(batch_size):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    rows.append(i + b * n_nodes)
                    cols.append(j + b * n_nodes)
        rows = torch.LongTensor(rows)
        cols = torch.LongTensor(cols)
        by_batch[batch_size] = (rows, cols)
        return rows.to(device), cols.to(device)

    def forward(
        self,
        coords: torch.Tensor,
        atomics: torch.Tensor,
        padding_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: (B, N, 3)
            atomics: (B, N, atom_dim) one-hot (or soft) atom-type features
            padding_mask: (B, N), TABASCO convention -- 1 = padded, 0 = real
            t: (B,) timestep in [0, 1]

        Returns:
            coords: (B, N, 3) endpoint prediction
            atom_logits: (B, N, atom_dim)
        """
        batch_size, n_nodes, _ = coords.shape
        device = coords.device

        # Invert to EGNN's own convention (1 = real).
        node_mask = (~padding_mask.bool()).float()

        t_expand = t.reshape(batch_size, 1, 1).expand(batch_size, n_nodes, 1).to(device=coords.device, dtype=coords.dtype)
        h = torch.cat([atomics, t_expand], dim=-1)

        node_mask_flat = node_mask.reshape(batch_size * n_nodes, 1)
        edge_mask_flat = (
            node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        ).reshape(batch_size * n_nodes * n_nodes, 1)

        h_flat = h.reshape(batch_size * n_nodes, -1) * node_mask_flat
        x_flat = coords.reshape(batch_size * n_nodes, 3) * node_mask_flat

        rows, cols = self._fully_connected_edges(n_nodes, batch_size, device)

        h_out, x_out = self.egnn(
            h_flat,
            x_flat,
            [rows, cols],
            node_mask=node_mask_flat,
            edge_mask=edge_mask_flat,
        )

        coords_out = x_out.reshape(batch_size, n_nodes, 3)
        atom_logits = h_out.reshape(batch_size, n_nodes, self.atom_dim)

        return coords_out, atom_logits
