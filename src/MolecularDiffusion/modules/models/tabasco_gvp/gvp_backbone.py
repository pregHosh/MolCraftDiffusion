"""GVP backbone for TABASCO's flow-matching model.

Wraps this platform's existing SE(3)-equivariant GVP building blocks
(``GVPConv``/``NodePositionUpdate``/``EdgeUpdate``, canonical location
``modules/layers/gvp/gvp.py``) so they can be substituted 1-for-1 for
TABASCO's ``TransformerModule``
(``modules/layers/tabasco/transformer_module.py``) with no change to
``FlowMatchingModel``, the interpolants, the Euler-Maruyama sampler, or the
pointcloud<->TensorDict adapters.

Backbone contract (fixed by ``FlowMatchingModel._call_net``,
``modules/models/tabasco/flow_model.py:84-97``):

    forward(coords, atomics, padding_mask, t) -> (coords, atom_logits)

``coords`` returned is the direct endpoint prediction ``x_1`` -- the same
endpoint-parameterization semantics ``EGNNBackbone`` already returns as-is
and that FlowMol's ``EndpointVectorField.denoise_graph`` predicts
(``modules/models/flowmol/vector_field.py:206-269``), whose interleaved
conv/update-stack structure this backbone reproduces closely, minus the
charge modality and the bond-token edge features (TABASCO's pointcloud
pipeline has neither), and importing the canonical
``NodePositionUpdate``/``EdgeUpdate`` from ``modules/layers/gvp/gvp.py``
instead of ``vector_field.py``'s own local duplicate copies of those two
classes (a pre-existing inconsistency in this repo, noted but deliberately
not fixed here -- see the ledger's Derivation Rung).

Mask convention: ``padding_mask`` follows TABASCO's own inverted convention
(``1 = padded``, see ``PointCloudToTensorDictAdapter``,
``modules/tasks/diffusion_tabasco.py:60-113``) -- inverted once at the
boundary, the same idiom ``EGNNBackbone.forward`` uses
(``modules/models/tabasco_egnn/egnn_backbone.py:147``).

Graph construction: per-molecule masked-slice -> fully-connected DIRECTED
graph WITHOUT self-loops (``build_edge_idxs``,
``modules/models/flowmol/graph_utils.py:14-20``) -> ``dgl.batch`` -- the same
idiom ``PointCloudToDGLAdapter`` already uses in production
(``modules/tasks/diffusion_flowmol.py:82-105``). No self-loops is
deliberate, matching ``EndpointVectorField``'s own established graph shape
(``vector_field.py:216-231``); this differs from the sibling
``tabasco_egnn``'s self-loop-including graph, which is not a bug in either
(see the ledger's Confound #4).

Dense reconstruction: this backbone's output must satisfy
``FlowMatchingModel``'s contract exactly -- ``_call_net`` reuses the
caller's ``padding_mask`` verbatim for the returned TensorDict
(``flow_model.py:90-97``), and both the loss (``_compute_loss``,
``flow_model.py:174-213``) and the Euler step (``_step``,
``flow_model.py:279-286``) combine ``pred["coords"]``/``pred["atomics"]``
elementwise against tensors shaped by that same ``padding_mask``. So the
dense width reconstructed here is always exactly the *input* ``coords``
width ``N`` -- independently confirmed to already equal "this batch's
max real-atom count" (the platform's pointcloud collator,
``data/dataloader.py:97-180``, slices every batch down to
``natoms.max()`` before returning it), so this is the same quantity
``DGLToPointCloudAdapter`` (``diffusion_flowmol.py:108-137``) recovers via
``dgl.unbatch`` + per-graph ``num_nodes()``. Reconstruction here uses a
boolean-mask scatter (``dense[node_mask] = flat_values``) instead of that
``dgl.unbatch`` loop -- provably correct regardless of whether padding
happens to be a contiguous per-molecule prefix, because the per-molecule
graph-construction loop below selects real atoms in increasing-column order
(``coords[b][mask_b]``), which is exactly the row-major enumeration order a
leading-dims boolean-mask scatter assignment also uses, and every GVP layer
below only ever transforms node features elementwise / via message-passing
without permuting node order.
"""

from typing import Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from MolecularDiffusion.modules.layers.gvp import (
    GVPConv,
    NodePositionUpdate,
    EdgeUpdate,
    _norm_no_nan,
    _rbf,
)
from MolecularDiffusion.modules.models.flowmol.graph_utils import (
    build_edge_idxs,
    get_node_batch_idxs,
)


class GVPBackbone(nn.Module):
    """TABASCO-compatible net: ``(coords, atomics, padding_mask, t) -> (coords, atom_logits)``.

    Hyperparameters default to FlowMol's own proven-stable QM9-scale GVP
    configuration (``configs/tasks/diffusion_flowmol.yaml``), reused as-is
    per the ledger's Hyperparameter Provenance table -- deliberately NOT
    capacity-matched to TABASCO's transformer width (see the ledger's
    Confound #2).
    """

    def __init__(
        self,
        atom_dim: int,
        n_hidden_scalars: int = 64,
        n_vec_channels: int = 16,
        n_hidden_edge_feats: int = 64,
        n_molecule_updates: int = 2,
        convs_per_update: int = 2,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        n_expansion_gvps: int = 3,
        attention: bool = False,
        message_norm: float = 100,
        rbf_dmax: float = 20,
        rbf_dim: int = 16,
        n_recycles: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.atom_dim = atom_dim
        self.n_hidden_scalars = n_hidden_scalars
        self.n_vec_channels = n_vec_channels
        self.n_molecule_updates = n_molecule_updates
        self.convs_per_update = convs_per_update
        self.n_recycles = n_recycles
        self.rbf_dmax = rbf_dmax
        self.rbf_dim = rbf_dim

        # atom-type one-hot + scalar time t -> n_hidden_scalars.
        # (mirrors EndpointVectorField.scalar_embedding, vector_field.py:93-102,
        # minus the formal-charge channel TABASCO's pointcloud pipeline has no
        # slot for)
        self.scalar_embedding = nn.Sequential(
            nn.Linear(atom_dim + 1, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_scalars),
        )

        # RBF(pairwise distance) -> n_hidden_edge_feats -- geometry-only edge
        # features (mirrors EndpointVectorField.edge_embedding,
        # vector_field.py:106-112; TABASCO has no bond-token modality to embed
        # instead)
        self.edge_embedding = nn.Sequential(
            nn.Linear(rbf_dim, n_hidden_edge_feats),
            nn.SiLU(),
            nn.Linear(n_hidden_edge_feats, n_hidden_edge_feats),
            nn.SiLU(),
            nn.LayerNorm(n_hidden_edge_feats),
        )

        self.conv_layers = nn.ModuleList(
            [
                GVPConv(
                    scalar_size=n_hidden_scalars,
                    vector_size=n_vec_channels,
                    edge_feat_size=n_hidden_edge_feats,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    n_expansion_gvps=n_expansion_gvps,
                    message_norm=message_norm,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    attention=attention,
                    dropout=dropout,
                )
                for _ in range(convs_per_update * n_molecule_updates)
            ]
        )

        # A single position/edge updater reused after every `convs_per_update`
        # conv layers (FlowMol's `separate_mol_updaters: false` default,
        # vector_field.py:139-148 -- not exposed as a knob here since neither
        # FlowMol's own config nor this ledger's Hyperparameter Provenance
        # table varies it).
        self.node_position_updater = NodePositionUpdate(
            n_hidden_scalars, n_vec_channels, n_gvps=3
        )
        self.edge_updater = EdgeUpdate(
            n_hidden_scalars,
            n_hidden_edge_feats,
            update_edge_w_distance=True,
            rbf_dim=rbf_dim,
        )

        # final node-scalar -> atom-type-logit head (no charge head -- see
        # the ledger's Hyperparameter Provenance, "charge modality" row)
        self.node_output_head = nn.Sequential(
            nn.Linear(n_hidden_scalars, n_hidden_scalars),
            nn.SiLU(),
            nn.Linear(n_hidden_scalars, atom_dim),
        )

    def _precompute_distances(
        self, g: dgl.DGLGraph, node_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalised pairwise displacement + RBF distance on every edge.

        Mirrors ``EndpointVectorField.precompute_distances``
        (``vector_field.py:271-285``).
        """
        with g.local_scope():
            g.ndata["_pos"] = node_positions
            g.apply_edges(fn.u_sub_v("_pos", "_pos", "_x_diff"))
            dij = _norm_no_nan(g.edata["_x_diff"], keepdims=True) + 1e-8
            x_diff = g.edata["_x_diff"] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        return x_diff, d

    def _build_graph(
        self,
        coords: torch.Tensor,
        atomics: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dgl.DGLGraph:
        """Per-molecule masked-slice -> fully-connected no-self-loop DGL
        graph -> ``dgl.batch``, following ``PointCloudToDGLAdapter``'s
        established loop (``diffusion_flowmol.py:82-105``)."""
        device = coords.device
        graphs = []
        for b in range(coords.size(0)):
            mask_b = node_mask[b]
            n = int(mask_b.sum().item())
            coords_b = coords[b][mask_b]
            atom_b = atomics[b][mask_b]
            edges = build_edge_idxs(n).to(device)
            g_i = dgl.graph((edges[0], edges[1]), num_nodes=n, device=device)
            g_i.ndata["pos"] = coords_b
            g_i.ndata["atom_oh"] = atom_b
            graphs.append(g_i)
        return dgl.batch(graphs)

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
        batch_size, n_dense, _ = coords.shape
        device = coords.device
        dtype = coords.dtype

        # Invert once at the boundary (same idiom EGNNBackbone.forward uses,
        # egnn_backbone.py:147).
        node_mask = ~padding_mask.bool()

        g = self._build_graph(coords, atomics, node_mask)
        node_batch_idx = get_node_batch_idxs(g)

        t_per_node = t.to(device=device, dtype=dtype)[node_batch_idx].unsqueeze(-1)
        scalar_in = torch.cat([g.ndata["atom_oh"], t_per_node], dim=-1)
        node_scalar_features = self.scalar_embedding(scalar_in)

        node_positions = g.ndata["pos"]
        num_nodes = g.num_nodes()
        node_vec_features = torch.zeros(
            (num_nodes, self.n_vec_channels, 3), device=device, dtype=dtype
        )

        x_diff, d = self._precompute_distances(g, node_positions)
        edge_features = self.edge_embedding(d)

        for _ in range(self.n_recycles):
            for conv_idx, conv in enumerate(self.conv_layers):
                node_scalar_features, node_vec_features = conv(
                    g,
                    scalar_feats=node_scalar_features,
                    coord_feats=node_positions,
                    vec_feats=node_vec_features,
                    edge_feats=edge_features,
                    x_diff=x_diff,
                    d=d,
                )

                if (conv_idx + 1) % self.convs_per_update == 0:
                    node_positions = self.node_position_updater(
                        node_scalar_features, node_positions, node_vec_features
                    )
                    x_diff, d = self._precompute_distances(g, node_positions)
                    edge_features = self.edge_updater(
                        g, node_scalar_features, edge_features, d=d
                    )

        atom_logits_flat = self.node_output_head(node_scalar_features)

        coords_out = torch.zeros(
            batch_size, n_dense, 3, device=device, dtype=dtype
        )
        logits_out = torch.zeros(
            batch_size, n_dense, self.atom_dim, device=device, dtype=dtype
        )
        coords_out[node_mask] = node_positions
        logits_out[node_mask] = atom_logits_flat

        return coords_out, logits_out
