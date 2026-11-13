"""DiffSMol's shape-conditioned SE(3)-equivariant backbone (``uni_o2``).

Ported from DiffSMol ``source/models/uni_transformer.py``, bond-free.

Dropped relative to upstream (per the approved INTEGRATION_PLAN gate
resolution): the ``ligand_bond_index``/``ligand_bond_type`` arguments, the
per-layer bond CE loss, and the ``cov_radius`` cutoff mode. None of these fed
the geometry path -- upstream already rebuilds a kNN graph from scratch inside
every block (``_connect_graph``), never using the ground-truth bond graph.

What is *kept* is ``_build_edge_type``: a purely distance-and-element based
bond-order heuristic computed from the current (noisy) coordinates. It reads
no ground-truth bonds, so it survives the bond drop intact and still supplies
the 5-wide edge feature the attention layers expect.

**``pred_bond_type`` (default ``False``, opt-in).** Upstream does not merely
*predict* bond types, it feeds them back: ``uni_transformer.py:504`` does
``edge_type = F.softmax(next_edge_type, dim=-1)`` inside the layer loop, so
layers 1..N-1 were trained on iteratively refined *predicted* bond types as
their 5-dim edge feature, not on the distance heuristic. Loading upstream's
released ``diffusion.pt`` without the ``b_inference`` head would therefore
feed every layer an input distribution it never saw -- a silent correctness
failure, not a crash. Switching ``pred_bond_type: true`` restores the head
and that feedback loop, which is what makes the port checkpoint-compatible.

Bond *supervision* (the bond-type CE / bond-distance / bond-angle / torsion
losses) still needs a ground-truth bond channel the data layer does not have,
and stays out of scope. It is not needed here: upstream's
``_pred_edge_type(..., if_test=True)`` path -- the only one this port
implements -- produces the feedback features without any ground truth.

The shape latent enters at every layer, twice: as an invariant scalar
(``InvariantShapeEmbLayer``) and as raw equivariant 3-vectors mixed into the
vector channel. That is why the latent must be in the same rotational frame
as the coordinates.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from MolecularDiffusion.modules.layers.vn import VNLinear, VNLinearLeakyReLU
from MolecularDiffusion.modules.models.diffsmol.bond_tables import (
    DIFFSMOL_BOND_TABLES,
)
from MolecularDiffusion.modules.models.diffsmol.common import (
    GVP,
    GaussianSmearing,
    GVPLayerNorm,
    MLP,
    ShiftedSoftplus,
    norm_no_nan,
    outer_product,
)
from MolecularDiffusion.utils.chem.geom_analyzer import bonds1, bonds2, bonds3

EPS = 1e-6

#: Distance tolerances (in 0.01 A units) for the single/double/triple/aromatic
#: bond-order heuristic. Upstream ``UniTransformerO2TwoUpdateGeneral``.
BOND_MARGINS = (10, 5, 3, 5)

#: ``len(utils_data.BOND_TYPES)`` upstream: unspecified/single/double/triple/
#: aromatic. Also the width of the edge feature, hence ``edge_feat_dim: 5``.
N_BOND_TYPES = 5


def build_bond_length_tensors(atom_vocab: list[str]) -> torch.Tensor:
    """Reference bond lengths per (element_i, element_j) and bond order.

    Returns ``[4, V, V]`` in 0.01 A units: single / double / triple /
    aromatic. Missing pairs get ``-1``, upstream's sentinel for "this bond
    order does not exist for this pair" (the heuristic's
    ``dist - ref < margin`` test then never fires for real distances).

    The aromatic slice is all ``-1``: aromaticity is a bond-graph property
    and was dropped with the bonds, so the vocab is plain elements.
    Reuses the platform's own bond-length tables rather than re-vendoring
    DiffSMol's copies of them.
    """
    v = len(atom_vocab)
    out = torch.full((4, v, v), -1.0)
    for order, table in enumerate((bonds1, bonds2, bonds3)):
        for i, ai in enumerate(atom_vocab):
            row = table.get(ai, {})
            for j, aj in enumerate(atom_vocab):
                out[order, i, j] = row.get(aj, -1)
    # symmetrise: the tables are only populated in one triangle for some pairs
    for order in range(3):
        tri = out[order]
        out[order] = torch.where(tri >= 0, tri, tri.T)
    return out


def build_diffsmol_bond_tensors(
    elements: list[str], aromatic_flags: list[bool]
) -> torch.Tensor:
    """Upstream ``construct_bond_tensors('add_aromatic')``, replicated exactly.

    ``elements[i]``/``aromatic_flags[i]`` describe class ``i`` of the 15-class
    ``(element, is_aromatic)`` vocabulary. A pair that is aromatic on *both*
    ends gets single + aromatic reference lengths and no double/triple; every
    other pair gets single/double/triple and no aromatic.

    Used only by the ``pred_bond_type`` / ``add_aromatic`` checkpoint-
    compatible path. ``build_bond_length_tensors`` above stays the element-
    vocab default and is untouched.

    # ponytail: upstream's loop writes ``triple_bond_tensor[i, j]`` twice and
    # never ``[j, i]`` (analyze.py:102-103), so its triple-bond table is
    # upper-triangular and a triple bond is only ever detected when
    # ``class(src) <= class(dst)``. Replicated deliberately: the released
    # checkpoint was trained against exactly that asymmetry. Do not "fix" it
    # without retraining.
    """
    if len(elements) != len(aromatic_flags):
        raise ValueError(
            f"elements ({len(elements)}) and aromatic_flags "
            f"({len(aromatic_flags)}) must have the same length"
        )
    single, double, triple, aromatic = DIFFSMOL_BOND_TABLES
    n = len(elements)
    out = torch.full((4, n, n), -1.0)
    for i in range(n):
        for j in range(i, n):
            ei, ej = elements[i], elements[j]
            out[0, i, j] = out[0, j, i] = single[ei][ej]
            if aromatic_flags[i] and aromatic_flags[j]:
                out[3, i, j] = out[3, j, i] = aromatic[ei][ej]
            else:
                out[1, i, j] = out[1, j, i] = double[ei][ej]
                out[2, i, j] = triple[ei][ej]
    return out


class BaseGVPAttLayer(nn.Module):
    """One shape-conditioned GVP attention layer."""

    def __init__(
        self,
        input_dim: tuple[int, int],
        hidden_dim: tuple[int, int],
        output_dim: tuple[int, int],
        shape_dim: int,
        shape_latent_dim: int,
        n_heads: int,
        edge_feat_dim: int,
        r_feat_dim: int,
        mess_gvp_layer_num: int = 3,
        node_gvp_layer_num: int = 3,
        act_fn: str = "relu",
        use_shape_vec_mul: bool = True,
        use_residue: bool = True,
        mlp_norm: bool = True,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.shape_dim = shape_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.use_shape_vec_mul = use_shape_vec_mul
        self.use_residue = use_residue
        self.edge_feat_dim = edge_feat_dim
        self.output_norm = output_norm

        # scalar_feat | <shape, vec> | |vec| | invariant shape embedding.
        # Upstream writes ``shape_dim`` for that last term; it means
        # ``shape_latent_dim`` (the width ``InvariantShapeEmbLayer`` emits).
        # The two are equal in upstream's shipped config, so the bug is
        # invisible there and only bites when they are configured apart.
        shape_emb_dim = (
            input_dim[0]
            + shape_dim * input_dim[1]
            + input_dim[1]
            + shape_latent_dim
        )
        self.shape_in = VNLinear(shape_dim, shape_dim)
        self.shape_emb_layer = nn.Sequential(
            nn.Linear(shape_emb_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[0]),
        )
        self.shape_scalar_layer = MLP(
            input_dim[0] + hidden_dim[0],
            hidden_dim[0],
            hidden_dim[0],
            norm=mlp_norm,
            act_fn=act_fn,
        )

        if self.use_shape_vec_mul:
            self.shape_vec_layer = nn.Sequential(
                nn.Linear(hidden_dim[0], 2 * hidden_dim[0]),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim[0], shape_dim * input_dim[1]),
            )
            self.vector_emb_shape_layer = VNLinearLeakyReLU(
                input_dim[1] * 2, hidden_dim[1], dim=3
            )
        else:
            self.vector_emb_shape_layer = VNLinearLeakyReLU(
                input_dim[1] + shape_dim, hidden_dim[1], dim=3
            )

        kv_input_dim = hidden_dim[0] + hidden_dim[1]
        self.hk_func = MLP(
            kv_input_dim, hidden_dim[0], hidden_dim[0],
            norm=mlp_norm, act_fn=act_fn,
        )
        self.hq_func = MLP(
            hidden_dim[0] + hidden_dim[1], hidden_dim[0], hidden_dim[0],
            norm=mlp_norm, act_fn=act_fn,
        )

        message_layers = []
        for i in range(mess_gvp_layer_num):
            mess_input_dim = (
                (hidden_dim[0] + r_feat_dim + edge_feat_dim, hidden_dim[1] + 1)
                if i == 0
                else hidden_dim
            )
            message_layers.append(
                GVP(mess_input_dim, hidden_dim[1], hidden_dim)
            )
        self.message_layer = nn.Sequential(*message_layers)

        node_output_layer_list = (
            [
                GVP(
                    (2 * hidden_dim[0] + input_dim[0], 2 * hidden_dim[1] + 1),
                    hidden_dim[0],
                    hidden_dim,
                )
            ]
            + [
                GVP(hidden_dim, hidden_dim[0], hidden_dim)
                for _ in range(node_gvp_layer_num - 2)
            ]
            + [GVP(hidden_dim, hidden_dim[0], output_dim)]
        )
        self.node_output_layer = nn.Sequential(*node_output_layer_list)

        if output_norm:
            self.node_norm = GVPLayerNorm(output_dim)

    def embed_ligand_shape(
        self,
        scalar_feat: torch.Tensor,
        vec_feat: torch.Tensor,
        ligand_shape: torch.Tensor,
        invar_ligand_shape: torch.Tensor,
    ) -> torch.Tensor:
        n = scalar_feat.size(0)
        vec_feat_norm = norm_no_nan(vec_feat)
        net_shape = torch.einsum(
            "bmi,bni->bmn", ligand_shape, vec_feat
        ).view(n, -1)
        shape_input = torch.cat(
            [scalar_feat, net_shape, vec_feat_norm, invar_ligand_shape], -1
        )
        return self.shape_emb_layer(shape_input)

    def embed_message_att_weight(
        self,
        mess_scalar_emb: torch.Tensor,
        mess_vec_emb: torch.Tensor,
        node_scalar_emb: torch.Tensor,
        node_vec_emb: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        _src, dst = edge_index
        kv_input = torch.cat(
            [mess_scalar_emb, norm_no_nan(mess_vec_emb)], -1
        )
        k = self.hk_func(kv_input)
        q = self.hq_func(
            torch.cat([node_scalar_emb, norm_no_nan(node_vec_emb)], -1)
        )
        return scatter_softmax(
            (q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0
        )

    def message_passing(
        self,
        scalar_emb: torch.Tensor,
        vec_emb: torch.Tensor,
        x: torch.Tensor,
        ligand_emb: torch.Tensor,
        edge_feat: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        r_feat: torch.Tensor,
        rel_x: torch.Tensor,
    ):
        n = scalar_emb.size(0)
        src, dst = edge_index

        mess_scalar_i_in = torch.cat([scalar_emb[src], r_feat], -1)
        if edge_feat is not None and self.edge_feat_dim > 0:
            mess_scalar_i_in = torch.cat([mess_scalar_i_in, edge_feat], -1)
        mess_vec_i_in = torch.cat([vec_emb[src], rel_x.unsqueeze(1)], -2)

        mess_scalar_i_out, mess_vec_i_out = self.message_layer(
            (mess_scalar_i_in, mess_vec_i_in)
        )

        att_weight = self.embed_message_att_weight(
            mess_scalar_i_out, mess_vec_i_out, scalar_emb, vec_emb, edge_index
        )

        scalar_output = scatter_sum(
            att_weight.unsqueeze(-1) * mess_scalar_i_out,
            dst, dim=0, dim_size=n,
        )
        vec_output = scatter_sum(
            att_weight.view(-1, 1, 1) * mess_vec_i_out,
            dst, dim=0, dim_size=n,
        )

        scalar_output = torch.cat([ligand_emb, scalar_emb, scalar_output], -1)
        vec_output = torch.cat([x, vec_emb, vec_output], -2)
        return self.node_output_layer((scalar_output, vec_output))

    def forward(
        self,
        scalar_feat: torch.Tensor,
        vec_feat: torch.Tensor,
        r_feat: torch.Tensor,
        rel_x: torch.Tensor,
        x: torch.Tensor,
        ligand_emb: torch.Tensor,
        edge_feat: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        ligand_shape: torch.Tensor,
        invar_ligand_shape: torch.Tensor,
    ):
        o = self.embed_ligand_shape(
            scalar_feat, vec_feat, ligand_shape, invar_ligand_shape
        )
        scalar_emb = self.shape_scalar_layer(torch.cat([scalar_feat, o], -1))

        if self.use_shape_vec_mul:
            vec_o = (
                self.shape_vec_layer(o)
                .unsqueeze(-1)
                .view(scalar_feat.shape[0], -1, self.shape_dim)
            )
            vec_emb_input = torch.cat([vec_feat, vec_o @ ligand_shape], 1)
        else:
            vec_emb_input = torch.cat([vec_feat, ligand_shape], 1)
        vec_emb = self.vector_emb_shape_layer(vec_emb_input)

        scalar_output, vec_output = self.message_passing(
            scalar_emb, vec_emb, x, ligand_emb, edge_feat,
            edge_index, r_feat, rel_x,
        )
        if (
            self.use_residue
            and self.output_dim[0] == self.input_dim[0]
            and self.output_dim[1] == self.input_dim[1]
        ):
            scalar_output = scalar_feat + scalar_output
            vec_output = vec_feat + vec_output

        if self.output_norm:
            scalar_output, vec_output = self.node_norm(
                (scalar_output, vec_output)
            )
        return scalar_output, vec_output


class InvariantShapeEmbLayer(nn.Module):
    """Collapse the equivariant ``(shape_dim, 3)`` latent to an invariant
    ``shape_latent_dim`` scalar by projecting onto its own mean direction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        act_fn: str = "relu",
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_layer = MLP(
            input_dim, output_dim, input_dim, norm=norm, act_fn=act_fn
        )

    def forward(self, shape_h: torch.Tensor) -> torch.Tensor:
        shape_mean = shape_h.mean(dim=1)
        shape_mean_norm = (shape_mean * shape_mean).sum(-1, keepdim=True)
        shape_mean_norm = shape_mean / (shape_mean_norm + EPS)
        invar = torch.einsum("bij,bj->bi", shape_h, shape_mean_norm)
        return self.hidden_layer(invar)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    """Thin wrapper: distance expansion + edge-feature outer product, then
    one ``BaseGVPAttLayer``."""

    #: Class-level default so modules pickled before the bond toggle existed
    #: (e.g. the smoke-test checkpoints) still resolve it after unpickling.
    pred_bond_type = False

    def __init__(
        self,
        input_dim: tuple[int, int],
        hidden_dim: tuple[int, int],
        output_dim: tuple[int, int],
        n_heads: int,
        num_r_gaussian: int,
        edge_feat_dim: int,
        shape_dim: int,
        shape_latent_dim: int,
        act_fn: str = "relu",
        norm: bool = True,
        r_min: float = 0.0,
        r_max: float = 10.0,
        output_norm: bool = True,
        use_shape_vec_mul: bool = True,
        use_residue: bool = True,
        pred_bond_type: bool = False,
    ) -> None:
        super().__init__()
        self.distance_expansion = GaussianSmearing(
            r_min, r_max, num_gaussians=num_r_gaussian
        )
        self.gvp_layer = BaseGVPAttLayer(
            input_dim,
            hidden_dim,
            output_dim,
            shape_dim,
            shape_latent_dim,
            n_heads,
            edge_feat_dim,
            r_feat_dim=self.distance_expansion.num_gaussians
            * max(edge_feat_dim, 1),
            use_shape_vec_mul=use_shape_vec_mul,
            use_residue=use_residue,
            act_fn=act_fn,
            mlp_norm=norm,
            output_norm=output_norm,
        )
        self.edge_feat_dim = edge_feat_dim
        self.output_dim = output_dim
        self.pred_bond_type = pred_bond_type
        if pred_bond_type:
            # Last layer has a single output 3-vector, so it has no vector
            # norms to compare and falls back to a distance expansion of the
            # *predicted* positions instead.
            b_in_dim = (
                2 * output_dim[0] + num_r_gaussian
                if output_dim[1] == 1
                else 2 * output_dim[0] + 2 * output_dim[1]
            )
            self.b_inference = nn.Sequential(
                nn.Linear(b_in_dim, hidden_dim[0]),
                ShiftedSoftplus(),
                nn.Linear(hidden_dim[0], N_BOND_TYPES),
            )

    def _pred_edge_type(
        self,
        h: torch.Tensor,
        vec: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Per-edge bond-type logits, scattered back onto ``edge_index``.

        Upstream ``_pred_edge_type(..., if_test=True)``: the head is evaluated
        once per *undirected* edge (kNN graphs contain both directions), then
        broadcast back to every directed edge. No ground-truth bonds are
        touched -- that branch only ever built the CE target, which is out of
        scope here.
        """
        swap = torch.where(edge_index[0] >= edge_index[1])[0]
        canonical = edge_index.clone()
        canonical[:, swap] = edge_index[:, swap].flip(0)
        unique_edge_index, inverse = torch.unique(
            canonical, sorted=True, return_inverse=True, dim=1
        )
        src, dst = unique_edge_index

        diff_h = torch.abs(h[src] - h[dst])
        sum_h = h[src] + h[dst]
        if self.output_dim[1] == 1:
            pred_new_x = vec.squeeze(1) + x
            dist = torch.norm(
                pred_new_x[dst] - pred_new_x[src], p=2, dim=-1, keepdim=True
            )
            bond_input = torch.cat(
                (self.distance_expansion(dist), diff_h, sum_h), dim=1
            )
        else:
            norm1, norm2 = norm_no_nan(vec[src]), norm_no_nan(vec[dst])
            bond_input = torch.cat(
                (torch.abs(norm1 - norm2), norm1 + norm2, diff_h, sum_h),
                dim=1,
            )
        return self.b_inference(bond_input)[inverse]

    def forward(
        self,
        h: torch.Tensor,
        vec: torch.Tensor,
        x: torch.Tensor,
        ligand_emb: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        ligand_shape_emb: torch.Tensor,
        invar_ligand_shape: torch.Tensor,
    ):
        edge_feat = edge_attr if self.edge_feat_dim > 0 else None
        src, dst = edge_index
        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)
        dist_feat = outer_product(edge_attr, self.distance_expansion(dist))
        new_h, new_vec = self.gvp_layer(
            h, vec, dist_feat, rel_x, x.unsqueeze(1), ligand_emb,
            edge_feat, edge_index, ligand_shape_emb, invar_ligand_shape,
        )
        if self.pred_bond_type:
            return new_h, new_vec, self._pred_edge_type(
                new_h, new_vec, x, edge_index
            )
        return new_h, new_vec


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    """``uni_o2`` backbone; bond-free unless ``pred_bond_type`` is set."""

    #: See ``AttentionLayerO2TwoUpdateNodeGeneral.pred_bond_type``.
    pred_bond_type = False

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        scalar_hidden_dim: int,
        vec_hidden_dim: int,
        shape_dim: int,
        shape_latent_dim: int,
        atom_vocab: list[str],
        n_heads: int = 1,
        k: int = 32,
        num_r_gaussian: int = 20,
        edge_feat_dim: int = 5,
        act_fn: str = "relu",
        norm: bool = True,
        r_max: float = 10.0,
        use_shape_vec_mul: bool = False,
        use_residue: bool = True,
        pred_bond_type: bool = False,
        aromatic_flags: Optional[list[bool]] = None,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.scalar_hidden_dim = scalar_hidden_dim
        self.vec_hidden_dim = vec_hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.r_max = r_max
        self.use_shape_vec_mul = use_shape_vec_mul
        self.use_residue = use_residue
        self.shape_dim = shape_dim
        self.shape_latent_dim = shape_latent_dim
        self.atom_vocab = list(atom_vocab)
        self.pred_bond_type = pred_bond_type
        self.aromatic_flags = (
            list(aromatic_flags) if aromatic_flags is not None else None
        )

        self.base_block = self._build_share_blocks()
        if self.edge_feat_dim > 0:
            # Buffer (not a bare cuda tensor as upstream) so it follows
            # ``.to(device)`` and lands in the checkpoint.
            self.register_buffer(
                "bond_tensors",
                build_bond_length_tensors(self.atom_vocab)
                if self.aromatic_flags is None
                else build_diffsmol_bond_tensors(
                    self.atom_vocab, self.aromatic_flags
                ),
            )
        self.invariant_shape_layer = InvariantShapeEmbLayer(
            shape_dim, shape_latent_dim
        )

    def _build_share_blocks(self) -> nn.ModuleList:
        hidden_dims = (self.scalar_hidden_dim, self.vec_hidden_dim)
        base_block = []
        for l_idx in range(self.num_layers - 1):
            input_dims = (
                (self.scalar_hidden_dim, 1) if l_idx == 0 else hidden_dims
            )
            base_block.append(
                AttentionLayerO2TwoUpdateNodeGeneral(
                    input_dims, hidden_dims, hidden_dims, self.n_heads,
                    self.num_r_gaussian, self.edge_feat_dim, self.shape_dim,
                    self.shape_latent_dim,
                    act_fn=self.act_fn, norm=self.norm, r_max=self.r_max,
                    use_shape_vec_mul=self.use_shape_vec_mul,
                    use_residue=self.use_residue, output_norm=True,
                    pred_bond_type=self.pred_bond_type,
                )
            )
        # last layer collapses the vector channel to a single 3-vector: the
        # predicted coordinate displacement.
        base_block.append(
            AttentionLayerO2TwoUpdateNodeGeneral(
                hidden_dims, hidden_dims, (self.scalar_hidden_dim, 1),
                self.n_heads, self.num_r_gaussian, self.edge_feat_dim,
                self.shape_dim, self.shape_latent_dim,
                act_fn=self.act_fn, norm=self.norm,
                r_max=self.r_max, use_shape_vec_mul=self.use_shape_vec_mul,
                use_residue=self.use_residue, output_norm=False,
                pred_bond_type=self.pred_bond_type,
            )
        )
        return nn.ModuleList(base_block)

    def _get_bond_order(
        self,
        atom1: torch.Tensor,
        atom2: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Heuristic bond order from element pair + interatomic distance."""
        edge_dist = edge_dist * 100
        edge_types = torch.zeros(
            atom1.shape[0], dtype=torch.long, device=atom1.device
        )
        for order in range(4):
            ref = self.bond_tensors[order][atom1, atom2]
            edge_types[(edge_dist - ref) < BOND_MARGINS[order]] = order + 1
        return edge_types

    def _build_edge_type(
        self,
        ligand_v: torch.Tensor,
        ligand_pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        a1 = torch.argmax(ligand_v[src], dim=1)
        a2 = torch.argmax(ligand_v[dst], dim=1)
        edge_dist = torch.norm(ligand_pos[src] - ligand_pos[dst], dim=1)
        return F.one_hot(
            self._get_bond_order(a1, a2, edge_dist), num_classes=5
        ).float()

    def _connect_graph(
        self,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch: torch.Tensor,
    ):
        edge_index = knn_graph(
            ligand_pos, k=self.k, batch=batch, flow="source_to_target"
        )
        return edge_index, self._build_edge_type(
            ligand_v, ligand_pos, edge_index
        )

    def forward(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        x: torch.Tensor,
        batch_ligand: torch.Tensor,
        ligand_shape: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        all_vec, all_h = [x], [h]

        invar_ligand_shape_emb = torch.index_select(
            self.invariant_shape_layer(ligand_shape), 0, batch_ligand
        )
        ligand_shape_emb = torch.index_select(ligand_shape, 0, batch_ligand)

        vec = x.unsqueeze(1)
        ligand_emb = h
        for _ in range(self.num_blocks):
            edge_index, edge_type = self._connect_graph(x, v, batch_ligand)
            for layer in self.base_block:
                if self.pred_bond_type:
                    h, vec, next_edge_type = layer(
                        h, vec, x, ligand_emb, edge_type, edge_index,
                        ligand_shape_emb, invar_ligand_shape_emb,
                    )
                    # The feedback loop: every layer after the first sees
                    # refined *predicted* bond types, not the distance
                    # heuristic. Upstream uni_transformer.py:504.
                    edge_type = F.softmax(next_edge_type, dim=-1)
                else:
                    h, vec = layer(
                        h, vec, x, ligand_emb, edge_type, edge_index,
                        ligand_shape_emb, invar_ligand_shape_emb,
                    )
            vec = vec.squeeze(1)
            all_vec.append(vec)
            all_h.append(h)

        outputs = {"x": vec + x, "h": h}
        if return_all:
            outputs.update({"all_vec": all_vec, "all_h": all_h})
        return outputs
