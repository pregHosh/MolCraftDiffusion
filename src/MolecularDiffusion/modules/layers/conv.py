import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn import init
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.dropout import Dropout
from MolecularDiffusion.utils import coord2diff, assert_correctly_masked, remove_mean_with_mask_v2
import math

#%% EGCL
class GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        attention=False,
        dropout=0.0,
        normalization=False,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.normalization = normalization

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        self.dropout = Dropout(dropout)
        if normalization:
            self.normalization = nn.LayerNorm(hidden_nf)

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        newh, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            newh = newh * node_mask
        newh_d = self.dropout(newh)
        if self.normalization:
            h = self.normalization(newh_d)
        else:
            h = newh_d
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        tanh=False,
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = (
                coord_diff
                * torch.tanh(self.coord_mlp(input_tensor))
                * self.coords_range
            )
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):

    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        act_fn=nn.SiLU(),
        n_layers=2,
        attention=True,
        norm_diff=True,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        sin_embedding=None,
        normalization_factor=100,
        aggregation_method="sum",
        dropout=0.0,
        normalization=False,
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    act_fn=act_fn,
                    attention=attention,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                    dropout=dropout,
                    normalization=normalization,
                ),
            )
        self.add_module(
            "gcl_equiv",
            EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                tanh=tanh,
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method,
            ),
        )

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](
                h,
                edge_index,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        x = self._modules["gcl_equiv"](
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        )

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


def unsorted_segment_sum(
    data, segment_ids, num_segments, normalization_factor, aggregation_method: str
):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == "sum":
        result = result / normalization_factor

    if aggregation_method == "mean":
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


#%% EGT
def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)



class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """E: bs, n, n, de
        Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.to(E.dtype) # assume E is float
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.to(E.dtype) # assume E is float
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(
            e_mask2, dim=2
        )
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """Note: There is a relatively similar layer implemented by NVIDIA:
        https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
        It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (1,)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(
            node_mask, dim=1, keepdim=True
        )  # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)



class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """X: bs, n, dx."""
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.to(X.dtype)
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(
            x_mask, dim=1
        )
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out




class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        last_layer=False,
    ) -> None:

        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)

        self.linX1 = Linear(dx, dim_ffX)
        self.linX2 = Linear(dim_ffX, dx)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps)

        self.linE1 = Linear(de, dim_ffE)
        self.linE2 = Linear(dim_ffE, de)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy)
            self.lin_y2 = Linear(dim_ffy, dy)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X, E, y, pos, node_mask):
        """Pass the input through the encoder layer.
        X: (bs, n, d)
        E: (bs, n, n, d)
        y: (bs, dy)
        pos: (bs, n, 3)
        node_mask: (bs, n) Mask for the src keys per batch (optional)
        Output: newX, newE, new_y with the same shape.
        """

        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        newX, newE, new_y, vel = self.self_attn(X, E, y, pos, node_mask=node_mask)
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        
        new_pos = self.norm_pos1(vel, x_mask) + pos
        if torch.isnan(new_pos).any():
            new_pos = torch.nan_to_num(new_pos, nan=0.0)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

        return X, E, y, new_pos, node_mask


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, last_layer=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)
        # self.lin_dist2 = Linear(dx, dx)

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)

        self.pos_x_mul = EtoX(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)  # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Process_pos
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)  # For EGNN v3: map to pi, pj

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, pos, node_mask):
        """:param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param pos: bs, n, 3
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape."""
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)  # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1)
        cosines = torch.sum(
            normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2),
            dim=-1,
            keepdim=True,
        )
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)  # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)  # bs, n, de
        dist1 = (
            F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1))
            * e_mask1
            * e_mask2
        )

        # 1. Process E
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2  # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(Y) * e_mask1 * e_mask2  # bs, n, n, de
        assert_correctly_masked(Eout, e_mask1 * e_mask2)

        # 2. Process the node features
        Q = (self.q(X) * x_mask).unsqueeze(2)  # bs, 1, n, dx
        K = (self.k(X) * x_mask).unsqueeze(1)  # bs, n, 1, dx
        prod = Q * K / math.sqrt(Y.size(-1))  # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2  # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a = a * e_mask1 * e_mask2

        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)  # bs, 1, n, 1, dx
        weighted_V = alpha * V  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)  # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask  # bs, n, dx

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)  # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        assert_correctly_masked(Xout, x_mask)

        # Process y based on X and E
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)  # bs, dy

        # Update the positions
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)  # bs, 1, n, 3
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)  # bs, n, 1, 3
        delta_pos = pos2 - pos1  # bs, n, n, 3

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))  # bs, n, n, 1, 2
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        vel = remove_mean_with_mask_v2(vel, node_mask.unsqueeze(-1))
        return Xout, Eout, y_out, vel

class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp(norm)  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos
