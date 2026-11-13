import warnings

import torch
import torch.nn as nn
from MolecularDiffusion.modules.layers.conv import XEyTransformerLayer, PositionsMLP
from MolecularDiffusion.utils import (
    remove_mean,
    remove_mean_with_mask, 
)

class GraphTransformer(nn.Module):
    """
    Graph Transformer model for processing graph-structured data with node, edge, and global features.

    This model applies a stack of transformer layers to update node, edge, and global features, supporting flexible input and output dimensions.
    It is suitable for tasks such as molecular property prediction, generative modeling, and other graph-based learning problems.

    Args:
        in_node_nf (int): Number of input node features.
        in_edge_nf (int): Number of input edge features.
        in_global_nf (int): Number of input global features.
        n_layers (int): Number of transformer layers.
        hidden_mlp_dims (dict): Hidden dimensions for MLPs (keys: 'X', 'E', 'y', 'pos').
        hidden_dims (dict): Hidden dimensions for transformer layers (keys: 'dx', 'de', 'dy', 'n_head', 'dim_ffX', 'dim_ffE').
        out_node_nf (int, optional): Number of output node features. Defaults to in_node_nf.
        out_edge_nf (int, optional): Number of output edge features. Defaults to in_edge_nf.
        dropout (float): Dropout probability.
        act_fn_in (nn.Module): Activation function for input MLPs.
        act_fn_out (nn.Module): Activation function for output MLPs.
    """

    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        in_global_nf: int,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        out_node_nf: int = None,
        out_edge_nf: int = None,
        dropout: float = 0.0,
        act_fn_in: nn.Module = nn.SiLU(),
        act_fn_out: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        if out_node_nf is None:
            self.out_dim_X = in_node_nf
        else:
            self.out_dim_X = out_node_nf

        if out_edge_nf is None:
            self.out_dim_E = in_edge_nf
        else:
            self.out_dim_E = out_edge_nf
        self.out_dim_y = in_global_nf
        self.out_dim_charges = 1

        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_node_nf, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(in_edge_nf, hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(in_global_nf, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims["pos"])

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dropout=dropout,
                    last_layer=False,
                )
                for i in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], self.out_dim_X),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], self.out_dim_E),
        )
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])

    def forward(self, X, E, y, pos, node_mask, get_emd=False):
        """
        X: node features (bs, n, in_node_nf)
        E: adjacncy matrixs (bs, n, n, edge features dim)
        y: global features (bs, n, global features dim)
        pos: positions (bs, n, 3)
        node_mask: node mask (bs, n)
        """

        if y.dtype != X.dtype:
            y = y.to(X.dtype)

        if E.dtype != X.dtype:
            E = E.to(X.dtype)

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        X = self.mlp_in_X(X)
        E = new_E
        y = self.mlp_in_y(y)
        pos = self.mlp_in_pos(pos, node_mask)

        for layer in self.tf_layers:
            X, E, y, pos, node_mask = layer(X, E, y, pos, node_mask)

        if not get_emd:
            X = self.mlp_out_X(X)
            E = self.mlp_out_E(E)
            pos = self.mlp_out_pos(pos, node_mask)

            E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        return (
            X,
            E,
            y,
            pos,
        )


class GraphDiffTransformer(nn.Module):
    """
    Args:
        in_node_nf (int): Number of input node features.
        in_edge_nf (int): Number of input edge features.
        in_global_nf (int): Number of input global features.
        n_layers (int): Number of transformer layers.
        hidden_mlp_dims (dict): Dictionary specifying hidden dimensions for MLPs for 'X' (node), 'E' (edge), 'y' (global), and 'pos' (position) features.
        hidden_dims (dict): Dictionary specifying hidden dimensions for the transformer layers, including 'dx', 'de', 'dy', 'n_head', 'dim_ffX', and 'dim_ffE'.
        out_node_nf (int, optional): Number of output node features. Defaults to `in_node_nf`.
        out_edge_nf (int, optional): Number of output edge features. Defaults to `in_edge_nf`.
        dropout (float): Dropout probability. Defaults to 0.0.
        act_fn_in (nn.Module): Activation function for input MLPs. Defaults to `nn.SiLU()`.
        act_fn_out (nn.Module): Activation function for output MLPs. Defaults to `nn.SiLU()`.
    """

    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        in_global_nf: int,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        out_node_nf: int = None,
        out_edge_nf: int = None,
        dropout: float = 0.0,
        act_fn_in: nn.Module = nn.SiLU(),
        act_fn_out: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf
        if out_node_nf is None:
            self.out_dim_X = in_node_nf
        else:
            self.out_dim_X = out_node_nf

        if out_edge_nf is None:
            self.out_dim_E = in_edge_nf
        else:
            self.out_dim_E = out_edge_nf
        self.out_dim_y = in_global_nf
        self.out_dim_charges = 1

        self.mlp_in_X = nn.Sequential(
            nn.Linear(in_node_nf, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )
        self.mlp_in_E = nn.Sequential(
            nn.Linear(in_edge_nf, hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(in_global_nf, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims["pos"])

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims["dim_ffX"],
                    dim_ffE=hidden_dims["dim_ffE"],
                    dropout=dropout,
                    last_layer=False,
                )
                for i in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], self.out_dim_X),
        )
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], self.out_dim_E),
        )
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])

    def forward(self, X, E, y, pos, node_mask, get_emd=False):
        """
        X: node features (bs, n, in_node_nf)
        E: adjacncy matrixs (bs, n, n, edge features dim)
        y: global features (bs, n, global features dim)
        pos: positions (bs, n, 3)
        node_mask: node mask (bs, n)
        """

        bs, n = X.shape[0], X.shape[1]
        if y.dtype != torch.float32:
            y = y.float()

        diag_mask = ~torch.eye(n, device=X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        new_E = self.mlp_in_E(E.float())
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        X = self.mlp_in_X(X)
        E = new_E
        y = self.mlp_in_y(y)
        pos = self.mlp_in_pos(pos, node_mask)

        for i, layer in enumerate(self.tf_layers):
            X, E, y, pos, node_mask = layer(X, E, y, pos, node_mask, i)

        if not get_emd:
            X = self.mlp_out_X(X)
            E = self.mlp_out_E(E)
            pos = self.mlp_out_pos(pos, node_mask)

            E = 1 / 2 * (E + torch.transpose(E, 1, 2))

        return (
            X,
            E,
            y,
            pos,
        )

class EGT_dynamics(nn.Module):
    """
    The dynamic function for EGT-based diffusion models.
    Args:
        in_node_nf (int): Number of input node features per node.
        in_edge_nf (int): Number of input edge features per edge.
        in_global_nf (int): Number of input global features.
        n_layers (int): Number of transformer layers.
        hidden_mlp_dims (dict): Hidden dimensions for MLPs.
        hidden_dims (dict): Hidden dimensions for transformer layers.
        context_node_nf (int): Number of context features per node.
        dropout (float): Dropout probability.
        n_dims (int): Number of spatial dimensions (e.g., 3 for 3D coordinates).
        condition_time (bool): Whether to condition on time.
        model (str): The name of the EGNN model to use ('GraphTransformer' or 'GraphDiffTransformer').
    """

    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        in_global_nf: int,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        context_node_nf: int,
        dropout: float = 0.0,
        n_dims: int = 3,
        condition_time=True,
        model: str = "GraphTransformer",
    ):
        super().__init__()

        if model == "GraphTransformer":
            model_class = GraphTransformer
        elif model == "GraphDiffTransformer":
            model_class = GraphDiffTransformer
        else:
            raise ValueError(f"Unknown model: {model}")

        self.egnn = model_class(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            in_global_nf=in_global_nf,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        """
        Performs a single forward pass of the dynamics model.

        Args:
            t (torch.Tensor): A tensor representing time, shape (bs,) or a single value.
            xh (torch.Tensor): The state tensor, containing node positions and features, shape (bs, n_nodes, dims).
            node_mask (torch.Tensor): A boolean mask for nodes, shape (bs, n_nodes).
            edge_mask (torch.Tensor): A boolean mask for edges, shape (bs, n_nodes, n_nodes, 1).
            context (torch.Tensor, optional): Context information to condition on. Defaults to None.

        Returns:
            torch.Tensor: The change in state (velocities and feature changes), shape (bs, n_nodes, dims).
        """
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        node_mask = node_mask.view(bs, n_nodes)
        edge_mask = edge_mask.view(bs, n_nodes, n_nodes, -1)

        node_mask_ = node_mask.view(bs * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask_
        x = xh[:, 0 : self.n_dims].clone()
        y = t.view(bs, 1)

        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims :].clone()

        if self.condition_time:
            if torch.numel(t) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
            y = torch.cat([y, context], dim=1)

        h = h.view(bs, n_nodes, -1)
        x = x.view(bs, n_nodes, -1)
        h_final, _, _, x_final = self.egnn(
            h, E=edge_mask, y=y, pos=x, node_mask=node_mask
        )

        h_final = h_final.view(bs * n_nodes, -1)
        x_final = x_final.view(bs * n_nodes, -1)
        x = x.view(bs * n_nodes, -1)
        vel = (
            x_final - x
        ) * node_mask_  # This masking operation is redundant but just in case

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            warnings.warn("detected nan, resetting EGNN output to zero.", RuntimeWarning, stacklevel=2)
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1).int())

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)
        