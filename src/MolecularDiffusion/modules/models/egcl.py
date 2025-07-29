import torch
import torch.nn as nn
from MolecularDiffusion.modules.layers.common import MLP, SinusoidsEmbeddingNew
from MolecularDiffusion.modules.layers.conv import EquivariantBlock
from MolecularDiffusion.utils import (
    coord2diff, 
    coord2cosine,
    remove_mean,
    remove_mean_with_mask, 
)



class  EGNN(nn.Module):
    """
    Equivariant Graph Neural Network (EGNN) module for processing graph-structured data with node features and coordinates.

    This model supports optional context conditioning, sinusoidal embeddings, cosine edge features, and adapter modules for context.
    It is designed for tasks where equivariance to geometric transformations is important, such as molecular modeling.

    Args:
        in_node_nf (int): Number of input node features.
        hidden_nf (int): Number of hidden features.
        act_fn (nn.Module): Activation function.
        in_context_nf (int): Number of context features (for adapter module).
        n_layers (int): Number of EGNN layers.
        n_mlp_layers (int): Number of layers in each MLP.
        attention (bool): Whether to use attention in the EGNN blocks.
        norm_diff (bool): Whether to normalize coordinate differences.
        out_node_nf (int, optional): Number of output node features. Defaults to in_node_nf.
        tanh (bool): Whether to use tanh activation in coordinate updates.
        coords_range (float): Range for coordinate normalization.
        norm_constant (float): Normalization constant for coordinates.
        inv_sublayers (int): Number of sublayers in each EGNN block.
        sin_embedding (bool): Whether to use sinusoidal embedding for edge features.
        include_cosine (bool): Whether to include cosine similarity as edge features.
        normalization_factor (float): Factor for normalization in aggregation.
        aggregation_method (str): Aggregation method ('sum' or 'mean').
        dropout (float): Dropout probability.
        normalization (bool): Whether to use batch normalization in MLPs.
        adapter_module (bool): Whether to use adapter modules for context.
    """

    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        in_context_nf=0,
        n_layers=3,
        n_mlp_layers=2,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        norm_constant=1,
        inv_sublayers=2,
        sin_embedding=False,
        include_cosine=False,
        normalization_factor=100,
        aggregation_method="sum",
        dropout=0.0,
        normalization=False,
        adapter_module=False, # for context
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf

        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.include_cosine = include_cosine
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        if include_cosine:
            edge_feat_nf += 1

        self.embedding = MLP(
            in_node_nf,
            [hidden_nf] * n_mlp_layers,
            batch_norm=normalization,
            dropout=dropout,
        )

        self.embedding_out = MLP(
            hidden_nf,
            [hidden_nf] * n_mlp_layers + [out_node_nf],
            batch_norm=normalization,
            dropout=dropout,
        )
        for i in range(0, n_layers):
            self.add_module(
                "e_block_%d" % i,
                EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    act_fn=act_fn,
                    n_layers=inv_sublayers,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    sin_embedding=self.sin_embedding,
                    normalization_factor=self.normalization_factor,
                    aggregation_method=self.aggregation_method,
                    dropout=dropout,
                    normalization=normalization,
                ),
            )
        # self.to(self.device)
        
        self.adapter_module = adapter_module    
        if self.adapter_module:
            assert in_context_nf > 0, "in_context_nf must be greater than 0"
            self.emb_c_in = MLP(
                in_context_nf,
                [hidden_nf] * n_mlp_layers,
                batch_norm=normalization,
                dropout=dropout,
            )
            for i in range(0, n_layers):  
                self.add_module(
                    "adapter_%d" % i,
                    MLP(
                        hidden_nf,
                        [hidden_nf] * n_mlp_layers,
                        batch_norm=normalization,
                        dropout=dropout,
                    ),
                )
            

    def forward(
        self, h, x, edge_index, node_mask=None, edge_mask=None, context=None, use_embed=False
    ):

        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        if self.include_cosine:
            cosines = coord2cosine(x, edge_index).unsqueeze(-1)
            distances = torch.cat([distances, cosines], dim=1)

        h = self.embedding(h)
        
        if self.adapter_module and context is not None:
            h_c = self.emb_c_in(context)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h,
                x,
                edge_index,
                node_mask=node_mask,
                edge_mask=edge_mask,
                edge_attr=distances,
            )
            if self.adapter_module and context is not None:
                h_c = self._modules["adapter_%d" % i](h_c) 
                h_c = h_c.view(-1, self.hidden_nf)
                h+=h_c

        # Important, the bias of the last linear might be non-zero
        if use_embed:
            return h, x
        else:
            h = self.embedding_out(h)
            if node_mask is not None:
                h = h * node_mask
            return h, x


class EGNN_dynamics(nn.Module):
    """
    Dynamics model for Equivariant Diffusion Models (EDMs) using EGNNs.

    This class wraps an EGNN to model the time evolution of node features and coordinates, supporting context conditioning, time conditioning, and adapter modules.
    It is suitable for molecular dynamics, generative modeling, and other tasks requiring equivariant dynamics on graphs.

    Args:
        in_node_nf (int): Number of input node features per node (including time if used).
        context_node_nf (int): Number of context features per node.
        n_dims (int): Number of spatial dimensions (e.g., 3 for 3D coordinates).
        hidden_nf (int): Number of hidden features in the EGNN.
        act_fn (nn.Module): Activation function.
        n_layers (int): Number of EGNN blocks.
        attention (bool): Whether to use attention in the EGNN.
        condition_time (bool): Whether to condition on time.
        tanh (bool): Whether to use tanh in the EGNN.
        norm_constant (float): Normalization constant for the EGNN.
        inv_sublayers (int): Number of sublayers in the EGNN.
        sin_embedding (bool): Whether to use sinusoidal embedding in the EGNN.
        include_cosine (bool): Whether to include cosine as edge features.
        normalization_factor (float): Normalization factor for the EGNN.
        aggregation_method (str): Aggregation method for the EGNN.
        dropout (float): Dropout probability.
        normalization (bool): Whether to use normalization in the EGNN.
        use_adapter_module (bool): Whether to use adapter module for context.
    """

    def __init__(
        self,
        in_node_nf,
        context_node_nf,
        n_dims,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        attention=False,
        condition_time=True,
        tanh=False,
        norm_constant=0,
        inv_sublayers=2,
        sin_embedding=False,
        include_cosine=False,
        normalization_factor=100,
        aggregation_method="sum",
        dropout=0.0,
        normalization=False,
        use_adapter_module=False,
    ):
        """
        Dynamics model for EDMs using EGNNs.
        in_node_nf: int -- number of ALL input features per node (including time)
        context_node_nf: int -- number of context features per node
        n_dims: int -- number of dimensions for the output (3)
        hidden_nf: int -- number of hidden features in the EGNN
        act_fn: torch.nn.Module -- activation function
        n_layers: int -- number of EGNN blocks
        attention: bool -- whether to use attention in the EGNN
        condition_time: bool -- whether to condition on time
        tanh: bool -- whether to use tanh in the EGNN
        norm_constant: float -- normalization constant for the EGNN
        inv_sublayers: int -- number of layers in the EGNN
        sin_embedding: bool -- whether to use sin embedding in the EGNN
        include_cosine: bool -- whether to include cosine along with distance as edge features
        normalization_factor: float -- normalization factor for the EGNN
        aggregation_method: str -- aggregation method for the EGNN
        dropout: float -- dropout probability
        normalization: bool -- whether to use normalization in the EGNN
        use_adapter_module: bool -- whether to use adapter module for context
        """
        super().__init__()

        if use_adapter_module:
            in_node_nf_model = in_node_nf 
        else:
            in_node_nf_model = in_node_nf + context_node_nf
        
        self.use_adapter_module = use_adapter_module    
        self.egnn = EGNN(
            in_node_nf=in_node_nf_model,
            hidden_nf=hidden_nf,
            in_context_nf=context_node_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            include_cosine=include_cosine,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            dropout=dropout,
            normalization=normalization,
            adapter_module=use_adapter_module,  
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
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()
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

        if context is not None and not(self.use_adapter_module):
            # We're conditioning, awesome!
            context = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(
            h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context=context
        )
        vel = (
            x_final - x
        ) * node_mask  # This masking operation is redundant but just in case

        if context is not None and not(self.use_adapter_module):
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)
            h_final = torch.zeros_like(h_final)
        else:
            if node_mask is None:
                vel = remove_mean(vel)
            else:
                vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
        return torch.cat([vel, h_final], dim=2)
    
    
    def _forward_pyG(self, mol_graph):

        x = mol_graph["graph"].pos
        h = mol_graph["graph"].x
        edge_index = mol_graph["graph"].edge_index
        edges = [edge_index[0], edge_index[1]]
      
        if self.condition_time:
            h_time = mol_graph["t"]
            h = torch.cat([h, h_time], dim=1)


        if hasattr(mol_graph, 'context') and not(self.use_adapter_module):
            # We're conditioning, awesome!
            context = mol_graph["graph"].context # (nnodes, n_contexts)
            h = torch.cat([h, context], dim=1)
        else:
            context = None
            
        h_final, x_final = self.egnn(
            h, x, edges, node_mask=None, edge_mask=None, context=context
        )
 
        vel = (
            x_final - x
        ) # This masking operation is redundant but just in case

        if context is not None and not(self.use_adapter_module):
            # Slice off context size:
            h_final = h_final[:, : -self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
        if torch.any(torch.isnan(vel)):
            print("Warning: detected nan, resetting EGNN output to zero.")
            vel = torch.zeros_like(vel)
            h_final = torch.zeros_like(h_final)
        else:
            vel = remove_mean(vel)
          
        return torch.cat([vel, h_final], dim=1)



    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(device),
                    torch.LongTensor(cols).to(device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)


