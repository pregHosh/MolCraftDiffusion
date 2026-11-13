import warnings

import torch
import torch.nn as nn
from MolecularDiffusion import core
from MolecularDiffusion.modules.layers.common import MLP, SinusoidsEmbeddingNew
from MolecularDiffusion.modules.layers.conv import EquivariantBlock
from MolecularDiffusion.utils import (
    coord2diff, 
    coord2cosine,
    remove_mean,
    remove_mean_with_mask,
    remove_mean_pyG,
)



class EGNN(nn.Module, core.Configurable):
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
        adapter_module (bool): Whether to use adapter modules for context. (Legacy, prefer n_adapter_context.)
    n_adapter_context (int): Number of context features routed through adapter MLPs.
    n_concat_context (int): Number of context features concatenated to input. (Informational; caller must widen in_node_nf.)
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
        adapter_module=False, # legacy bool for backward compat
        n_adapter_context=0,  # hybrid: dims routed through adapter
        n_concat_context=0,   # hybrid: dims concatenated to input
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
        
        # Hybrid adapter support: resolve legacy bool vs new int args
        if adapter_module and n_adapter_context == 0:
            # Legacy: adapter_module=True means all context through adapter
            n_adapter_context = in_context_nf
        self.n_adapter_context = n_adapter_context
        self.n_concat_context = n_concat_context
        # Keep legacy attribute for backward compat checks
        self.adapter_module = n_adapter_context > 0

        if self.n_adapter_context > 0:
            self.emb_c_in = MLP(
                n_adapter_context,
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
            

    def __setstate__(self, state):
        """Backward compat: patch missing hybrid attributes for old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, 'n_adapter_context'):
            # Old checkpoint: infer from legacy adapter_module flag
            if getattr(self, 'adapter_module', False):
                # Old EGNN never stored in_context_nf, so recover the true adapter
                # input width from the existing emb_c_in MLP (dims[0] == in_context_nf).
                # A bare getattr here yields 0 and silently disables conditioning.
                emb_c = getattr(self, 'emb_c_in', None)
                width = getattr(self, 'in_context_nf', 0)
                if width == 0 and emb_c is not None and getattr(emb_c, 'dims', None):
                    width = emb_c.dims[0]
                self.n_adapter_context = width
                self.in_context_nf = width
                self.n_concat_context = 0
            else:
                self.n_adapter_context = 0
                self.n_concat_context = 0

    def forward(
        self, h, x, edge_index, node_mask=None, edge_mask=None, context=None, use_embed=False
    ):
        """Forward pass.
        
        Args:
            context: For adapter mode, this should contain ONLY the adapter-routed
                     columns (n_adapter_context dims). Concat-routed columns should
                     already be part of h before calling this method.
        """
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        if self.include_cosine:
            cosines = coord2cosine(x, edge_index).unsqueeze(-1)
            distances = torch.cat([distances, cosines], dim=1)

        h = self.embedding(h)
        
        if self.n_adapter_context > 0 and context is not None:
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
            if self.n_adapter_context > 0 and context is not None:
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


class EGNN_dynamics(nn.Module, core.Configurable):
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
        use_adapter_module (bool): Whether to use adapter module for context. (Legacy, prefer adapter_indices.)
    adapter_indices (list): Column indices of context tensor routed through adapter MLPs.
    concat_indices (list): Column indices of context tensor concatenated to input features.
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
        adapter_indices=None,
        concat_indices=None,
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
        use_adapter_module: bool -- (legacy) whether to use adapter for ALL context
        adapter_indices: list -- indices of context columns for adapter routing
        concat_indices: list -- indices of context columns for concatenation routing
        """
        super().__init__()

        # Resolve hybrid adapter/concat indices
        if adapter_indices is not None:
            self.adapter_indices = list(adapter_indices)
            self.concat_indices = list(concat_indices) if concat_indices is not None else []
        elif use_adapter_module:
            # Legacy: all context through adapter
            self.adapter_indices = list(range(context_node_nf))
            self.concat_indices = []
        else:
            # Legacy: all context through concat
            self.adapter_indices = []
            self.concat_indices = list(range(context_node_nf))

        n_adapter_context = len(self.adapter_indices)
        n_concat_context = len(self.concat_indices)
        # Legacy attribute for backward compat in diffusion models
        self.use_adapter_module = n_adapter_context > 0
        self.n_adapter_context = n_adapter_context
        self.n_concat_context = n_concat_context

        in_node_nf_model = in_node_nf + n_concat_context

        self.egnn = EGNN(
            in_node_nf=in_node_nf_model,
            hidden_nf=hidden_nf,
            in_context_nf=n_adapter_context,
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
            n_adapter_context=n_adapter_context,
            n_concat_context=n_concat_context,
        )
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.attention = attention
        self.tanh = tanh
        self.norm_constant = norm_constant
        self.inv_sublayers = inv_sublayers
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.dropout = dropout
        self.normalization = normalization
        self.include_cosine = include_cosine
        # act_fn is usually a module, might need string representation later
        self.act_fn_obj = act_fn

    def __setstate__(self, state):
        """Backward compat: patch missing hybrid attributes for old checkpoints."""
        super().__setstate__(state)
        if not hasattr(self, 'n_adapter_context'):
            # Old checkpoint: infer from legacy use_adapter_module flag
            context_nf = getattr(self, 'context_node_nf', 0)
            if getattr(self, 'use_adapter_module', False):
                self.adapter_indices = list(range(context_nf))
                self.concat_indices = []
                self.n_adapter_context = context_nf
                self.n_concat_context = 0
            else:
                self.adapter_indices = []
                self.concat_indices = list(range(context_nf))
                self.n_adapter_context = 0
                self.n_concat_context = context_nf

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)

        return fwd

    def unwrap_forward(self):
        return self._forward

    def _split_context(self, context):
        """Split context tensor into adapter and concat slices by stored indices."""
        if context is None:
            return None, None
        adapter_ctx = context[..., self.adapter_indices] if self.n_adapter_context > 0 else None
        concat_ctx = context[..., self.concat_indices] if self.n_concat_context > 0 else None
        return adapter_ctx, concat_ctx

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

        # Split context into adapter and concat slices
        adapter_ctx, concat_ctx = self._split_context(context)

        if concat_ctx is not None:
            concat_ctx = concat_ctx.view(bs * n_nodes, self.n_concat_context)
            h = torch.cat([h, concat_ctx], dim=1)

        # Prepare adapter context for EGNN
        egnn_context = None
        if adapter_ctx is not None:
            egnn_context = adapter_ctx.view(bs * n_nodes, self.n_adapter_context)

        h_final, x_final = self.egnn(
            h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context=egnn_context
        )
        vel = (
            x_final - x
        ) * node_mask  # This masking operation is redundant but just in case

        if self.n_concat_context > 0 and concat_ctx is not None:
            # Slice off concat context size:
            h_final = h_final[:, : -self.n_concat_context]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)
        h_final = h_final.view(bs, n_nodes, -1)

        # Reset per molecule: a batch-wide reset lets one diverging molecule zero
        # the prediction of every other molecule in the batch.
        nan_mol = torch.isnan(vel).flatten(1).any(dim=1)
        if nan_mol.any():
            warnings.warn(
                f"detected nan in {int(nan_mol.sum())}/{bs} molecule(s), "
                "resetting their EGNN output to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            nan_mol = nan_mol.view(bs, 1, 1)
            vel = torch.where(nan_mol, torch.zeros_like(vel), vel)
            h_final = torch.where(nan_mol, torch.zeros_like(h_final), h_final)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        return torch.cat([vel, h_final], dim=2)
    
    
    def _forward_pyG(self, mol_graph, return_hidden=False):
        """
        PyG-native forward pass through EGNN dynamics.
        
        Args:
            mol_graph: Dict with 'graph' (PyG Batch), 't' (timestep), 'context' (optional)
            return_hidden: If True, also return hidden features before final projection
            
        Returns:
            If return_hidden=False: continuous_out [N, 3 + in_node_nf]
            If return_hidden=True: (continuous_out, hidden_features [N, hidden_nf])
        """
        x = mol_graph["graph"].pos
        h = mol_graph["graph"].x
        atomic_numbers = mol_graph["graph"].atomic_numbers
        edge_index = mol_graph["graph"].edge_index
        edges = [edge_index[0], edge_index[1]]
        
        # Build h: [atomic_numbers, extra_features (if any), time]
        # atomic_numbers is always prepended as base feature for EGNN
        atom_feat = atomic_numbers.unsqueeze(-1).float()
        if h is None:
            h = atom_feat
        else:
            h = torch.cat([atom_feat, h], dim=1)
      
        if self.condition_time:
            h_time = mol_graph["t"]
            h = torch.cat([h, h_time], dim=1)

        # Correctly retrieve context from the input dictionary
        context = mol_graph.get("context")

        # Split context into adapter and concat slices
        adapter_ctx, concat_ctx = self._split_context(context)

        # Concatenate concat-routed context to h
        if concat_ctx is not None:
            h = torch.cat([h, concat_ctx], dim=1)
        
        # Get hidden features (before final projection) if requested
        if return_hidden:
            h_hidden, x_final = self.egnn(
                h, x, edges, node_mask=None, edge_mask=None, context=adapter_ctx, use_embed=True
            )
            h_final = self.egnn.embedding_out(h_hidden)
        else:
            h_final, x_final = self.egnn(
                h, x, edges, node_mask=None, edge_mask=None, context=adapter_ctx
            )
            h_hidden = None
 
        vel = (
            x_final - x
        ) # This masking operation is redundant but just in case

        if self.n_concat_context > 0 and concat_ctx is not None:
            # Slice off concat context size:
            h_final = h_final[:, : -self.n_concat_context]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
        if torch.any(torch.isnan(vel)):
            warnings.warn("detected nan, resetting EGNN output to zero.", RuntimeWarning, stacklevel=2)
            vel = torch.zeros_like(vel)
            h_final = torch.zeros_like(h_final)
        else:
            vel = remove_mean_pyG(vel, mol_graph["graph"].batch)
        
        continuous_out = torch.cat([vel, h_final], dim=1)
        
        if return_hidden:
            return continuous_out, h_hidden
        return continuous_out



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


