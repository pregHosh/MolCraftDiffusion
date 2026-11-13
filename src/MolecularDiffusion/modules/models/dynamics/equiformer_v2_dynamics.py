"""
EquiformerV2_dynamics — wraps EquiformerV2 encoder as a drop-in denoising network
for EnVariationalDiffusion and EnVariationalDiffusionPyG in en_diffusion.py.

The EquiformerV2 encoder and all layer files are NOT modified.
"""
import warnings

import torch
import torch.nn as nn
from MolecularDiffusion import core
from MolecularDiffusion.modules.layers.equiformer_v2_s.so3 import SO3_Embedding, SO3_Grid
from MolecularDiffusion.modules.layers.equiformer_v2_s.edge_rot_mat import init_edge_rot_mat
from MolecularDiffusion.modules.layers.equiformer_v2_s.transformer_block import FeedForwardNetwork
from MolecularDiffusion.modules.layers.equiformer_v2_s.module_list import ModuleListInfo
from MolecularDiffusion.utils import remove_mean_pyG


class EquiformerV2_dynamics(nn.Module, core.Configurable):
    """
    Dynamics wrapper that adapts EquiformerV2 for use with EnVariationalDiffusion(PyG).

    Args:
        equiformer: Instantiated EquiformerV2 model (from shepherd_arch).
        in_node_nf (int): Number of output node features expected by the diffusion model
                          (atomic_number_dim + extra_features + optional_charge).
                          Does NOT include timestep — that is handled internally.
        n_dims (int): Spatial dimensions (default 3).
        condition_time (bool): Whether to condition on timestep (default True).
        context_node_nf (int): Total context feature dimension (default 0).
        adapter_indices (list[int]): Indices of context columns routed through additive adapter MLPs.
        concat_indices (list[int]): Indices of context columns concatenated to input features.
        sphere_channels (int): Must match equiformer.sphere_channels.
        lmax_list (list[int]): Must match equiformer.lmax_list.
    """

    def __init__(
        self,
        equiformer: nn.Module,
        in_node_nf: int,
        n_dims: int = 3,
        condition_time: bool = True,
        context_node_nf: int = 0,
        adapter_indices=None,
        concat_indices=None,
        sphere_channels: int = 128,
        lmax_list=None,
    ):
        super().__init__()

        if lmax_list is None:
            lmax_list = [6]

        self.equiformer = equiformer
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.condition_time = condition_time
        self.context_node_nf = context_node_nf
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list

        # ── Resolve context routing ──────────────────────────────────────────
        if adapter_indices is not None:
            self.adapter_indices = list(adapter_indices)
            self.concat_indices = list(concat_indices) if concat_indices else []
        else:
            # default: all context concatenated
            self.adapter_indices = []
            self.concat_indices = list(range(context_node_nf))

        self.n_adapter_context = len(self.adapter_indices)
        self.n_concat_context = len(self.concat_indices)
        self.use_adapter_module = self.n_adapter_context > 0

        # ── Input dimension for projection ───────────────────────────────────
        # atomic_numbers(1) + extra_features(in_node_nf - 1) + time(1 if condition_time)
        # + concat_context
        # Note: in_node_nf already counts atomic_numbers as 1 dim.
        in_proj_dim = in_node_nf + int(condition_time) + self.n_concat_context

        self.input_proj = nn.Linear(in_proj_dim, sphere_channels)

        # ── SO3 grid for FeedForwardNetwork (needed for S2 activation) ────────
        lmax = max(lmax_list)
        self.vel_SO3_grid = ModuleListInfo('({}, {})'.format(lmax, lmax))
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(SO3_Grid(l, m, resolution=18, normalization='component'))
            self.vel_SO3_grid.append(SO3_m_grid)

        # ── Velocity head: SO3 FFN → l=1 extraction (like shepherd) ──────────
        # output_channels=1 so l=1 embedding[:,1:4,:] has shape [N,3,1] → squeeze → [N,3]
        self.head_vel_ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=sphere_channels,
            output_channels=1,
            lmax_list=lmax_list,
            mmax_list=lmax_list,   # use lmax as mmax for full resolution
            SO3_grid=self.vel_SO3_grid,
            activation='silu',
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
        )

        # ── Feature head: l=0 scalars → in_node_nf (unchanged) ───────────────
        self.head_h = nn.Linear(sphere_channels, in_node_nf)

        # ── Adapter MLP for context ───────────────────────────────────────────
        if self.n_adapter_context > 0:
            self.adapter_proj = nn.Linear(self.n_adapter_context, sphere_channels)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_so3_input(self, h_flat: torch.Tensor) -> SO3_Embedding:
        """
        Project flat node features [N, in_proj_dim] into SO3_Embedding.

        Args:
            h_flat: [N, in_proj_dim] tensor (atomic features + time + concat_ctx)

        Returns:
            SO3_Embedding with l=0 channel set to self.input_proj(h_flat),
            all higher-l channels zero.
        """
        num_nodes = h_flat.size(0)
        device = h_flat.device
        dtype = h_flat.dtype

        x = SO3_Embedding(
            num_nodes,
            self.lmax_list,
            self.sphere_channels,
            device,
            dtype,
        )
        # Populate l=0 channel only; higher-l are zero-initialized by SO3_Embedding
        x.embedding[:, 0, :] = self.input_proj(h_flat)
        return x

    @staticmethod
    def _compute_edge_geometry(pos: torch.Tensor, edge_index: torch.Tensor):
        """
        Compute edge distances and displacement vectors.

        Args:
            pos: [N, 3] Cartesian coordinates
            edge_index: [2, E] source/target indices

        Returns:
            edge_distance: [E] scalar distances
            edge_distance_vec: [E, 3] displacement vectors (target - source)
        """
        src, tgt = edge_index[0], edge_index[1]
        edge_distance_vec = pos[tgt] - pos[src]          # [E, 3]
        edge_distance = edge_distance_vec.norm(dim=-1)   # [E]
        return edge_distance, edge_distance_vec

    def _split_context(self, context):
        """
        Split context into adapter-routed and concat-routed slices.

        Returns (adapter_ctx, concat_ctx) — either may be None.
        """
        if context is None:
            return None, None
        adapter_ctx = context[..., self.adapter_indices] if self.n_adapter_context > 0 else None
        concat_ctx = context[..., self.concat_indices] if self.n_concat_context > 0 else None
        return adapter_ctx, concat_ctx

    # ─────────────────────────────────────────────────────────────────────────
    # Forward passes
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_core(self, h, pos, edge_index, batch, t=None, context=None):
        """
        Shared tail: h [N, in_node_nf] (atom/feature block, no time or
        context yet) -> [N, 3 + in_node_nf] velocity + feature update.
        """
        # ── Timestep conditioning ─────────────────────────────────────────────
        if self.condition_time:
            h = torch.cat([h, t], dim=1)

        # ── Context routing ───────────────────────────────────────────────────
        adapter_ctx, concat_ctx = self._split_context(context)

        if concat_ctx is not None:
            h = torch.cat([h, concat_ctx], dim=1)

        # ── Build SO3_Embedding input ─────────────────────────────────────────
        x_so3 = self._build_so3_input(h)  # l=0 = input_proj(h), higher-l = 0

        # Additive adapter injection into l=0
        if adapter_ctx is not None and self.n_adapter_context > 0:
            adapter_emb = self.adapter_proj(adapter_ctx)  # [N, sphere_channels]
            x_so3.embedding[:, 0, :] = x_so3.embedding[:, 0, :] + adapter_emb

        # ── Edge geometry ─────────────────────────────────────────────────────
        edge_distance, edge_distance_vec = self._compute_edge_geometry(pos, edge_index)

        # ── Run EquiformerV2 encoder ──────────────────────────────────────────
        x_out, _ = self.equiformer(
            x_so3, pos, edge_index, edge_distance, edge_distance_vec, batch
        )

        # ── Velocity: SO3 FFN → l=1 extraction (shepherd-style) ─────────────
        # Run a full equivariant FFN so higher-l geometry informs the output,
        # then read off the l=1 (vector) components as the position delta.
        vel_so3 = self.head_vel_ffn(x_out)              # SO3_Embedding, output_channels=1
        vel = vel_so3.embedding[:, 1:4, :].squeeze(-1)  # [N, 3]

        # ── Feature update: l=0 scalars → in_node_nf ─────────────────────────
        scalar_out = x_out.embedding[:, 0, :]   # [N, sphere_channels]
        h_final = self.head_h(scalar_out)        # [N, in_node_nf]

        # ── Remove centre-of-gravity drift from velocity ──────────────────────
        vel = remove_mean_pyG(vel, batch)

        if torch.any(torch.isnan(vel)):
            warnings.warn("EquiformerV2_dynamics detected nan, resetting to zero.", RuntimeWarning, stacklevel=2)
            vel = torch.zeros_like(vel)
            h_final = torch.zeros_like(h_final)

        return torch.cat([vel, h_final], dim=1)  # [N, 3 + in_node_nf]

    def _forward_pyG_impl(self, mol_graph: dict) -> torch.Tensor:
        """
        PyG-native forward pass.

        Args:
            mol_graph: dict with keys:
                'graph': PyG Batch — .pos [N,3], .x [N,h_cat_dim]|None,
                                     .atomic_numbers [N], .edge_index [2,E], .batch [N]
                't':     [N, 1] per-node timestep
                'context': [N, ctx_dim] or None

        Returns:
            [N, 3 + in_node_nf] tensor: concatenation of velocity and feature update
        """
        g = mol_graph["graph"]
        pos = g.pos                        # [N, 3]
        batch = g.batch                    # [N]
        edge_index = g.edge_index          # [2, E]
        atomic_numbers = g.atomic_numbers  # [N] or [N,1]
        extra_h = g.x                      # [N, h_cat_dim] or None

        if atomic_numbers.dim() == 1:
            atomic_numbers = atomic_numbers.unsqueeze(-1)
        atom_feat = atomic_numbers.float()  # [N, 1]

        if extra_h is None:
            h = atom_feat
        else:
            h = torch.cat([atom_feat, extra_h], dim=1)  # [N, in_node_nf]

        t = mol_graph["t"] if self.condition_time else None  # [N, 1]
        context = mol_graph.get("context", None)

        return self._forward_core(
            h, pos, edge_index, batch, t=t, context=context
        )

    def _forward_dense(self, t, xh, node_mask, edge_mask, context):
        """
        Dense-batch path for EnVariationalDiffusion (non-PyG).

        Args:
            t:          scalar or [B, 1] timestep
            xh:         [B, N, 3 + h_dims] positions + features (padded)
            node_mask:  [B, N, 1] binary mask
            edge_mask:  [B*N*N, 1] (unused — edges built from valid node pairs)
            context:    [B, N, ctx_dim] or None

        Returns:
            [B, N, 3 + h_dims] same layout as xh
        """
        B, N, dims = xh.shape
        device = xh.device

        node_mask_flat = node_mask.view(B * N, 1)  # [B*N, 1]

        # Unpack positions and features
        x_flat = xh[:, :, :self.n_dims].reshape(B * N, self.n_dims) * node_mask_flat
        h_flat_in = xh[:, :, self.n_dims:].reshape(B * N, dims - self.n_dims)

        # Build batch index
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N).reshape(B * N)

        # Valid (non-padded) nodes only
        mask_bool = node_mask_flat.squeeze(1).bool()  # [B*N]
        valid_nodes = torch.where(mask_bool)[0]
        node_batch = batch_idx[valid_nodes]

        # Build fully-connected intra-molecule edges for valid nodes, indexed
        # into the COMPACTED [0, n_valid) space -- pos/h/batch below are all
        # `[valid_nodes]`-gathered, not the original padded [0, B*N) space.
        src_list, tgt_list = [], []
        for mol_id in range(B):
            mol_nodes = torch.where(node_batch == mol_id)[0]
            if mol_nodes.numel() < 2:
                continue
            grid = torch.meshgrid(mol_nodes, mol_nodes, indexing='ij')
            s, tg = grid[0].reshape(-1), grid[1].reshape(-1)
            mask_self = s != tg
            src_list.append(s[mask_self])
            tgt_list.append(tg[mask_self])

        if src_list:
            src = torch.cat(src_list)
            tgt = torch.cat(tgt_list)
        else:
            src = tgt = torch.zeros(0, dtype=torch.long, device=device)
        edge_index = torch.stack([src, tgt], dim=0)

        # Flatten timestep to [B*N, 1]
        if torch.numel(t) == 1:
            t_flat = t.expand(B * N, 1)
        else:
            t_flat = t.view(B, 1).expand(B, N).reshape(B * N, 1)

        ctx_flat = context.reshape(B * N, -1) if context is not None else None

        # EnVariationalDiffusion's dense `h` (h_flat_in) is already the full
        # [n_valid, in_node_nf] feature block (categorical one-hot + optional
        # charge) -- there is no separate "atomic number" scalar to prepend
        # here, unlike the real-PyG-graph contract in _forward_pyG_impl.
        out_valid = self._forward_core(
            h_flat_in[valid_nodes],
            x_flat[valid_nodes],
            edge_index,
            batch_idx[valid_nodes],
            t=t_flat[valid_nodes],
            context=ctx_flat[valid_nodes] if ctx_flat is not None else None,
        )  # [n_valid, 3 + in_node_nf]

        # Scatter back to [B*N, 3 + in_node_nf]
        out_full = torch.zeros(B * N, 3 + self.in_node_nf, device=device, dtype=out_valid.dtype)
        out_full[valid_nodes] = out_valid

        return out_full.view(B, N, 3 + self.in_node_nf)

    def _forward(self, t_or_mol_graph, xh=None, node_mask=None, edge_mask=None, context=None):
        """
        Unified forward — dispatches to PyG or dense-batch path.

        PyG path:   _forward(mol_graph: dict)
        Dense path: _forward(t, xh, node_mask, edge_mask, context)
        """
        if isinstance(t_or_mol_graph, dict):
            return self._forward_pyG_impl(t_or_mol_graph)
        else:
            return self._forward_dense(t_or_mol_graph, xh, node_mask, edge_mask, context)

    def _forward_pyG(self, *args, **kwargs):
        """Alias — maintains API compatibility with EGNN_dynamics."""
        return self._forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)
