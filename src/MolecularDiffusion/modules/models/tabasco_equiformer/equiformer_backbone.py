"""EquiformerV2 backbone for TABASCO's flow-matching model.

Wraps ``MolecularDiffusion.modules.models.shepherd_arch.equiformer_v2_encoder.
EquiformerV2`` (the SO(3) tensor-product transformer encoder, ``equiformer_v2_s``
layer family -- the same encoder class ``EquiformerV2_dynamics``,
``EquiformerV2Backbone``, and ``shepherd_arch`` already use, imported here
unmodified) so it can be substituted 1-for-1 for TABASCO's
``TransformerModule`` (``modules/layers/tabasco/transformer_module.py``) with
no change to ``FlowMatchingModel``, the interpolants, the Euler-Maruyama
sampler, or the pointcloud<->TensorDict adapters.

Backbone contract (fixed by ``FlowMatchingModel._call_net``,
``modules/models/tabasco/flow_model.py:84-97``)::

    forward(coords, atomics, padding_mask, t) -> (coords, atom_logits)

``coords`` returned is the direct endpoint prediction ``x_1`` -- both
``CenteredMetricInterpolant.compute_loss`` and ``SDEMetricInterpolant.step``
(``modules/models/tabasco/flow/interpolate.py``) treat ``pred["coords"]`` as
an x_1 estimate, not a velocity/residual.

Mask convention: ``padding_mask`` follows TABASCO's own inverted convention
(``1 = padded``, see ``PointCloudToTensorDictAdapter``,
``modules/tasks/diffusion_tabasco.py:60-113``) -- the opposite of the
platform's usual ``node_mask`` (``1 = real``). This class inverts once at the
boundary, the same idiom ``EGNNBackbone``/``GVPBackbone`` already use.

Six-step mechanism (see docs/model_novel/tabasco_equiformer/INTEGRATION_PLAN.md,
"Hypothesis" / "Integration Plan"):

1. Compact the padded ``(B, N, ...)`` batch to its real (unmasked) atoms and
   build a per-molecule fully-connected, no-self-loop edge index -- reusing
   the exact valid-node-compaction and edge-construction pattern
   ``EquiformerV2_dynamics._forward_dense`` implements for a dense pointcloud
   batch (``modules/models/equiformer_v2_dynamics.py:266-343``). Reproduced
   locally (a two-line body, boilerplate index bookkeeping, not a
   layer/schedule) rather than importing ``EquiformerV2_dynamics`` itself,
   which this backbone otherwise never touches.
2. Project one-hot atom type + scalar time ``t`` into the l=0 (scalar)
   channel of a fresh ``SO3_Embedding``, following ``_build_so3_input``'s
   pattern (``equiformer_v2_dynamics.py:118-142``) -- omitting the
   context/adapter machinery ``EquiformerV2_dynamics`` carries for EDM
   conditioning, since TABASCO has no conditioning context.
3. Run the wrapped ``EquiformerV2`` encoder over the compacted graph.
4. Read a per-atom displacement ``d`` off the l=1 channel via a
   ``FeedForwardNetwork`` SO3 head (same construction as
   ``EquiformerV2_dynamics.head_vel_ffn``, including its own ``SO3_Grid``).
   **Deliberate, load-bearing departure from ``EquiformerV2_dynamics``'s own
   convention:** returns ``coords_in + d`` as the endpoint prediction, not
   ``d`` alone, and does **not** call ``remove_mean_pyG``. ``d`` is built
   purely from ``edge_distance_vec`` (translation-invariant relative
   geometry), so on its own it carries no absolute position information -- it
   is a displacement, not an endpoint. ``EquiformerV2_dynamics`` returns it
   bare because EDM's dynamics contract wants a velocity/noise prediction
   added externally by the caller; TABASCO's contract instead wants the
   endpoint directly. The ground-truth ``x_1`` target is already COM-centered
   by ``mask_and_zero_com`` before the MSE loss is computed, so -- exactly as
   neither ``EGNNBackbone`` nor ``GVPBackbone`` needed a forced centering
   step -- the loss alone is sufficient to teach a centered output.
5. Read the l=0 scalar channel through a fresh ``nn.Linear(sphere_channels,
   atom_dim)`` into atom-type logits -- NOT ``EquiformerV2_dynamics.head_h``,
   which is sized for EDM's ``in_node_nf`` feature convention.
6. Scatter both outputs back into the padded ``(B, N, 3)``/``(B, N,
   atom_dim)`` shape, mirroring ``_forward_dense``'s own scatter-back step.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from MolecularDiffusion.modules.layers.equiformer_v2_s.module_list import (
    ModuleListInfo,
)
from MolecularDiffusion.modules.layers.equiformer_v2_s.so3 import (
    SO3_Embedding,
    SO3_Grid,
)
from MolecularDiffusion.modules.layers.equiformer_v2_s.transformer_block import (
    FeedForwardNetwork,
)
from MolecularDiffusion.modules.models.shepherd_arch.equiformer_v2_encoder import (
    EquiformerV2,
)


class EquiformerV2TabascoBackbone(nn.Module):
    """TABASCO-compatible net: ``(coords, atomics, padding_mask, t) -> (coords, atom_logits)``.

    Architecture kwargs below default to the exact values
    ``configs/tasks/diffusion_equiformer.yaml`` uses (this platform's own
    proven QM9-scale EquiformerV2 config) -- see the ledger's "Hyperparameter
    Provenance" table.
    """

    def __init__(
        self,
        atom_dim: int,
        sphere_channels: int = 128,
        input_sphere_channels: int = 128,
        num_layers: int = 8,
        lmax_list: Optional[List[int]] = None,
        mmax_list: Optional[List[int]] = None,
        grid_resolution: int = 18,
        num_sphere_samples: int = 128,
        attn_hidden_channels: int = 64,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 128,
        num_heads: int = 8,
        norm_type: str = "layer_norm_sh",
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = True,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        proj_drop: float = 0.0,
        cutoff: float = 9.0,
        weight_init: str = "uniform",
    ):
        super().__init__()
        if lmax_list is None:
            lmax_list = [2]
        if mmax_list is None:
            mmax_list = [1]

        self.atom_dim = atom_dim
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

        # Step 3's encoder. A fresh, analogous helper mirroring
        # `tasks_equiformer.py._build_equiformer`'s kwarg-forwarding pattern
        # (reused pattern, not reused code -- importing `_build_equiformer`
        # itself would mean instantiating an unrelated `ModelTaskFactory`).
        self.equiformer = self._build_equiformer(
            num_layers=num_layers,
            input_sphere_channels=input_sphere_channels,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            norm_type=norm_type,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=grid_resolution,
            num_sphere_samples=num_sphere_samples,
            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,
            attn_activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            cutoff=cutoff,
            weight_init=weight_init,
        )

        # Step 2: project one-hot atom type (atom_dim) + scalar time (1) into
        # the l=0 channel -- no context/adapter machinery (TABASCO has none).
        self.input_proj = nn.Linear(atom_dim + 1, sphere_channels)

        # SO3 grid for the equivariant FFN head's S2 activation, same
        # construction as `EquiformerV2_dynamics.vel_SO3_grid`
        # (equiformer_v2_dynamics.py:83-90).
        lmax = max(lmax_list)
        self.so3_grid = ModuleListInfo("({}, {})".format(lmax, lmax))
        for l in range(lmax + 1):
            m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                m_grid.append(
                    SO3_Grid(l, m, resolution=18, normalization="component")
                )
            self.so3_grid.append(m_grid)

        # Step 4: displacement head -- SO3 FFN reading the l=1 (vector)
        # channel. output_channels=1 so the l=1 slice has shape [N, 3, 1].
        self.head_disp_ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=sphere_channels,
            output_channels=1,
            lmax_list=lmax_list,
            mmax_list=lmax_list,
            SO3_grid=self.so3_grid,
            activation="silu",
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,
        )

        # Step 5: fresh atom-type logit head off the l=0 channel -- sized for
        # this backbone's atom_dim, not EquiformerV2_dynamics.head_h's
        # in_node_nf.
        self.head_atom = nn.Linear(sphere_channels, atom_dim)

    @staticmethod
    def _build_equiformer(**kwargs) -> EquiformerV2:
        return EquiformerV2(**kwargs)

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
        dtype = coords.dtype

        # Step 1: invert to platform convention (1 = real) and compact to
        # valid nodes -- following `EquiformerV2_dynamics._forward_dense`
        # (equiformer_v2_dynamics.py:266-343).
        node_mask_flat = (~padding_mask.bool()).reshape(batch_size * n_nodes)

        coords_flat = coords.reshape(batch_size * n_nodes, 3)
        atomics_flat = atomics.reshape(batch_size * n_nodes, self.atom_dim)

        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, n_nodes)
            .reshape(-1)
        )

        valid_nodes = torch.where(node_mask_flat)[0]
        node_batch = batch_idx[valid_nodes]
        pos = coords_flat[valid_nodes]
        atomics_valid = atomics_flat[valid_nodes]

        # Fully-connected, no-self-loop edges within each molecule, indexed
        # into the COMPACTED [0, n_valid) space.
        src_list, tgt_list = [], []
        for mol_id in range(batch_size):
            mol_nodes = torch.where(node_batch == mol_id)[0]
            if mol_nodes.numel() < 2:
                continue
            grid = torch.meshgrid(mol_nodes, mol_nodes, indexing="ij")
            s, tg = grid[0].reshape(-1), grid[1].reshape(-1)
            keep = s != tg
            src_list.append(s[keep])
            tgt_list.append(tg[keep])

        if src_list:
            src = torch.cat(src_list)
            tgt = torch.cat(tgt_list)
        else:
            src = tgt = torch.zeros(0, dtype=torch.long, device=device)
        edge_index = torch.stack([src, tgt], dim=0)

        # Step 2: project atom-type one-hot + scalar time into l=0.
        # Device/dtype move must happen BEFORE indexing with `valid_nodes`
        # (which is CUDA-resident) -- `t` itself can arrive on CPU (a known
        # pre-existing platform quirk in TABASCO's shared sampling loop).
        # Mirrors GVPBackbone.forward's working order
        # (gvp_backbone.py:260): `t.to(device=device, dtype=dtype)[node_batch_idx]`.
        t_flat = (
            t.to(device=device, dtype=dtype)
            .reshape(batch_size, 1)
            .expand(batch_size, n_nodes)
            .reshape(-1)[valid_nodes]
            .unsqueeze(-1)
        )
        h = torch.cat([atomics_valid, t_flat], dim=-1)

        n_valid = h.size(0)
        x_so3 = SO3_Embedding(
            n_valid, self.lmax_list, self.sphere_channels, device, dtype
        )
        x_so3.embedding[:, 0, :] = self.input_proj(h)

        # Edge geometry -- translation-invariant relative-geometry
        # quantities (equiformer_v2_dynamics.py:_compute_edge_geometry).
        edge_distance_vec = pos[tgt] - pos[src]
        edge_distance = edge_distance_vec.norm(dim=-1)

        # Step 3: run the wrapped EquiformerV2 encoder.
        x_out, _ = self.equiformer(
            x_so3, pos, edge_index, edge_distance, edge_distance_vec, node_batch
        )

        # Step 4: endpoint = coords_in + displacement read off the l=1
        # channel. No `remove_mean_pyG` -- see module docstring, point 4.
        disp_so3 = self.head_disp_ffn(x_out)
        d = disp_so3.embedding[:, 1:4, :].squeeze(-1)  # [n_valid, 3]
        coords_valid = pos + d

        # Step 5: atom-type logits off the l=0 channel.
        atom_logits_valid = self.head_atom(x_out.embedding[:, 0, :])

        # Step 6: scatter back to the padded (B, N, ...) shape.
        coords_out = torch.zeros(
            batch_size * n_nodes, 3, device=device, dtype=coords_valid.dtype
        )
        coords_out[valid_nodes] = coords_valid
        atom_logits_out = torch.zeros(
            batch_size * n_nodes,
            self.atom_dim,
            device=device,
            dtype=atom_logits_valid.dtype,
        )
        atom_logits_out[valid_nodes] = atom_logits_valid

        return (
            coords_out.reshape(batch_size, n_nodes, 3),
            atom_logits_out.reshape(batch_size, n_nodes, self.atom_dim),
        )
