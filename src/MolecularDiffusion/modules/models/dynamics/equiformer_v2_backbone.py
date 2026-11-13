"""
EquiformerV2Backbone — plain per-node scalar-feature backbone for property
prediction. Unlike EquiformerV2_dynamics (built for denoising: mandatory
timestep, velocity + feature-delta output), this exposes per-node l=0
features via forward(data) -> {"x": ...}, mirroring eSEN_Backbone's contract
so it plugs into ProperyPrediction.

The EquiformerV2 encoder and all layer files are NOT modified.
"""
import torch
import torch.nn as nn

from MolecularDiffusion.modules.layers.equiformer_v2_s.so3 import SO3_Embedding


class EquiformerV2Backbone(nn.Module):
    """
    Args:
        equiformer: Instantiated EquiformerV2 model (from shepherd_arch).
        in_node_channels (int): Width of the per-node input feature, i.e.
                                 atomic_number(1) + data.x width.
        sphere_channels (int): Must match equiformer.sphere_channels.
        lmax_list (list[int]): Must match equiformer.lmax_list.
    """

    def __init__(
        self,
        equiformer: nn.Module,
        in_node_channels: int,
        sphere_channels: int = 128,
        lmax_list=None,
    ):
        super().__init__()
        self.equiformer = equiformer
        self.sphere_channels = sphere_channels
        self.lmax_list = lmax_list or [6]
        # hidden dim exposed for the task wrapper, mirrors eSEN_Backbone.d_model
        self.d_model = sphere_channels
        self.input_proj = nn.Linear(in_node_channels, sphere_channels)

    def forward(self, data) -> dict[str, torch.Tensor]:
        pos = data.pos
        batch = data.batch
        edge_index = data.edge_index

        atomic_numbers = data.atomic_numbers
        if atomic_numbers.dim() == 1:
            atomic_numbers = atomic_numbers.unsqueeze(-1)
        atom_feat = atomic_numbers.float()
        h = torch.cat([atom_feat, data.x], dim=-1) if data.x is not None else atom_feat

        x_so3 = SO3_Embedding(
            h.size(0), self.lmax_list, self.sphere_channels, h.device, h.dtype
        )
        x_so3.embedding[:, 0, :] = self.input_proj(h)

        src, tgt = edge_index[0], edge_index[1]
        edge_distance_vec = pos[tgt] - pos[src]
        edge_distance = edge_distance_vec.norm(dim=-1)

        x_out, _ = self.equiformer(
            x_so3, pos, edge_index, edge_distance, edge_distance_vec, batch
        )
        return {"x": x_out.embedding[:, 0, :]}
