from typing import Any

import torch
from torch import Tensor, nn

from MolecularDiffusion.modules.tasks.diffusion_ldm import VAETask


class _RecordingEncoder(nn.Module):
    d_model = 4

    def __init__(self) -> None:
        super().__init__()
        self.last_pos: Tensor | None = None

    def forward(self, batch: Any) -> dict[str, Tensor]:
        self.last_pos = batch.pos.detach().clone()
        ones = torch.ones(batch.pos.shape[0], 1, device=batch.pos.device)
        return {
            "x": torch.cat([batch.pos, ones], dim=-1),
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }


class _UnusedDecoder(nn.Module):
    d_model = 4


def _pointcloud_batch(translation: list[float]) -> dict[str, Tensor]:
    coords = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    node_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    coords = (
        coords
        + torch.tensor(translation).view(1, 1, 3) * node_mask.unsqueeze(-1)
    )
    return {
        "coords": coords,
        "charges": torch.tensor([[6, 8, 0], [6, 6, 8]]),
        "node_mask": node_mask,
        "natoms": node_mask.sum(dim=1),
    }


def test_vae_encoder_centers_each_molecule_without_mutating_input() -> None:
    encoder = _RecordingEncoder()
    task = VAETask(encoder=encoder, decoder=_UnusedDecoder(), latent_dim=2)
    centered = _pointcloud_batch([0.0, 0.0, 0.0])
    translated = _pointcloud_batch([10.0, 20.0, 30.0])
    translated_before = translated["coords"].clone()

    posterior_centered, _ = task.encode(centered)
    assert encoder.last_pos is not None
    encoded_centered_pos = encoder.last_pos.clone()
    posterior_translated, _ = task.encode(translated)

    assert encoder.last_pos is not None
    torch.testing.assert_close(encoder.last_pos, encoded_centered_pos)
    torch.testing.assert_close(
        posterior_translated.mean, posterior_centered.mean
    )
    torch.testing.assert_close(
        posterior_translated.logvar, posterior_centered.logvar
    )
    torch.testing.assert_close(translated["coords"], translated_before)

    batch_index = torch.tensor([0, 0, 1, 1, 1])
    molecule_means = torch.stack(
        [encoder.last_pos[batch_index == i].mean(dim=0) for i in range(2)]
    )
    torch.testing.assert_close(
        molecule_means, torch.zeros_like(molecule_means)
    )
