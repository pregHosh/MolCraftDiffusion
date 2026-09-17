import torch
from tensordict import TensorDict

from MolecularDiffusion.modules.models.tabasco.flow.interpolate import (
    CenteredMetricInterpolant,
)
from MolecularDiffusion.modules.models.tabasco.flow.path import FlowPath


def _batch(coords):
    padding_mask = torch.tensor([[False, False, True]])
    return TensorDict(
        {
            "coords": torch.tensor([coords], dtype=torch.float32),
            "padding_mask": padding_mask,
        },
        batch_size=1,
    )


def _path(batch):
    return FlowPath(
        x_0=batch,
        x_t=batch,
        dx_t=batch,
        x_1=batch,
        t=torch.tensor([0.5]),
    )


def test_centered_coordinate_loss_is_translation_invariant():
    interpolant = CenteredMetricInterpolant(key="coords", centered=True)
    centered = _batch([[-1, 0, 0], [1, 0, 0], [0, 0, 0]])
    translated = _batch([[9, 20, 30], [11, 20, 30], [0, 0, 0]])

    prediction = centered.clone()
    centered_loss, _ = interpolant.compute_loss(_path(centered), prediction, False)
    translated_loss, _ = interpolant.compute_loss(
        _path(translated), prediction, False
    )

    torch.testing.assert_close(centered_loss, torch.tensor(0.0))
    torch.testing.assert_close(translated_loss, centered_loss)


def test_uncentered_coordinate_loss_preserves_translation():
    interpolant = CenteredMetricInterpolant(key="coords", centered=False)
    centered = _batch([[-1, 0, 0], [1, 0, 0], [0, 0, 0]])
    translated = _batch([[9, 20, 30], [11, 20, 30], [0, 0, 0]])

    prediction = centered.clone()
    translated_loss, _ = interpolant.compute_loss(
        _path(translated), prediction, False
    )

    torch.testing.assert_close(
        translated_loss, torch.tensor((10**2 + 20**2 + 30**2) / 3)
    )


def test_create_path_respects_centered_flag():
    batch = _batch([[9, 20, 30], [11, 20, 30], [0, 0, 0]])
    noise = _batch([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    t = torch.tensor([1.0])

    centered = CenteredMetricInterpolant(key="coords", centered=True)
    uncentered = CenteredMetricInterpolant(key="coords", centered=False)
    _, centered_xt, _ = centered.create_path(batch, t, noise)
    _, uncentered_xt, _ = uncentered.create_path(batch, t, noise)

    torch.testing.assert_close(
        centered_xt, torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0], [0, 0, 0]]])
    )
    torch.testing.assert_close(uncentered_xt, batch["coords"])
