"""The VAE named by `tasks.autoencoder_ckpt` must survive any state_dict load.

An LDM checkpoint stores `autoencoder.*` beside `denoiser.*`. Those tensors have the same
names and shapes as the VAE the task just built and froze from the config, so loading such a
checkpoint as a warm start (`trainer.load_weights_from`) silently replaced the configured
VAE -- no error, no shape mismatch -- and the denoiser then trained through the wrong latent
space for 100k steps.
"""
import torch
from torch import nn

from MolecularDiffusion.modules.tasks.diffusion_ldm import LDMTask


class _Stub(nn.Module):
    """Stands in for VAETask / the denoiser: one weight we can fingerprint."""

    def __init__(self, fill):
        super().__init__()
        self.w = nn.Parameter(torch.full((2, 2), float(fill)))


def _task(vae_fill, denoiser_fill, **kw):
    return LDMTask(
        autoencoder=_Stub(vae_fill),
        denoiser=_Stub(denoiser_fill),
        interpolant_config={},
        autoencoder_ckpt="outputs/adit_vae_ft/formed.ckpt",
        **kw,
    )


def _warm_start(vae_fill, denoiser_fill):
    """A stage-1 checkpoint: a stale VAE next to the denoiser we do want."""
    return _task(vae_fill, denoiser_fill).state_dict()


def test_stale_vae_in_a_warm_start_is_ignored():
    task = _task(vae_fill=1.0, denoiser_fill=0.0)          # 1.0 = the configured (FORMED) VAE
    task.load_state_dict(_warm_start(9.0, 7.0), strict=False)   # 9.0 = the stale (GEOM) VAE

    assert torch.allclose(task.autoencoder.w, torch.full((2, 2), 1.0)), (
        "the checkpoint's VAE replaced the configured one"
    )
    assert torch.allclose(task.denoiser.w, torch.full((2, 2), 7.0)), (
        "the denoiser was not warm-started"
    )


def test_override_flag_restores_the_old_behaviour():
    task = _task(vae_fill=1.0, denoiser_fill=0.0, allow_autoencoder_override=True)
    task.load_state_dict(_warm_start(9.0, 7.0), strict=False)

    assert torch.allclose(task.autoencoder.w, torch.full((2, 2), 9.0))
    assert torch.allclose(task.denoiser.w, torch.full((2, 2), 7.0))


def test_matching_vae_loads_unchanged():
    """The common case: checkpoint and config hold the same VAE. Nothing to protect."""
    task = _task(vae_fill=1.0, denoiser_fill=0.0)
    task.load_state_dict(_warm_start(1.0, 7.0), strict=False)

    assert torch.allclose(task.autoencoder.w, torch.full((2, 2), 1.0))
    assert torch.allclose(task.denoiser.w, torch.full((2, 2), 7.0))


def test_task_without_a_configured_vae_is_untouched():
    """No autoencoder_ckpt (e.g. a task built some other way) -> no interference."""
    task = LDMTask(
        autoencoder=_Stub(1.0), denoiser=_Stub(0.0), interpolant_config={},
    )
    task.load_state_dict(_warm_start(9.0, 7.0), strict=False)

    assert torch.allclose(task.autoencoder.w, torch.full((2, 2), 9.0))
    assert torch.allclose(task.denoiser.w, torch.full((2, 2), 7.0))
