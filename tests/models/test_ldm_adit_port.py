"""ADiT port from the dev tree: import, molecule cell, pos_scale, task_type.

CPU only, toy batches, no data files.
"""

from typing import Any

import pytest
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from MolecularDiffusion.cli.generate import _validate_task_type
from MolecularDiffusion.modules.models.ldm.encoders.equiformer import (
    EquiformerEncoder,
)
from MolecularDiffusion.modules.tasks.diffusion_ldm import (
    LDMTask,
    LDMTaskFactory,
    VAETask,
    VAETaskFactory,
)

ELEMENTS = torch.tensor([1, 6, 6, 7, 8])


def _dense_batch(pos: Tensor) -> dict[str, Tensor]:
    return {
        "coords": pos[None],
        "charges": ELEMENTS[None],
        "node_mask": torch.ones(1, len(ELEMENTS), dtype=torch.bool),
        "natoms": torch.tensor([len(ELEMENTS)]),
    }


def _tiny_equiformer() -> EquiformerEncoder:
    torch.manual_seed(0)
    return EquiformerEncoder(
        use_pbc=False, max_neighbors=20, max_radius=6.0, max_num_elements=10,
        num_layers=1, sphere_channels=8, attn_hidden_channels=8, num_heads=2,
        attn_alpha_channels=8, attn_value_channels=4, ffn_hidden_channels=8,
        lmax_list=[1], mmax_list=[1], num_distance_basis=16, edge_channels=8,
        alpha_drop=0.0, drop_path_rate=0.0,
    ).eval()


class _FixedDecoder(nn.Module):
    """Returns whatever positions the test puts in ``self.pos``."""

    d_model = 4

    def __init__(self, pos: Tensor) -> None:
        super().__init__()
        self.pos = pos

    def forward(self, inputs: dict[str, Any]) -> dict[str, Tensor]:
        n = inputs["x"].shape[0]
        return {"pos": self.pos[:n], "atom_types": torch.zeros(n, 10)}


class _PosEncoder(nn.Module):
    d_model = 4

    def forward(self, batch: Any) -> dict[str, Tensor]:
        ones = torch.ones(batch.pos.shape[0], 1)
        return {
            "x": torch.cat([batch.pos, ones], dim=-1),
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }


def test_equiformer_encoder_imports() -> None:
    assert EquiformerEncoder.__name__ == "EquiformerEncoder"


def test_molecules_get_zero_cell_on_both_batch_paths() -> None:
    task = VAETask(encoder=_PosEncoder(), decoder=_FixedDecoder(None))
    dense = task._adapt_batch(_dense_batch(torch.randn(5, 3)))
    assert dense.cell.shape == (1, 3, 3)
    assert float(dense.cell.abs().sum()) == 0.0

    data = Data(
        pos=torch.randn(5, 3), atomic_numbers=ELEMENTS,
        natoms=len(ELEMENTS), token_idx=torch.arange(5),
    )
    pyg = task._adapt_batch(Batch.from_data_list([data, data.clone()]))
    assert pyg.cell.shape == (2, 3, 3)
    assert float(pyg.cell.abs().sum()) == 0.0


def test_equiformer_invariant_channel_is_rotation_invariant() -> None:
    # A fixed non-zero molecule cell leaks absolute orientation into the
    # encoder's edge features (a 100*I cell gives a ~0.9 l=0 shift here).
    encoder = _tiny_equiformer()
    task = VAETask(encoder=encoder, decoder=_FixedDecoder(None))
    torch.manual_seed(1)
    pos = torch.randn(5, 3) * 1.2
    rot = torch.tensor(
        Rotation.random(random_state=3).as_matrix(), dtype=torch.float32
    )

    def l0(p: Tensor) -> Tensor:
        with torch.no_grad():
            out = encoder(task._adapt_batch(_dense_batch(p)))["x"]
        return out.view(5, 4, 8)[:, 0]  # (lmax+1)^2 = 4 coeffs, l=0 first

    ref, rotated = l0(pos), l0(pos @ rot.T)
    assert torch.isfinite(ref).all()
    # The S2 grid activation is only approximately equivariant (~1e-3).
    assert (ref - rotated).abs().max() < 1e-2 * ref.abs().max()


def _vae(pos_scale: float | None, decoded: Tensor) -> VAETask:
    kwargs = {} if pos_scale is None else {"pos_scale": pos_scale}
    return VAETask(
        encoder=_PosEncoder(), decoder=_FixedDecoder(decoded), latent_dim=2,
        loss_weights={"pos": 1.0, "atom_types": 0.0, "kl": 0.0}, **kwargs,
    ).eval()


class _Recorder:
    def __init__(self) -> None:
        self.pred: list[dict] = []

    def append_pred_array(self, arr: dict) -> None:
        self.pred.append(arr)

    def append_gt_array(self, arr: dict) -> None:
        pass


def test_pos_scale_divides_target_and_rescales_outputs() -> None:
    pos = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10]]
    )
    # Decoder emitting the target in the 1/10 unit: zero loss only at 10.
    _, stats = _vae(10.0, pos / 10)(_dense_batch(pos))
    assert stats["pos_loss"] == pytest.approx(0.0, abs=1e-10)
    _, stats = _vae(None, pos / 10)(_dense_batch(pos))
    assert stats["pos_loss"] > 1.0  # default 1.0: target is Angstrom

    for scale, factor in ((10.0, 10.0), (None, 1.0)):
        vae = _vae(scale, pos / 10)
        vae.reconstruction_evaluator = _Recorder()
        batch = vae._adapt_batch(_dense_batch(pos))
        _, encoded = vae.encode(batch)
        recon = vae.decode(torch.zeros(5, 2), encoded)
        vae._collect_reconstruction_samples(batch, recon, encoded)
        got = torch.as_tensor(vae.reconstruction_evaluator.pred[0]["pos"])
        want = pos / 10 * factor
        torch.testing.assert_close(got, want - want.mean(0))


class _ZeroDenoiser(nn.Module):
    def forward(self, x, t, mask, x_sc=None):  # noqa: ANN001, ANN201
        return torch.zeros_like(x)


@pytest.mark.parametrize("scale", [10.0, None, "missing"])
def test_ldm_sample_rescales_decoded_positions(scale) -> None:  # noqa: ANN001
    decoded = torch.arange(15.0).view(5, 3)
    vae = _vae(None if scale == "missing" else scale, decoded)
    if scale == "missing":  # VAE object pickled before pos_scale existed
        del vae.pos_scale
    task = LDMTask(vae, _ZeroDenoiser(), {"num_timesteps": 2})
    _, _, pos, _ = task.sample(nodesxsample=torch.tensor([5]))
    factor = 10.0 if scale == 10.0 else 1.0
    torch.testing.assert_close(pos[0], decoded * factor)


def test_task_type_is_configured_name(tmp_path) -> None:  # noqa: ANN001
    tiny = {"max_num_elements": 10, "d_model": 8, "nhead": 2,
            "dim_feedforward": 8, "num_layers": 1}
    vae_config = {
        "_target_": "MolecularDiffusion.modules.tasks.diffusion_ldm"
                    ".VAETaskFactory",
        "task_type": "vae_transformer",
        "encoder": {"_target_": "MolecularDiffusion.modules.models.ldm"
                                ".encoders.transformer.TransformerEncoder",
                    **tiny},
        "decoder": {"_target_": "MolecularDiffusion.modules.models.ldm"
                                ".decoders.transformer.TransformerDecoder",
                    **tiny},
        "latent_dim": 2,
        "pos_scale": 10.0,
    }
    vae = VAETaskFactory(**vae_config).build()
    assert vae.task_type == "vae_transformer"
    assert VAETask(encoder=_PosEncoder(), decoder=_FixedDecoder(None)) \
        .task_type == "vae"  # direct construction keeps the legacy name

    ckpt = tmp_path / "vae.ckpt"
    torch.save({
        "state_dict": {f"task.{k}": v for k, v in vae.state_dict().items()},
        "hyper_parameters": {"model_config": vae_config},
    }, ckpt)
    ldm = LDMTaskFactory(
        task_type="diffusion_adit",
        autoencoder_ckpt=str(ckpt),
        denoiser={"_target_": "MolecularDiffusion.modules.models.ldm"
                              ".denoisers.dit.DiT",
                  "d_model": 8, "num_layers": 1, "nhead": 2},
        interpolant={"num_timesteps": 2},
        # D-only keys: accepted and ignored by this tree's factory.
        modality_conditioning=True,
        cfg_scale=2.0,
    ).build()
    assert ldm.task_type == "diffusion_adit"
    assert ldm.autoencoder.pos_scale == 10.0
    for key, value in vae.state_dict().items():
        torch.testing.assert_close(ldm.autoencoder.state_dict()[key], value)


def test_generate_accepts_legacy_ldm_vae_stamps() -> None:
    for legacy in ("ldm", "vae"):
        _validate_task_type({"task_type": legacy}, "diffusion_adit")
    _validate_task_type({"task_type": "diffusion_adit"}, "diffusion_adit")
    with pytest.raises(ValueError, match="Task type mismatch"):
        _validate_task_type({"task_type": "diffusion"}, "diffusion_adit")

