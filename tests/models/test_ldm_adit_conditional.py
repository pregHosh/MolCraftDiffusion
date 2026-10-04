"""ADiT property conditioning + CFG, and the sibling cfg_scale_schedule /
unconditional-null fixes (TABASCO, FlowMol). CPU only, toy tensors."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from MolecularDiffusion.modules.models.ldm.denoisers.dit import DiT
from MolecularDiffusion.modules.tasks.diffusion_ldm import (
    LDMTask,
    LDMTaskFactory,
)

from .test_ldm_adit_port import ELEMENTS, _vae


def _dit(d_context: int = 0) -> DiT:
    torch.manual_seed(0)
    dit = DiT(d_x=2, d_model=8, num_layers=1, nhead=2, d_context=d_context)
    # adaLN-Zero makes a fresh DiT output 0: randomise so outputs are informative.
    g = torch.Generator().manual_seed(1)
    for p in dit.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.3
    return dit


def _task(cond=("gap",), rate=0.0, mask_value=5.0, norm="value_10"):
    vae = _vae(None, torch.arange(30.0).view(10, 3))
    task = LDMTask(
        vae,
        _dit(len(cond)),
        {
            "num_timesteps": 4,
            "self_condition": True,
            "self_condition_prob": 1.0,
        },
        condition_names=list(cond),
        context_mask_rate=rate,
        mask_value=mask_value,
        normalize_condition=norm,
    )
    return task.eval()


def _batch(gap):
    pos = torch.randn(2, 5, 3)
    return {
        "coords": pos,
        "charges": ELEMENTS[None].repeat(2, 1),
        "node_mask": torch.ones(2, 5, dtype=torch.bool),
        "natoms": torch.tensor([5, 5]),
        "gap": torch.tensor(gap),
    }


def test_absent_key_is_inert() -> None:
    """(1) condition_names=[]: no new tensors, identical outputs to a plain DiT."""
    plain = _dit(0)
    assert not any("context" in k for k in plain.state_dict())
    task = _task(cond=())
    assert task.condition == [] and task.property_norms is None
    assert set(task.state_dict()) == {
        *(f"autoencoder.{k}" for k in task.autoencoder.state_dict()),
        *(f"denoiser.{k}" for k in plain.state_dict()),
    }
    # d_context DiT loaded from an unconditional state_dict is the SAME function
    # (zero-init context MLP output layer) whatever the context is.
    cond = DiT(d_x=2, d_model=8, num_layers=1, nhead=2, d_context=1)
    missing, unexpected = cond.load_state_dict(
        plain.state_dict(), strict=False
    )
    assert not unexpected and all("context_embedder" in k for k in missing)
    nn.init.constant_(cond.context_embedder[-1].weight, 0)
    x, t, m = (
        torch.randn(2, 5, 2),
        torch.rand(2),
        torch.ones(2, 5, dtype=torch.bool),
    )
    ref = plain.eval()(x, t, m)
    out = cond.eval()(x, t, m, context=torch.tensor([[3.0], [-7.0]]))
    assert (ref - out).abs().max() == 0
    # Unconditional task: forward/sample never pass `context` (old denoisers keep working).
    torch.manual_seed(3)
    loss, _ = task(_batch([1.0, 2.0]))
    assert torch.isfinite(loss)
    task.sample(nodesxsample=torch.tensor([5]))


def test_conditional_forward_depends_on_condition() -> None:
    """(2)"""
    task = _task()
    losses = []
    for gap in ([1.0, 2.0], [9.0, -4.0]):
        torch.manual_seed(5)
        batch = _batch(gap)
        torch.manual_seed(5)
        losses.append(task(batch)[0])
    assert (losses[0] - losses[1]).abs() > 1e-6


def test_dropout_rate_extremes() -> None:
    """(4)"""
    batch = _batch([1.0, 2.0])
    dev = torch.device("cpu")
    all_null = _task(rate=1.0)._train_context(batch, 2, dev)
    assert (all_null == 5.0).all()
    none = _task(rate=0.0)._train_context(batch, 2, dev)
    torch.testing.assert_close(none, torch.tensor([[0.1], [0.2]]))


def _spy(task):
    seen = []
    task.denoiser.register_forward_pre_hook(
        lambda mod, a, kw: seen.append(kw["context"].clone()), with_kwargs=True
    )
    return seen


def test_cfg_reduces_to_cond_and_null() -> None:
    """(3) w=0 == conditional only; target == null branch == unconditional."""
    task = _task()
    it, model = task.interpolant, task.denoiser
    it.device = "cpu"
    x0 = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    ctx = torch.tensor([[0.4], [0.4]])
    null = torch.full((2, 1), 5.0)

    def run(**kw):
        return it.sample(
            2, 5, 2, model, mask=mask, num_timesteps=4, x_0=x0, **kw
        )["clean_traj"][-1]

    cond_only = run(context=ctx)
    torch.testing.assert_close(
        run(context=ctx, negative_context=null, cfg_scale=0.0), cond_only
    )
    # uncond branch == cond branch -> any w gives the conditional result
    torch.testing.assert_close(
        run(context=ctx, negative_context=ctx, cfg_scale=3.0),
        cond_only,
        atol=1e-4,  # doubled batch: float noise only
        rtol=1e-3,
    )
    # a real guidance term changes it; schedule ramps it (t=min_t start ~ 0)
    guided = run(context=ctx, negative_context=null, cfg_scale=3.0)
    assert (guided - cond_only).abs().max() > 1e-5
    ramped = run(
        context=ctx,
        negative_context=null,
        cfg_scale=3.0,
        cfg_scale_schedule="linear",
    )
    assert (ramped - guided).abs().max() > 1e-6
    # unconditional sample() on a conditional task feeds the null (mask_value)
    seen = _spy(task)
    task.sample(nodesxsample=torch.tensor([5, 5]))
    assert seen and all((c == 5.0).all() and c.shape == (2, 1) for c in seen)
    # negative_target_value replaces the null branch; CFG entry point runs
    seen.clear()
    task.sample_guidance_conitional(
        target_value=[4.0],
        negative_target_value=[8.0],
        nodesxsample=torch.tensor([5, 5]),
        cfg_scale=2.0,
    )
    assert seen[0].shape == (4, 1)
    torch.testing.assert_close(
        seen[0][:, 0], torch.tensor([0.4, 0.4, 0.8, 0.8])
    )


def test_factory_rejects_adapter_and_logs(capsys) -> None:
    kw = dict(
        task_type="diffusion_adit",
        autoencoder_ckpt="x",
        denoiser={},
        interpolant={},
    )
    with pytest.raises(ValueError, match="adapter_conditions"):
        LDMTaskFactory(**kw, adapter_conditions=["gap"])
    LDMTaskFactory(**kw)
    assert "condition_names=[]" in capsys.readouterr().out


def test_cfg_scale_schedule_default_is_none() -> None:
    """(5) omitted key -> None reaches the task (an int crashed `.lower()`)."""
    from MolecularDiffusion.runmodes.generate.tasks_generate import (
        GenerativeFactory,
    )

    seen = {}

    class _Task:
        def sample_guidance_conitional(self, **kw):
            seen.update(kw)
            raise RuntimeError("stop after recording")

    g = object.__new__(GenerativeFactory)
    g.__dict__.update(
        target_values=[1.0],
        negative_target_values=None,
        task=_Task(),
        num_generate=1,
        batch_size=1,
        mol_size=[3],
        task_type="cfg",
        condition_configs={},
        n_frames=0,
        visualize_trajectory=False,
        _fail_count=0,
        _first_exc=None,
    )
    g.conditional_generation()
    assert seen["cfg_scale_schedule"] is None


def test_flowmol_unconditional_sample_passes_null_condition() -> None:
    """(6) conditional checkpoint + sample() -> condition of width D, cfg 0."""
    from MolecularDiffusion.modules.tasks.diffusion_flowmol import (
        FlowMolFlowMatchingTask,
    )

    calls = {}
    t = object.__new__(FlowMolFlowMatchingTask)
    nn.Module.__init__(t)
    t.register_parameter("p", nn.Parameter(torch.zeros(1)))
    t.condition, t.mask_value = ["gap", "mu"], 5.0
    t.n_adapter_context, t.adapter_indices, t.concat_indices = 0, [], [0, 1]
    t.fm_num_timesteps = 3
    t._sample_prior = lambda g, idx: g
    t.to_pc = lambda g, n: {
        k: None for k in ("one_hot", "charges", "coords", "node_mask")
    }
    t.n_atom_types = 4

    def integrate(g, idx, n_timesteps, **kw):
        calls.update(kw)
        return g

    t.vector_field = types.SimpleNamespace(integrate=integrate)
    t.sample(nodesxsample=torch.tensor([2, 3]))
    assert (
        calls["condition"].shape == (5, 2)
        and (calls["condition"] == 5.0).all()
    )
    assert calls["cfg_scale"] == 0.0

    t.condition = []  # unconditional checkpoint: unchanged (no condition)
    calls.clear()
    t.sample(nodesxsample=torch.tensor([2, 3]))
    assert calls["condition"] is None


def test_tabasco_unconditional_sample_passes_null_condition() -> None:
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
        TabascoDiffusionTask,
    )

    task = TabascoDiffusionTask(
        transformer_config=dict(
            spatial_dim=3, atom_dim=4, num_heads=2, num_layers=1, hidden_dim=8
        ),
        coords_interpolant_config={"key": "coords"},
        atomics_interpolant_config={"key": "atomics"},
        flow_matching_config={},
        num_atom_types=4,
        dataset_stats={"max_atoms": 5},
        condition_names=["gap"],
        mask_value=5.0,
    )
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    task.tabasco_model.sample = fake
    with pytest.raises(RuntimeError, match="stop"):
        task.sample(nodesxsample=torch.tensor([2, 3]))
    cond = seen["condition"]
    assert cond.shape == (2, 3, 1)
    assert (
        cond[0, :2].eq(5).all() and cond[0, 2].eq(0).all()
    )  # padding stays 0


def test_tabasco_eval_sample_without_batch_still_gets_null_condition() -> None:
    """In-training eval calls sample(batch_size=...) with no batch."""
    from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
        TabascoDiffusionTask,
    )

    task = TabascoDiffusionTask(
        transformer_config=dict(
            spatial_dim=3, atom_dim=4, num_heads=2, num_layers=1, hidden_dim=8
        ),
        coords_interpolant_config={"key": "coords"},
        atomics_interpolant_config={"key": "atomics"},
        flow_matching_config={},
        num_atom_types=4,
        dataset_stats={"max_atoms": 5},
        condition_names=["gap"],
        mask_value=5.0,
    )
    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    task.tabasco_model.sample = fake
    with pytest.raises(RuntimeError, match="stop"):
        task.sample(batch_size=3)
    cond = seen["condition"]
    pad = seen["batch"]["padding_mask"]
    assert cond.shape == (3, pad.shape[1], 1)
    assert cond[~pad].eq(5).all()  # null on real atoms
    assert cond[pad].eq(0).all()  # padding stays 0
