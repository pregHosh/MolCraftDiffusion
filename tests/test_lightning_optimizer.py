"""Lightning optimizer/scheduler construction matches the original engine."""

import importlib.util
import logging

import pytest
import torch
from torch import nn, optim

from MolecularDiffusion.core.engine_lightning import EngineLightning

BASE_LR = 1e-3
needs_param_groups = pytest.mark.skipif(
    importlib.util.find_spec("MolecularDiffusion.core.optim_utils") is None,
    reason="LoRA/adapter param groups are not in this checkout",
)


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        self.adapter_0 = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4)
        self.frozen.requires_grad_(False)


def _engine(scheduler=None, scheduler_kwargs=None, **opt_extra):
    return EngineLightning(
        optimizer_config={"optimizer_choice": "adamw", "lr": BASE_LR, "weight_decay": 0.01, **opt_extra},
        task=_Toy(),
        scheduler_config={"scheduler": scheduler, "scheduler_kwargs": scheduler_kwargs or {}},
    )


def _n(params):
    return sum(p.numel() for p in params)


def test_default_is_one_group_with_every_parameter() -> None:
    eng = _engine()
    opt = eng.configure_optimizers()
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == BASE_LR
    assert _n(opt.param_groups[0]["params"]) == _n(eng.task.parameters())  # frozen ones included, as before
    ref = optim.AdamW(_Toy().parameters(), lr=BASE_LR, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    for key in ("lr", "betas", "eps", "weight_decay"):
        assert opt.param_groups[0][key] == ref.param_groups[0][key]


@needs_param_groups
def test_lora_lr_gets_its_own_group_and_frozen_params_are_dropped() -> None:
    eng = _engine(lora_lr=5e-3)
    groups = eng.configure_optimizers().param_groups
    by_lr = {g["lr"]: _n(g["params"]) for g in groups}
    assert by_lr[5e-3] == 4 * 2 + 2 * 4                      # lora_A + lora_B
    assert by_lr[BASE_LR] == (4 * 4 + 4) + (4 * 4 + 4)       # backbone + adapter_0 (frozen excluded)


@needs_param_groups
def test_adapter_lr_gets_its_own_group() -> None:
    groups = _engine(adapter_lr=2e-4).configure_optimizers().param_groups
    by_lr = {g["lr"]: _n(g["params"]) for g in groups}
    assert by_lr[2e-4] == 4 * 4 + 4                          # adapter_0
    assert BASE_LR in by_lr


@needs_param_groups
def test_both_fine_tuning_rates_together() -> None:
    groups = _engine(lora_lr=5e-3, adapter_lr=2e-4).configure_optimizers().param_groups
    assert sorted(g["lr"] for g in groups) == sorted([5e-3, 2e-4, BASE_LR])


@pytest.mark.parametrize(
    ("name", "kwargs", "cls"),
    [
        ("steplr", {"step_size": 1}, optim.lr_scheduler.StepLR),
        ("multisteplr", {"milestones": [1]}, optim.lr_scheduler.MultiStepLR),
        ("exponentiallr", {"gamma": 0.9}, optim.lr_scheduler.ExponentialLR),
        ("exponentiarlr", {"gamma": 0.9}, optim.lr_scheduler.ExponentialLR),   # original engine's spelling
        ("cosineannealing", {"T_max": 3}, optim.lr_scheduler.CosineAnnealingLR),
        ("caws", {"T_0": 2}, optim.lr_scheduler.CosineAnnealingWarmRestarts),
        ("lambdalr", {"lr_lambda": lambda epoch: 1.0}, optim.lr_scheduler.LambdaLR),
    ],
)
def test_every_scheduler_name_the_original_engine_accepts(name, kwargs, cls) -> None:
    out = _engine(name, kwargs).configure_optimizers()
    assert isinstance(out["lr_scheduler"]["scheduler"], cls)


def test_unknown_scheduler_still_falls_back_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        out = _engine("not_a_scheduler").configure_optimizers()
    assert isinstance(out, torch.optim.Optimizer)
    assert any("Unknown scheduler" in r.getMessage() for r in caplog.records)
