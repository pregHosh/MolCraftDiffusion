"""Smoke tests for the ShEPhERD-2 task (construction, aliasing, device move)."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("shepherd_score")
from omegaconf import OmegaConf  # noqa: E402

import MolecularDiffusion
from MolecularDiffusion.modules.tasks.diffusion_shepherd2 import (  # noqa: E402
    Shepherd2Generative,
)

CFG = (
    Path(MolecularDiffusion.__file__).parent
    / "configs/tasks/diffusion_shepherd2.yaml"
)


@pytest.fixture(scope="module")
def task():
    cfg = OmegaConf.load(CFG)
    return Shepherd2Generative(
        model_params=cfg.model_params,
        modality_weights=cfg.modality_weights,
    )


def test_build_returns_self(task):
    assert task.build() is task
    assert task.task is task


def test_dynamics_alias_not_registered(task):
    # alias must live in __dict__, else state_dict() recurses forever
    assert "dynamics" not in task.model._modules
    assert task.model.dynamics is task.model
    assert len(task.state_dict()) > 0


def test_to_moves_inner_module(task):
    task.to(torch.device("cpu"))
    assert task.device.type == "cpu"
    assert task.model.device.type == "cpu"


def test_missing_params_raises():
    with pytest.raises(ValueError):
        Shepherd2Generative()
