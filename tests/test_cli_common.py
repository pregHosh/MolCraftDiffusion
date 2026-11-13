"""``cli._common`` holds the helpers; each CLI module keeps its own variant."""

import types

import pytest
import torch
from omegaconf import OmegaConf

from MolecularDiffusion.cli import _common, eval_predict, generate, predict, train


class _Cap:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str, *_a, **_k) -> None:
        self.lines.append(msg)


def _objects():
    net = torch.nn.Sequential(torch.nn.Linear(2, 3))
    task = types.SimpleNamespace(task=net, enc=torch.nn.Linear(2, 3), _hidden=1)
    return {"cfg": OmegaConf.create({"a": 1}), "task": task}


def test_is_rank_zero_is_one_shared_function() -> None:
    for mod in (train, generate, predict, eval_predict):
        assert mod.is_rank_zero is _common.is_rank_zero
    assert _common.is_rank_zero() is True  # no process group in tests


# (module, prints parameter counts, prints nn.Module attributes by class name)
VARIANTS = [
    (train, True, True),
    (generate, False, False),
    (predict, True, False),
    (eval_predict, True, False),
]


@pytest.mark.parametrize(("mod", "params", "collapsed"), VARIANTS)
def test_each_cli_keeps_its_own_output(monkeypatch, mod, params, collapsed) -> None:
    cap = _Cap()
    monkeypatch.setattr(mod, "log", cap)  # lines must go through the module's own logger
    mod.log_hyperparameters(_objects())
    text = "\n".join(cap.lines)
    assert ("MODEL PARAMS" in text) is params
    assert ("enc: Linear\n" in text + "\n") is collapsed
    assert ("enc: Linear(in_features=2" in text) is not collapsed
    assert text.rstrip().endswith("End of Hyperparameters ==========")
