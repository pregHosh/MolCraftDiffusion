"""Which direction is "best" when the original engine selects a checkpoint (G13)."""

import math
import os

import pytest

from MolecularDiffusion.runmodes.train.eval import (
    TASK_REGISTRY,
    _resolve_task_config,
    evaluate,
    initial_best_metric,
    resolve_higher_is_better,
)


class _Solver:
    """Just enough of ``Engine`` for the loss-only evaluation path."""

    ema_decay = 0
    model = object()
    ema_model = object()

    def __init__(self, losses):
        self.losses = iter(losses)
        self._current = None

    def evaluate(self, split, use_amp=False, precision=None):
        if split == "valid":
            self._current = next(self.losses)
        return {}, [self._current], None

    def save(self, path, compact=False, full_state=True):
        open(path, "w").close()


def _run(tmp_path, losses, task="diffusion", **kwargs):
    """The loop `cli.train.engine_wrapper` runs: sentinel first, then one `evaluate` per epoch."""
    solver = _Solver(losses)
    best = initial_best_metric(task, kwargs.get("generative_analysis", False), kwargs.get("eval_higher_is_better"))
    ckpts = []
    for epoch in range(len(losses)):
        best, ckpts = evaluate(
            task, solver, epoch, best, ckpts, output_path=os.fspath(tmp_path), save_top_k=1, **kwargs
        )
    return best, ckpts


def test_loss_only_generative_run_keeps_the_lowest_loss(tmp_path) -> None:
    best, ckpts = _run(tmp_path, [1.0, 0.5, 0.2])
    assert best == pytest.approx(0.2)
    assert [round(m, 3) for m, _ in ckpts] == [0.2]


def test_explicit_override_still_wins(tmp_path) -> None:
    best, _ = _run(tmp_path, [1.0, 0.5, 0.2], eval_higher_is_better=True)
    assert best == pytest.approx(1.0)  # the user asked for higher-is-better
    best, _ = _run(tmp_path, [1.0, 0.5, 0.2], eval_higher_is_better=False)
    assert best == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("task", "gen", "override", "expected"),
    [
        ("diffusion", True, None, True),     # validity metric: registry direction, unchanged
        ("diffusion", False, None, False),   # loss only: lowest is best (the fix)
        ("diffusion", False, True, True),    # override wins
        ("diffusion", True, False, False),   # override wins
        ("diffusion_adit", False, None, False),
    ],
)
def test_direction_table(task, gen, override, expected) -> None:
    cfg = _resolve_task_config(task)
    assert resolve_higher_is_better(cfg, gen, override) is expected


@pytest.mark.parametrize("task", sorted(TASK_REGISTRY))
def test_tasks_that_are_not_loss_only_generative_are_unchanged(task) -> None:
    cfg = TASK_REGISTRY[task]
    if cfg.needs_generative:
        pytest.skip("generative tasks are covered above")
    for gen in (False, True):
        assert resolve_higher_is_better(cfg, gen, None) is cfg.higher_is_better


def test_initial_sentinel_matches_the_direction() -> None:
    assert initial_best_metric("diffusion", True) == -math.inf
    assert initial_best_metric("diffusion", False) == math.inf
    assert initial_best_metric("diffusion", False, override=True) == -math.inf
