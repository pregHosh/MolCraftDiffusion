"""``GenerativeEvalCallback``: raw weights by default, the EMA copy on request."""

import logging
from types import SimpleNamespace

import pytest

from MolecularDiffusion.core.lightning_callbacks import generative_eval_callback as mod


@pytest.fixture
def seen(monkeypatch):
    models = []

    def fake_analyze(model, epoch, **kwargs):
        models.append(model)
        return {"Validity": 1.0}

    monkeypatch.setattr(mod, "analyze_and_save", fake_analyze)
    return models


def _run(cb, raw, ema, tmp_path):
    trainer = SimpleNamespace(is_global_zero=True, current_epoch=3)
    module = SimpleNamespace(task=raw, ema_model=ema, log=lambda *a, **k: None)
    cb.on_validation_epoch_end(trainer, module)


def test_default_generates_from_the_raw_weights_and_says_so_once(seen, tmp_path, caplog) -> None:
    raw, ema = object(), object()
    cb = mod.GenerativeEvalCallback(n_samples=1, output_dir=str(tmp_path))
    with caplog.at_level(logging.INFO):
        _run(cb, raw, ema, tmp_path)
        _run(cb, raw, ema, tmp_path)
    assert seen == [raw, raw]
    notes = [r for r in caplog.records if "raw weights" in r.getMessage()]
    assert len(notes) == 1


def test_use_ema_generates_from_the_ema_copy(seen, tmp_path) -> None:
    raw, ema = object(), object()
    _run(mod.GenerativeEvalCallback(n_samples=1, output_dir=str(tmp_path), use_ema=True), raw, ema, tmp_path)
    assert seen == [ema]


def test_use_ema_without_an_ema_copy_falls_back_to_raw(seen, tmp_path, caplog) -> None:
    raw = object()
    with caplog.at_level(logging.INFO):
        _run(mod.GenerativeEvalCallback(n_samples=1, output_dir=str(tmp_path), use_ema=True), raw, None, tmp_path)
    assert seen == [raw]
    assert not [r for r in caplog.records if "raw weights" in r.getMessage()]
