"""``trainer.precision`` -> Lightning precision."""

import logging

import pytest

from MolecularDiffusion.cli.train import resolve_lightning_precision


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (32, 32), (None, 32), (16, "16-mixed"), ("16", "16-mixed"), ("bf16", "bf16-mixed"),   # unchanged
        ("bf16-mixed", "bf16-mixed"), ("16-mixed", "16-mixed"), ("32-true", "32-true"),        # now honoured
        (64, 64), ("64-true", "64-true"),
    ],
)
def test_known_values(value, expected) -> None:
    assert resolve_lightning_precision(value) == expected


@pytest.mark.parametrize("value", ["fp8", 8, "half", "x"])
def test_unknown_values_fall_back_to_32_and_say_so(value, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert resolve_lightning_precision(value) == 32
    assert any("falling back to 32" in r.getMessage() for r in caplog.records)
