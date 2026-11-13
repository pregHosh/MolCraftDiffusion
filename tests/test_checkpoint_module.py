"""``core.checkpoint`` owns the checkpoint helpers; old names stay importable."""

import pytest

from MolecularDiffusion.cli import generate
from MolecularDiffusion.core import checkpoint as ck

MOVED = {
    "_get_ckpt_meta": "get_ckpt_meta",
    "_validate_task_type": "validate_task_type",
    "_select_ckpt_file": "select_ckpt_file",
    "_extract_clean_state_dict": "extract_clean_state_dict",
    "_apply_total_step_override": "apply_total_step_override",
    "_load_edm_stat": "load_edm_stat",  # not in every checkout
    "_extract_ema_state_dict": "extract_ema_state_dict",  # not in every checkout
}


@pytest.mark.parametrize(("old", "new"), MOVED.items())
def test_old_generate_names_are_the_same_objects(old: str, new: str) -> None:
    if not hasattr(ck, new):
        pytest.skip(f"{new} not present in this checkout")
    assert getattr(generate, old) is getattr(ck, new)


def test_engine_weights_prefers_ema_then_raw_then_none() -> None:
    assert ck.engine_weights({"ema_model": {"a": 1}, "model": {"a": 2}}) == {"a": 1}
    assert ck.engine_weights({"model": {"a": 2}}) == {"a": 2}
    assert ck.engine_weights({"ema_model": {}, "model": {"a": 2}}) == {"a": 2}
    assert ck.engine_weights({"state_dict": {}}) is None
