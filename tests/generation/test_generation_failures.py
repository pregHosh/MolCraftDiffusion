"""Batch-failure bookkeeping and progress reporting of ``GenerativeFactory``."""

import logging
import os

import pytest

from MolecularDiffusion.runmodes.generate.tasks_generate import GenerativeFactory

from .test_generation_execution_compat import (  # noqa: F401  (fixtures)
    FakeGenerationTask,
    fake_xyz_io,
)

pytestmark = pytest.mark.generation


class _Bar:
    def __init__(self):
        self.posts = []

    def set_postfix(self, d):
        self.posts.append(d)


class _FailsFirst(FakeGenerationTask):
    """Sampling raises on the first call only (or on every call)."""

    def __init__(self, every_call=False):
        super().__init__()
        self.every_call = every_call
        self.n = 0

    def sample(self, nodesxsample, **kwargs):
        self.n += 1
        if self.every_call or self.n == 1:
            raise RuntimeError("boom")
        return super().sample(nodesxsample, **kwargs)


def _factory(task, tmp_path):
    return GenerativeFactory(
        task=task, task_type="unconditional", num_generate=4, batch_size=2,
        mol_size=[0, 0], max_mol_size=5, output_path=os.fspath(tmp_path),
    )


def test_record_batch_failure_counts_and_keeps_the_first_error(capsys):
    f = GenerativeFactory.__new__(GenerativeFactory)
    f._fail_count, f._first_exc = 0, None
    first, second = ValueError("one"), KeyError("two")
    f._record_batch_failure(0, first)
    f._record_batch_failure(3, second)
    assert (f._fail_count, f._first_exc) == (2, first)
    out = capsys.readouterr().out
    assert "[Batch 0] Sampling failed: one" in out
    assert "[Batch 3] Sampling failed: 'two'" in out


@pytest.mark.parametrize(
    ("per_batch", "expected"),
    [
        (1, {"completed": 3, "failed": 1, "success": 2, "success_rate": "66.7%"}),
        (4, {"completed": 12, "failed": 1, "success": 8, "success_rate": "66.7%"}),
    ],
)
def test_set_batch_postfix(per_batch, expected):
    bar = _Bar()
    GenerativeFactory._set_batch_postfix(bar, 2, 1, per_batch=per_batch)
    assert bar.posts == [expected]
    assert list(bar.posts[0]) == list(expected)  # key order is part of the output


def test_one_failed_batch_is_counted_and_the_rest_still_written(
    tmp_path, fake_xyz_io, capsys  # noqa: F811
):
    factory = _factory(_FailsFirst(), tmp_path)
    factory.unconditional_generation()
    assert factory._fail_count == 1
    assert str(factory._first_exc) == "boom"
    assert "[Batch 0] Sampling failed: boom" in capsys.readouterr().out
    assert len(fake_xyz_io.moves) == 2  # batch 1 (2 molecules) went through


def test_all_batches_failing_still_raises_with_the_first_error(
    tmp_path, fake_xyz_io  # noqa: F811
):
    factory = _factory(_FailsFirst(every_call=True), tmp_path)
    factory.unconditional_generation()
    with pytest.raises(RuntimeError, match="First failure was RuntimeError: boom"):
        factory._assert_produced()


def test_traceback_is_logged_at_debug_only(tmp_path, fake_xyz_io, caplog):  # noqa: F811
    with caplog.at_level(logging.INFO):
        _factory(_FailsFirst(), tmp_path).unconditional_generation()
    assert not [r for r in caplog.records if "traceback" in r.getMessage()]
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _factory(_FailsFirst(), tmp_path).unconditional_generation()
    rec = [r for r in caplog.records if "traceback" in r.getMessage()]
    assert rec and rec[0].exc_info[1].args == ("boom",)
