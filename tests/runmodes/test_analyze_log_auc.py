"""Unit tests for `log_auc` (adjusted LogAUC; formula source unverified)."""

import math

import pytest


def _one_active(position):
    scores = [float(i) for i in range(1000)]  # lower is better
    labels = [0] * 1000
    labels[position] = 1
    return scores, labels


class TestLogAuc:
    """Hand-computed: random ranking integrates to 0.14462 at lambda 0.001."""

    def test_active_ranked_first(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import log_auc

        assert log_auc(*_one_active(0)) == pytest.approx(
            1 - 0.14462, abs=1e-4
        )

    def test_active_ranked_last(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import log_auc

        assert log_auc(*_one_active(999)) == pytest.approx(-0.14462, abs=1e-4)

    def test_random_ranking_is_near_zero(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import log_auc

        n = 20000
        scores = [float(i) for i in range(n)]
        labels = [i % 2 for i in range(n)]  # actives spread evenly: y ~ x
        assert log_auc(scores, labels) == pytest.approx(0.0, abs=0.002)

    def test_no_actives_is_nan(self, caplog):
        from MolecularDiffusion.runmodes.analyze.dock_library import log_auc

        assert math.isnan(log_auc([-1.0, -2.0], [0, 0]))
        assert "no scored actives" in caplog.text

    def test_ranking_metrics_carries_the_key(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        out = ranking_metrics([-10.0, -9.0, -2.0, -1.0], [1, 1, 0, 0])
        assert out["log_auc"] > 0
