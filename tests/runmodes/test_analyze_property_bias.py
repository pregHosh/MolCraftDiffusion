"""Unit tests for `analyze bias` -- score-vs-property bias diagnostics."""

import argparse
import logging

import numpy as np
import pytest


def _frame(smiles, scores):
    import pandas as pd

    return pd.DataFrame(
        {
            "name": [f"m{i}" for i in range(len(smiles))],
            "smiles": smiles,
            "score": scores,
        }
    )


def _props(smiles, scores):
    from MolecularDiffusion.runmodes.analyze.property_bias import (
        _with_properties,
    )

    return _with_properties(_frame(smiles, scores))[0]


def _write_pool(path, n, charge_col=False):
    path.write_text(
        "name,smiles,vina_dock\n"
        + "".join(f"m{i},{'C' * (i + 3)},-{i}.0\n" for i in range(n))
    )


ALKANES = ["C" * n for n in (10, 12, 14, 16, 18, 20)]


class TestAlkanes:
    def test_perfect_negative_correlation_and_topk(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            score_correlations,
            topk_vs_pool,
        )

        props = _props(ALKANES, [-5.0, -6.0, -7.0, -8.0, -9.0, -10.0])
        corr = score_correlations(props, n_boot=50)
        row = corr[corr["property"] == "heavy_atoms"].iloc[0]
        assert row["spearman"] == pytest.approx(-1.0)
        assert row["pearson"] == pytest.approx(-1.0)

        top = topk_vs_pool(props, 2)
        heavy = next(r for r in top if r["property"] == "heavy_atoms")
        assert heavy["subset_mean"] == pytest.approx(19.0)
        assert heavy["pool_mean"] == pytest.approx(15.0)
        # exact p over all 15 pairs is 1/15; the sampled one must be close
        assert heavy["p_perm"] == pytest.approx(1 / 15, abs=0.02)


class TestNoBias:
    def test_shuffled_scores_give_no_warning(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            bias_warnings,
            score_correlations,
            topk_vs_pool,
        )

        rng = np.random.default_rng(1)
        smiles = [
            "C" * int(rng.integers(3, 25)) + "O" * int(rng.integers(0, 3))
            for _ in range(200)
        ]
        props = _props(smiles, rng.permutation(200).astype(float))
        corr = score_correlations(props, n_boot=100)
        assert (corr["spearman"].abs().fillna(0) < 0.2).all()
        top = topk_vs_pool(props, 10, n_perm=2000)
        assert bias_warnings(corr, top) == []

    def test_size_bias_warns(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            bias_warnings,
            score_correlations,
            topk_vs_pool,
        )

        smiles = ["C" * n for n in range(3, 63)]
        props = _props(smiles, [-float(n) for n in range(3, 63)])
        corr = score_correlations(props, n_boot=50)
        top = topk_vs_pool(props, 10, n_perm=2000)
        text = "\n".join(bias_warnings(corr, top))
        assert "The score is picking size" in text


class TestRows:
    def test_largest_fragment_drops_the_counter_ion(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            molecule_properties,
        )

        out = molecule_properties("[Na+].CC(=O)[O-]")
        assert out["heavy_atoms"] == 4
        assert out["charge"] == -1
        whole = molecule_properties("[Na+].CC(=O)[O-]", largest_fragment=False)
        assert whole["heavy_atoms"] == 5
        assert whole["charge"] == 0

    def test_empty_score_dropped_and_duplicate_keeps_lowest(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            load_scores,
        )

        path = tmp_path / "d.csv"
        path.write_text(
            "name,smiles,vina_dock\n"
            "a,CCO,-5.0\n"
            "b,CCC,\n"
            "c,CCCC,oops\n"
            "a,CCO,-7.5\n"
            "d,CCCCC,-1.0\n"
        )
        frame, counts = load_scores([str(path)])
        assert counts["errors"] == 2
        assert counts["duplicates"] == 1
        assert counts["scored"] == 2
        assert frame.set_index("name").loc["a", "score"] == -7.5

    def test_unparseable_smiles_dropped(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            _with_properties,
        )

        frame, n_bad = _with_properties(
            _frame(["CCO", "not_a_smiles"], [-1, -2])
        )
        assert n_bad == 1
        assert len(frame) == 1


class TestRunner:
    @staticmethod
    def _args(path, **kw):
        base = {
            "inputs": [str(path)],
            "score_col": "vina_dock",
            "labels": None,
            "id_col": "name",
            "label_col": "label",
            "top": None,
            "sd_flag": 0.5,
            "out_dir": None,
        }
        base.update(kw)
        return argparse.Namespace(**base)

    def test_too_few_molecules_does_not_crash(self, tmp_path, caplog):
        from MolecularDiffusion.runmodes.analyze.property_bias import runner

        path = tmp_path / "d.csv"
        _write_pool(path, 5)
        with caplog.at_level(logging.INFO):
            runner(self._args(path))
        assert "too few" in caplog.text

    def test_constant_charge_is_flagged_and_csv_has_no_nan(
        self, tmp_path, caplog
    ):
        from MolecularDiffusion.runmodes.analyze.property_bias import runner

        path = tmp_path / "d.csv"
        _write_pool(path, 40)
        out = tmp_path / "out"
        with caplog.at_level(logging.INFO):
            runner(self._args(path, out_dir=str(out)))
        assert "charge is nearly constant" in caplog.text
        for name in (
            "property_bias_correlations.csv",
            "property_bias_topk.csv",
            "property_bias_props.csv",
        ):
            assert "nan" not in (out / name).read_text().lower()

    def test_no_files_without_out_dir(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.property_bias import runner

        path = tmp_path / "d.csv"
        _write_pool(path, 30)
        runner(self._args(path))
        assert [p.name for p in tmp_path.iterdir()] == ["d.csv"]

    def test_labels_section(self, tmp_path, caplog):
        from MolecularDiffusion.runmodes.analyze.property_bias import runner

        path = tmp_path / "d.csv"
        _write_pool(path, 40)
        master = tmp_path / "m.csv"
        master.write_text("cid,hit\nm39,1\nm38,1\nm0,0\n")
        with caplog.at_level(logging.INFO):
            runner(
                self._args(
                    path, labels=str(master), id_col="cid", label_col="hit"
                )
            )
        assert "2 of them in the scored pool" in caplog.text


class TestCli:
    def test_registered_and_help_renders(self):
        from click.testing import CliRunner

        from MolecularDiffusion.cli.analyze import analyze

        res = CliRunner().invoke(analyze, ["bias", "--help"])
        assert res.exit_code == 0
        assert "--labels" in res.output

    def test_runs_on_a_tiny_csv(self, tmp_path):
        from click.testing import CliRunner

        from MolecularDiffusion.cli.analyze import analyze

        path = tmp_path / "d.csv"
        _write_pool(path, 30)
        res = CliRunner().invoke(analyze, ["bias", str(path), "--top", "5"])
        assert res.exit_code == 0, res.output


class TestReviewFixes:
    """Cases the independent review found uncovered."""

    def test_empty_smiles_is_dropped_not_a_crash(self):
        from MolecularDiffusion.runmodes.analyze.property_bias import (
            molecule_properties,
        )

        assert molecule_properties("") is None

    def test_constant_column_correlation_is_nan_for_both_methods(self):
        import numpy as np

        from MolecularDiffusion.runmodes.analyze.property_bias import (
            _corr_pairs,
            _corr_rows_ranked,
        )

        x = np.full((1, 30), 0.1)
        y = np.arange(30, dtype=float)[None, :]
        assert np.isnan(_corr_pairs(x, y)).all()
        assert np.isnan(_corr_rows_ranked(x, y)).all()

    def test_top_must_be_at_least_one(self, tmp_path):
        from click.testing import CliRunner

        from MolecularDiffusion.cli.analyze import analyze

        csv = tmp_path / "d.csv"
        csv.write_text("name,smiles,vina_dock\na,CC,-1\nb,CCC,-2\n")
        for bad in ("0", "-3"):
            res = CliRunner().invoke(analyze, ["bias", str(csv), "--top", bad])
            assert res.exit_code == 2, res.output
