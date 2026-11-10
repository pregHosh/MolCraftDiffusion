"""Unit tests for `analyze dock` -- library docking, RMSD and enrichment.

`ranking_metrics` needs no optional dependency, so those tests always run;
the rest import the module lazily and monkeypatch the Vina call, so nothing
here actually docks.
"""

import argparse

import pytest


class TestRankingMetrics:
    """Hand-computed numbers -- the point is to catch a sign flip."""

    def test_perfect_ranking(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        # vina: lower is better, so actives must carry the lower scores
        scores = [-10.0, -9.0, -2.0, -1.0]
        labels = [1, 1, 0, 0]
        out = ranking_metrics(scores, labels)
        assert out["roc_auc"] == pytest.approx(1.0)
        assert out["n_actives"] == 2
        assert out["n_decoys"] == 2

    def test_inverted_ranking_scores_zero(self):
        """Guards the negation: actives last must give AUC 0, not 1."""
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        out = ranking_metrics([-1.0, -2.0, -9.0, -10.0], [1, 1, 0, 0])
        assert out["roc_auc"] == pytest.approx(0.0)

    def test_enrichment_factor_is_hand_checkable(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        # 100 molecules, 10 actives, and the 5 best scores are all actives.
        scores = [-10.0 - i for i in range(5)]          # 5 actives, best
        labels = [1] * 5
        scores += [-1.0] * 5                            # 5 actives, worst
        labels += [1] * 5
        scores += [-2.0] * 90                           # 90 decoys
        labels += [0] * 90

        out = ranking_metrics(scores, labels, fractions=(0.05,))
        # top 5% = 5 molecules, all active -> 1.0 / (10/100) = 10x
        assert out["ef_5pct"] == pytest.approx(10.0)

    def test_single_class_returns_nothing(self):
        """Enrichment is undefined with no decoys; 0.5 would be a lie."""
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        assert ranking_metrics([-1.0, -2.0], [1, 1]) == {}
        assert ranking_metrics([], []) == {}

    def test_non_finite_scores_are_dropped(self):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            ranking_metrics,
        )

        out = ranking_metrics(
            [-10.0, float("nan"), -1.0], [1, 1, 0],
        )
        assert out["n_actives"] == 1
        assert out["n_decoys"] == 1


class TestLoadLibrary:
    def test_reads_an_sdf_keeping_coordinates(self, tmp_path):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", "ethanol")
        sdf = tmp_path / "lib.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)

        loaded = load_library(sdf)
        assert len(loaded) == 1
        name, got = loaded[0]
        assert name == "ethanol"
        assert got.GetNumConformers() == 1

    def test_unsatisfiable_bridgehead_stereo_still_embeds(self, tmp_path):
        """A CACHE #3 compound whose bridged-ring stereo cannot exist in 3D.

        RDKit fails at every seed; the loader retries without stereo so the
        molecule is docked rather than silently lost.
        """
        from rdkit import Chem

        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        smi = tmp_path / "lib.smi"
        smi.write_text(
            "C1[C@H]2CN(C(c3ccccn3)=O)[C@H]1CN2C(c1ccc(cc1)C(N)=O)=O bad\n"
            "CCO ok\n"
        )
        loaded = dict(load_library(smi))
        assert loaded["bad"] is not None
        assert loaded["bad"].GetNumConformers() == 1
        assert "@" not in Chem.MolToSmiles(Chem.RemoveHs(loaded["bad"]))
        assert loaded["ok"] is not None          # the ordinary path is untouched

    def test_embeds_smiles_lines(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        smi = tmp_path / "lib.smi"
        smi.write_text("CCO ethanol\nc1ccccc1 benzene\n")
        loaded = load_library(smi)
        assert [n for n, _ in loaded] == ["ethanol", "benzene"]
        assert all(m.GetNumConformers() == 1 for _, m in loaded)

    def test_unparseable_smiles_becomes_none_not_an_exception(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        smi = tmp_path / "lib.smi"
        smi.write_text("CCO\nnot_a_smiles\n")
        loaded = load_library(smi)
        assert len(loaded) == 2
        assert loaded[1][1] is None

    def test_csv_prefers_a_smiles_column(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        csv = tmp_path / "lib.csv"
        csv.write_text("id,smiles\nm1,CCO\n")
        loaded = load_library(csv)
        assert loaded[0][0] == "m1"
        assert loaded[0][1] is not None

    def test_unknown_suffix_is_rejected(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.dock_library import (
            load_library,
        )

        bad = tmp_path / "lib.mol2"
        bad.write_text("")
        with pytest.raises(ValueError, match="unsupported library"):
            load_library(bad)


class TestPoseRmsd:
    def test_reports_top_and_best_separately(self):
        """The whole lesson: finding the pose != ranking it first."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.dock_library import (
            _pose_rmsd,
        )

        ref = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMultipleConfs(ref, numConfs=2, randomSeed=42)
        exact = Chem.Mol(ref, confId=ref.GetConformers()[0].GetId())
        other = Chem.Mol(ref, confId=ref.GetConformers()[1].GetId())

        # the exact match is ranked SECOND
        out = _pose_rmsd([(-8.0, other), (-7.0, exact)], exact)
        assert out["rmsd_best_rank"] == 1
        assert out["rmsd_best"] == pytest.approx(0.0, abs=1e-6)
        assert out["rmsd_top"] > out["rmsd_best"]


class TestRunnerEndToEnd:
    """Drives `runner` with the Vina call faked out."""

    def test_writes_a_csv_with_enrichment(self, tmp_path, monkeypatch):
        pytest.importorskip("vina", reason="needs the [sbdd] extra")
        pytest.importorskip("meeko", reason="needs the [sbdd] extra")

        import pandas as pd

        from MolecularDiffusion.runmodes.analyze import dock_library, docking

        actives = tmp_path / "actives.smi"
        actives.write_text("CCO a1\nCCCO a2\n")
        decoys = tmp_path / "decoys.smi"
        decoys.write_text("C d1\nCC d2\n")

        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM\n")

        # actives score better than decoys, by atom count
        def fake_score_pose(mol, _receptor, **_kwargs):
            return {"vina_score": -float(mol.GetNumHeavyAtoms())}

        monkeypatch.setattr(docking, "score_pose", fake_score_pose)

        out_csv = tmp_path / "out.csv"
        dock_library.runner(argparse.Namespace(
            receptor=str(rec),
            actives=[str(actives)],
            decoys=[str(decoys)],
            ligands=None,
            box_ref_ligand=None,
            box_center=[0.0, 0.0, 0.0],
            box_size=[20.0, 20.0, 20.0],
            mode="score",
            exhaustiveness=1,
            n_poses=1,
            rmsd_ref=None,
            seed=42,
            allow_bad_res=False,
            default_altloc=None,
            out_dir=str(tmp_path),
            output=str(out_csv),
        ))

        frame = pd.read_csv(out_csv)
        assert len(frame) == 4
        assert set(frame["name"]) == {"a1", "a2", "d1", "d2"}
        assert frame["vina_score"].notna().all()

        summary = tmp_path / "out_summary.json"
        assert summary.is_file()

        import json

        payload = json.loads(summary.read_text())
        # ethanol/propanol have more heavy atoms -> better faked scores
        assert payload["roc_auc"] == pytest.approx(1.0)
        assert payload["n_molecules"] == 4

    def test_empty_input_is_a_clear_error(self, tmp_path):
        pytest.importorskip("vina", reason="needs the [sbdd] extra")

        from MolecularDiffusion.runmodes.analyze import dock_library

        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM\n")
        with pytest.raises(ValueError, match="nothing to dock"):
            dock_library.runner(argparse.Namespace(
                receptor=str(rec), actives=None, decoys=None, ligands=None,
                box_ref_ligand=None, box_center=None, box_size=None,
                mode="score", exhaustiveness=1, n_poses=1, rmsd_ref=None,
                seed=42, allow_bad_res=False, default_altloc=None,
                out_dir=str(tmp_path), output=str(tmp_path / "x.csv"),
            ))
