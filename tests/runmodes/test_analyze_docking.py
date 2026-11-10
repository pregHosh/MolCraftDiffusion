"""Unit tests for the `analyze metrics --metrics sbdd` building blocks."""

import numpy as np
import pytest

vina = pytest.importorskip("vina", reason="needs the [sbdd] extra")
meeko = pytest.importorskip("meeko", reason="needs the [sbdd] extra")

METHANE_XYZ = (
    "5\n\n"
    "C 0.000 0.000 0.000\n"
    "H 0.629 0.629 0.629\n"
    "H -0.629 -0.629 0.629\n"
    "H -0.629 0.629 -0.629\n"
    "H 0.629 -0.629 -0.629\n"
)


class TestLoadPose:
    def test_keeps_the_conformer(self, tmp_path):
        """Docking needs the generated coordinates, not just the topology."""
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        xyz = tmp_path / "methane.xyz"
        xyz.write_text(METHANE_XYZ)
        smiles, mol = load_pose(str(xyz))
        assert mol is not None
        assert mol.GetNumConformers() == 1
        assert smiles

    def test_unreadable_file_returns_none(self, tmp_path):
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        bad = tmp_path / "corrupt.xyz"
        bad.write_text("not\nan xyz file\n")
        assert load_pose(str(bad)) == (None, None)

    def test_bad_valence_returns_none_without_raising(self, tmp_path):
        """A structure RDKit cannot sanitize is data, not an exception."""
        from MolecularDiffusion.runmodes.analyze.docking import load_pose

        # five atoms crowded around one N -> hypervalent after perception
        xyz = tmp_path / "bad.xyz"
        xyz.write_text(
            "6\n\n"
            "N 0.000 0.000 0.000\n"
            "C 1.470 0.000 0.000\n"
            "C -1.470 0.000 0.000\n"
            "C 0.000 1.470 0.000\n"
            "C 0.000 -1.470 0.000\n"
            "C 0.000 0.000 1.470\n"
        )
        smiles, mol = load_pose(str(xyz))
        assert mol is None


class TestBox:
    def test_box_is_extent_plus_buffer(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import box_for

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        center, size = box_for(mol)

        pos = mol.GetConformer(0).GetPositions()
        assert center == pytest.approx(((pos.max(0) + pos.min(0)) / 2).tolist())
        assert size == pytest.approx(((pos.max(0) - pos.min(0)) + 5.0).tolist())

    def test_buffer_is_configurable(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import box_for

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        _, small = box_for(mol, buffer=0.0)
        _, big = box_for(mol, buffer=10.0)
        assert np.allclose(np.array(big) - np.array(small), 10.0)


class TestLigandPdbqt:
    def test_writes_pdbqt_for_a_sane_molecule(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import ligand_pdbqt

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        text = ligand_pdbqt(mol)
        assert "ATOM" in text or "ROOT" in text


class TestScoreMode:
    def test_rejects_unknown_mode(self):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from MolecularDiffusion.runmodes.analyze.docking import score_pose

        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        with pytest.raises(ValueError, match="mode must be one of"):
            score_pose(mol, "unused.pdbqt", mode="nope")


class TestPrepareReceptor:
    def test_pdbqt_is_passed_through_untouched(self, tmp_path):
        """An already-prepared receptor must be reused, not re-derived."""
        from MolecularDiffusion.runmodes.analyze.docking import prepare_receptor

        rec = tmp_path / "rec.pdbqt"
        rec.write_text("ATOM\n")
        assert prepare_receptor(str(rec)) == str(rec)


def _ethanol():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


class _FakeVina:
    """Records the box it was handed; runs no docking."""

    last_maps = None

    def __init__(self, **_kwargs):
        pass

    def set_receptor(self, _path):
        pass

    def set_ligand_from_string(self, _text):
        pass

    def compute_vina_maps(self, center=None, box_size=None):
        type(self).last_maps = (list(center), list(box_size))

    def score(self):
        return [-7.5]

    def optimize(self):
        return [-8.0]

    def dock(self, **_kwargs):
        pass

    def energies(self, n_poses=1):
        return [[-8.5]] * n_poses


class TestFixedBox:
    """The pocket-box override -- the part catalogue compounds need."""

    def test_omitting_the_override_keeps_the_per_ligand_box(self, monkeypatch):
        import vina as vina_mod

        from MolecularDiffusion.runmodes.analyze import docking

        monkeypatch.setattr(vina_mod, "Vina", _FakeVina)
        mol = _ethanol()
        docking.score_pose(mol, "unused.pdbqt", mode="score")

        from rdkit import Chem

        expected = docking.box_for(Chem.AddHs(mol, addCoords=True))
        assert _FakeVina.last_maps[0] == pytest.approx(expected[0])
        assert _FakeVina.last_maps[1] == pytest.approx(expected[1])

    def test_explicit_box_overrides_the_ligand_extent(self, monkeypatch):
        import vina as vina_mod

        from MolecularDiffusion.runmodes.analyze import docking

        monkeypatch.setattr(vina_mod, "Vina", _FakeVina)
        docking.score_pose(
            _ethanol(), "unused.pdbqt", mode="score",
            center=[1.0, 2.0, 3.0], box_size=[20.0, 20.0, 20.0],
        )
        assert _FakeVina.last_maps == ([1.0, 2.0, 3.0], [20.0, 20.0, 20.0])

    def test_box_from_ligand_matches_box_for(self, tmp_path):
        from rdkit import Chem

        from MolecularDiffusion.runmodes.analyze import docking

        mol = _ethanol()
        sdf = tmp_path / "ref.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            writer.write(mol)

        center, size = docking.box_from_ligand(str(sdf))
        # round-trips through SDF, so compare on the re-read molecule
        reread = Chem.MolFromMolFile(str(sdf), sanitize=True)
        expected = docking.box_for(Chem.AddHs(reread, addCoords=True))
        assert center == pytest.approx(expected[0])
        assert size == pytest.approx(expected[1])


class TestReceptorFlags:
    """The altloc/bad-residue flags real PDB files need."""

    def _argv_for(self, monkeypatch, tmp_path, **kwargs):
        import subprocess

        from MolecularDiffusion.runmodes.analyze import docking

        seen = {}

        def fake_run(argv, **_kw):
            seen["argv"] = argv
            # pretend meeko wrote the output so no exception is raised
            (tmp_path / "rec.pdbqt").write_text("ATOM\n")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM\n")
        docking.prepare_receptor(str(pdb), out_dir=str(tmp_path), **kwargs)
        return seen["argv"]

    def test_flags_absent_by_default(self, monkeypatch, tmp_path):
        argv = self._argv_for(monkeypatch, tmp_path)
        assert "--allow_bad_res" not in argv
        assert "--default_altloc" not in argv

    def test_flags_added_when_asked(self, monkeypatch, tmp_path):
        argv = self._argv_for(
            monkeypatch, tmp_path, allow_bad_res=True, default_altloc="A",
        )
        assert "--allow_bad_res" in argv
        assert argv[argv.index("--default_altloc") + 1] == "A"
