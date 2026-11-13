"""``utils`` imports its helpers lazily and keeps the old public surface."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import MolecularDiffusion.utils as utils

SRC = Path(__file__).resolve().parents[1] / "src"
# every public name the eager package exposed (functions and submodules), recorded before the change
OLD_NAMES = Path(__file__).with_name("utils_public_names.lst").read_text().split()


def test_recorded_names_cover_the_surface() -> None:
    assert len(OLD_NAMES) > 60


@pytest.mark.parametrize("name", OLD_NAMES)
def test_every_old_public_name_still_resolves(name: str) -> None:
    assert getattr(utils, name) is not None


def test_exports_are_the_defining_objects() -> None:
    for name, (module, attr) in utils._EXPORTS.items():
        mod = importlib.import_module(module, utils.__name__)
        assert getattr(utils, name) is (mod if attr is None else getattr(mod, attr))


def test_star_import_resolves_all() -> None:
    ns: dict = {}
    exec("from MolecularDiffusion.utils import *", ns)  # noqa: S102
    assert set(utils.__all__) <= set(ns)


def test_submodules_are_reachable_as_attributes() -> None:
    assert utils.comm.get_world_size() == 1
    assert callable(utils.file.download)
    assert utils.torch is importlib.import_module("MolecularDiffusion.utils.torch")


def test_unknown_names_raise_attribute_error() -> None:
    with pytest.raises(AttributeError):
        utils.definitely_not_a_helper  # noqa: B018


def _modules_after(code: str) -> set[str]:
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code + "; import sys; print('\\n'.join(sys.modules))"],
        env=dict(os.environ, PYTHONPATH=str(SRC), CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True, check=True, timeout=300,
    ).stdout.split()
    return set(out)


def test_light_import_does_not_load_the_chemistry_stack() -> None:
    mods = _modules_after("import MolecularDiffusion.utils.runtime.comm")
    assert "rdkit" not in mods
    assert "MolecularDiffusion.utils.chem" not in mods
    assert "MolecularDiffusion.utils.metrics.geom_stability" not in mods
