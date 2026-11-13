"""``modules.models`` imports its classes lazily and keeps the old API."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import MolecularDiffusion.modules.models as models

SRC = Path(__file__).resolve().parents[1] / "src"


def test_public_names_resolve_to_the_defining_class() -> None:
    for name in models.__all__:
        mod = importlib.import_module(models._LAZY[name], models.__name__)
        assert getattr(models, name) is getattr(mod, name)


def test_star_import_and_submodule_attribute() -> None:
    ns: dict = {}
    exec("from MolecularDiffusion.modules.models import *", ns)  # noqa: S102
    assert set(models.__all__) <= set(ns)
    assert models.en_diffusion.EnVariationalDiffusion is models.EnVariationalDiffusion


def test_importing_one_model_does_not_load_the_others() -> None:
    code = (
        "import sys, MolecularDiffusion.modules.models.noisemodel;"
        "bad = [m for m in sys.modules if m.endswith('models.en_diffusion')];"
        "sys.exit(bool(bad))"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC), CUDA_VISIBLE_DEVICES="")
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
