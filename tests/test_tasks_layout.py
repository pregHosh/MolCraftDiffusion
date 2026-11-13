"""``modules/tasks`` groups its flat task modules into family packages.

The old flat paths stay as aliases (same module object), so ``_target_``
strings in configs, saved checkpoints and imports keep working.
"""

import importlib
from pathlib import Path

import pytest

# Imported first on purpose: the task modules hit a known circular import otherwise.
import MolecularDiffusion.data.dataset  # noqa: F401

PKG = "MolecularDiffusion.modules.tasks"
TASKS = Path(__file__).resolve().parents[1] / "src/MolecularDiffusion/modules/tasks"
FAMILIES = {"flow", "graph", "synth", "latent", "conditional", "reaction", "crystal", "predict", "rl"}
# core modules that stay flat on purpose
FLAT = {"_preprocess_cache", "diffusion", "diffusion_fm_pyg", "metrics", "task"}
ALIASES = sorted(p.stem for p in TASKS.glob("*.py") if p.stem != "__init__" and p.stem not in FLAT)


def test_found_aliases() -> None:
    assert len(ALIASES) >= 30


@pytest.mark.parametrize("old", ALIASES)
def test_flat_path_is_an_alias_of_a_family_module(old: str) -> None:
    mod = importlib.import_module(f"{PKG}.{old}")
    family, short = mod.__name__.removeprefix(f"{PKG}.").split(".")
    assert family in FAMILIES, mod.__name__
    assert old in (short, f"diffusion_{short}")  # the `diffusion_` prefix is dropped inside a family
    assert mod is importlib.import_module(f"{PKG}.{family}.{short}")


def test_module_style_import_through_the_package_still_works() -> None:
    old = ALIASES[0]
    ns: dict = {}
    exec(f"from {PKG} import {old} as m", ns)  # noqa: S102
    assert ns["m"] is importlib.import_module(f"{PKG}.{old}")
    assert ns["m"].__name__ != f"{PKG}.{old}"  # it is the grouped module, not a copy


def test_classes_report_the_canonical_module() -> None:
    from MolecularDiffusion.modules.tasks.regression import ProperyPrediction

    assert ProperyPrediction.__module__ == f"{PKG}.predict.regression"
