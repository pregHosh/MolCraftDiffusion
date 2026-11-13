"""``data`` groups the old flat ``component/`` modules into ``datasets/``, ``features/``, ``integrations/``.

The old ``data.component.*`` paths stay as aliases (same module object), so imports,
``_target_`` strings in configs and caches pickled under them keep working.
"""

import importlib
from pathlib import Path

import pytest

# Imported first on purpose: component.dataset first hits a known circular import.
import MolecularDiffusion.data.dataset  # noqa: F401

PKG = "MolecularDiffusion.data"
COMPONENT = Path(__file__).resolve().parents[1] / "src/MolecularDiffusion/data/component"
GROUPS = {"datasets", "features", "integrations"}
STUBS = sorted(p.stem for p in COMPONENT.glob("*.py") if p.stem != "__init__")


def test_component_only_holds_aliases() -> None:
    assert len(STUBS) >= 14


@pytest.mark.parametrize("name", STUBS)
def test_component_path_is_an_alias_of_a_grouped_module(name: str) -> None:
    mod = importlib.import_module(f"{PKG}.component.{name}")
    group = mod.__name__.removeprefix(f"{PKG}.").split(".")[0]
    assert group in GROUPS, mod.__name__
    assert mod is importlib.import_module(f"{PKG}.{group}.{name}")


def test_classes_report_the_canonical_module() -> None:
    from MolecularDiffusion.data.component.graph3d_dataset import Graph3DDataset

    assert Graph3DDataset.__module__ == f"{PKG}.datasets.graph3d_dataset"
