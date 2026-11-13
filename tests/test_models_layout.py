"""``modules/models`` groups its flat model files into ``edm/`` and ``dynamics/``.

The old flat paths stay as aliases (same module object), so imports,
``_target_`` strings and old checkpoints keep resolving.
"""

import importlib
from pathlib import Path

import pytest

PKG = "MolecularDiffusion.modules.models"
MODELS = Path(__file__).resolve().parents[1] / "src/MolecularDiffusion/modules/models"
# private helper modules (``_protocols``) are real modules, not aliases
FLAT = sorted(p.stem for p in MODELS.glob("*.py") if p.stem != "__init__" and not p.stem.startswith("_"))


def test_flat_files_exist() -> None:
    assert len(FLAT) >= 10


@pytest.mark.parametrize("name", FLAT)
def test_flat_path_is_an_alias_of_a_grouped_module(name: str) -> None:
    mod = importlib.import_module(f"{PKG}.{name}")
    group = mod.__name__.removeprefix(f"{PKG}.").split(".")[0]
    assert group in {"edm", "dynamics"}, mod.__name__
    assert mod is importlib.import_module(f"{PKG}.{group}.{name}")


def test_public_classes_live_under_the_new_paths() -> None:
    from MolecularDiffusion.modules.models import EGNN, EnVariationalDiffusion

    assert EGNN.__module__ == f"{PKG}.dynamics.egcl"
    assert EnVariationalDiffusion.__module__ == f"{PKG}.edm.en_diffusion"
