"""``utils`` groups its flat modules into ``geometry/``, ``chem/``, ``metrics/``, ``runtime/``.

The old flat paths stay as aliases (same module object), so imports and
``_target_`` strings written against them keep working.
"""

import importlib
from pathlib import Path

import pytest

PKG = "MolecularDiffusion.utils"
UTILS = Path(__file__).resolve().parents[1] / "src/MolecularDiffusion/utils"
GROUPS = {"geometry", "chem", "metrics", "runtime"}
# files that stay flat on purpose
FLAT = {"diffusion_utils", "discrete_diffusion", "file", "geom_constraint", "io", "plot_function", "pretty"}
ALIASED = sorted(p.stem for p in UTILS.glob("*.py") if p.stem != "__init__" and p.stem not in FLAT)


def test_found_aliases() -> None:
    assert len(ALIASED) >= 10


@pytest.mark.parametrize("name", ALIASED)
def test_flat_path_is_an_alias_of_a_grouped_module(name: str) -> None:
    mod = importlib.import_module(f"{PKG}.{name}")
    group = mod.__name__.removeprefix(f"{PKG}.").split(".")[0]
    assert group in GROUPS, mod.__name__
    assert mod is importlib.import_module(f"{PKG}.{group}.{name}")


def test_functions_report_the_canonical_module() -> None:
    from MolecularDiffusion.utils import coord2diff

    assert coord2diff.__module__ == f"{PKG}.geometry.geom_utils"


def test_data_files_still_found_after_the_move() -> None:
    # these two modules locate their data through __file__; the data stays in utils/
    from MolecularDiffusion.utils.chem import sascore

    assert (Path(sascore.path) / "fpscores.pkl.gz").is_file()
    if (UTILS / "scscore.py").is_file():  # not in every checkout
        scscore = importlib.import_module(f"{PKG}.chem.scscore")
        assert scscore._MODEL.is_file()
