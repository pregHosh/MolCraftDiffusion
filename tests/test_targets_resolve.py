"""Every ``_target_`` in the bundled Hydra configs resolves to a symbol.

Guards renames and moves: a stale dotted path only fails at launch time
otherwise. ``scripts.*`` targets are user-local classes resolved from the
caller's working directory (see ``cli/_hydra.py``), so they are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from hydra.utils import get_object

CONFIGS = (
    Path(__file__).resolve().parents[1] / "src/MolecularDiffusion/configs"
)
_TARGET = re.compile(r"^\s*(?:-\s*)?_target_:\s*['\"]?([\w.]+)", re.M)


def _targets() -> dict[str, str]:
    found: dict[str, str] = {}
    for f in sorted(CONFIGS.rglob("*.yaml")):
        for t in _TARGET.findall(f.read_text()):
            found.setdefault(t, f.relative_to(CONFIGS).as_posix())
    return {t: f for t, f in found.items() if not t.startswith("scripts.")}


TARGETS = _targets()


def test_found_targets() -> None:
    assert len(TARGETS) > 80


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_target_resolves(target: str) -> None:
    assert get_object(target) is not None, f"in {TARGETS[target]}"
