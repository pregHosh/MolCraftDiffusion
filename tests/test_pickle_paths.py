"""Old checkpoints pickle ``nn.Module`` classes by dotted path.

``tests/pickle_paths.lst`` lists ``module:Class`` for every top-level
``nn.Module`` under ``modules/`` (one snapshot, taken before the cleanup).
Each path must keep resolving, whatever moves later (a moved module keeps a
``sys.modules`` alias at its old path). Regenerate only to ADD new classes:

    python tests/test_pickle_paths.py --update
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
SNAPSHOT = Path(__file__).parent / "pickle_paths.lst"


def _current() -> set[str]:
    import torch.nn as nn

    root = SRC / "MolecularDiffusion/modules"
    names = sorted(
        ".".join(p.relative_to(SRC).with_suffix("").parts)
        for p in root.rglob("*.py")
    )
    out = set()
    for n in names:
        n = n.removesuffix(".__init__")
        try:
            m = importlib.import_module(n)
        except Exception:  # noqa: BLE001, S112  (broken orphans, see test_import_all_modules)
            continue
        for k, v in vars(m).items():
            if (
                inspect.isclass(v)
                and v.__module__ == n
                and v.__qualname__ == k
                and issubclass(v, nn.Module)
            ):
                out.add(f"{n}:{k}")
    return out


def test_pickled_class_paths_resolve() -> None:
    paths = SNAPSHOT.read_text().split()
    assert len(paths) > 500
    missing = []
    for p in paths:
        mod, cls = p.split(":")
        try:
            getattr(importlib.import_module(mod), cls)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{p} ({type(e).__name__})")
    assert not missing, f"old checkpoint paths no longer resolve: {missing}"


if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.exit(__doc__)
    sys.path.insert(0, str(SRC))
    keep = set(SNAPSHOT.read_text().split()) if SNAPSHOT.exists() else set()
    SNAPSHOT.write_text("\n".join(sorted(keep | _current())) + "\n")
