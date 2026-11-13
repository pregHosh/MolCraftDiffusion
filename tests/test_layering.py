"""Package layering: lower layers must not import higher ones.

Order, lowest first (it matches the imports as they are, not an aspirational plan)::

    utils, callbacks  <  data  <  core  <  modules  <  runmodes  <  cli

A module may import from its own layer or any lower one. Imports that point
upward are recorded in ``layering_baseline.lst`` (``<file> -> <layer> [module|lazy]``;
``lazy`` = inside a function). The test fails on a NEW upward import, and on a
recorded one that has since been fixed (delete its line). Regenerate with::

    python tests/test_layering.py --update
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "MolecularDiffusion"
BASELINE = Path(__file__).with_name("layering_baseline.lst")
RANK = {"utils": 0, "callbacks": 0, "data": 1, "core": 2, "modules": 3, "runmodes": 4, "cli": 5}
SKIP_DIRS = {"__pycache__", "shepherd_score", "xyz2mol"}  # vendored code keeps its own layout


def _absolute(path: Path, level: int, module: str | None) -> str:
    """Dotted module a (possibly relative) ImportFrom points at."""
    if level == 0:
        return module or ""
    pkg = ".".join(path.relative_to(SRC).with_suffix("").parts[:-1])
    base = pkg.rsplit(".", level - 1)[0] if level > 1 else pkg
    return f"{base}.{module}" if module else base


def _layer(dotted: str) -> str | None:
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "MolecularDiffusion" and parts[1] in RANK else None


def _imports(tree: ast.AST, path: Path):
    """Yield (imported layer, lazy) for every import of a MolecularDiffusion layer."""

    def walk(node: ast.AST, lazy: bool):
        for child in ast.iter_child_nodes(node):
            inner = lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            if isinstance(child, ast.ImportFrom):
                mod = _absolute(path, child.level, child.module)
                layers = {_layer(mod)}
                if mod == "MolecularDiffusion":  # `from MolecularDiffusion import utils`
                    layers = {a.name for a in child.names if a.name in RANK}
                for layer in layers - {None}:
                    yield layer, lazy
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    layer = _layer(alias.name)
                    if layer:
                        yield layer, lazy
            yield from walk(child, inner)

    yield from walk(tree, False)


def upward_imports() -> set[str]:
    found = set()
    for path in sorted(PKG.rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        rel = path.relative_to(PKG)
        mine = rel.parts[0]
        if mine not in RANK:
            continue
        for layer, lazy in _imports(ast.parse(path.read_text()), path):
            if RANK[layer] > RANK[mine]:
                found.add(f"{rel.as_posix()} -> {layer} [{'lazy' if lazy else 'module'}]")
    return found


def test_no_new_upward_imports() -> None:
    new = upward_imports() - set(BASELINE.read_text().splitlines())
    assert not new, "new upward imports (lower layer importing a higher one):\n  " + "\n  ".join(sorted(new))


def test_baseline_has_no_stale_entries() -> None:
    stale = set(BASELINE.read_text().splitlines()) - upward_imports()
    assert not stale, "fixed upward imports, delete these lines from layering_baseline.lst:\n  " + "\n  ".join(sorted(stale))


if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.exit(__doc__)
    BASELINE.write_text("\n".join(sorted(upward_imports())) + "\n")
