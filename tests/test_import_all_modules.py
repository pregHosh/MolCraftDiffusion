"""Every module under ``src/MolecularDiffusion`` imports.

Walks the file tree (not ``pkgutil``, which skips folders without an
``__init__.py``) in a child process, so import-time side effects such as the
banner and global warning filters stay out of the pytest process.

``KNOWN_BROKEN`` is a ratchet: a module listed there must still fail, so the
entry is deleted together with the fix or the file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Failures caused by a missing optional dependency are not bugs.
OPTIONAL_DEPS = {"posecheck", "ocpmodels", "pymatgen", "dgl", "openbabel", "shepherd_score"}

# Modules that cannot be imported today: {dotted name: reason}.
KNOWN_BROKEN: dict[str, str] = {}

_WALK = r"""
import importlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
names = sorted(
    ".".join(p.relative_to(root).with_suffix("").parts)
    for p in (root / "MolecularDiffusion").rglob("*.py")
)
names = [n[:-9] if n.endswith(".__init__") else n for n in names]
bad = {}
for n in names:
    try:
        importlib.import_module(n)
    except BaseException as e:  # noqa: BLE001
        bad[n] = [type(e).__name__, getattr(e, "name", None) or "", str(e)[:200]]
sys.stdout.write("\n@@RESULT@@" + json.dumps({"n": len(names), "bad": bad}))
"""


def _walk() -> dict:
    env = dict(
        os.environ,
        PYTHONPATH=str(SRC),
        CUDA_VISIBLE_DEVICES="",
        MPLCONFIGDIR="/tmp/molcraft-mpl",
    )
    out = subprocess.run(
        [sys.executable, "-c", _WALK, str(SRC)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    ).stdout
    return json.loads(out.rsplit("@@RESULT@@", 1)[1])


def test_every_module_imports() -> None:
    res = _walk()
    assert res["n"] > 500, "module walk found too few files"
    unexpected = {}
    for name, (exc, missing, msg) in res["bad"].items():
        if name in KNOWN_BROKEN:
            continue
        if (
            exc == "ModuleNotFoundError"
            and missing.split(".")[0] in OPTIONAL_DEPS
        ):
            continue
        unexpected[name] = f"{exc}: {msg}"
    assert not unexpected, f"modules fail to import: {unexpected}"
    fixed = [n for n in KNOWN_BROKEN if n not in res["bad"]]
    assert not fixed, f"now import fine, drop from KNOWN_BROKEN: {fixed}"
