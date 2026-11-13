"""The banner prints once per CLI run and not on library import."""

import os
import subprocess
import sys
from pathlib import Path

import MolecularDiffusion.cli.main as cli_main

SRC = Path(__file__).resolve().parents[1] / "src"
GLYPH = max((l.strip() for l in cli_main._BANNER.splitlines()), key=len)  # one art line
ENV = dict(os.environ, PYTHONPATH=str(SRC), CUDA_VISIBLE_DEVICES="")
# what the `MolCraftDiff` console script does
ENTRY = "from MolecularDiffusion.cli.main import cli; cli()"


def _run(*args: str) -> str:
    return subprocess.run(
        [sys.executable, "-W", "ignore", *args],
        env=ENV, capture_output=True, text=True, timeout=300, check=True,
    ).stdout


def test_library_import_is_silent() -> None:
    assert GLYPH not in _run("-c", "import MolecularDiffusion")


def test_cli_help_and_version_show_the_banner_once() -> None:
    for flag in ("--help", "--version"):
        assert _run("-c", ENTRY, flag).count(GLYPH) == 1, flag
