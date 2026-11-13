"""Every name in ``MolecularDiffusion.core.__all__`` exists."""

import MolecularDiffusion.core as core


def test_core_all_resolves() -> None:
    missing = [n for n in core.__all__ if not hasattr(core, n)]
    assert not missing, f"core.__all__ lists missing names: {missing}"
