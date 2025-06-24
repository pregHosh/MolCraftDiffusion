"""Test that the package can be imported correctly."""

import pytest


def test_import_moleculardiffusion():
    """Test that MolecularDiffusion can be imported."""
    import MolecularDiffusion
    assert MolecularDiffusion.__version__ == "0.1.0"


def test_import_submodules():
    """Test that submodules can be imported."""
    from MolecularDiffusion import core, data, modules, utils, callbacks, runmodes
    assert core is not None
    assert data is not None
    assert modules is not None
    assert utils is not None
    assert callbacks is not None
    assert runmodes is not None





if __name__ == "__main__":
    pytest.main([__file__]) 