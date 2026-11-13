"""
MolecularDiffusion - A unified generative AI framework for 3D molecular generation.
"""

import os as _os
import sys as _sys
from importlib.metadata import PackageNotFoundError, version

# macOS only: let aten ops Apple MPS lacks fall back to CPU instead of raising.
# torch reads this when it is first imported, so it has to be set here.
if _sys.platform == "darwin":
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    __version__ = version("molcraftdiffusion")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Thanapat Worakul"
__email__ = "thanapat.worakul@epfl.ch"

# Submodules are loaded lazily to avoid import errors when optional
# dependencies (xyz2mol, xtb, openbabel) are not installed.
_submodules = ["core", "data", "modules", "utils", "callbacks", "runmodes"]


def __getattr__(name: str):
    if name in _submodules:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = _submodules

# Register the ${asset:...} OmegaConf resolver at package import, so it is
# live for both the CLI (cli/_hydra.py compose) and plain library use
# (OmegaConf.load in a notebook). assets.py imports only stdlib + yaml, so
# this does not defeat the lazy-submodule scheme above.
from MolecularDiffusion.assets import register as _register_asset_resolver

_register_asset_resolver()
