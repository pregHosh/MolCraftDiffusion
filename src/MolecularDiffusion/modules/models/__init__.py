"""Model classes, imported on first use (PEP 562).

``import MolecularDiffusion.modules.models.<anything>`` runs this file, and
loading every EDM variant eagerly cost ~6 s and ~900 MB. The public names and
``from MolecularDiffusion.modules.models import EGNN`` keep working; the
submodule behind a name is imported when the name is first read.
"""

import importlib

_LAZY = {
    "EGNN": ".egcl",
    "EGNN_dynamics": ".egcl",
    "GraphTransformer": ".egt",
    "EGT_dynamics": ".egt",
    "EnVariationalDiffusion": ".en_diffusion",
    "NoiseModel": ".noisemodel",
}
_SUBMODULES = {m[1:] for m in _LAZY.values()}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    elif name in _SUBMODULES:  # `models.en_diffusion` as an attribute
        value = importlib.import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_LAZY, *_SUBMODULES})
