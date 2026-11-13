"""Model classes, imported on first use (PEP 562).

Layout: ``edm/`` holds the diffusion / flow-matching wrappers, ``dynamics/`` the
denoising networks they call, and every other sub-package is one architecture.
The old flat paths (``models.egcl``, ``models.en_diffusion``, ...) are aliases
of the moved modules, so existing imports, ``_target_`` strings and pickled
checkpoints keep working.

``import MolecularDiffusion.modules.models.<anything>`` runs this file, and
loading every EDM variant eagerly cost ~6 s and ~900 MB. The public names and
``from MolecularDiffusion.modules.models import EGNN`` keep working; the
submodule behind a name is imported when the name is first read.
"""

import importlib

_LAZY = {
    "EGNN": ".dynamics.egcl",
    "EGNN_dynamics": ".dynamics.egcl",
    "GraphTransformer": ".dynamics.egt",
    "EGT_dynamics": ".dynamics.egt",
    "EnVariationalDiffusion": ".edm.en_diffusion",
    "NoiseModel": ".edm.noisemodel",
}
# `models.en_diffusion` as an attribute: short name -> module path
_SUBMODULES = {path.rsplit(".", 1)[1]: path for path in _LAZY.values()}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(_SUBMODULES[name], __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_LAZY, *_SUBMODULES})
