"""Alias of ``MolecularDiffusion.utils.geometry.geom_constant``.

The module moved into ``geometry/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.geometry import geom_constant as _impl

sys.modules[__name__] = _impl
