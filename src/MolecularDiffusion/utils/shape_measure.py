"""Alias of ``MolecularDiffusion.utils.geometry.shape_measure``.

The module moved into ``geometry/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.geometry import shape_measure as _impl

sys.modules[__name__] = _impl
