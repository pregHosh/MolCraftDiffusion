"""Alias of ``MolecularDiffusion.utils.metrics.geom_metrics``.

The module moved into ``metrics/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.metrics import geom_metrics as _impl

sys.modules[__name__] = _impl
