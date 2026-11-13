"""Alias of ``MolecularDiffusion.utils.runtime.comm``.

The module moved into ``runtime/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.runtime import comm as _impl

sys.modules[__name__] = _impl
