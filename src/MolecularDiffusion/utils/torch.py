"""Alias of ``MolecularDiffusion.utils.runtime.torch``.

The module moved into ``runtime/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.runtime import torch as _impl

sys.modules[__name__] = _impl
