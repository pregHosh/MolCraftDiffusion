"""Alias of ``MolecularDiffusion.data.features.feature``.

The module moved into ``data/features/``. This path keeps working: it is the same
module object, so imports and ``_target_`` strings written against the old path
still resolve.
"""

import sys

from MolecularDiffusion.data.features import feature as _impl

sys.modules[__name__] = _impl
