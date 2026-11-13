"""Alias of ``MolecularDiffusion.modules.models.dynamics.equiformer_v2_backbone``.

The module moved into ``dynamics/``. This path keeps working: it is the same module
object, so imports, ``_target_`` strings and checkpoints pickled under the old
path still resolve.
"""

import sys

from MolecularDiffusion.modules.models.dynamics import equiformer_v2_backbone as _impl

sys.modules[__name__] = _impl
