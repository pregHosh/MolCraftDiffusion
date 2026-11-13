"""Alias of ``MolecularDiffusion.modules.models.dynamics.transformer_dynamics``.

The module moved into ``dynamics/``. This path keeps working: it is the same module
object, so imports, ``_target_`` strings and checkpoints pickled under the old
path still resolve.
"""

import sys

from MolecularDiffusion.modules.models.dynamics import transformer_dynamics as _impl

sys.modules[__name__] = _impl
