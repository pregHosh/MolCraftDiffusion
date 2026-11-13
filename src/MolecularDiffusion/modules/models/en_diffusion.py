"""Alias of ``MolecularDiffusion.modules.models.edm.en_diffusion``.

The module moved into ``edm/``. This path keeps working: it is the same module
object, so imports, ``_target_`` strings and checkpoints pickled under the old
path still resolve.
"""

import sys

from MolecularDiffusion.modules.models.edm import en_diffusion as _impl

sys.modules[__name__] = _impl
