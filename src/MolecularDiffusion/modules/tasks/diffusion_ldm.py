"""Alias of ``MolecularDiffusion.modules.tasks.latent.ldm``.

The module moved into ``tasks/latent/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.latent import ldm as _impl

sys.modules[__name__] = _impl
