"""Alias of ``MolecularDiffusion.modules.tasks.conditional.kgdiff``.

The module moved into ``tasks/conditional/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.conditional import kgdiff as _impl

sys.modules[__name__] = _impl
