"""Alias of ``MolecularDiffusion.modules.tasks.flow.tabasco``.

The module moved into ``tasks/flow/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.flow import tabasco as _impl

sys.modules[__name__] = _impl
