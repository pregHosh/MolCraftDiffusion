"""Alias of ``MolecularDiffusion.modules.tasks.reaction.ts_generator``.

The module moved into ``tasks/reaction/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.reaction import ts_generator as _impl

sys.modules[__name__] = _impl
