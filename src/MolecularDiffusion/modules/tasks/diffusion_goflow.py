"""Alias of ``MolecularDiffusion.modules.tasks.reaction.goflow``.

The module moved into ``tasks/reaction/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.reaction import goflow as _impl

sys.modules[__name__] = _impl
