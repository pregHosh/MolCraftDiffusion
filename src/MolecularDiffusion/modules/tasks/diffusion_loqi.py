"""Alias of ``MolecularDiffusion.modules.tasks.graph.loqi``.

The module moved into ``tasks/graph/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.graph import loqi as _impl

sys.modules[__name__] = _impl
