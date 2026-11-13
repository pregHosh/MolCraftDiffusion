"""Alias of ``MolecularDiffusion.modules.tasks.graph.midi``.

The module moved into ``tasks/graph/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.graph import midi as _impl

sys.modules[__name__] = _impl
