"""Alias of ``MolecularDiffusion.modules.tasks.flow.flowmol_graph3d``.

The module moved into ``tasks/flow/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.flow import flowmol_graph3d as _impl

sys.modules[__name__] = _impl
