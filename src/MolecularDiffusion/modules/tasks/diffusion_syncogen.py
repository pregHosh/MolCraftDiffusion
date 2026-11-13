"""Alias of ``MolecularDiffusion.modules.tasks.synth.syncogen``.

The module moved into ``tasks/synth/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.synth import syncogen as _impl

sys.modules[__name__] = _impl
