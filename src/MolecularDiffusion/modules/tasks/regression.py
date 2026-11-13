"""Alias of ``MolecularDiffusion.modules.tasks.predict.regression``.

The module moved into ``tasks/predict/``. This path keeps working: it is the same
module object, so imports, ``_target_`` strings and checkpoints written against
the old path still resolve.
"""

import sys

from MolecularDiffusion.modules.tasks.predict import regression as _impl

sys.modules[__name__] = _impl
