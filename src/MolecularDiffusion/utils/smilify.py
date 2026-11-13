"""Alias of ``MolecularDiffusion.utils.chem.smilify``.

The module moved into ``chem/``. This path keeps working: it is the same module
object, so imports and ``_target_`` strings written against the old path still
resolve.
"""

import sys

from MolecularDiffusion.utils.chem import smilify as _impl

sys.modules[__name__] = _impl
