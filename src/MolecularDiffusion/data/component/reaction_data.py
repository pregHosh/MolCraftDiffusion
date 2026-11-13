"""Alias of ``MolecularDiffusion.data.datasets.reaction_data``.

The module moved into ``data/datasets/``. This path keeps working: it is the same
module object, so imports and ``_target_`` strings written against the old path
still resolve.
"""

import sys

from MolecularDiffusion.data.datasets import reaction_data as _impl

sys.modules[__name__] = _impl
