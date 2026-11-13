"""Alias of ``MolecularDiffusion.data.datasets.graph3d_dataset``.

The module moved into ``data/datasets/``. This path keeps working: it is the same
module object, so imports and ``_target_`` strings written against the old path
still resolve.
"""

import sys

from MolecularDiffusion.data.datasets import graph3d_dataset as _impl

sys.modules[__name__] = _impl
