"""Alias of ``MolecularDiffusion.data.datasets.pointcloud``.

The module moved into ``data/datasets/``. This path keeps working: it is the same
module object, so imports and ``_target_`` strings written against the old path
still resolve.
"""

import sys

from MolecularDiffusion.data.datasets import pointcloud as _impl

sys.modules[__name__] = _impl
