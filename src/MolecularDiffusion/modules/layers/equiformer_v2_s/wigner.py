"""Alias of ``equiformer_v2.wigner``.

``equiformer_v2_s`` carried a copy of this file that is identical in behaviour.
Both import paths now give the same module object, so classes keep one identity
and checkpoints pickled under either path keep loading.
"""

import sys

from MolecularDiffusion.modules.layers.equiformer_v2 import wigner as _impl

sys.modules[__name__] = _impl
