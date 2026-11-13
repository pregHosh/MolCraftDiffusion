"""Alias of ``MolecularDiffusion.data.integrations.shepherd2``.

The module moved into ``data/integrations/``. This path keeps working: it is the same
module object, so imports and ``_target_`` strings written against the old path
still resolve.
"""

import sys

from MolecularDiffusion.data.integrations import shepherd2 as _impl

sys.modules[__name__] = _impl
