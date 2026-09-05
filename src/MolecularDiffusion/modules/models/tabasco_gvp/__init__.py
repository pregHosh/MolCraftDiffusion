"""tabasco_gvp: TABASCO's flow-matching recipe with an equivariant GVP
backbone in place of its plain augmented ``TransformerModule``.

Novel-model ablation track. See
docs/model_novel/tabasco_gvp/INTEGRATION_PLAN.md for the hypothesis,
ablation contract, and hyperparameter provenance.

The backbone is built from this platform's existing
``GVPConv``/``NodePositionUpdate``/``EdgeUpdate``
(``modules/layers/gvp/gvp.py``) -- nothing here reimplements those layers.
"""

from .gvp_backbone import GVPBackbone

__all__ = ["GVPBackbone"]
