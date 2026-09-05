"""tabasco_egnn: TABASCO's flow-matching recipe with an equivariant EGNN
backbone in place of its plain augmented ``TransformerModule``.

Novel-model ablation track. See
docs/model_novel/tabasco_egnn/INTEGRATION_PLAN.md for the hypothesis,
ablation contract, and hyperparameter provenance.

The backbone is a thin wrapper around this platform's existing
``MolecularDiffusion.modules.models.egcl.EGNN`` (built from ``GCL`` /
``EquivariantUpdate`` / ``EquivariantBlock``,
``modules/layers/conv.py``) -- nothing here reimplements those layers.
"""

from .egnn_backbone import EGNNBackbone

__all__ = ["EGNNBackbone"]
