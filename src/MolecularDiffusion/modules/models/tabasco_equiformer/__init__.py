"""tabasco_equiformer: TABASCO's flow-matching recipe with an SO(3)
tensor-product equivariant EquiformerV2 backbone in place of its plain
augmented ``TransformerModule``.

Novel-model ablation track. See
docs/model_novel/tabasco_equiformer/INTEGRATION_PLAN.md for the hypothesis,
ablation contract, and hyperparameter provenance.

The backbone is a thin wrapper around this platform's existing
``MolecularDiffusion.modules.models.shepherd_arch.equiformer_v2_encoder.
EquiformerV2`` (``equiformer_v2_s`` layer family) -- nothing here
reimplements those layers.
"""

from .equiformer_backbone import EquiformerV2TabascoBackbone

__all__ = ["EquiformerV2TabascoBackbone"]
