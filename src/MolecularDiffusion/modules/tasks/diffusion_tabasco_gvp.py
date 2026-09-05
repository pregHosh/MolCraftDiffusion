"""TABASCO + GVP backbone: a novel-model derivation of ``diffusion_tabasco``.

Replaces TABASCO's ``TransformerModule`` backbone with ``GVPBackbone``
(``modules/models/tabasco_gvp/gvp_backbone.py``, wrapping this platform's
existing ``GVPConv``/``NodePositionUpdate``/``EdgeUpdate`` layers). Every
other TABASCO component -- ``FlowMatchingModel``, ``SDEMetricInterpolant``,
``DiscreteInterpolant``, the Euler-Maruyama sampler,
``TabascoNodeDistribution``, the pointcloud<->TensorDict adapters -- is
imported unmodified from
``MolecularDiffusion.modules.tasks.diffusion_tabasco``.

See docs/model_novel/tabasco_gvp/INTEGRATION_PLAN.md ("Integration Plan",
"Derivation Rung") for why this is a subclass that overrides only
``__init__`` rather than a smaller override: ``TabascoDiffusionTask.__init__``
has no seam to override just the backbone-construction line, so this
reproduces its assembly sequence (``diffusion_tabasco.py:280-326``) with one
substitution -- ``GVPBackbone(**gvp_config)`` in place of
``TransformerModule(**transformer_config)``. Same pattern the sibling
``tabasco_egnn`` track already used
(``modules/tasks/diffusion_tabasco_egnn.py``).
"""

from typing import Optional

import torch
import torch.nn as nn

from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
    ModelTaskFactory as TabascoModelTaskFactory,
    PointCloudToTensorDictAdapter,
    TabascoDiffusionTask,
    TensorDictToPointCloudAdapter,
)
from MolecularDiffusion.modules.models.tabasco.flow_model import FlowMatchingModel
from MolecularDiffusion.modules.models.tabasco.flow.interpolate import (
    DiscreteInterpolant,
    SDEMetricInterpolant,
)
from MolecularDiffusion.modules.models.tabasco_gvp.gvp_backbone import GVPBackbone


class TabascoGVPDiffusionTask(TabascoDiffusionTask):
    """TABASCO flow-matching diffusion with a GVP backbone.

    Every method other than ``__init__`` (``forward``, ``predict_and_target``,
    ``evaluate``, ``sample``, ``node_dist_model``, ``n_node_dist``, ``model``,
    ``device``) is inherited unchanged from ``TabascoDiffusionTask`` -- they
    only ever call through ``self.tabasco_model``/``self.net``, which is
    backbone-agnostic.
    """

    def __init__(
        self,
        gvp_config: dict,
        coords_interpolant_config: dict,
        atomics_interpolant_config: dict,
        flow_matching_config: dict,
        num_atom_types: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None,
    ):
        # Reproduces TabascoDiffusionTask.__init__ (diffusion_tabasco.py:280-326)
        # verbatim except for the backbone line below -- there is no hook to
        # call `super().__init__()` and override just that line (see the
        # ledger's Derivation Rung), so `nn.Module.__init__` is called directly.
        nn.Module.__init__(self)

        self.to_tensordict = PointCloudToTensorDictAdapter(num_atom_types)
        self.to_pointcloud = TensorDictToPointCloudAdapter()

        # The one substitution: GVPBackbone in place of TransformerModule.
        net = GVPBackbone(**gvp_config)
        coords_interpolant = SDEMetricInterpolant(**coords_interpolant_config)
        atomics_interpolant = DiscreteInterpolant(**atomics_interpolant_config)

        self.tabasco_model = FlowMatchingModel(
            net=net,
            coords_interpolant=coords_interpolant,
            atomics_interpolant=atomics_interpolant,
            **flow_matching_config,
        )

        self.tabasco_model.set_data_stats({
            'max_num_atoms': dataset_stats.get('max_atoms', 100),
            'num_atoms_histogram': dataset_stats.get('atom_count_histogram', {}),
            'spatial_dim': 3,
            'atom_dim': num_atom_types,
            'all_smiles': dataset_stats.get('all_smiles', [])
        })

        self.atom_vocab = atom_vocab
        self.num_atom_types = num_atom_types
        self.task_type = "diffusion_tabasco_gvp"
        self._dataset_stats = dataset_stats

        # EDM compatibility attributes
        self.prop_dist_model = None
        self.max_n_nodes = dataset_stats.get('max_atoms', 100)
        self._node_dist_model = None


class ModelTaskFactory(TabascoModelTaskFactory):
    """Factory for ``TabascoGVPDiffusionTask``.

    ``compute_dataset_stats`` is inherited unchanged from
    ``TabascoModelTaskFactory`` (diffusion_tabasco.py:191-245) -- it only
    touches ``self.dataset_stats``/``self.train_set``, neither of which
    changes shape here. Only the constructor's config surface
    (``gvp_config`` in place of ``transformer_config``) and ``build()``'s
    target class differ.
    """

    def __init__(
        self,
        task_type: str,
        gvp_config: dict,
        coords_interpolant_config: dict,
        atomics_interpolant_config: dict,
        flow_matching_config: dict,
        num_atom_types: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.gvp_config = gvp_config
        self.coords_interpolant_config = coords_interpolant_config
        self.atomics_interpolant_config = atomics_interpolant_config
        self.flow_matching_config = flow_matching_config
        self.num_atom_types = num_atom_types
        self.dataset_stats = dataset_stats
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab", None)
        self.train_set = train_set
        self.kwargs = kwargs

    def build(self):
        """Build and return the TabascoGVPDiffusionTask."""
        needs_stats = (
            not self.dataset_stats.get("atom_count_histogram")
            or not self.dataset_stats.get("all_smiles")
            or not self.dataset_stats.get("max_atoms")
        )

        if needs_stats:
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set)
            else:
                print(
                    "WARNING: Dataset stats missing and no train_set "
                    "provided. Using defaults/placeholders. This may "
                    "affect generation quality."
                )

        self.task = TabascoGVPDiffusionTask(
            gvp_config=self.gvp_config,
            coords_interpolant_config=self.coords_interpolant_config,
            atomics_interpolant_config=self.atomics_interpolant_config,
            flow_matching_config=self.flow_matching_config,
            num_atom_types=self.num_atom_types,
            dataset_stats=self.dataset_stats,
            atom_vocab=self.atom_vocab,
        )
        return self.task
