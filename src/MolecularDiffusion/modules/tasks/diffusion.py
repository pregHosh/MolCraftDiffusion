import logging
import math
import time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from ase.data import chemical_symbols
from rdkit import Chem

from MolecularDiffusion import core
from MolecularDiffusion.callbacks import SP_regularizer
from MolecularDiffusion.modules.layers import common, functional
from MolecularDiffusion.modules.tasks.regression import MLPRegressor_padded, MLPRegressor_pernode
from MolecularDiffusion.modules.models.en_diffusion import (
    DistributionNodes,
    DistributionProperty,
    split_frame,
)
from MolecularDiffusion.utils import (
    assert_correctly_masked,
    assert_mean_zero_with_mask,
    check_mask_correct,
    check_stability,
    compute_mean_mad_from_dataloader,
    compute_mean_mad_from_dataloader,
    prepare_context,
    prepare_context_pyG,
    random_rotation,
    remove_mean_pyG,
    remove_mean_with_mask,
    sample_center_gravity_zero_gaussian_with_mask,
    sample_gaussian_with_mask,
    sample_uniform_rotation_matrices,
)

from . import _preprocess_cache as _ppcache
from .task import Task, _get_criterion_name, _get_metric_name


logger = logging.getLogger(__name__)

_GEOM_CONSTRAINT_CFG_KEYS = {
    "connector_dicts",
    "constraint_strength",
    "scale_factor",
}


def _without_geometric_constraint_cfgs(cfgs):
    if not cfgs:
        return {}
    return {
        key: value
        for key, value in cfgs.items()
        if key not in _GEOM_CONSTRAINT_CFG_KEYS
    }


@core.Registry.register("GeomMolecularGenerative")
class GeomMolecularGenerative(Task, core.Configurable):

    _option_members = {"task"}

    def __init__(
        self,
        diffusion_model,
        node_dist_model=None,
        prop_dist_model=None,
        n_node_dist: Dict = {},
        augment_noise: float = 0,
        data_augmentation: bool = False,
        num_random_augmentations: int = 0,
        condition: List = [],
        normalize_condition: str = None,
        sp_regularizer: SP_regularizer = None,
        reference_indices: List = None, # For OP task
        reference_freeze_mode: str = "all",
    ):
        """
        Generative Diffusion model for molecular structures.
        Parameters:
        - diffusion_model: The dynamic functional model for diffusion.
        - node_dist_model (Optional[NodeDistributionModel]): The model for number of node distribution. Default is None.
        - prop_dist_model (Optional[PropertyDistributionModel]): The model for property distribution. Default is None.
        - n_node_dist (Dict): The distribution of number of nodes. Default is {}.
        - augment_noise (float): The amount of noise to add to the coordinates for data augmentation. Default is 0.
        - data_augmentation (bool): Whether to apply data augmentation by symmetry operations. Default is False.
        - num_random_augmentations (int): Dense/pointcloud path only. When
          > 0, replaces the single in-place rotation `data_augmentation`
          applies with TABASCO's batch-expansion scheme
          (`modules/models/tabasco/flow_model.py::FlowMatchingModel.forward`):
          each molecule in the batch is replicated into
          `num_random_augmentations + 1` copies, each independently
          rotated by a Haar-uniform SO(3) sample
          (`sample_uniform_rotation_matrices`, Haar-uniform unlike
          `random_rotation`'s Euler-angle composition). Every copy
          contributes its own loss term in the same step -- this
          multiplies the per-step compute cost by
          `num_random_augmentations + 1`. Takes precedence over
          `data_augmentation` when both are set (mutually exclusive, not
          additive). Default 0 preserves every existing config's
          behavior exactly. Not implemented for the PyG (`"graph" in
          batch`) path -- out of scope, no current caller needs it.
        - condition (List): The list of conditions for the model. Default is [].
        - normalize_condition (str): The normalization method for the condition. Default is None. [None, "maxmin", "mad"]
        - sp_regularizer (SP_regularizer): The self-pace learning regularizer for the model. Default is None.
        """
        super(GeomMolecularGenerative, self).__init__()
        if reference_freeze_mode not in {"all", "features_only"}:
            raise ValueError(
                "reference_freeze_mode must be one of {'all', 'features_only'}, "
                f"got {reference_freeze_mode!r}"
            )
        self.model = diffusion_model
        self.node_dist_model = node_dist_model
        self.prop_dist_model = prop_dist_model
        self.n_node_dist = n_node_dist
        self.augment_noise = augment_noise
        self.data_augmentation = data_augmentation
        self.num_random_augmentations = num_random_augmentations
        self.condition = condition
        self.sp_regularizer = sp_regularizer
        self.reference_indices = reference_indices
        self.reference_freeze_mode = reference_freeze_mode
        self.reference_scaffold = None  # Mean scaffold geometry, set during preprocess
        self.reference_feature_stats = None
        self.normalize_condition = normalize_condition
        
        self.n_dim_data = self.model.in_node_nf 
        self.n_atom_types = self.model.in_node_nf - len(self.model.extra_norm_values)
        if self.model.include_charges:
            self.n_atom_types -= 1
        
        
        
    def _split_frame(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split unnormalized chain frames along their feature axis.

        Frames are laid out ``[x | h_cat | h_int]``; ``h_int`` is the
        model's integer (atomic-number) block, which is empty for models
        built with ``include_charges=False``. Derived from the model's own
        dimensions rather than assuming the last column is the charge, and
        kept on the task so it works for every model class this task and
        its subclasses drive. ``include_charges`` is read with a ``True``
        default so a model that never declares it keeps the historical
        one-column layout.
        """
        return split_frame(self.model, frames)

    def _charge_channel(self, charges: torch.Tensor) -> torch.Tensor:
        """Shape ``batch["charges"]`` into the model's integer feature block.

        ``(B, N) -> (B, N, 1)`` for models that carry an atomic-number
        channel, and ``(B, N, 0)`` for those built with
        ``include_charges=False`` — an empty block the model concatenates,
        masks and sums without any further branching.
        """
        charges = charges.unsqueeze(2)
        if getattr(self.model, "include_charges", True):
            return charges
        return charges[:, :, :0]

    def _compute_mean_scaffold(self, train_set):
        """Find the medoid scaffold from training molecules.

        Iterates the training set, collects scaffold positions at
        self.reference_indices, computes the mean position, then returns the
        actual molecule whose scaffold is closest to that mean (the medoid).
        Using a real molecule's geometry avoids the coordinate collapse that
        occurs when averaging positions across diverse orientations.

        Returns a normalized tensor of shape (1, n_ref, 3 + n_feat + 1), or
        None when no valid molecules are found or reference_indices is not set.
        """
        ref_idx = self.reference_indices
        if not ref_idx:
            return None
        max_idx = max(ref_idx)
        nc, nf, nch = self.model.norm_values

        all_pos = []   # list of (n_ref, 3) tensors
        all_x = []     # list of (n_ref, n_feat) tensors
        all_ch = []    # list of (n_ref,) tensors

        _t = time.perf_counter()
        for sample in train_set:
            if "graph" in sample:
                g = sample["graph"]
                pos = g.pos
                x = g.x
                charges = g.atomic_numbers
            else:
                pos = sample["coords"]
                x = sample["node_feature"]
                charges = sample.get("charges", None)
                if pos.dim() == 3:
                    pos = pos.squeeze(0)
                if x.dim() == 3:
                    x = x.squeeze(0)
                if charges is not None and charges.dim() == 2:
                    charges = charges.squeeze(0)

            if pos.shape[0] <= max_idx:
                continue

            all_pos.append(pos[ref_idx].float().cpu())
            all_x.append(x[ref_idx].float().cpu())
            all_ch.append(
                charges[ref_idx].float().cpu()
                if charges is not None
                else torch.zeros(len(ref_idx))
            )

        if not all_pos:
            logger.warning(
                "preprocess: no valid molecules found for scaffold computation"
            )
            return None

        pos_stack = torch.stack(all_pos)   # (N, n_ref, 3)
        mean_pos = pos_stack.mean(0)        # (n_ref, 3)

        # Find medoid: molecule whose scaffold has the smallest mean per-atom
        # distance to the mean position.  This gives a real, non-collapsed
        # geometry that is representative of the training distribution.
        sq_dists = (pos_stack - mean_pos.unsqueeze(0)).pow(2).sum(-1).mean(-1)
        medoid_idx = int(sq_dists.argmin())

        med_pos = all_pos[medoid_idx]   # (n_ref, 3)
        x_ref = all_x[medoid_idx]       # (n_ref, n_feat)
        ch_ref = all_ch[medoid_idx]     # (n_ref,)

        med_pos_n = med_pos / nc
        x_ref_n = x_ref / nf
        ch_ref_n = (ch_ref / nch).unsqueeze(-1)

        xh_ref = torch.cat([med_pos_n, x_ref_n, ch_ref_n], dim=-1)
        logger.info(
            f"preprocess: medoid scaffold (idx={medoid_idx}) from "
            f"{len(all_pos)}/{len(train_set)} molecules "
            f"in {time.perf_counter() - _t:.2f}s, shape {list(xh_ref.shape)}"
        )
        return xh_ref.unsqueeze(0).detach().cpu()  # (1, n_ref, D)

    def _compute_reference_feature_stats(self, train_set):
        """Compute modal frozen node features and atomic numbers."""
        ref_idx = self.reference_indices
        if not ref_idx:
            return None
        max_idx = max(ref_idx)
        _t = time.perf_counter()
        feature_rows = [[] for _ in ref_idx]
        charge_rows = [[] for _ in ref_idx]

        for sample in train_set:
            if "graph" in sample:
                g = sample["graph"]
                x = g.x
                charges = g.atomic_numbers
            else:
                x = sample["node_feature"]
                charges = sample.get("charges", None)
                if x.dim() == 3:
                    x = x.squeeze(0)
                if charges is not None and charges.dim() == 2:
                    charges = charges.squeeze(0)

            if x.shape[0] <= max_idx:
                continue
            if charges is None:
                charges = torch.zeros(x.shape[0], device=x.device)

            for out_idx, atom_idx in enumerate(ref_idx):
                feature_rows[out_idx].append(x[atom_idx].detach().float().cpu())
                charge_rows[out_idx].append(charges[atom_idx].detach().float().cpu().view(()))

        if not feature_rows or any(len(rows) == 0 for rows in feature_rows):
            logger.warning(
                "preprocess: no valid molecules found for reference feature stats"
            )
            return None

        modal_features = []
        modal_charges = []
        for features_at_idx, charges_at_idx in zip(feature_rows, charge_rows):
            features = torch.stack(features_at_idx, dim=0)
            charges = torch.stack(charges_at_idx, dim=0)

            uniq_features, feature_counts = torch.unique(
                features, dim=0, return_counts=True
            )
            modal_features.append(uniq_features[torch.argmax(feature_counts)])

            uniq_charges, charge_counts = torch.unique(charges, return_counts=True)
            modal_charges.append(uniq_charges[torch.argmax(charge_counts)])

        _, nf, nch = self.model.norm_values
        node_feature = torch.stack(modal_features, dim=0) / nf
        atomic_numbers = (torch.stack(modal_charges, dim=0) / nch).unsqueeze(-1)
        stats = {
            "reference_indices": list(ref_idx),
            "node_feature": node_feature.unsqueeze(0).detach().cpu(),
            "atomic_numbers": atomic_numbers.unsqueeze(0).detach().cpu(),
        }
        logger.info(
            f"preprocess: modal reference feature stats from "
            f"{len(feature_rows[0])}/{len(train_set)} molecules in "
            f"{time.perf_counter() - _t:.2f}s"
        )
        return stats

    def preprocess(
        self,
        train_set=None,
    ):
        if train_set is None:
            self.atomic_numbers = []
            self.atom_decoder = []
            self.atom_encoder = {}
            self.dataset_smiles_list = []
            self.max_n_nodes = 0
            self.n_node_dist = {}
            return

        if len(train_set) == 0:
            raise ValueError("Training set is empty. check the data path and format.")

        # Atom vocab (cheap, always recompute)
        self.atomic_numbers = train_set.atom_types()
        self.atom_decoder = [
            chemical_symbols[number]
            for number in self.atomic_numbers
            if number < len(chemical_symbols)
        ]
        self.atom_encoder = {symbol: i for i, symbol in enumerate(self.atom_decoder)}

        # Try disk cache first — covers the slow paths (n_node_dist
        # iteration, full-chunk property scan, RDKit canonicalisation)
        cached = _ppcache.try_load(train_set, list(self.condition))
        if cached is not None:
            self.n_node_dist = cached["n_node_dist"]
            self.max_n_nodes = cached["max_n_nodes"]
            self.dataset_smiles_list = cached["dataset_smiles_list"]
            if self.node_dist_model is None:
                self.node_dist_model = DistributionNodes(self.n_node_dist)
            if (self.prop_dist_model is None
                    and len(self.condition) > 0
                    and cached.get("prop_dist_model") is not None):
                self.prop_dist_model = cached["prop_dist_model"]
                self.property_norms = cached["property_norms"]
                self.prop_dist_model.set_normalizer(self.property_norms)
            if self.reference_indices is not None:
                self.reference_scaffold = self._compute_mean_scaffold(train_set)
                self.reference_feature_stats = self._compute_reference_feature_stats(
                    train_set
                )
            return

        # Bulk path: derive n_node_dist from dataset attributes when available.
        # Falls back to a per-sample loop only when the dataset does not
        # expose `n_atoms` (e.g. the legacy padded-tensor format).
        smiles_raw: List = []
        bulk = _ppcache.bulk_n_node_stats(train_set)
        if bulk is not None:
            self.n_node_dist, self.max_n_nodes, smiles_raw = bulk
        elif "graph" in train_set[0]:
            logger.info(f"preprocess: falling back to per-sample loop for n_node_dist ({len(train_set):,} samples)")
            _t = time.perf_counter()
            self.max_n_nodes = 0
            self.n_node_dist = {}
            for sample in train_set:
                smiles_raw.append(getattr(sample["graph"], "smiles", None))
                n_node = int(sample["graph"].natoms)
                if n_node > self.max_n_nodes:
                    self.max_n_nodes = n_node
                self.n_node_dist[n_node] = self.n_node_dist.get(n_node, 0) + 1
            logger.info(f"preprocess: per-sample loop (graph) done in {time.perf_counter() - _t:.2f}s")
        else:
            self.max_n_nodes = train_set[0]["coords"].size()[0]
            if not self.n_node_dist:
                logger.info(f"preprocess: falling back to per-sample loop for n_node_dist ({len(train_set):,} samples)")
                _t = time.perf_counter()
                self.n_node_dist = {}
                for sample in train_set:
                    n_node = int(sample["node_mask"].nonzero().size(0))
                    self.n_node_dist[n_node] = self.n_node_dist.get(n_node, 0) + 1
                logger.info(f"preprocess: per-sample loop (pointcloud) done in {time.perf_counter() - _t:.2f}s")

        if self.node_dist_model is None:
            logger.info("Creating node distribution model")
            self.node_dist_model = DistributionNodes(self.n_node_dist)

        if (self.prop_dist_model is None) and len(self.condition) > 0:
            logger.info("Creating property distribution model")
            _t = time.perf_counter()
            base, subset_indices = _ppcache.resolve_dataset_and_indices(train_set)
            prop_indices = _ppcache.property_sample_indices(len(train_set), subset_indices)
            if prop_indices is not subset_indices:
                logger.info(
                    f"preprocess: using {len(prop_indices):,}/{len(train_set):,} "
                    "samples for property distribution"
                )
            num_atoms = _ppcache.subset_tensor(base.num_atoms, prop_indices)
            props = []
            for task in self.condition:
                if task not in base.targets.keys():
                    raise ValueError(f"Task {task} not found in dataset")
                try:
                    props.append(_ppcache.get_property_subset(base, task, prop_indices))
                except Exception as e:
                    raise ValueError(f"Fail {task} to get property from dataset due to {e}")

            props = torch.stack(props)
            self.prop_dist_model = DistributionProperty(
                num_atoms, props, self.condition, num_bins=10
            )
            self.property_norms = compute_mean_mad_from_dataloader(props, self.condition)
            self.prop_dist_model.set_normalizer(self.property_norms)
            logger.info(f"preprocess: property distribution model built in {time.perf_counter() - _t:.2f}s")

        # Canonicalise SMILES (parallel for large sets)
        if not smiles_raw:
            base, subset_indices = _ppcache.resolve_dataset_and_indices(train_set)
            smiles_raw = _ppcache.subset_sequence(getattr(base, "smiles_list", None), subset_indices) or []
        self.dataset_smiles_list = _ppcache.canonical_smiles_set(smiles_raw)

        if self.reference_indices is not None:
            self.reference_scaffold = self._compute_mean_scaffold(train_set)
            self.reference_feature_stats = self._compute_reference_feature_stats(
                train_set
            )

        # Persist for subsequent runs
        _ppcache.save(
            train_set,
            list(self.condition),
            {
                "n_node_dist": self.n_node_dist,
                "max_n_nodes": self.max_n_nodes,
                "dataset_smiles_list": self.dataset_smiles_list,
                "prop_dist_model": self.prop_dist_model,
                "property_norms": getattr(self, "property_norms", None),
            },
        )


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        _loss, _metric = self.density_estimation(batch)
        all_loss += _loss
        metric.update(_metric)

        return all_loss, metric

    def _apply_n_fold_rotation_augmentation(
        self, x, h, node_mask, charges, edge_mask
    ):
        """TABASCO-style N-fold augmentation: replicate the batch and
        rotate each copy independently, instead of one in-place rotation
        per molecule (see `__init__`'s `num_random_augmentations`
        docstring). `.repeat(naug, ...)` tiles the WHOLE batch naug times
        end-to-end (not interleaved) -- every tensor here uses this exact
        pattern on its own natural shape to stay row-aligned with the
        others; `context`, computed later from the raw batch dict, is
        repeated the same way by the caller.
        """
        naug = self.num_random_augmentations + 1
        x = x.repeat(naug, 1, 1)
        h = h.repeat(naug, 1, 1)
        node_mask = node_mask.repeat(naug, 1, 1)
        charges = charges.repeat(naug, *([1] * (charges.dim() - 1)))
        edge_mask = edge_mask.repeat(naug, *([1] * (edge_mask.dim() - 1)))
        rotations = sample_uniform_rotation_matrices(
            x.shape[0], x.device, x.dtype
        )
        x = torch.matmul(x, rotations)
        x = remove_mean_with_mask(x, node_mask)
        return x, h, node_mask, charges, edge_mask

    def density_estimation(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        
        sp_reg_handled = False
        if "graph" in batch.keys():
            # NOTE ignore remove mean over mask for now
            if len(self.condition) > 0:
                context = prepare_context_pyG(self.condition, batch, self.property_norms,
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
            else:
                context = None

            batch["context"] = context
            batch["graph"].pos = remove_mean_pyG(batch["graph"].pos, batch["graph"].batch)

            if self.sp_regularizer is not None and self.training and hasattr(self.model, 'compute_loss_per_graph'):
                # Mirror FM: get per-graph training losses and apply SP before log_pN correction.
                # This avoids the scale mismatch between raw training losses and full NLL.
                loss_per_graph = self.model.compute_loss_per_graph(
                    batch,
                    context,
                    self.reference_indices,
                    reference_freeze_mode=self.reference_freeze_mode,
                )
                nll = self.sp_regularizer(loss_per_graph)
                sp_reg_handled = True
            else:
                nll = self.model(
                    mol_graph=batch,
                    context=context,
                    reference_indices=self.reference_indices,
                    reference_freeze_mode=self.reference_freeze_mode,
                )
            N = batch["graph"].natoms
        else:
            node_mask = batch["node_mask"].unsqueeze(2)
            edge_mask = batch["edge_mask"]
            x = batch["coords"]
            h = batch["node_feature"]
            charges = self._charge_channel(batch["charges"])
            

            x = remove_mean_with_mask(x, node_mask)
            if self.augment_noise > 0 and not(self.reference_indices):
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * self.augment_noise
                x = remove_mean_with_mask(x, node_mask)
            if self.num_random_augmentations > 0:
                x, h, node_mask, charges, edge_mask = (
                    self._apply_n_fold_rotation_augmentation(
                        x, h, node_mask, charges, edge_mask
                    )
                )
            elif self.data_augmentation:
                x = random_rotation(x).detach()

            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            bs, n_nodes, n_dims = x.size()
            assert_correctly_masked(x, node_mask)
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            h = {"categorical": h, "integer": charges}

            if len(self.condition) > 0:
                context = prepare_context(self.condition, batch, self.property_norms,
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                if self.num_random_augmentations > 0:
                    # `context` was computed from the raw (unexpanded)
                    # batch above -- repeat it the same way x/h/node_mask
                    # already were, so each augmented copy of a molecule
                    # keeps that molecule's own condition values.
                    context = context.repeat(
                        self.num_random_augmentations + 1, 1, 1
                    )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            nll = self.model(
                x,
                h,
                node_mask,
                edge_mask,
                context,
                reference_indices=self.reference_indices,
                reference_freeze_mode=self.reference_freeze_mode,
            )

            N = node_mask.squeeze(2).sum(1).long()
        if not sp_reg_handled:
            log_pN = self.node_dist_model.log_prob(N)
            assert nll.size() == log_pN.size()
            nll = nll - log_pN
            if self.sp_regularizer is not None and self.training:
                nll = self.sp_regularizer(nll)
        loss = nll.mean(0)
        metric["train_negative_log_likelihood"] = loss
        all_loss += loss

        return all_loss, metric

    def predict_and_target(self, batch):
        all_loss = self._evaluate(batch)
        all_loss = all_loss.unsqueeze(0)
        dummy_tensor = torch.zeros_like(all_loss)
        return all_loss, dummy_tensor

    def evaluate(self, all_loss, dummy_tensor):
        metric = {}
        metric["val_negative_log_likelihood"] = all_loss.mean()
        return metric


    def _evaluate(self, batch):

        if "graph" in batch.keys():
            if len(self.condition) > 0:
                context = prepare_context_pyG(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
            else:
                context = None
            
            batch["context"] = context
            batch["graph"].pos = remove_mean_pyG(batch["graph"].pos, batch["graph"].batch)
            nll = self.model(
                mol_graph=batch,
                context=context,
                reference_indices=self.reference_indices,
                reference_freeze_mode=self.reference_freeze_mode,
            )
            N = batch["graph"].natoms
        else:
            node_mask = batch["node_mask"].unsqueeze(2)
            edge_mask = batch["edge_mask"]
            x = batch["coords"]
            h = batch["node_feature"]
            charges = self._charge_channel(batch["charges"])
                        
            if self.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(
                    x.size(), x.device, node_mask
                )
                x = x + eps * self.augment_noise
    
            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, h], node_mask)
            assert_mean_zero_with_mask(x, node_mask)
            bs, n_nodes, n_dims = x.size()
            assert_correctly_masked(x, node_mask)
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            h = {"categorical": h, "integer": charges}

            if len(self.condition) > 0:
                context = prepare_context(self.condition, batch, self.property_norms, 
                                        normalization_method=self.normalize_condition).to(
                    dtype=torch.float32, device=self.device
                )
                assert_correctly_masked(context, node_mask)
            else:
                context = None
            N = node_mask.squeeze(2).sum(1).long()
  
            nll = self.model(x, h, node_mask, edge_mask, context, 
                            reference_indices=self.reference_indices,
                            reference_freeze_mode=self.reference_freeze_mode)

        log_pN = self.node_dist_model.log_prob(N)
        # Handle shape mismatch: can occur when data collator has edge cases
        if nll.dim() == 0:
            # nll is already reduced to scalar
            nll = nll - log_pN.mean()
        elif nll.size() != log_pN.size():
            # Batch size mismatch - use minimum size
            min_size = min(nll.size(0), log_pN.size(0))
            nll = nll[:min_size] - log_pN[:min_size]
        else:
            nll = nll - log_pN
        loss = nll.mean() if nll.dim() > 0 else nll

        return loss



    def sample_chain(self, n_nodes: int, n_tries: int, keep_frames: int = 100):
        """
        Sample a molecule for visualizing the diffusion process.

        Parameters:
        - n_nodes (int): Number of nodes in the molecular graph.
        - n_tries (int): Number of attempts to find a stable molecule.
        - keep_frames (int): Number of frames to keep. Default is 100.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, and positions.
        """
        N_SAMPLE = 1
        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            context = self.prop_dist_model.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(self.device)
        else:
            context = None

        node_mask = torch.ones(N_SAMPLE, n_nodes, 1).to(self.device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(N_SAMPLE, 1, 1).view(-1, 1).to(self.device)

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = self.model.sample_chain(
                N_SAMPLE,
                n_nodes,
                node_mask,
                edge_mask,
                context,
                keep_frames=keep_frames,
            )
            chain = reverse_tensor(chain)

            if self.model.ndim_extra > 0:
                n_core = self.model.in_node_nf - self.model.ndim_extra - 1
                start  = self.model.n_dims
                mid    = start + n_core
            
            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            if  self.model.ndim_extra > 0:
                one_hot = chain[-1:, :, start:mid]
            else:
                one_hot = self._split_frame(chain[-1:])[1]
                
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()

            zs = [self.atomic_numbers[i] for i in atom_type]

            mol_stable = check_stability(x_squeeze, zs, self.atom_decoder)[0]

            # Prepare entire chain.
            x = chain[:, :, 0:3]
            if  self.model.ndim_extra > 0:
                one_hot = chain[-1:, :, start:mid]
            else:
                one_hot = self._split_frame(chain[-1:])[1]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=2), num_classes=len(self.atom_decoder)
            )
            charges = torch.round(self._split_frame(chain)[2]).long()

            if mol_stable:
                print("ლ(́◉◞౪◟◉‵ლ) Found stable molecule to visualize -(๑☆‿ ☆#)ᕗ")
                break
            elif i == n_tries - 1:
                print("Did not find stable molecule, showing last sample. ༼ಢ_ಢ༽")

        return one_hot, charges, x



    def sample(self, 
               nodesxsample=torch.tensor([10]),
               context=None, 
               condition_tensor=None,
               condition_mode=None,
               fix_noise=False,
               n_frames=0,
               n_retrys=0,
               t_retry=180,
               mode="ddpm",
               use_noised_conditioning=False,
               **kwargs):
        """
        Sample molecular structures.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - context (Optional[Tensor]): Context tensor for sampling. Default is None.
        - condition_tensor (Optional[Tensor]): Condition tensor for sampling. Default is None.
            Note that it has to be normalized the same way as the training set.
            Size = [batch size, n_atom, n_features]
        - condition_mode (Optional[str]): Mode for conditioning. Default is None.
            Format: [condition_name]_[component_alg]
            component name can be x, h, or xh
            component_alg: SSGD, ...    
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - n_frames (int): Number of frames to keep. Default is 0.
        - n_retrys (int): Number of retry attempts in the event of bad molecules . Default is 0.
        - t_retrys (int): Timestep to start retrying. Default is 180.
        - mode (str): Mode for sampling. Default is "ddpm ["ddpm", "ddim"].

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        
        
        batch_size = nodesxsample.size(0)
        nnode = int(torch.max(nodesxsample).item())
        node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            if context is None:
                context = self.prop_dist_model.sample_batch(nodesxsample)
                context = context.unsqueeze(1)
                context = context.expand(-1, nnode, -1)
            else:
                context = (
                context.unsqueeze(1).repeat(batch_size, nnode, 1).to(self.device) * node_mask
            )
       
        else:
            context = None

        if mode == "ddpm":
            x, h, chain = self.model.sample(
                batch_size,
                nnode,
                node_mask,
                edge_mask,
                context,
                condition_tensor,
                condition_mode,
                fix_noise=fix_noise,
                n_frames=n_frames,
                n_retrys=n_retrys,
                t_retry=t_retry,
                use_noised_conditioning=use_noised_conditioning,
                **kwargs
            )
        elif mode == "ddim":    
            x, h, chain = self.model.sample_ddim(
                batch_size,
                nnode,
                node_mask,
                edge_mask,
                context,
                eta=1,
                fix_noise=fix_noise,
                n_frames=n_frames,
                **kwargs # eta, n_steps, save_frame
            )

        if self.model.ndim_extra > 0:
            n_core = self.model.in_node_nf - self.model.ndim_extra - 1
            start  = self.model.n_dims
            mid    = start + n_core
            
        if chain is not None:

            # chain = chain.reshape(batch_size, n_frames, nnode, -1)
            # Prepare entire chain.
            if isinstance(chain, torch.Tensor):
                x = chain[:, :, :, 0:3]

                if  self.model.ndim_extra > 0:
                    one_hot = chain[:, :, :, start:mid]
                else:
                    one_hot = self._split_frame(chain)[1]
                    
                one_hot = F.one_hot(
                    torch.argmax(one_hot, dim=3), num_classes=self.n_atom_types
                )
                charges = torch.round(self._split_frame(chain)[2]).long()
                
            #TODO how to deal with batch in case of retry here
            elif isinstance(chain, list):
                x_0 = chain[0][:, :, 0:3]
                one_hot_0 = self._split_frame(chain[0])[1]
                one_hot_0 = F.one_hot(
                    torch.argmax(one_hot_0, dim=2), num_classes=self.n_atom_types
                )
                charges_0 = torch.round(self._split_frame(chain[0])[2]).long()
                
                x_retrys = []
                one_hot_retrys = []
                charges_retrys = []
                for i in range(chain[1].shape[0]):
                    x_i = chain[1][i][:, :, 0:3]
                    one_hot_i = self._split_frame(chain[1][i])[1]
                    one_hot_i = F.one_hot(
                        torch.argmax(one_hot_i, dim=2), num_classes=self.n_atom_types
                    )
                    charges_i = torch.round(self._split_frame(chain[1][i])[2]).long()
                    x_retrys.append(x_i)
                    one_hot_retrys.append(one_hot_i)
                    charges_retrys.append(charges_i)
                one_hot = [one_hot_0, one_hot_retrys]
                charges = [charges_0, charges_retrys]
                x = [x_0, x_retrys]
        else:
            one_hot = h.get("categorical", torch.zeros_like(x))
            charges = h["integer"]
        return one_hot, charges, x, node_mask


    def sample_around_xh_target(self, nodesxsample=torch.tensor([10]), 
                                xh_target=None, context=None, fix_noise=False):
                            
        """
        Sample molecular structures.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - xh_target (Tensor): target xh: [batch size, n_atom, n_features]
        - context (Optional[Tensor]): Context tensor for sampling. Default is None.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        nodesxsample = torch.where(
            nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        )
        batch_size = len(nodesxsample)

        if batch_size > 1:
            node_mask = torch.zeros(batch_size, self.max_n_nodes)
            nnode = self.max_n_nodes
        else:
            nnode = int(nodesxsample[0])
            node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            if context is None:
                context = self.prop_dist_model.sample_batch(nodesxsample)
            context = (
                context.unsqueeze(1).repeat(1, nnode, 1).to(self.device) * node_mask
            )
        else:
            context = None

        x, h = self.model.sample_around_xh(
            batch_size,
            nnode,
            node_mask,
            edge_mask,
            context,
            xh_target,
            fix_noise=fix_noise,
        )

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h["categorical"]
        charges = h["integer"]

        assert_correctly_masked(one_hot.float(), node_mask)
        assert_correctly_masked(charges.float(), node_mask)

        return one_hot, charges, x, node_mask
    
    def sample_conditonal(
        self,
        nodesxsample=torch.tensor([10]),
        target_value=[0],
        fix_noise=False,
        mode="ddpm",
        n_frames=0,
    ):
        """
        Sample molecular structures conditioned on a property value.
        Only works if the model is trained with a property distribution.

        The interval should be wider than the bin width of the property distribution.
        If the interval is too narrow, the model might just get the same molecule.

        Parameters:
        - nodesxsample (Tensor): Number of nodes per sample.
        - target_value (List[float]): Target values for conditional sampling.
        - fix_nose (bool): Fix noise for visualization purposes. Default is False.
        - mode (str): Mode for sampling. Default is "ddpm ["ddpm", "ddim"].
        - n_frames (int): Number of frames to keep. Default is 0.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, positions, and node mask.
        """
  

        context = []
        for i, key in enumerate(self.prop_dist_model.distributions):
            if self.normalize_condition is not None:
                if self.normalize_condition == "mad":
                    mean, mad = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (target_value[i] - mean) / (mad)
                elif self.normalize_condition == "maxmin":   
                    mean, min, max = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["min"],
                        self.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (target_value[i] - min) / (max - min) - 1    
                elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                    value = float(self.normalize_condition.split("_")[1])
                    val = target_value[i] / value
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalize_condition}")
                   
            else:
                val = target_value[i]
            context_row = torch.tensor(
                 [val]
            ).unsqueeze(1)
            context.append(context_row)
        context = torch.cat(context, dim=1).float().to(self.device)
        one_hot, charges, x, node_mask = self.sample(nodesxsample, context=context, fix_noise=fix_noise, mode=mode, n_frames=n_frames)
        return one_hot, charges, x, node_mask

    #GG
    def sample_guidance(
        self,
        target_function,
        nodesxsample=torch.tensor([10]),
        scale=1,
        max_norm=10,
        std=1.0,
        fix_noise=False,
        scheduler=None,
        guidance_at=0,
        guidance_stop=1,    
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        n_frames=0,
        debug=False,
    ):
        """
        Sample molecular structures with guidance from target function.

        Parameters:
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - nodesxsample (Tensor): Number of nodes per sample. Default is torch.tensor([10]).
        - scale (float): Scale factor for gradient guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - context (Optional[Tensor]): Context tensor for sampling. Default is None.
        - condition_tensor (Optional[Tensor]): Condition tensor for sampling. Default is None.
        - n_frames (int): Number of frames to keep. Default is 0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Positions, one-hot encoding of atoms, node mask, and edge mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        batch_size = nodesxsample.size(0)
        nnode = int(torch.max(nodesxsample).item())
        node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)


        # sample from the EDM model
        x, h, chain = self.model.sample_guidance(
            batch_size,
            target_function,
            node_mask,
            edge_mask,
            None,
            context_negative=None,
            gg_scale=scale,
            cfg_scale=None,
            max_norm=max_norm,
            fix_noise=fix_noise,
            std=std,
            scheduler=scheduler,
            guidance_at=guidance_at,
            guidance_stop=guidance_stop,
            guidance_ver=guidance_ver,
            n_backwards=n_backwards,
            h_weight=h_weight,
            x_weight=x_weight,
            debug=debug,
            n_frames=n_frames,
        )

        
        if self.model.ndim_extra > 0:
            n_core = self.model.in_node_nf - self.model.ndim_extra - 1
            start  = self.model.n_dims
            mid    = start + n_core

        if chain is not None:

            # chain = chain.reshape(batch_size, n_frames, nnode, -1)
            # Prepare entire chain.
            if isinstance(chain, torch.Tensor):
                if  self.model.ndim_extra > 0:
                    one_hot = chain[:, :, :, start:mid]
                else:
                    one_hot = self._split_frame(chain)[1]
                x = chain[:, :, :, 0:3]
                one_hot = F.one_hot(
                    torch.argmax(one_hot, dim=3), num_classes=self.n_atom_types
                )
                charges = torch.round(self._split_frame(chain)[2]).long()
                
            #TODO how to deal with batch in case of retry here
            elif isinstance(chain, list):
                x_0 = chain[0][:, :, 0:3]
                if  self.model.ndim_extra > 0:
                    one_hot = chain[:, :, :, start:mid]
                else:
                    one_hot = self._split_frame(chain)[1]
                one_hot_0 = F.one_hot(
                    torch.argmax(one_hot_0, dim=2), num_classes=self.n_atom_types
                )
                charges_0 = torch.round(self._split_frame(chain[0])[2]).long()
                
                x_retrys = []
                one_hot_retrys = []
                charges_retrys = []
                for i in range(chain[1].shape[0]):
                    x_i = chain[1][i][:, :, 0:3]
                    one_hot_i = self._split_frame(chain[1][i])[1]
                    one_hot_i = F.one_hot(
                        torch.argmax(one_hot_i, dim=2), num_classes=self.n_atom_types
                    )
                    charges_i = torch.round(self._split_frame(chain[1][i])[2]).long()
                    x_retrys.append(x_i)
                    one_hot_retrys.append(one_hot_i)
                    charges_retrys.append(charges_i)
                one_hot = [one_hot_0, one_hot_retrys]
                charges = [charges_0, charges_retrys]
                x = [x_0, x_retrys]
        else:
            one_hot = h["categorical"]
            charges = h["integer"]
        return one_hot, charges, x, node_mask

    #CFG/CFGGG
    def sample_guidance_conitional(
        self,
        target_function,
        target_value=[0],
        negative_target_value=[],
        nodesxsample=torch.tensor([10]),
        gg_scale=1,
        cfg_scale=1,
        cfg_scale_schedule=None,
        max_norm=10,
        std=1.0,
        fix_noise=False,
        scheduler=None,
        guidance_at=1,
        guidance_stop=0,
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        n_frames=0,
        debug=False,
    ):
        """
        Sample molecular structures with guidance from target function and conditional property.

        Parameters:
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - target_value (List[float]): Target values for conditional sampling.
        - nodesxsample (Tensor): Number of nodes per sample. Default is torch.tensor([10]).
        - gg_scale (float): Scale factor for gradient guidance. Default is 1.0.
        - cfg_scale (float): Scale factor for classifier-free guidance. Default is 1.0.
        - cfg_scale_schedule (str, optional): Scheduler for cfg scale. Default is None. [linear, exponential, cosine]
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - n_frames (int): Number of frames to keep. Default is 0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Positions, one-hot encoding of atoms, node mask, and edge mask.
        """
        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        batch_size = nodesxsample.size(0)
        nnode = int(torch.max(nodesxsample).item())
        node_mask = torch.zeros(batch_size, nnode)

        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)
        n_node = node_mask.size(1)  
        context = []
        for i, key in enumerate(self.prop_dist_model.distributions):
            if self.normalize_condition is not None:
                if self.normalize_condition == "mad":
                    mean, mad = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["mad"],
                    )
                    val = (target_value[i] - mean) / (mad)
                elif self.normalize_condition == "maxmin":   
                    mean, min, max = (
                        self.prop_dist_model.normalizer[key]["mean"],
                        self.prop_dist_model.normalizer[key]["min"],
                        self.prop_dist_model.normalizer[key]["max"],
                    )
                    val = 2 * (target_value[i] - min) / (max - min) - 1   

                elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                    value = float(self.normalize_condition.split("_")[1])
                    val = target_value[i] / value    
                else:
                    raise ValueError(f"Unknown normalization method: {self.normalize_condition}")    
            else:
                val = target_value[i]
            context_row = torch.tensor(
                 [val]
            ).unsqueeze(1)
            context.append(context_row)

        context = torch.cat(context, dim=1).float().to(self.device)
        context = context.repeat(batch_size, n_node, 1)

        if negative_target_value:
            context_negative = []
            for i, key in enumerate(self.prop_dist_model.distributions):
                if i < len(negative_target_value):
                    if self.normalize_condition is not None:
                        if self.normalize_condition == "mad":
                            mean, mad = (
                                self.prop_dist_model.normalizer[key]["mean"],
                                self.prop_dist_model.normalizer[key]["mad"],
                            )
                            val = (negative_target_value[i] - mean) / (mad)
                        elif self.normalize_condition == "maxmin":   
                            mean, min, max = (
                                self.prop_dist_model.normalizer[key]["mean"],
                                self.prop_dist_model.normalizer[key]["min"],
                                self.prop_dist_model.normalizer[key]["max"],
                            )
                            val = 2 * (negative_target_value[i] - min) / (max - min) - 1   

                        elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                            value = float(self.normalize_condition.split("_")[1])
                            val = negative_target_value[i] / value    
                        else:
                            raise ValueError(f"Unknown normalization method: {self.normalize_condition}")    
                    else:
                        val = negative_target_value[i]
                    context_row = torch.tensor(
                         [val]
                    ).unsqueeze(1)
                    context_negative.append(context_row)
            context_negative = torch.cat(context_negative, dim=1).float().to(self.device)
            context_negative = context_negative.repeat(batch_size, n_node, 1)
        else:
            context_negative = None

        # sample from the EDM model
        x, h, chain = self.model.sample_guidance(
            batch_size,
            target_function,
            node_mask,
            edge_mask,
            context,
            context_negative=context_negative,
            gg_scale=gg_scale,
            cfg_scale=cfg_scale,
            cfg_scale_schedule=cfg_scale_schedule,
            max_norm=max_norm,
            fix_noise=fix_noise,
            std=std,
            scheduler=scheduler,
            guidance_at=guidance_at,
            guidance_stop=guidance_stop,
            guidance_ver=guidance_ver,
            n_backwards=n_backwards,
            h_weight=h_weight,
            x_weight=x_weight,
            debug=debug,
            n_frames=n_frames,
        )

        if self.model.ndim_extra > 0:
            n_core = self.model.in_node_nf - self.model.ndim_extra - 1
            start  = self.model.n_dims
            mid    = start + n_core
            
        if chain is not None:

            # chain = chain.reshape(batch_size, n_frames, nnode, -1)
            # Prepare entire chain.
            if isinstance(chain, torch.Tensor):
                x = chain[:, :, :, 0:3]
                if  self.model.ndim_extra > 0:
                    one_hot = chain[:, :, :, start:mid]
                else:
                    one_hot = self._split_frame(chain)[1]
                one_hot = F.one_hot(
                    torch.argmax(one_hot, dim=3), num_classes=self.n_atom_types
                )
                charges = torch.round(self._split_frame(chain)[2]).long()
                
            #TODO how to deal with batch in case of retry here
            elif isinstance(chain, list):
                x_0 = chain[0][:, :, 0:3]
                one_hot_0 = self._split_frame(chain[0])[1]
                one_hot_0 = F.one_hot(
                    torch.argmax(one_hot_0, dim=2), num_classes=self.n_atom_types
                )
                charges_0 = torch.round(self._split_frame(chain[0])[2]).long()
                
                x_retrys = []
                one_hot_retrys = []
                charges_retrys = []
                for i in range(chain[1].shape[0]):
                    x_i = chain[1][i][:, :, 0:3]
                    one_hot_i = self._split_frame(chain[1][i])[1]
                    one_hot_i = F.one_hot(
                        torch.argmax(one_hot_i, dim=2), num_classes=self.n_atom_types
                    )
                    charges_i = torch.round(self._split_frame(chain[1][i])[2]).long()
                    x_retrys.append(x_i)
                    one_hot_retrys.append(one_hot_i)
                    charges_retrys.append(charges_i)
                one_hot = [one_hot_0, one_hot_retrys]
                charges = [charges_0, charges_retrys]
                x = [x_0, x_retrys]
        else:
            one_hot = h["categorical"]
            charges = h["integer"]
        return one_hot, charges, x, node_mask

    # Structure+property guidance
    def sample_hybrid_guidance(
        self,
        target_function,
        target_value=[0],
        negative_target_value=[],
        nodesxsample=torch.tensor([10]),
        gg_scale=1,
        cfg_scale=1,
        cfg_scale_schedule=None,
        max_norm=10,
        std=1.0,
        fix_noise=False,
        scheduler=None,
        guidance_at=1,
        guidance_stop=0,
        guidance_ver=1,
        n_backwards=0,
        h_weight=1,
        x_weight=1,
        condition_tensor=None,
        condition_mode=None,
        inpaint_cfgs={},
        outpaint_cfgs={},
        use_noised_conditioning=False,
        n_frames=0,
        debug=False,
    ):
        """
        Sample molecular structures with guidance from target function and conditional property.

        Parameters:
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - target_value (List[float]): Target values for conditional sampling.
        - nodesxsample (Tensor): Number of nodes per sample. Default is torch.tensor([10]).
        - gg_scale (float): Scale factor for gradient guidance. Default is 1.0.
        - cfg_scale (float): Scale factor for classifier-free guidance. Default is 1.0.
        - cfg_scale_schedule (str, optional): Scheduler for cfg scale. Default is None. [linear, exponential, cosine]
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - fix_noise (bool): Fix noise for visualization purposes. Default is False.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - guidance_at (int): The timestep at which to apply guidance [0-1]  0 = since beginning. Default is 1.
        - guidance_stop (int): The timestep at which to stop applying guidance [0-1]  1 = until the end. Default is 0.  
        - guidance_ver (int): The version of the guidance. Default is 1. [0,1,2,cfg,cfg_gg]
        - n_backwards (int): Number of backward steps. Default is 0.
        - h_weight (float): Weight for the gradient of atom feature. Default is 1.0.
        - x_weight (float): Weight for the gradient of cartesian coordinate. Default is 1.0.
        - debug (bool): Debug mode. Default is False.
            Save gradient norms, max gradients, clipping coefficients, and energies to files.
        - condition_tensor (torch.Tensor, optional): Tensor for conditional guidance. Defaults to None.
        - condition_mode (str, optional): Mode for conditional guidance. Defaults to None.
        - inpaint_cfgs (dict, optional): Configuration for inpainting. 
            The dictionary must contains:        
                - mask_node_index (torch.Tensor, optional): Indices of nodes to be inpainted. Defaults to an empty tensor.
                - denoising_strength (float, optional): Strength of denoising for inpainting
                - noise_initial_mask (bool, optional): Whether to noise the initial masked region. Defaults to False.
        - outpaint_cfgs (dict, optional): Configuration for outpainting. 
            The dictionary must contains:
                - t_start (float, optional): Timestep to start the generation. Defaults to 1.0.
                - t_critical (float, optional): Timestep threshold for applying reference tensor constraints. Defaults to None.
`               - connector_index (torch.Tensor, optional): Indices of connector nodes for outpainting. Defaults to an empty tensor.
                - seed_dist (float, optional): Distance of the seed from the connector atom (used if n_bq_atom == 0)..
                - min_dist (float, optional): Minimum distance from any existing atom in xh_cond (except the connector itself). Defaults to 1.
                - spread (float, optional): Random-walk angular dispersion, or
                    legacy seed-cloud position spread. Defaults to 1.
                - jitter_scale (float): Explicit positional noise magnitude
                    when forward_noise is "jitter".
                - n_bq_atom (int, optional): Number of dummy atoms. Defaults is 0.

        - n_frames (int, optional): Number of frames to keep. Defaults to 0.

        Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Positions, one-hot encoding of atoms, node mask, and edge mask.
        """
        if guidance_ver == "cfg" and condition_mode is not None:
            mode_prefix = condition_mode.split("_")[0]
            if mode_prefix in ("inpaint", "outpaint"):
                inpaint_cfgs = _without_geometric_constraint_cfgs(inpaint_cfgs)
                outpaint_cfgs = _without_geometric_constraint_cfgs(outpaint_cfgs)

        # assert int(torch.max(nodesxsample)) <= self.max_n_nodes
        # nodesxsample = torch.where(
        #     nodesxsample > self.max_n_nodes, self.max_n_nodes, nodesxsample
        # )
        batch_size = nodesxsample.size(0)
        nnode = int(torch.max(nodesxsample).item())
        node_mask = torch.zeros(batch_size, nnode)
            
        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * nnode * nnode, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)
        n_node = node_mask.size(1)  
        
        if target_value is not None:
            context = []
            for i, key in enumerate(self.prop_dist_model.distributions):
                if self.normalize_condition is not None:
                    if self.normalize_condition == "mad":
                        mean, mad = (
                            self.prop_dist_model.normalizer[key]["mean"],
                            self.prop_dist_model.normalizer[key]["mad"],
                        )
                        val = (target_value[i] - mean) / (mad)
                    elif self.normalize_condition == "maxmin":   
                        mean, min, max = (
                            self.prop_dist_model.normalizer[key]["mean"],
                            self.prop_dist_model.normalizer[key]["min"],
                            self.prop_dist_model.normalizer[key]["max"],
                        )
                        val = 2 * (target_value[i] - min) / (max - min) - 1   

                    elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                        value = float(self.normalize_condition.split("_")[1])
                        val = target_value[i] / value    
                    else:
                        raise ValueError(f"Unknown normalization method: {self.normalize_condition}")    
                else:
                    val = target_value[i]
                context_row = torch.tensor(
                    [val]
                ).unsqueeze(1)
                context.append(context_row)

            context = torch.cat(context, dim=1).float().to(self.device)
            context = context.repeat(batch_size, n_node, 1)

            if negative_target_value:
                context_negative = []
                for i, key in enumerate(self.prop_dist_model.distributions):
                    if i < len(negative_target_value):
                        if self.normalize_condition is not None:
                            if self.normalize_condition == "mad":
                                mean, mad = (
                                    self.prop_dist_model.normalizer[key]["mean"],
                                    self.prop_dist_model.normalizer[key]["mad"],
                                )
                                val = (negative_target_value[i] - mean) / (mad)
                            elif self.normalize_condition == "maxmin":   
                                mean, min, max = (
                                    self.prop_dist_model.normalizer[key]["mean"],
                                    self.prop_dist_model.normalizer[key]["min"],
                                    self.prop_dist_model.normalizer[key]["max"],
                                )
                                val = 2 * (negative_target_value[i] - min) / (max - min) - 1   

                            elif "value" in self.normalize_condition: # "value_n where n is the value to normalize"
                                value = float(self.normalize_condition.split("_")[1])
                                val = negative_target_value[i] / value    
                            else:
                                raise ValueError(f"Unknown normalization method: {self.normalize_condition}")    
                        else:
                            val = negative_target_value[i]
                        context_row = torch.tensor(
                            [val]
                        ).unsqueeze(1)
                        context_negative.append(context_row)
                context_negative = torch.cat(context_negative, dim=1).float().to(self.device)
                context_negative = context_negative.repeat(batch_size, n_node, 1)
            else:
                context_negative = None
        else:
            context = None


        # sample from the EDM model
        x, h, chain = self.model.sample_guidance(
            batch_size,
            target_function,
            node_mask,
            edge_mask,
            context,
            context_negative=context_negative,
            gg_scale=gg_scale,
            cfg_scale=cfg_scale,
            max_norm=max_norm,
            fix_noise=fix_noise,
            std=std,
            scheduler=scheduler,
            guidance_at=guidance_at,
            guidance_stop=guidance_stop,
            guidance_ver=guidance_ver,
            n_backwards=n_backwards,
            h_weight=h_weight,
            x_weight=x_weight,
            debug=debug,
            condition_tensor=condition_tensor,
            condition_mode=condition_mode,
            inpaint_cfgs=inpaint_cfgs,
            outpaint_cfgs=outpaint_cfgs,
            use_noised_conditioning=use_noised_conditioning,
            n_frames=n_frames,
            cfg_scale_schedule=cfg_scale_schedule
        )

        if self.model.ndim_extra > 0:
            n_core = self.model.in_node_nf - self.model.ndim_extra - 1
            start  = self.model.n_dims
            mid    = start + n_core
            
        if chain is not None:

            # chain = chain.reshape(batch_size, n_frames, nnode, -1)
            # Prepare entire chain.
            if isinstance(chain, torch.Tensor):
                x = chain[:, :, :, 0:3]
                if  self.model.ndim_extra > 0:
                    one_hot = chain[:, :, :, start:mid]
                else:
                    one_hot = self._split_frame(chain)[1]
                one_hot = F.one_hot(
                    torch.argmax(one_hot, dim=3), num_classes=self.n_atom_types
                )
                charges = torch.round(self._split_frame(chain)[2]).long()
                
            elif isinstance(chain, list):
                x_0 = chain[0][:, :, 0:3]
                one_hot_0 = self._split_frame(chain[0])[1]
                one_hot_0 = F.one_hot(
                    torch.argmax(one_hot_0, dim=2), num_classes=self.n_atom_types
                )
                charges_0 = torch.round(self._split_frame(chain[0])[2]).long()
                
                x_retrys = []
                one_hot_retrys = []
                charges_retrys = []
                for i in range(chain[1].shape[0]):
                    x_i = chain[1][i][:, :, 0:3]
                    one_hot_i = self._split_frame(chain[1][i])[1]
                    one_hot_i = F.one_hot(
                        torch.argmax(one_hot_i, dim=2), num_classes=self.n_atom_types
                    )
                    charges_i = torch.round(self._split_frame(chain[1][i])[2]).long()
                    x_retrys.append(x_i)
                    one_hot_retrys.append(one_hot_i)
                    charges_retrys.append(charges_i)
                one_hot = [one_hot_0, one_hot_retrys]
                charges = [charges_0, charges_retrys]
                x = [x_0, x_retrys]
        else:
            one_hot = h["categorical"]
            charges = h["integer"]
        return one_hot, charges, x, node_mask
    

    def sample_chain_guide(
        self,
        n_nodes: int,
        n_tries: int,
        target_function,
        scale: float = 1,
        max_norm=10,
        std: float = 1.0,
        scheduler=None,
        keep_frames: int = 100,
    ):
        """
        Sample a molecule for visualizing the diffusion process.

        Parameters:
        - n_nodes (int): Number of nodes in the molecular graph.
        - n_tries (int): Number of attempts to find a stable molecule.
        - target_function (Callable[[Tensor], Tensor]): Target function for guidance. Higher value, better
        - scale (float): Scale factor for guidance. Default is 1.0.
        - max_norm (float): Initial maximum norm for the gradients. Default is 10.0.
        - std (float): Standard deviation of the noise. Default is 1.0.
        - scheduler (RateScheduler): Rate scheduler. Default is None.
            The scheduler should have a step method that takes the energy and the current scale as input.
        - keep_frames (int): Number of frames to keep. Default is 100.

        Returns:
        Tuple[Tensor, Tensor, Tensor]: One-hot encoding of atoms, charges, and positions.
        """
        N_SAMPLE = 1
        if (len(self.condition) > 0) and (self.prop_dist_model is not None):
            context = self.prop_dist_model.sample(n_nodes).unsqueeze(1).unsqueeze(0)
            context = context.repeat(1, n_nodes, 1).to(self.device)
        else:
            context = None

        node_mask = torch.ones(N_SAMPLE, n_nodes, 1).to(self.device)

        edge_mask = (1 - torch.eye(n_nodes)).unsqueeze(0)
        edge_mask = edge_mask.repeat(N_SAMPLE, 1, 1).view(-1, 1).to(self.device)

        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = self.model.sample_chain_guidance(
                N_SAMPLE,
                target_function,
                node_mask,
                edge_mask,
                scale,
                max_norm,
                std=std,
                keep_frames=keep_frames,
                scheduler=scheduler,
            )
            chain = reverse_tensor(chain)

            # Repeat last frame to see final sample better.
            chain = torch.cat([chain, chain[-1:].repeat(10, 1, 1)], dim=0)
            x = chain[-1:, :, 0:3]
            one_hot = self._split_frame(chain[-1:])[1]
            one_hot = torch.argmax(one_hot, dim=2)

            atom_type = one_hot.squeeze(0).cpu().detach().numpy()
            x_squeeze = x.squeeze(0).cpu().detach().numpy()

            zs = [self.atomic_numbers[i] for i in atom_type]

            mol_stable = check_stability(x_squeeze, zs, self.atom_decoder)[0]

            if self.model.ndim_extra > 0:
                n_core = self.model.in_node_nf - self.model.ndim_extra - 1
                start  = self.model.n_dims
                mid    = start + n_core
            
            # Prepare entire chain.
            x = chain[:, :, 0:3]
            if  self.model.ndim_extra > 0:
                one_hot = chain[:, :, :, start:mid]
            else:
                one_hot = self._split_frame(chain)[1]
            one_hot = F.one_hot(
                torch.argmax(one_hot, dim=2), num_classes=len(self.atom_decoder)
            )
            charges = torch.round(self._split_frame(chain)[2]).long()

            if mol_stable:
                print("ლ(́◉◞౪◟◉‵ლ) Found stable molecule to visualize -(๑☆‿ ☆#)ᕗ")
                break
            elif i == n_tries - 1:
                print("Did not find stable molecule, showing last sample. ༼ಢ_ಢ༽")

        return one_hot, charges, x


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]



@core.Registry.register("GuidanceModelPrediction")
class GuidanceModelPrediction(Task, core.Configurable):
    eps = 1e-10
    SNR_CLAMP_MAX = 5.0
    _option_members = {"task", "criterion", "metric"}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __init__(
        self,
        model,
        noisemodel,
        task=(),
        include_charge=True,
        metric=("mae", "rmse"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        mlp_batch_norm=None,  # None, 'layernorm', 'batchnorm'
        readout="mean",
        mlp_dropout=0,
        std_mean=None,
        load_mlps_layer=0,
        nextra_nf=0,
        norm_values=(1.0, 1.0, 1.0),
        extra_norm_values=(),
        norm_biases=(None, 0.0, 0.0),
        weight_classes=None,
        t_max=1,
        verbose=0,
        prediction_mlp_type="pernode",  # 'pernode' or 'padded'
        prediction_activation="relu",  # 'relu' or 'silu'
        loss_weighting="none",
        **kwargs
    ):

        super(GuidanceModelPrediction, self).__init__()
        self.model = model

        if self.model.__class__.__name__ in [
            "GraphTransformer",
            "GraphDiffTransformer",
        ]:
            self.architecture = "egt"
        elif self.model.__class__.__name__ in ["EGNN"]:
            self.architecture = "egcn"
        elif self.model.__class__.__name__ in ["PaiNN", "GemNetOC"]:
            self.architecture = "egnn_extra"
        elif self.model.__class__.__name__ in ["eSEN_Backbone", "eSEN_Encoder"]:
            self.architecture = "esen"
        else:
            # Default fallback
            self.architecture = "egcn"
        
        self.metric = metric
        self.criterion = {"mse":1}

        self.gamma = noisemodel
        self.task = task
        self.include_charge = include_charge
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.verbose = verbose
        self.std_mean = std_mean
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.readout = readout
        self.t_max = t_max
        self.prediction_mlp_type = prediction_mlp_type
        self.t_max = t_max
        self.prediction_mlp_type = prediction_mlp_type
        self.prediction_activation = prediction_activation
        
        self.loss_weighting = loss_weighting
        self._last_sampled_t = None

        self.num_targets = len(self.task)
        self.ndim_extra = nextra_nf
        self.n_dims = 3
        
        # Map attributes based on architecture
        if self.architecture == "esen":
            # eSEN stores input dimension as in_node_channels (set during construction)
            # This should be: len(atom_vocab) + n_extra + 1 (charge) + 1 (time)
            # We need in_node_nf = input_dim - 1 (excluding time which is added later)
            if hasattr(self.model, 'in_node_channels'):
                # eSEN was constructed with in_node_channels = in_node_nf + 1 (for time)
                self.in_node_nf = self.model.in_node_channels - 1
            else:
                # Fallback: estimate from sphere_embedding if available
                # This shouldn't happen if eSEN was constructed correctly
                self.in_node_nf = getattr(self.model, 'sphere_channels', 128)
            
            # Set hidden_nf for MLP compatibility (d_model is the output dim after message passing)
            self.model.hidden_nf = getattr(self.model, 'd_model', 
                                           self.model.sphere_channels * ((self.model.lmax + 1) ** 2))
            # Also set in_node_nf on model for compatibility
            self.model.in_node_nf = self.in_node_nf + 1
        else:
            self.in_node_nf = self.model.in_node_nf - 1
        if weight_classes is None:
            self.weight_classes = torch.ones(len(self.task))
        else:
            self.weight_classes = torch.tensor(weight_classes, dtype=torch.float32)
        self.T = self.gamma.T
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.extra_norm_values = extra_norm_values

        self.mlp = None
        self.mlp_final = None
        if std_mean:
            self.std = std_mean[0]
            self.mean = std_mean[1]

        if load_mlps_layer > 0:
            # MLP will be created below in the num_class block using MLPRegressor
            pass
        self.load_mlps_layer = load_mlps_layer


        if self.num_class:
            # Select MLP class based on prediction_mlp_type
            # 'legacy' uses common.MLP for backward compatibility with old checkpoints
            if self.prediction_mlp_type == "legacy":
                # Legacy common.MLP mode for old checkpoints
                hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)
                if load_mlps_layer > 0:
                    self.mlp = common.MLP(
                        self.model.hidden_nf,
                        hidden_dims,
                        batch_norm=self.mlp_batch_norm if isinstance(self.mlp_batch_norm, bool) else (self.mlp_batch_norm is not None),
                        dropout=self.mlp_dropout,
                    )
                    n_layer_final = self.num_mlp_layer - load_mlps_layer - 1
                    self.mlp_final = common.MLP(
                        hidden_dims[n_layer_final:-1] if n_layer_final < len(hidden_dims) else [self.model.hidden_nf],
                        [sum(self.num_class)],
                        batch_norm=self.mlp_batch_norm if isinstance(self.mlp_batch_norm, bool) else (self.mlp_batch_norm is not None),
                        dropout=self.mlp_dropout,
                    )
                else:
                    self.mlp = common.MLP(
                        self.model.hidden_nf,
                        hidden_dims + [sum(self.num_class)],
                        batch_norm=self.mlp_batch_norm if isinstance(self.mlp_batch_norm, bool) else (self.mlp_batch_norm is not None),
                        dropout=self.mlp_dropout,
                    )
                self._use_legacy_mlp = True
            else:
                # New MLPRegressor mode
                if self.prediction_mlp_type == "padded":
                    MLPRegressor = MLPRegressor_padded
                else:
                    MLPRegressor = MLPRegressor_pernode
                    
                if load_mlps_layer > 0:
                    n_layer_final = self.num_mlp_layer - load_mlps_layer - 1
                    self.mlp_final = MLPRegressor(
                        self.model.hidden_nf,
                        sum(self.num_class),
                        hidden_dim=self.model.hidden_nf,
                        num_layers=n_layer_final,
                        normalization=self.mlp_batch_norm,
                        dropout=self.mlp_dropout,
                        activation=self.prediction_activation,
                        readout_method=self.readout,
                        prediction_level="graph",
                    )
                else:
                    self.mlp = MLPRegressor(
                        self.model.hidden_nf,
                        sum(self.num_class),
                        hidden_dim=self.model.hidden_nf,
                        num_layers=self.num_mlp_layer,
                        normalization=self.mlp_batch_norm,
                        dropout=self.mlp_dropout,
                        activation=self.prediction_activation,
                        readout_method=self.readout,
                        prediction_level="graph",
                )


    def preprocess(self, train_set, valid_set=None, test_set=None):
        """
        Compute the mean and derivation for each task on the training set.
        """
        # if len(train_set) == 0:
        #     raise ValueError("Training set is empty. check the data path and format.")
        values = defaultdict(list)

        if train_set is not None:

            for sample in train_set:
                if not sample.get("labeled", True):
                    continue
                for task in self.task:
                    if not math.isnan(sample[task]):
                        values[task].append(sample[task])
            mean = []
            std = []
            weight = []
            num_class = []
            for task, w in self.task.items():
                if task not in train_set.targets.keys():
                    raise ValueError(f"Task {task} not found in dataset")
                value = torch.tensor(values[task])
                mean.append(value.float().mean())
                std.append(value.float().std())
                weight.append(w)
                if value.ndim > 1:
                    num_class.append(value.shape[1])
                elif value.dtype == torch.long:
                    task_class = value.max().item()
                    if task_class == 1 and "bce" in self.criterion:
                        num_class.append(1)
                    else:
                        num_class.append(task_class + 1)
                else:
                    num_class.append(1)
            if not hasattr(self, "mean"):
                print("mean and std not found, registering buffer")
                self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))

            if not hasattr(self, "std"):
                self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
            self.num_class = self.num_class or num_class

            hidden_dims = [self.model.hidden_nf] * (self.num_mlp_layer - 1)

            if self.mlp is None:
                # Select MLP class based on prediction_mlp_type
                if self.prediction_mlp_type == "legacy":
                    # Legacy common.MLP mode
                    self.mlp = common.MLP(
                        self.model.hidden_nf,
                        hidden_dims + [sum(self.num_class)],
                        batch_norm=self.mlp_batch_norm if isinstance(self.mlp_batch_norm, bool) else (self.mlp_batch_norm is not None),
                        dropout=self.mlp_dropout,
                    )
                    self._use_legacy_mlp = True
                else:
                    if self.prediction_mlp_type == "padded":
                        MLPRegressor = MLPRegressor_padded
                    else:
                        MLPRegressor = MLPRegressor_pernode
                        
                    self.mlp = MLPRegressor(
                        self.model.hidden_nf,
                        sum(self.num_class),
                        hidden_dim=self.model.hidden_nf,
                        num_layers=self.num_mlp_layer,
                        normalization=self.mlp_batch_norm,
                        dropout=self.mlp_dropout,
                        activation=self.prediction_activation,
                        readout_method=self.readout,
                        prediction_level="graph",
                    )
            if self.load_mlps_layer > 0:
                if self.prediction_mlp_type == "legacy":
                    n_layer_final = self.num_mlp_layer - self.load_mlps_layer - 1
                    self.mlp_final = common.MLP(
                        hidden_dims[n_layer_final:-1] if n_layer_final < len(hidden_dims) else [self.model.hidden_nf],
                        [sum(self.num_class)],
                        batch_norm=self.mlp_batch_norm if isinstance(self.mlp_batch_norm, bool) else (self.mlp_batch_norm is not None),
                        dropout=self.mlp_dropout,
                    )
                else:
                    if self.prediction_mlp_type == "padded":
                        MLPRegressor = MLPRegressor_padded
                    else:
                        MLPRegressor = MLPRegressor_pernode
                    n_layer_final = self.num_mlp_layer - self.load_mlps_layer - 1
                    self.mlp_final = MLPRegressor(
                        self.model.hidden_nf,
                        sum(self.num_class),
                        hidden_dim=self.model.hidden_nf,
                        num_layers=n_layer_final,
                        normalization=self.mlp_batch_norm,
                        dropout=self.mlp_dropout,
                        activation=self.prediction_activation,
                        readout_method=self.readout,
                        prediction_level="graph",
                    )

            self.train_set_size = len(train_set)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.weight_classes = self.weight_classes.to(device)
        else:
            self.train_set_size = 0
            
            
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0
        
        if self.normalization:
            loss = F.mse_loss(
                (pred - self.mean) / self.std,
                (target - self.mean) / self.std,
                reduction="none",
            )
        else:
            loss = F.mse_loss(pred, target, reduction="none")

        # Apply loss weighting if configured
        if self.loss_weighting != "none" and self._last_sampled_t is not None:
            w = self.get_loss_weight(self._last_sampled_t)
            if w.size(0) == loss.size(0):
                 if w.dim() == 1 and loss.dim() == 2:
                     w = w.unsqueeze(1)
                 loss = loss * w
            
        name = _get_criterion_name("mse")
        if self.verbose > 0:
            for t, l in zip(self.task, loss):
                metric["%s [%s]" % (name, t)] = l
        loss = functional.masked_mean(loss, labeled, dim=0).sum()
        metric[name] = loss
        all_loss += loss

        return all_loss, metric        


    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get(
            "labeled", torch.ones(len(target), dtype=torch.bool, device=target.device)
        )
        target[~labeled] = math.nan
        return target
        
    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            name = _get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric     
                
    def predict(self, batch, all_loss=None, metric=None, evaluate=False):
        """"
        If evaluate is True, the data must be normalized beforehand.
        """
        h = batch["graph"].x
        charges = batch["graph"].atomic_numbers.unsqueeze(-1)
        x = batch["graph"].pos

        bs = batch["graph"].batch.max().item() + 1
        n_atoms  = batch["graph"].natoms
        n_nodes = h.size(0) 
        node_mask = torch.ones((n_nodes, 1), device=x.device)

        if self.include_charge:
            if self.ndim_extra > 0:
                h_cat = h[:, :, : self.ndim_extra]
                h_extra = h[:, :, -self.ndim_extra :]
                h = torch.cat(
                    [h_cat, charges, h_extra], dim=1
                )  # NOTE not sure about dim
            else:
                h = torch.cat([h, charges], dim=1)
        x, h  = self.normalize(x, h, node_mask)    
        if evaluate:
            z_h, z_x = h, x
        else:
            t_upper = int(self.T * self.t_max)
            t_int = torch.zeros((n_nodes, 1), device=x.device, dtype=torch.long)
            t_int_value = torch.randint(
                0, t_upper + 1, size=(bs, 1),  dtype=torch.long, device=x.device
            )

            n_atom_cum = 0
            for i, n_atom in enumerate(n_atoms):
                t_int[n_atom_cum : n_atom_cum + n_atom] = t_int_value[i]
                n_atom_cum += n_atom

            t = t_int / self.T
            
            # Store sampled t for loss weighting in forward
            self._last_sampled_t = t_int_value.float() / self.T if not evaluate else None
            
            eps_x, eps_h = self.sample_combined_position_feature_noise(
                n_samples=1, n_nodes=n_nodes, node_mask=node_mask
            )
            eps_x = eps_x.view(n_nodes, 3)
            eps_h = eps_h.view(n_nodes, -1)
            
            if self.ndim_extra > 0:
                s_eps_hint = eps_h[
                    :, self.in_node_nf - self.ndim_extra - 1
                ].unsqueeze(-1) * self.gamma.get_sigma_bar(
                    t_int=t_int, key="integer"
                )
                s_eps_hcat = eps_h[
                    : self.in_node_nf - self.ndim_extra - 1
                ] * self.gamma.get_sigma_bar(t_int=t_int, key="categorical")

                s_eps_hextra = eps_h[
                    :, -self.ndim_extra :
                ] * self.gamma.get_sigma_bar(t_int=t_int, key="extra")

                s_eps_hs = torch.cat([s_eps_hcat, s_eps_hint, s_eps_hextra], dim=1)
            else:
                s_eps_hint = eps_h[:,-1].unsqueeze(-1) * self.gamma.get_sigma_bar(
                    t_int=t_int, key="integer"
                )
                s_eps_hcat = eps_h[:, :-1] * self.gamma.get_sigma_bar(
                    t_int=t_int, key="categorical"
                )
                s_eps_hs = torch.cat([s_eps_hcat, s_eps_hint], dim=1)

            s_eps_x = eps_x * self.gamma.get_sigma_bar(
                t_int=t_int, key="pos"
            )

            s_eps = torch.cat([s_eps_x, s_eps_hs], dim=1)

            h_catp = h[
                :, : self.in_node_nf - self.ndim_extra - 1
            ] * self.gamma.get_alpha_bar(t_int=t_int, key="categorical")
            h_intp = h[
                :, self.in_node_nf - self.ndim_extra - 1
            ].unsqueeze(-1) * self.gamma.get_alpha_bar(t_int=t_int, key="integer")

            if self.ndim_extra > 0:
                h_extrap = h[:, -self.ndim_extra :] * self.gamma.get_alpha_bar(
                    t_int=t_int, key="extra"
                )
                hs = torch.cat([h_catp, h_intp, h_extrap], dim=1)
            else:
                hs = torch.cat([h_catp, h_intp], dim=1)

            xp = x * self.gamma.get_alpha_bar(t_int=t_int, key="pos")
            xh = torch.cat([xp, hs], dim=1)
            z_t = xh + s_eps
            z_h = z_t[:, 3:]
            z_x = z_t[:, :3]   


        if self.architecture == "egcn":
            
            # if not(self.include_charge):
            #     z_h = z_h[:, :-1]
            if evaluate:
                t = batch["graph"].times # size: (n_nodes, 1)
                if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
                    t = torch.full((n_nodes, 1), t, device=x.device)
                elif isinstance(t, torch.Tensor) and t.ndim == 1:
                    t = t.unsqueeze(-1).repeat(n_nodes // t.shape[0], 1)

            z_h = torch.cat([z_h, t], dim=1)
              
            edge_index = batch["graph"].edge_index
            edges = [edge_index[0], edge_index[1]]
            node_mask = None
            edge_mask = None
            h_final, _ = self.model(
                z_h, z_x, edges, node_mask=node_mask, edge_mask=edge_mask, use_embed=True
            )  
        elif self.architecture == "egnn_extra":
            # MUST USE ONLY MLP embedding 
            if not(self.include_charge):
                z_h = z_h[:, :-1]

            if evaluate:
                t = batch["graph"].times
            
            z_h = torch.cat([z_h, t], dim=1)
            batch["graph"].x = z_h
            batch["graph"].pos = z_x    
            batch["graph"].num_atoms = batch["graph"].natoms
            batch["graph"].token_idx = torch.zeros(batch["graph"].num_nodes, device=self.device).long()
            h_final, _ = self.model(batch["graph"])    
        elif self.architecture == "esen":
            # eSEN architecture: expects PyG Data object
            # IMPORTANT: eSEN's sphere_embedding already concatenates atomic_numbers to data.x
            # So we must NOT include charge in z_h to avoid double-counting
            # Strip the charge dimension (last dim) from z_h
            z_h = z_h[:, :-1]  # Remove charge - eSEN will use atomic_numbers separately

            if evaluate:
                t = batch["graph"].times
            
            # Concatenate time to features
            z_h = torch.cat([z_h, t], dim=1)
            
            # Build eSEN-compatible data object
            batch["graph"].x = z_h
            batch["graph"].pos = z_x
            batch["graph"].num_atoms = batch["graph"].natoms
            batch["graph"].token_idx = torch.zeros(batch["graph"].num_nodes, device=self.device).long()
            # For guidance mode with noisy atomic numbers, keep as float
            # eSEN's sphere_embedding already calls .float() on atomic_numbers
            # if hasattr(batch["graph"], 'atomic_numbers'):
            #     batch["graph"].atomic_numbers = batch["graph"].atomic_numbers.long()
            
            # Forward pass through eSEN
            emb_dict = self.model(batch["graph"])
            
            # Extract node features from eSEN output dict
            # eSEN returns {"x": [N, d_model], "num_atoms": ..., "batch": ..., ...}
            h_final = emb_dict["x"]

        # Handle MLP prediction based on mode
        if getattr(self, '_use_legacy_mlp', False):
            # Legacy common.MLP: does its own readout via pad_data + readout_f
            h_final = self.pad_data(h_final, batch, self.model.hidden_nf)   
            graph_embedding = self.readout_f(h_final)
            if self.load_mlps_layer > 0:
                x = self.mlp(graph_embedding)
                pred = self.mlp_final(x)
            else:
                pred = self.mlp(graph_embedding)
        else:
            # New MLPRegressor: pass node features + batch indices
            batch_indices = batch["graph"].batch
            if self.load_mlps_layer > 0:
                x = self.mlp(h_final, batch_indices)
                pred = self.mlp_final(x, batch_indices)
            else:
                pred = self.mlp(h_final, batch_indices)
            
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


    def get_adj_matrix(self, _edges_dict, n_nodes, batch_size):
        if n_nodes in _edges_dict:
            edges_dic_b = _edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [
                    torch.LongTensor(rows).to(self.device),
                    torch.LongTensor(cols).to(self.device),
                ]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            _edges_dict[n_nodes] = {}
            return self.get_adj_matrix(_edges_dict, n_nodes, batch_size)

    def pad_data(self, array, batch, dim):
        """"
        array: torch.Tensor of shape (n_atoms, n_features)
        batch: pytorch_geometric.data.Batch
        """
        bs = batch["graph"].batch.max().item() + 1
        natoms = batch["graph"].natoms   
        n_nodes = natoms.max().item()
        array_paddded = torch.zeros(bs, n_nodes, array.shape[1]).to(self.device)
        if natoms.dim() == 0:
            natoms = natoms.unsqueeze(0)
        natom_cum = 0
        for i, natom in enumerate(natoms):
            array_mol = array[natom_cum:natom_cum+natom]
            array_mol = torch.cat([array_mol, torch.zeros(n_nodes-natom, array.shape[1]).to(self.device)], dim=0)    
            array_paddded[i] = array_mol
            natom_cum += natom

        array = array_paddded.view(bs, n_nodes, dim)
        
        return array

    def readout_f(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Perform readout operation over nodes in each molecule.

        Parameters:
        - embeddings (torch.Tensor): Tensor of size (x, y, z) where x is the batch size, y is the number of nodes, and z is the feature size.

        Returns:
        torch.Tensor: Aggregated tensor of size (x, z).
        """
        if self.readout == "sum":
            return embeddings.sum(dim=1)
        elif self.readout == "mean":
            return embeddings.mean(dim=1)
        else:
            raise ValueError("Unsupported method. Choose either 'sum' or 'mean'.")
    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        # delta_log_px = -self.subspace_dimensionality(node_mask) * torch.log(
        #     torch.tensor(self.norm_values[0])
        # )

        # Casting to float in case h still has long or int type.
        h_cat = (
            (
                h[:, : self.in_node_nf - self.ndim_extra - 1].float()
                - self.norm_biases[1]
            )
            / self.norm_values[1]
            * node_mask
        )
        h_int = (
            (
                h[:, self.in_node_nf - self.ndim_extra - 1].float().unsqueeze(-1)
                - self.norm_biases[2]
            )
            / self.norm_values[2]
            * node_mask
        )

        h = torch.cat([h_cat, h_int], dim=1)
        if len(self.extra_norm_values) > 0:
            h_extra = (
                h[:, -self.ndim_extra :].float()
                / torch.tensor(self.extra_norm_values, device=x.device).view(1, -1)
                * node_mask
            )
            h = torch.cat([h, h_extra], dim=2)

        return x, h

    def sample_combined_position_feature_noise(
        self, n_samples, n_nodes, node_mask, std=1.0
    ):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
            std=std,
        )

        if self.ndim_extra > 0:

            z_h = sample_gaussian_with_mask(
                size=(
                    n_samples,
                    n_nodes,
                    self.in_node_nf - self.ndim_extra,
                ),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )
            z_h_extra = sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.ndim_extra),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )
            z_h = torch.cat([z_h, z_h_extra], dim=2)
        else:
            z_h = sample_gaussian_with_mask(
                size=(n_samples, n_nodes, self.in_node_nf),
                device=node_mask.device,
                node_mask=node_mask,
                std=std,
            )

        return z_x, z_h

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims



    def get_loss_weight(self, t):
        """
        Compute importance weights for the loss based on the timestep t.
        
        Args:
            t: Tensor of shape (B, 1) or (B,) containing normalized timesteps in [0, 1].
            
        Returns:
            w: Tensor of importance weights.
        """
        if self.loss_weighting == "none":
            return torch.ones_like(t)
            
        elif self.loss_weighting == "linear":
            # Linear decay: weight = 1 - t
            # Emphasizes t=0 (clean data)
            # Strict clamping to [0, 1) range to avoid exactly 1.0
            return torch.clamp(1.0 - t, min=0.0, max=1.0 - 1e-6)
            
        elif self.loss_weighting == "snr":
            # SNR weighting: alpha^2 / sigma^2
            # Closely related to VLB weighting for diffusion models.
            # We use the noise schedule from self.gamma.
            
            # t is normalized [0, 1]. Map to integer steps.
            t_int = torch.round(t * self.T).long()
            
            # Get alpha_bar and sigma_bar
            # Note: We use "pos" key as representative schedule (usually they are similar/same)
            alpha_bar = self.gamma.get_alpha_bar(t_int=t_int, key="pos")
            sigma_bar = self.gamma.get_sigma_bar(t_int=t_int, key="pos")
            
            # Avoid division by zero at t=0 (sigma=0)
            snr = (alpha_bar ** 2) / (sigma_bar ** 2 + 1e-8)
            
            # Clip to avoid extreme values at t=0 and rescale to [0, 1)
            # Common practice in diffusion guidance
            w = torch.clamp(snr, max=self.SNR_CLAMP_MAX) / self.SNR_CLAMP_MAX
            return torch.clamp(w, max=1.0 - 1e-6)
            
        else:
            raise ValueError(f"Unknown loss weighting scheme: {self.loss_weighting}")


class GuidanceModelPredictionPointCloud(GuidanceModelPrediction):
    """
    PointCloud-optimized subclass of GuidanceModelPrediction for EGCL/EGNN.
    
    Accepts dense tensor inputs (B, N, D) directly without requiring PyG Data objects.
    Generates fully-connected edge indices on-the-fly for the EGNN backbone.
    
    Works with both pointcloud batch format (training) and raw tensors (inference).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edges_dict = {}  # Cache for edge indices

    def target(self, batch):
        """Override target extraction for pointcloud batch format."""
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get(
            "labeled", torch.ones(len(target), dtype=torch.bool, device=target.device)
        )
        target[~labeled] = math.nan
        return target

    def predict(self, batch, all_loss=None, metric=None, evaluate=False):
        """
        Prediction for pointcloud batch format.
        
        Batch format: {coords, node_feature, charges, node_mask, edge_mask, natoms, ...}
        """
        if self.architecture != "egcn":
            raise NotImplementedError(
                f"PointCloud predict only supports EGCL/EGNN architecture, got {self.architecture}"
            )
        
        # Extract from pointcloud batch format
        x = batch["coords"]  # (B, N, 3)
        h = batch["node_feature"]  # (B, N, F)
        charges = batch["charges"]  # (B, N)
        node_mask = batch["node_mask"]  # (B, N)
        
        bs, n_nodes, _ = x.shape
        device = x.device
        
        # Ensure node_mask has correct shape
        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)  # (B, N) -> (B, N, 1)
        
        # Add charges to features
        if self.include_charge:
            charges_expanded = charges.unsqueeze(-1) if charges.dim() == 2 else charges
            h = torch.cat([h, charges_expanded], dim=-1)
        
        # Normalize
        x = x / self.norm_values[0]
        h = h / self.norm_values[1]
        
        # Apply node mask to features
        h = h * node_mask
        
        if evaluate:
            # For evaluation, use provided time
            t = batch.get("times", torch.zeros(bs, 1, device=device))
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            z_h, z_x = h, x
        else:
            # For training, sample random timestep and add noise
            t_upper = int(self.T * self.t_max)
            t_int_value = torch.randint(0, t_upper + 1, size=(bs, 1), dtype=torch.long, device=device)
            t = t_int_value.float() / self.T
            
            # Store sampled t for loss weighting in forward
            self._last_sampled_t = t if not evaluate else None
            
            # Sample noise
            eps = torch.randn_like(torch.cat([x, h], dim=-1)) * node_mask
            eps_x = eps[:, :, :3]
            eps_h = eps[:, :, 3:]
            
            # Get noise scales
            alpha_bar = self.gamma.get_alpha_bar(t_int=t_int_value, key="pos")
            sigma_bar = self.gamma.get_sigma_bar(t_int=t_int_value, key="pos")
            
            # alpha and sigma are (B, 1) - expand to (B, N, 1)
            alpha_bar = alpha_bar.unsqueeze(1)
            sigma_bar = sigma_bar.unsqueeze(1)
            
            # Add noise to coordinates
            z_x = x * alpha_bar + eps_x * sigma_bar
            
            # For features, use same schedule (simplified)
            alpha_h = self.gamma.get_alpha_bar(t_int=t_int_value, key="categorical").unsqueeze(1)
            sigma_h = self.gamma.get_sigma_bar(t_int=t_int_value, key="categorical").unsqueeze(1)
            z_h = h * alpha_h + eps_h * sigma_h
        
        # Flatten to (B*N, ...)
        x_flat = z_x.view(bs * n_nodes, 3)
        h_flat = z_h.view(bs * n_nodes, -1)
        node_mask_flat = node_mask.view(bs * n_nodes, 1)
        
        # Expand time to per-node: (B, 1) -> (B*N, 1)
        t_expanded = t.unsqueeze(1).expand(bs, n_nodes, 1).reshape(bs * n_nodes, 1)
        
        # Concatenate time to features
        h_with_time = torch.cat([h_flat, t_expanded], dim=1)
        
        # Generate fully-connected edge index
        edges = self.get_adj_matrix(self._edges_dict, n_nodes, bs)
        
        # Get edge mask if available
        edge_mask = batch.get("edge_mask")
        if edge_mask is not None:
            edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        
        # Forward through EGNN
        h_final, _ = self.model(
            h_with_time, x_flat, edges, 
            node_mask=node_mask_flat, 
            edge_mask=edge_mask, 
            use_embed=True
        )
        
        # Reshape back to (B, N, hidden_nf)
        h_final = h_final.view(bs, n_nodes, -1)
        
        # Apply node mask
        h_final = h_final * node_mask
        
        # Flatten back for MLPRegressor: (B, N, D) -> (B*N, D)
        h_final_flat = h_final.view(bs * n_nodes, -1)
        
        # Create batch indices for scatter (0,0,..0, 1,1,..1, ...)
        batch_indices = torch.arange(bs, device=device).repeat_interleave(n_nodes)
        
        # MLP prediction - MLPRegressor handles readout internally
        if self.load_mlps_layer > 0:
            x_out = self.mlp(h_final_flat, batch_indices)
            pred = self.mlp_final(x_out, batch_indices)
        else:
            pred = self.mlp(h_final_flat, batch_indices)
        
        # Denormalize if needed
        if self.normalization:
            pred = pred * self.std + self.mean
        
        return pred


    def predict_dense(self, x, h, node_mask, t):
        """
        Dense tensor prediction for gradient guidance.
        
        Args:
            x: Coordinates (B, N, 3)
            h: Node features (B, N, F) - should include atom types, charges, etc.
            node_mask: Valid node mask (B, N, 1) or (B, N)
            t: Timestep (B, 1) or scalar - normalized float in [0, 1]
            
        Returns:
            pred: Model predictions (B, num_targets)
        """
        if self.architecture != "egcn":
            raise NotImplementedError(
                f"predict_dense only supports EGCL/EGNN architecture, got {self.architecture}"
            )
        
        # Ensure proper shapes
        bs, n_nodes, _ = x.shape
        device = x.device
        
        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)  # (B, N) -> (B, N, 1)
        
        # Handle time tensor
        if isinstance(t, (int, float)):
            t = torch.full((bs, 1), t, device=device)
        elif t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0).expand(bs, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,) -> (B, 1)
        
        # Flatten to (B*N, ...)
        x_flat = x.view(bs * n_nodes, 3)
        h_flat = h.view(bs * n_nodes, -1)
        node_mask_flat = node_mask.view(bs * n_nodes, 1)
        
        # Expand time to per-node: (B, 1) -> (B*N, 1)
        t_expanded = t.unsqueeze(1).expand(bs, n_nodes, 1).reshape(bs * n_nodes, 1)
        
        # Concatenate time to features
        h_with_time = torch.cat([h_flat, t_expanded], dim=1)
        
        # Generate fully-connected edge index
        edges = self.get_adj_matrix(self._edges_dict, n_nodes, bs)
        
        # Forward through EGNN
        h_final, _ = self.model(
            h_with_time, x_flat, edges, 
            node_mask=node_mask_flat, 
            edge_mask=None, 
            use_embed=True
        )
        
        # Reshape back to (B, N, hidden_nf)
        h_final = h_final.view(bs, n_nodes, -1)
        
        # Apply node mask
        h_final = h_final * node_mask
        
        # Flatten for MLPRegressor: (B, N, D) -> (B*N, D)
        h_final_flat = h_final.view(bs * n_nodes, -1)
        
        # Create batch indices for scatter (0,0,..0, 1,1,..1, ...)
        batch_indices = torch.arange(bs, device=x.device).repeat_interleave(n_nodes)
        
        # MLP prediction - MLPRegressor handles readout internally
        if self.load_mlps_layer > 0:
            x_out = self.mlp(h_final_flat, batch_indices)
            pred = self.mlp_final(x_out, batch_indices)
        else:
            pred = self.mlp(h_final_flat, batch_indices)
        
        # Denormalize if needed
        if self.normalization:
            pred = pred * self.std + self.mean
        
        return pred
