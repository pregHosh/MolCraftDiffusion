"""Coordinate-only FlowMol flow-matching task integration.

Wraps the SE(3)-equivariant GVP endpoint vector field
(``modules/models/flowmol``) in the duck-typed ``Task`` contract
(docs/adding_new_models.md §2.1) and adapts the platform's PointCloud dict
batch to / from FlowMol's batched-DGL-graph container.

Coordinate-only scope (per the approved INTEGRATION_PLAN): the generated
modalities are ``x`` (coords), ``a`` (atom types), ``c`` (formal charge).
Bonds (the ``e`` modality) are NOT generated -- edges are a geometry-only
message-passing convenience. Since the PointCloud pipeline carries no
formal-charge channel, ``c`` is trained against a constant neutral target
(QM9-style neutral molecules) and is not surfaced in the sample() return.
"""

from collections import Counter
from typing import Dict, List, Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from MolecularDiffusion.modules.models.flowmol.graph_utils import (
    build_edge_idxs,
    get_node_batch_idxs,
)
from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)
from MolecularDiffusion.modules.models.flowmol.priors import (
    centered_normal_prior_batched_graph,
    uniform_simplex_prior,
)
from MolecularDiffusion.modules.models.flowmol.vector_field import (
    EndpointVectorField,
)

# Histogram-backed size sampler already implemented for TABASCO; reuse it
# rather than re-implementing the same interface.
from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
    TabascoNodeDistribution,
)
from MolecularDiffusion.utils import prepare_context, compute_mean_mad_from_dataloader


def _atom_onehot(batch: Dict[str, torch.Tensor], n_atom_types: int):
    """Pull the atom-type one-hot out of a PointCloud batch."""
    if "node_feature" in batch:
        feat = batch["node_feature"]
    elif "node_features" in batch:
        feat = batch["node_features"]
    elif "x" in batch:
        feat = batch["x"]
    else:
        feat = F.one_hot(batch["charges"].long(), n_atom_types).float()
    # keep only the atom-type OHE columns (drop any extra per-atom features)
    return feat[..., :n_atom_types].float()


class PointCloudToDGLAdapter(nn.Module):
    """PointCloud dict batch -> batched, fully-connected DGL graph.

    Padding is stripped via ``node_mask``; per-molecule COM is removed from
    coordinates so ``x_1_true`` lives in the same zero-COM subspace as the
    Gaussian prior. No ``edata`` is set -- bonds are dropped.
    """

    def __init__(
        self, n_atom_types: int, n_charges: int, neutral_charge_index: int
    ):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index

    def forward(
        self, batch: Dict[str, torch.Tensor], condition: Optional[torch.Tensor] = None
    ) -> dgl.DGLGraph:
        coords = batch["coords"]
        node_mask = batch["node_mask"].bool()
        atom_oh = _atom_onehot(batch, self.n_atom_types)
        device = coords.device

        graphs = []
        for b in range(coords.size(0)):
            mask_b = node_mask[b]
            n = int(mask_b.sum().item())
            if n == 0:
                continue
            coords_b = coords[b][mask_b]
            coords_b = coords_b - coords_b.mean(dim=0, keepdim=True)
            a_b = atom_oh[b][mask_b]
            c_b = F.one_hot(
                torch.full((n,), self.neutral_charge_index, device=device),
                self.n_charges,
            ).float()

            edges = build_edge_idxs(n).to(device)
            g_i = dgl.graph(
                (edges[0], edges[1]), num_nodes=n, device=device
            )
            g_i.ndata["x_1_true"] = coords_b
            g_i.ndata["a_1_true"] = a_b
            g_i.ndata["c_1_true"] = c_b
            if condition is not None:
                g_i.ndata["cond"] = condition[b][mask_b]
            graphs.append(g_i)

        return dgl.batch(graphs)


class DGLToPointCloudAdapter(nn.Module):
    """Integrated DGL graph -> padded PointCloud sample tuple pieces."""

    def forward(
        self, g: dgl.DGLGraph, n_atom_types: int
    ) -> Dict[str, torch.Tensor]:
        device = g.device
        mols = dgl.unbatch(g)
        n_max = max(int(gi.num_nodes()) for gi in mols)
        bsz = len(mols)

        one_hot = torch.zeros(bsz, n_max, n_atom_types, device=device)
        coords = torch.zeros(bsz, n_max, 3, device=device)
        charges = torch.zeros(bsz, n_max, dtype=torch.long, device=device)
        node_mask = torch.zeros(bsz, n_max, device=device)

        for i, gi in enumerate(mols):
            n = int(gi.num_nodes())
            a_idx = gi.ndata["a_1"].argmax(dim=-1)
            one_hot[i, :n] = F.one_hot(a_idx, n_atom_types).float()
            coords[i, :n] = gi.ndata["x_1"]
            charges[i, :n] = a_idx
            node_mask[i, :n] = 1

        return {
            "one_hot": one_hot,
            "coords": coords,
            "charges": charges,
            "node_mask": node_mask,
        }


class FlowMolTaskFactory:
    """Factory instantiated by ``cli/train.py`` (declares ``train_set`` so the
    declarative seam injects the dataset for atom-count histogram stats)."""

    def __init__(
        self,
        task_type: str,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        n_charges: int = 6,
        neutral_charge_index: int = 0,
        prior_std: float = 1.0,
        time_scaled_loss: bool = True,
        total_loss_weights: Optional[dict] = None,
        default_n_timesteps: int = 250,
        dataset_stats: Optional[dict] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.interpolant_scheduler_config = interpolant_scheduler_config
        self.vector_field_config = vector_field_config
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.total_loss_weights = total_loss_weights or {}
        self.default_n_timesteps = default_n_timesteps
        self.dataset_stats = dataset_stats or {}
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab", None)
        self.train_set = train_set
        self.kwargs = kwargs

    def compute_dataset_stats(self, dataset):
        """Build the atom-count histogram from the training set (like
        TABASCO's ``compute_dataset_stats``)."""
        if hasattr(dataset, "n_atoms"):
            num_atoms_list = list(dataset.n_atoms)
        else:
            num_atoms_list = []
            for i in range(len(dataset)):
                item = dataset[i]
                if "natoms" in item:
                    n = item["natoms"]
                    num_atoms_list.append(
                        int(n.item()) if torch.is_tensor(n) else int(n)
                    )
                elif "node_mask" in item:
                    num_atoms_list.append(int(item["node_mask"].sum().item()))
        histogram = {int(k): int(v) for k, v in Counter(num_atoms_list).items()}
        self.dataset_stats["num_atoms_histogram"] = histogram
        print(
            f"[flowmol] atom-count histogram: {len(histogram)} unique sizes "
            f"from {len(num_atoms_list)} molecules."
        )

    def build(self):
        # Uniform conditioning signature; fallbacks match every other model's
        # factory and only matter when condition_names is non-empty.
        _names = list(self.kwargs.get("condition_names") or [])
        # Hydra ignores unknown keys silently: make the effective values visible.
        print(
            f"[FlowMolTaskFactory] condition_names={_names} "
            f"context_mask_rate={self.kwargs.get('context_mask_rate', 0.2)} "
            f"mask_value={self.kwargs.get('mask_value', 5)} "
            f"normalize_condition={self.kwargs.get('normalize_condition', 'value_10')}"
            + ("" if _names else " (unconditional)")
        )

        if not self.dataset_stats.get("num_atoms_histogram"):
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set)
            else:
                print(
                    "[flowmol] WARNING: no train_set and no histogram; "
                    "generation size sampling will fall back to a default."
                )

        if not self.atom_vocab:
            raise ValueError(
                "FlowMolTaskFactory requires atom_vocab (injected from "
                "data.atom_vocab) to size the atom-type head."
            )

        self.task = FlowMolFlowMatchingTask(
            n_atom_types=len(self.atom_vocab),
            n_charges=self.n_charges,
            neutral_charge_index=self.neutral_charge_index,
            interpolant_scheduler_config=self.interpolant_scheduler_config,
            vector_field_config=self.vector_field_config,
            prior_std=self.prior_std,
            time_scaled_loss=self.time_scaled_loss,
            total_loss_weights=self.total_loss_weights,
            default_n_timesteps=self.default_n_timesteps,
            dataset_stats=self.dataset_stats,
            atom_vocab=self.atom_vocab,
            condition_names=self.kwargs.get("condition_names", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.2),
            mask_value=self.kwargs.get("mask_value", 5),
            normalize_condition=self.kwargs.get("normalize_condition", "value_10"),
            adapter_conditions=self.kwargs.get("adapter_conditions", None),
            use_adapter_module=self.kwargs.get("use_adapter_module", False),
        )
        return self.task


class FlowMolFlowMatchingTask(nn.Module):

    def __init__(
        self,
        n_atom_types: int,
        n_charges: int,
        neutral_charge_index: int,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        prior_std: float,
        time_scaled_loss: bool,
        total_loss_weights: dict,
        default_n_timesteps: int,
        dataset_stats: dict,
        atom_vocab: Optional[list] = None,
        condition_names: list = [],
        context_mask_rate: float = 0.0,
        mask_value: float = 0.0,
        normalize_condition: Optional[str] = None,
        adapter_conditions: Optional[list] = None,
        use_adapter_module: bool = False,
    ):
        super().__init__()

        # Property-conditioning / CFG setup -- same config signature as
        # en_diffusion.py's GeomMolecularGenerative / TABASCO's
        # TabascoDiffusionTask, mirroring runmodes/train/tasks_egcl.py's
        # adapter/concat validation exactly.
        self.condition = condition_names
        self.context_mask_rate = context_mask_rate
        self.mask_value = mask_value
        self.normalize_condition = normalize_condition
        self.property_norms = None  # built in preprocess()

        if adapter_conditions:
            for ac in adapter_conditions:
                if ac not in condition_names:
                    raise ValueError(
                        f"adapter_conditions entry '{ac}' not found in "
                        f"condition_names {condition_names}"
                    )
            self.adapter_indices = [condition_names.index(ac) for ac in adapter_conditions]
            self.concat_indices = [
                i for i in range(len(condition_names)) if i not in self.adapter_indices
            ]
        elif use_adapter_module:
            self.adapter_indices = list(range(len(condition_names)))
            self.concat_indices = []
        else:
            self.adapter_indices = []
            self.concat_indices = list(range(len(condition_names)))
        self.n_adapter_context = len(self.adapter_indices)
        self.n_concat_context = len(self.concat_indices)

        self.canonical_feat_order = ["x", "a", "c"]
        self.n_atom_types = n_atom_types
        self.n_charges = n_charges
        self.neutral_charge_index = neutral_charge_index
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.task_type = "diffusion_flowmol"
        self.fm_num_timesteps = default_n_timesteps
        self.atom_vocab = atom_vocab

        self.total_loss_weights = {"x": 1.0, "a": 1.0, "c": 1.0}
        self.total_loss_weights.update(total_loss_weights or {})

        self.interpolant_scheduler = InterpolantScheduler(
            canonical_feat_order=self.canonical_feat_order,
            **interpolant_scheduler_config,
        )
        self.vector_field = EndpointVectorField(
            n_atom_types=n_atom_types,
            canonical_feat_order=self.canonical_feat_order,
            interpolant_scheduler=self.interpolant_scheduler,
            n_charges=n_charges,
            adapter_indices=self.adapter_indices,
            concat_indices=self.concat_indices,
            **vector_field_config,
        )

        self.to_dgl = PointCloudToDGLAdapter(
            n_atom_types, n_charges, neutral_charge_index
        )
        self.to_pc = DGLToPointCloudAdapter()

        self.loss_x = nn.MSELoss(reduction="none")
        self.loss_a = nn.CrossEntropyLoss(reduction="none")
        self.loss_c = nn.CrossEntropyLoss(reduction="none")

        self.prop_dist_model = None
        self._dataset_stats = dataset_stats
        self._node_dist_model = None

    # ------------------------------------------------------------------ #
    # prior sampling                                                     #
    # ------------------------------------------------------------------ #
    def _sample_prior(self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor):
        n = g.num_nodes()
        g.ndata["x_0"] = centered_normal_prior_batched_graph(
            g, node_batch_idx, std=self.prior_std
        ).to(g.device)
        g.ndata["a_0"] = uniform_simplex_prior(n, self.n_atom_types).to(
            g.device
        )
        g.ndata["c_0"] = uniform_simplex_prior(n, self.n_charges).to(g.device)
        return g

    def preprocess(self, train_set=None, valid_set=None, test_set=None):
        """Build self.property_norms for CFG conditioning (train-side only).

        Called generically by cli/train.py if this attribute exists. Does
        NOT touch node_dist_model/n_node_dist -- those come from
        dataset_stats at __init__ time via FlowMolTaskFactory, a separate
        mechanism. Deliberately skips DistributionProperty/prop_dist_model
        (out of scope -- generation always takes an explicit target_value).
        """
        if train_set is None or len(self.condition) == 0:
            return
        from . import _preprocess_cache as _ppcache

        base, subset_indices = _ppcache.resolve_dataset_and_indices(train_set)
        prop_indices = _ppcache.property_sample_indices(len(train_set), subset_indices)
        props = torch.stack([
            _ppcache.get_property_subset(base, name, prop_indices) for name in self.condition
        ])
        self.property_norms = compute_mean_mad_from_dataloader(props, self.condition)

    # ------------------------------------------------------------------ #
    # training / evaluation                                              #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]):
        condition = None
        if len(self.condition) > 0:
            if self.property_norms is None:
                raise RuntimeError(
                    "condition_names is set but property_norms is None -- "
                    "did preprocess() run? (cli/train.py calls it only if "
                    "hasattr(task, 'preprocess'))"
                )
            condition = prepare_context(
                self.condition, batch, self.property_norms, self.normalize_condition
            ).to(batch["coords"].device)
            if self.context_mask_rate > 0:
                drop = torch.rand(condition.size(0), device=condition.device) < self.context_mask_rate
                if self.n_adapter_context > 0:
                    null_value = torch.empty(
                        condition.shape[-1], device=condition.device, dtype=condition.dtype
                    )
                    null_value[self.adapter_indices] = 0.0
                    null_value[self.concat_indices] = self.mask_value
                else:
                    null_value = torch.full(
                        (condition.shape[-1],), self.mask_value,
                        device=condition.device, dtype=condition.dtype,
                    )
                condition = torch.where(drop.view(-1, 1, 1), null_value.view(1, 1, -1), condition)
                condition = condition * batch["node_mask"].unsqueeze(-1).to(condition.dtype)

        g = self.to_dgl(batch, condition=condition)
        node_batch_idx = get_node_batch_idxs(g)
        batch_size = g.batch_size

        g = self._sample_prior(g, node_batch_idx)

        t = torch.rand(batch_size, device=g.device).float()
        g = self.vector_field.sample_conditional_path(g, t, node_batch_idx)

        vf_output = self.vector_field(g, t, node_batch_idx)

        # endpoint targets
        x_target = g.ndata["x_1_true"]
        a_target = g.ndata["a_1_true"].argmax(dim=-1)
        c_target = g.ndata["c_1_true"].argmax(dim=-1)

        raw = {
            "x": self.loss_x(vf_output["x"], x_target).mean(dim=-1),
            "a": self.loss_a(vf_output["a"], a_target),
            "c": self.loss_c(vf_output["c"], c_target),
        }

        if self.time_scaled_loss:
            time_weights = self.interpolant_scheduler.loss_weights(t)
        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            if self.time_scaled_loss:
                w = time_weights[:, feat_idx][node_batch_idx]
                losses[feat] = (raw[feat] * w).mean()
            else:
                losses[feat] = raw[feat].mean()

        total_loss = sum(
            self.total_loss_weights[feat] * losses[feat]
            for feat in self.canonical_feat_order
        )
        stats = {f"{feat}_loss": losses[feat].detach() for feat in losses}
        stats["total_loss"] = total_loss.detach()
        return total_loss, stats

    def predict_and_target(self, batch: Dict[str, torch.Tensor]):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        return {"val_loss": pred.mean()}

    # ------------------------------------------------------------------ #
    # generation                                                         #
    # ------------------------------------------------------------------ #
    def _build_graphs(self, sizes: torch.Tensor) -> dgl.DGLGraph:
        graphs = []
        for n in sizes.tolist():
            n = int(n)
            edges = build_edge_idxs(n).to(self.device)
            graphs.append(
                dgl.graph((edges[0], edges[1]), num_nodes=n, device=self.device)
            )
        return dgl.batch(graphs)

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        mode=None,
        n_frames: int = 0,
        **kwargs,
    ):
        if num_steps is None:
            num_steps = self.fm_num_timesteps

        if nodesxsample is not None:
            sizes = nodesxsample.to(self.device).long()
        elif batch is not None:
            sizes = batch["natoms"].to(self.device).long()
        else:
            if batch_size is None:
                raise ValueError(
                    "sample() needs nodesxsample, batch, or batch_size."
                )
            sizes = self.node_dist_model.sample(batch_size).to(self.device)

        g = self._build_graphs(sizes)
        node_batch_idx = get_node_batch_idxs(g)
        g = self._sample_prior(g, node_batch_idx)

        # A conditional checkpoint's first layer expects the context columns, so
        # unconditional sampling must feed the same null training's dropout used
        # (single pass, cfg_scale=0) rather than omit `condition` (width crash).
        condition = None
        if len(self.condition) > 0:
            condition = self._null_context(len(sizes))[node_batch_idx]
        g = self.vector_field.integrate(
            g, node_batch_idx, n_timesteps=num_steps,
            condition=condition, cfg_scale=0.0,
        )

        pc = self.to_pc(g, self.n_atom_types)
        return pc["one_hot"], pc["charges"], pc["coords"], pc["node_mask"]

    def _null_context(self, n_mol: int) -> torch.Tensor:
        """(n_mol, D) null context: the value training's context_mask_rate dropout used."""
        if self.n_adapter_context > 0:
            null_value = torch.empty(len(self.condition), device=self.device)
            null_value[self.adapter_indices] = 0.0
            null_value[self.concat_indices] = self.mask_value
        else:
            null_value = torch.full((len(self.condition),), self.mask_value, device=self.device)
        return null_value.unsqueeze(0).expand(n_mol, -1)

    def _normalize(self, value, key):
        if self.normalize_condition is None:
            return value
        norms = self.property_norms[key]
        if self.normalize_condition == "mad":
            return (value - norms["mean"]) / norms["mad"]
        elif self.normalize_condition == "maxmin":
            return 2 * (value - norms["min"]) / (norms["max"] - norms["min"]) - 1
        elif "value" in self.normalize_condition:
            return value / float(self.normalize_condition.split("_")[1])
        raise ValueError(f"Unknown normalization method: {self.normalize_condition}")

    @torch.no_grad()
    def sample_guidance_conitional(
        self,
        target_function=None,
        target_value=None,
        negative_target_value=None,
        nodesxsample: Optional[torch.Tensor] = None,
        cfg_scale: float = 1,
        cfg_scale_schedule: Optional[str] = None,
        guidance_ver: str = "cfg",
        n_frames: int = 0,
        num_steps: Optional[int] = None,
        **kwargs,
    ):
        """
        Classifier-free-guidance generation. Matches the call signature
        GenerativeFactory.conditional_generation() hardcodes for
        task_type == "cfg" (runmodes/generate/tasks_generate.py), and returns
        (one_hot, charges, x, node_mask) like sample() -- "EDM compatibility".

        Simpler than TABASCO's version: FlowMol's DGL graphs are node-native
        (no per-molecule padding dimension), so a per-molecule condition
        value broadcasts via `context_per_mol[node_batch_idx]` directly --
        no dense (B,N,D) intermediate needed.
        """
        if guidance_ver != "cfg":
            raise NotImplementedError(
                f"sample_guidance_conitional only supports guidance_ver='cfg' (got {guidance_ver!r}); "
                "gradient-guidance variants are out of scope."
            )
        if n_frames:
            print(f"WARNING: n_frames={n_frames} is not supported for FlowMol CFG sampling; ignoring.")
        if num_steps is None:
            num_steps = self.fm_num_timesteps

        sizes = nodesxsample.to(self.device).long()

        vals = [self._normalize(target_value[i], key) for i, key in enumerate(self.condition)]
        context_per_mol = torch.tensor(vals, dtype=torch.float, device=self.device).unsqueeze(0).expand(len(sizes), -1)

        if negative_target_value:
            neg = [self._normalize(negative_target_value[i], key) for i, key in enumerate(self.condition)]
            negative_context_per_mol = torch.tensor(neg, dtype=torch.float, device=self.device).unsqueeze(0).expand(len(sizes), -1)
        else:
            # No explicit negative given -- reuse the same null value training's
            # context_mask_rate dropout used (mask_value / 0.0 for adapter cols).
            negative_context_per_mol = self._null_context(len(sizes))

        g = self._build_graphs(sizes)
        node_batch_idx = get_node_batch_idxs(g)
        g = self._sample_prior(g, node_batch_idx)

        condition = context_per_mol[node_batch_idx]
        negative_condition = negative_context_per_mol[node_batch_idx]

        g = self.vector_field.integrate(
            g, node_batch_idx, n_timesteps=num_steps,
            condition=condition, negative_condition=negative_condition,
            cfg_scale=cfg_scale, cfg_scale_schedule=cfg_scale_schedule,
        )

        pc = self.to_pc(g, self.n_atom_types)
        return pc["one_hot"], pc["charges"], pc["coords"], pc["node_mask"]

    # ------------------------------------------------------------------ #
    # properties (generation contract)                                   #
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self):
        if self._node_dist_model is None:
            self._node_dist_model = TabascoNodeDistribution(
                self._dataset_stats
            )
        return self._node_dist_model

    @property
    def n_node_dist(self):
        return self.node_dist_model.n_node_dist
