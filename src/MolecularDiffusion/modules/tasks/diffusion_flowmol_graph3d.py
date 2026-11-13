"""FlowMol3 (bond-generating, CTMC discrete flow matching) task.

The GEOM-Drugs-scale, bond-generating successor to the platform's
coordinate-only ``diffusion_flowmol`` task, which this file does not touch. Four
modalities are generated jointly: ``x`` (coordinates, continuous flow matching)
and ``a``/``c``/``e`` (atom types, formal charges, **bond orders** -- CTMC
discrete flow matching from an all-mask prior).

Wraps ``modules/models/flowmol_graph3d`` in the duck-typed ``Task`` contract
(docs/adding_new_models.md §2.1) and adapts the platform's ``graph3d`` PyG
``Batch`` (``bond_collate: raw``) to / from FlowMol's batched DGL graph.

Three atom-type widths, all real, all easy to confuse (this cost the first
FlowMol port two test attempts):

===============================  =====  ==========================================
width                            value  where
===============================  =====  ==========================================
``len(atom_vocab)``              10     stored data, and the ``sample()`` one-hot
``n_atom_types``                 11     output heads (+1 fake-atom column)
``a`` input token embedding      12     ``n_atom_types`` + 1 CTMC mask token
===============================  =====  ==========================================

``kekulize: true`` is **mandatory** in every config that reaches this task. The
released FlowMol3 weights have ``explicit_aromaticity: False``: their
``to_edge_logits`` is 4 logits wide and ``token_embeddings.e`` is 4 classes plus
one mask row. Canonical bond class 4 (AROMATIC) therefore has **no input row and
no output logit** -- a class-4 label is a hard index crash, not a distribution
wrinkle. Because ``kekulize_bonds`` returns bonds unchanged when RDKit cannot
kekulize a molecule (``graph3d_dataset.py:213-216``), which does happen on
GEOM-Drugs, :meth:`Graph3DToDGLAdapter.forward` asserts the invariant rather
than trusting it.
"""

import logging
import math
import os
from typing import Any

import dgl
import torch
from MolecularDiffusion.device import get_device
import torch.nn.functional as F  # noqa: N812
from torch import nn

from MolecularDiffusion.data.datasets.graph3d_dataset import (
    build_rdkit_mol,
    remap_bonds_after_atom_removal,
)
from MolecularDiffusion.modules.models.flowmol.interpolant_scheduler import (
    InterpolantScheduler,
)
from MolecularDiffusion.modules.models.flowmol_graph3d import (
    CTMCVectorField,
    build_edge_idxs,
    centered_normal_prior_batched_graph,
    ctmc_masked_edge_prior,
    ctmc_masked_prior,
    get_batch_idxs,
    get_upper_edge_mask,
)

# Histogram-backed size sampler, already used by TABASCO, FlowMol and MiDi.
from MolecularDiffusion.modules.tasks.diffusion_tabasco import (
    TabascoNodeDistribution,
)
from MolecularDiffusion.utils import prepare_context_pyG, compute_mean_mad_from_dataloader

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:  # only the optional .sdf sidecar needs RDKit
    Chem = None



def _plain(cfg: Any) -> dict:
    """OmegaConf node (or plain mapping) -> plain nested ``dict``."""
    from omegaconf import OmegaConf

    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


#: ``canonical_feat_order`` -- fixed by the released weights' module ordering.
CANONICAL_FEAT_ORDER = ["x", "a", "c", "e"]

#: FlowMol3 kekulizes at featurization, so only classes 0-3 exist.
N_BOND_TYPES_KEKULIZED = 4

#: ``ignore_index`` of ``nn.CrossEntropyLoss``; CTMC suppresses the loss on
#: already-unmasked positions with it.
_IGNORE_INDEX = -100


class Graph3DToDGLAdapter(nn.Module):
    """``graph3d`` PyG ``Batch`` -> batched, fully-connected DGL graph.

    Per molecule, in upstream's order (``data_processing/dataset.py:99-149``):

    1. inject fake atoms **first**, so the fully-connected edge set covers them;
    2. re-remove the centre of mass (upstream re-centres *after* the injection);
    3. build the edge set with ``build_edge_idxs`` -- never re-derived, because
       ``get_upper_edge_mask`` infers the upper/lower split from that ordering
       alone and silently returns a wrong mask under any other order;
    4. densify the stored upper-triangular bonds into an ``(n, n)`` integer
       adjacency and read the labels back along ``triu_indices``. This is the
       step that materializes bond class 0 ("no bond"), which is never stored;
    5. mirror the labels onto both edge directions.
    """

    def __init__(  # noqa: PLR0913
        self,
        n_atom_types: int,
        n_charge_classes: int,
        charge_offset: int,
        n_bond_types: int,
        fake_atom_p: float,
        fake_atom_std: float,
    ) -> None:
        super().__init__()
        self.n_atom_types = n_atom_types
        self.n_charge_classes = n_charge_classes
        self.charge_offset = charge_offset
        self.n_bond_types = n_bond_types
        self.fake_atom_p = fake_atom_p
        self.fake_atom_std = fake_atom_std
        #: the fake-atom column is the **last** one, matching upstream's dataset
        #: and molecule builder (``dataset.py:120-122``, ``:64,228``).
        self.fake_atom_index = n_atom_types - 1

    def forward(
        self,
        batch: Any,
        *,
        use_fake_atoms: bool,
        condition: torch.Tensor | None = None,
    ) -> dgl.DGLGraph:
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        device = pyg.pos.device

        graphs = []
        for b, item in enumerate(pyg.to_data_list()):
            cond_b = condition[b] if condition is not None else None
            graphs.append(
                self._one_molecule(
                    item, device, use_fake_atoms=use_fake_atoms, condition=cond_b
                )
            )
        return dgl.batch(graphs)

    def _one_molecule(
        self,
        item: Any,
        device: torch.device,
        *,
        use_fake_atoms: bool,
        condition: torch.Tensor | None = None,
    ) -> dgl.DGLGraph:
        pos = item.pos.to(device).float()
        atom_idx = item.atom_idx.to(device).long()
        fc = item.fc.to(device).long()
        bond_index = item.bond_index.to(device).long().reshape(2, -1)
        bond_type = item.bond_type.to(device).long().reshape(-1)

        # The residual-aromatic guard. kekulize_bonds fails *silently* when RDKit
        # cannot kekulize, so a class-4 survivor would otherwise reach a 4-wide
        # head as an opaque mid-epoch index error.
        if bond_type.numel() and int(bond_type.max()) >= self.n_bond_types:
            smiles = getattr(item, "smiles", "<no smiles>")
            msg = (
                f"bond class {int(bond_type.max())} >= n_bond_types="
                f"{self.n_bond_types} in molecule {smiles!r}. This model has no "
                "aromatic slot (explicit_aromaticity=False): set kekulize: true "
                "in the data config, and filter rows RDKit cannot kekulize "
                "(scripts/convert_dataset.py does both)."
            )
            raise ValueError(msg)

        # Charges are stored raw and signed; each model applies its own offset.
        # Raise rather than clip -- upstream does the same (dataset.py:150-155)
        # and GEOM is charged enough that clipping would be a chemistry lie.
        shifted = fc + self.charge_offset
        if fc.numel() and (
            int(shifted.min()) < 0
            or int(shifted.max()) >= self.n_charge_classes
        ):
            msg = (
                f"formal charges ({int(fc.min())}, {int(fc.max())}) do not fit "
                f"{self.n_charge_classes} classes at offset "
                f"{self.charge_offset} (FlowMol3/GEOM: +2 / 6 classes, i.e. "
                "[-2, +3])"
            )
            raise ValueError(msg)

        n_real = int(pos.shape[0])
        if use_fake_atoms and self.fake_atom_p > 0:
            pos, shifted, atom_type_oh = self._inject_fake_atoms(
                pos, shifted, atom_idx, n_real, device
            )
        else:
            atom_type_oh = F.one_hot(atom_idx, self.n_atom_types).float()

        # Upstream re-centres *after* fake-atom injection (dataset.py:125), so
        # the prior and the data share a zero-COM subspace over all n nodes.
        pos = pos - pos.mean(dim=0, keepdim=True)

        n = int(pos.shape[0])
        edges = build_edge_idxs(n).to(device)
        g = dgl.graph((edges[0], edges[1]), num_nodes=n, device=device)

        g.ndata["x_1_true"] = pos
        g.ndata["a_1_true"] = atom_type_oh
        g.ndata["c_1_true"] = F.one_hot(shifted, self.n_charge_classes).float()
        g.edata["e_1_true"] = self._edge_labels(
            bond_index, bond_type, n, device
        )
        if condition is not None:
            # condition is per-molecule (a single (D,) vector) -- broadcast
            # to every node including fake atoms, same as plain FlowMol's
            # per-molecule condition attachment.
            g.ndata["cond"] = condition.unsqueeze(0).expand(n, -1)
        return g

    def _inject_fake_atoms(
        self,
        pos: torch.Tensor,
        shifted: torch.Tensor,
        atom_idx: torch.Tensor,
        n_real: int,
        device: torch.device,
    ):
        """Append ``n_fake`` decoy atoms the model may decline to use.

        Deviation from upstream, deliberately: ``dataset.py:103`` draws
        ``randint(low=0, high=ceil(n*p))``, which can return **0**, and then
        ``:122``'s ``atom_types[-0:, -1] = 1`` -- Python's ``[-0:]`` is ``[0:]``
        -- flags **every** atom in the molecule as fake. Roughly a
        ``1/ceil(n*p)`` fraction of upstream training samples are poisoned this
        way. The ``if n_fake > 0`` guard below fixes it. No shape changes, so
        state-dict compatibility with the released weights is unaffected; the
        port matches the *intent* those weights were trained toward.
        """
        max_n_fake = math.ceil(n_real * self.fake_atom_p)
        n_fake = int(torch.randint(low=0, high=max(max_n_fake, 1), size=(1,)))

        atom_type_oh = F.one_hot(atom_idx, self.n_atom_types).float()
        if n_fake == 0:
            return pos, shifted, atom_type_oh

        anchors = torch.randint(
            low=0, high=n_real, size=(n_fake,), device=device
        )
        fake_pos = pos[anchors] + torch.randn_like(pos[anchors]) * (
            self.fake_atom_std
        )
        pos = torch.cat((pos, fake_pos), dim=0)

        # fake atoms are neutral
        fake_charges = torch.full(
            (n_fake,), self.charge_offset, dtype=shifted.dtype, device=device
        )
        shifted = torch.cat((shifted, fake_charges), dim=0)

        # atom type = the extra last column, all-zero elsewhere
        fake_types = torch.zeros(
            (n_fake, self.n_atom_types),
            dtype=atom_type_oh.dtype,
            device=device,
        )
        fake_types[:, self.fake_atom_index] = 1.0
        atom_type_oh = torch.cat((atom_type_oh, fake_types), dim=0)

        return pos, shifted, atom_type_oh

    def _edge_labels(
        self,
        bond_index: torch.Tensor,
        bond_type: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Stored upper-triangular bonds -> mirrored one-hot over all edges."""
        adj = torch.zeros((n, n), dtype=torch.long, device=device)
        if bond_index.shape[1]:
            adj[bond_index[0], bond_index[1]] = bond_type

        upper = torch.triu_indices(n, n, offset=1, device=device)
        upper_labels = adj[upper[0], upper[1]]
        edge_labels = torch.cat((upper_labels, upper_labels))
        return F.one_hot(edge_labels, self.n_bond_types).float()


class FlowMolGraph3DTaskFactory:
    """Factory instantiated by ``cli/train.py``.

    Declaring ``train_set`` is what makes the declarative injection seam
    (``cli/train.py:624-637``) hand over the dataset, which carries
    ``graph3d_stats`` and therefore the molecule-size histogram FlowMol needs
    for sampling.

    ``sdf_output_path`` is declared generation-time (docs §2.5b): the task is
    rebuilt from the checkpoint's training-time config, where it is ``null``,
    so without this declaration the generate config's value never arrives and
    the bond sidecar -- the whole 2D half of the model -- is silently dropped.
    """

    generation_time_keys = ("sdf_output_path",)

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_flowmol_graph3d",
        interpolant_scheduler_config: dict | None = None,
        vector_field_config: dict | None = None,
        n_charge_classes: int = 6,
        charge_offset: int = 2,
        n_bond_types: int = N_BOND_TYPES_KEKULIZED,
        fake_atom_p: float = 0.3,
        fake_atom_std: float = 1.0,
        distort_p: float = 0.7,
        distort_t: float = 0.25,
        prior_std: float = 1.0,
        time_scaled_loss: bool = True,
        total_loss_weights: dict | None = None,
        default_n_timesteps: int = 250,
        stochasticity: float = 30.0,
        high_confidence_threshold: float = 0.9,
        sdf_output_path: str | None = None,
        dataset_stats: dict | None = None,
        atom_vocab: list | None = None,
        train_set: torch.utils.data.Dataset | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        # Hydra hands these over as OmegaConf nodes, and the generate path hands
        # over a whole DictConfig. The existing InterpolantScheduler type-checks
        # `isinstance(schedule_type, (str, dict))`, which a DictConfig fails, so
        # resolve to plain containers here rather than touching that file.
        self.interpolant_scheduler_config = _plain(
            interpolant_scheduler_config
            or {"schedule_type": dict.fromkeys(CANONICAL_FEAT_ORDER, "linear")}
        )
        self.vector_field_config = _plain(vector_field_config or {})
        self.n_charge_classes = n_charge_classes
        self.charge_offset = charge_offset
        self.n_bond_types = n_bond_types
        self.fake_atom_p = fake_atom_p
        self.fake_atom_std = fake_atom_std
        self.distort_p = distort_p
        self.distort_t = distort_t
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.total_loss_weights = dict(
            total_loss_weights or {"x": 3.0, "a": 0.4, "c": 1.0, "e": 2.0}
        )
        self.default_n_timesteps = default_n_timesteps
        self.stochasticity = stochasticity
        self.high_confidence_threshold = high_confidence_threshold
        self.sdf_output_path = sdf_output_path
        self.dataset_stats = dict(dataset_stats or {})
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: FlowMolGraph3DTask | None = None

    def build(self) -> "FlowMolGraph3DTask":
        if not self.atom_vocab:
            msg = (
                "FlowMol3 needs atom_vocab (it sizes the atom-type head). Set "
                "data.atom_vocab, or tasks.atom_vocab explicitly. FlowMol3's "
                "GEOM order is [C, H, N, O, F, P, S, Cl, Br, I] -- NOT the "
                "platform's usual H-first order, and it is load-bearing for "
                "the released weights."
            )
            raise ValueError(msg)

        n_atom_types = len(self.atom_vocab) + int(self.fake_atom_p > 0)
        expected = self.kwargs.get("expected_n_atom_types")
        if expected is not None and int(expected) != n_atom_types:
            msg = (
                f"n_atom_types={n_atom_types} (len(atom_vocab)="
                f"{len(self.atom_vocab)} + fake column) disagrees with the "
                f"expected width {int(expected)}. A width mismatch loads "
                "silently under strict=False and produces plausible garbage."
            )
            raise ValueError(msg)

        hist = dict(self.dataset_stats.get("num_atoms_histogram") or {})
        if not hist:
            stats = getattr(self.train_set, "graph3d_stats", None)
            if stats is not None:
                hist = {int(k): int(v) for k, v in stats.n_atoms_hist.items()}
                self._log_build_stats(stats, hist)
            else:
                # Expected on the generation path: cli/generate.py builds no
                # DataModule. The real histogram arrives a moment later, either
                # from checkpoint['node_dist_model'] (cli/generate.py:343) or
                # from the edm_stat.pkl sidecar (:437) -- both assign
                # task.node_dist_model directly.
                logger.warning(
                    "No train_set.graph3d_stats -- building FlowMol3 with an "
                    "EMPTY size histogram. Expected when loading a checkpoint "
                    "for generation (the histogram is restored from it); if "
                    "this appears during TRAINING, set data.data_type=graph3d "
                    "with graph3d_stats: true."
                )

        self.task = FlowMolGraph3DTask(
            atom_vocab=self.atom_vocab,
            n_atom_types=n_atom_types,
            n_charge_classes=self.n_charge_classes,
            charge_offset=self.charge_offset,
            n_bond_types=self.n_bond_types,
            fake_atom_p=self.fake_atom_p,
            fake_atom_std=self.fake_atom_std,
            distort_p=self.distort_p,
            distort_t=self.distort_t,
            prior_std=self.prior_std,
            time_scaled_loss=self.time_scaled_loss,
            total_loss_weights=self.total_loss_weights,
            default_n_timesteps=self.default_n_timesteps,
            stochasticity=self.stochasticity,
            high_confidence_threshold=self.high_confidence_threshold,
            sdf_output_path=self.sdf_output_path,
            interpolant_scheduler_config=self.interpolant_scheduler_config,
            vector_field_config=self.vector_field_config,
            n_atoms_hist=hist,
            task_type=self.task_type,
            condition_names=self.kwargs.get("condition_names", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.0),
            mask_value=self.kwargs.get("mask_value", 0.0),
            normalize_condition=self.kwargs.get("normalize_condition", None),
            adapter_conditions=self.kwargs.get("adapter_conditions", None),
            use_adapter_module=self.kwargs.get("use_adapter_module", False),
        )
        return self.task

    def _log_build_stats(self, stats: Any, hist: dict) -> None:
        """Sanity-check the dataset against this model's assumptions, once."""
        bond_counts = list(stats.bond_type_counts)
        if len(bond_counts) > self.n_bond_types and bond_counts[4]:
            logger.error(
                "graph3d_stats reports %d AROMATIC (class 4) bonds, but this "
                "model has only %d bond classes. Set kekulize: true and drop "
                "the rows RDKit cannot kekulize.",
                bond_counts[4],
                self.n_bond_types,
            )
        atom_counts = list(stats.atom_type_counts)
        unpopulated = [
            self.atom_vocab[i]
            for i in range(min(len(self.atom_vocab), len(atom_counts)))
            if atom_counts[i] == 0
        ]
        logger.info(
            "FlowMol3 built from graph3d_stats over %d molecules: sizes "
            "%d-%d (modal %d), bond counts %s, charge range %s, unpopulated "
            "atom_vocab entries %s",
            stats.n_molecules,
            min(hist) if hist else -1,
            max(hist) if hist else -1,
            max(hist, key=hist.get) if hist else -1,
            bond_counts,
            stats.charge_range,
            unpopulated or "none",
        )


class FlowMolGraph3DTask(nn.Module):
    """FlowMol3 wrapped in the platform's duck-typed Task contract (§2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        atom_vocab: list,
        n_atom_types: int,
        n_charge_classes: int,
        charge_offset: int,
        n_bond_types: int,
        fake_atom_p: float,
        fake_atom_std: float,
        distort_p: float,
        distort_t: float,
        prior_std: float,
        time_scaled_loss: bool,
        total_loss_weights: dict,
        default_n_timesteps: int,
        stochasticity: float,
        high_confidence_threshold: float,
        sdf_output_path: str | None,
        interpolant_scheduler_config: dict,
        vector_field_config: dict,
        n_atoms_hist: dict,
        task_type: str = "diffusion_flowmol_graph3d",
        condition_names: list = [],
        context_mask_rate: float = 0.0,
        mask_value: float = 0.0,
        normalize_condition: str | None = None,
        adapter_conditions: list | None = None,
        use_adapter_module: bool = False,
    ) -> None:
        super().__init__()

        # Property-conditioning / CFG setup -- same config signature as
        # en_diffusion.py's GeomMolecularGenerative / TABASCO / plain
        # FlowMol, mirroring runmodes/train/tasks_egcl.py's adapter/concat
        # validation exactly.
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

        self.task_type = task_type
        self.canonical_feat_order = CANONICAL_FEAT_ORDER
        self.atom_vocab = list(atom_vocab)
        self.n_real_atom_types = len(atom_vocab)
        self.n_atom_types = n_atom_types
        self.fake_atom_index = n_atom_types - 1
        self.n_charge_classes = n_charge_classes
        self.charge_offset = charge_offset
        self.n_bond_types = n_bond_types
        self.fake_atom_p = fake_atom_p
        self.distort_p = distort_p
        self.distort_t = distort_t
        self.prior_std = prior_std
        self.time_scaled_loss = time_scaled_loss
        self.total_loss_weights = dict(total_loss_weights)
        self.stochasticity = stochasticity
        self.high_confidence_threshold = high_confidence_threshold

        #: read by ``runmodes/generate/tasks_generate.py:257`` for a default
        #: step count when the generate config does not set one.
        self.fm_num_timesteps = default_n_timesteps

        self.interpolant_scheduler = InterpolantScheduler(
            canonical_feat_order=self.canonical_feat_order,
            **interpolant_scheduler_config,
        )
        self.vector_field = CTMCVectorField(
            n_atom_types=n_atom_types,
            canonical_feat_order=self.canonical_feat_order,
            interpolant_scheduler=self.interpolant_scheduler,
            n_charges=n_charge_classes,
            n_bond_types=n_bond_types,
            fake_atoms=fake_atom_p > 0,
            stochasticity=stochasticity,
            high_confidence_threshold=high_confidence_threshold,
            adapter_indices=self.adapter_indices,
            concat_indices=self.concat_indices,
            **vector_field_config,
        )

        self.to_dgl = Graph3DToDGLAdapter(
            n_atom_types=n_atom_types,
            n_charge_classes=n_charge_classes,
            charge_offset=charge_offset,
            n_bond_types=n_bond_types,
            fake_atom_p=fake_atom_p,
            fake_atom_std=fake_atom_std,
        )

        # reduction='none' throughout: the per-example time weight is applied
        # before the reduction when time_scaled_loss is on.
        self.loss_x = nn.MSELoss(reduction="none")
        self.loss_cat = {
            feat: nn.CrossEntropyLoss(
                reduction="none", ignore_index=_IGNORE_INDEX
            )
            for feat in ("a", "c", "e")
        }

        self.node_dist_model = TabascoNodeDistribution(
            {"num_atoms_histogram": dict(n_atoms_hist)}
        )
        self.prop_dist_model = None  # unconditional-only
        self.last_bond_types: list | None = None
        self.sdf_output_path = sdf_output_path

    # -- properties required by the contract ---------------------------------

    @property
    def model(self) -> "FlowMolGraph3DTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        """``{n_atoms: count}``, used by ``GenerativeFactory`` to clamp sizes."""
        return self.node_dist_model.n_node_dist

    def preprocess(self, train_set=None, valid_set=None, test_set=None):
        """Build self.property_norms for CFG conditioning (train-side only).

        Called generically by cli/train.py if this attribute exists. Does
        NOT touch node_dist_model/n_node_dist -- those come from
        dataset_stats/graph3d_stats at __init__ time via
        FlowMolGraph3DTaskFactory, a separate mechanism. Deliberately skips
        DistributionProperty/prop_dist_model (out of scope -- generation
        always takes an explicit target_value).
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

    def _normalize_target(self, value, key):
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

    # -- priors --------------------------------------------------------------

    def _sample_prior(
        self, g: dgl.DGLGraph, node_batch_idx: torch.Tensor, upper_edge_mask
    ) -> dgl.DGLGraph:
        """Centered-normal for ``x``, all-mask CTMC for ``a``/``c``/``e``."""
        n = g.num_nodes()
        g.ndata["x_0"] = centered_normal_prior_batched_graph(
            g, node_batch_idx, std=self.prior_std
        ).to(g.device)
        g.ndata["a_0"] = ctmc_masked_prior(n, self.n_atom_types).to(g.device)
        g.ndata["c_0"] = ctmc_masked_prior(n, self.n_charge_classes).to(
            g.device
        )
        g.edata["e_0"] = ctmc_masked_edge_prior(
            upper_edge_mask, self.n_bond_types
        ).to(g.device)
        return g

    # -- training ------------------------------------------------------------

    def forward(self, batch: Any) -> tuple[torch.Tensor, dict]:
        """One training step: interpolate, distort, denoise, weight the losses."""
        condition_per_mol = None
        if len(self.condition) > 0:
            if self.property_norms is None:
                raise RuntimeError(
                    "condition_names is set but property_norms is None -- "
                    "did preprocess() run? (cli/train.py calls it only if "
                    "hasattr(task, 'preprocess'))"
                )
            pyg = batch["graph"] if isinstance(batch, dict) else batch
            condition_per_node = prepare_context_pyG(
                self.condition, batch, self.property_norms, self.normalize_condition
            )
            # Molecule-level property, broadcast to every node of that
            # molecule by prepare_context_pyG -- take back the one row per
            # molecule (pyg.ptr[:-1] is the first node index of each graph).
            condition_per_mol = condition_per_node[pyg.ptr[:-1]]
            if self.context_mask_rate > 0:
                drop = torch.rand(condition_per_mol.size(0), device=condition_per_mol.device) < self.context_mask_rate
                if self.n_adapter_context > 0:
                    null_value = torch.empty(
                        condition_per_mol.shape[-1], device=condition_per_mol.device, dtype=condition_per_mol.dtype
                    )
                    null_value[self.adapter_indices] = 0.0
                    null_value[self.concat_indices] = self.mask_value
                else:
                    null_value = torch.full(
                        (condition_per_mol.shape[-1],), self.mask_value,
                        device=condition_per_mol.device, dtype=condition_per_mol.dtype,
                    )
                condition_per_mol = torch.where(drop.unsqueeze(-1), null_value.unsqueeze(0), condition_per_mol)

        g = self.to_dgl(batch, use_fake_atoms=True, condition=condition_per_mol)
        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_mask = get_upper_edge_mask(g)

        # The one cheap symmetry guard: the edge ordering (not the labels) is
        # the invariant everything downstream leans on.
        if int(upper_edge_mask.sum()) * 2 != int(g.num_edges()):
            msg = (
                "upper_edge_mask does not cover exactly half the edges -- the "
                "graph was not built with build_edge_idxs"
            )
            raise RuntimeError(msg)

        g = self._sample_prior(g, node_batch_idx, upper_edge_mask)

        t = torch.rand(g.batch_size, device=g.device).float()
        g = self.vector_field.sample_conditional_path(
            g, t, node_batch_idx, edge_batch_idx, upper_edge_mask
        )

        # Geometry distortion (train-only, upstream flowmol.py:333-337): jitter
        # a per-node Bernoulli(distort_p) subset once t is past distort_t, so the
        # model learns to correct its own late-trajectory coordinate errors.
        if self.distort_p > 0.0:
            t_mask = (t > self.distort_t)[node_batch_idx].unsqueeze(-1)
            distort_mask = (
                torch.rand(g.num_nodes(), 1, device=g.device) < self.distort_p
            ) & t_mask
            g.ndata["x_t"] = (
                g.ndata["x_t"]
                + torch.randn_like(g.ndata["x_t"]) * distort_mask * 0.5
            )

        vf_output = self.vector_field(
            g,
            t,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )

        losses = self._losses(
            g, t, vf_output, node_batch_idx, edge_batch_idx, upper_edge_mask
        )
        total_loss = sum(
            self.total_loss_weights[feat] * losses[feat]
            for feat in self.canonical_feat_order
        )
        stats = {f"{feat}_loss": losses[feat].detach() for feat in losses}
        stats["total_loss"] = total_loss.detach()
        return total_loss, stats

    def _losses(  # noqa: PLR0913
        self,
        g: dgl.DGLGraph,
        t: torch.Tensor,
        vf_output: dict,
        node_batch_idx: torch.Tensor,
        edge_batch_idx: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> dict:
        """Endpoint targets and per-modality, time-scaled losses."""
        time_weights = (
            self.interpolant_scheduler.loss_weights(t)
            if self.time_scaled_loss
            else None
        )

        losses = {}
        for feat_idx, feat in enumerate(self.canonical_feat_order):
            data_src = g.edata if feat == "e" else g.ndata
            target = data_src[f"{feat}_1_true"]
            if feat == "e":
                target = target[upper_edge_mask]

            if feat == "x":
                raw = self.loss_x(vf_output["x"], target).mean(dim=-1)
            else:
                target = target.argmax(dim=-1)
                # CTMC applies no loss on positions that are already unmasked:
                # there is nothing left to predict there, and including them
                # would drown the signal (upstream flowmol.py:378-385).
                xt = data_src[f"{feat}_t"]
                if feat == "e":
                    xt = xt[upper_edge_mask]
                target = target.clone()
                target[xt.argmax(-1) != self.vector_field.mask_idxs[feat]] = (
                    _IGNORE_INDEX
                )
                raw = self.loss_cat[feat](vf_output[feat], target)

            if time_weights is None:
                losses[feat] = raw.mean()
            else:
                w = time_weights[:, feat_idx]
                w = (
                    w[edge_batch_idx][upper_edge_mask]
                    if feat == "e"
                    else w[node_batch_idx]
                )
                losses[feat] = (raw * w).mean()

        return losses

    def predict_and_target(self, batch: Any):
        """Pure-generative stub: the loss is the only scalar to report."""
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # -- generation ----------------------------------------------------------

    def _ensure_accelerated(self) -> None:
        """Move self onto the GPU if it is sitting on the CPU by accident.

        ``cli/generate.py`` only places a task when it has no ``device``
        attribute; this class exposes ``device`` as a read-only *report* of
        where its parameters are, so that placement is skipped and the task
        stays wherever ``torch.load`` left it -- the CPU. Every device in
        ``sample()`` derives from ``self.device``, so the whole 250-step CTMC
        loop then runs on the CPU with the GPU idle (~10-40x slower), silently.

        No-op during training (Lightning has already placed the module) and on
        CPU-only machines. Honours ``CUDA_VISIBLE_DEVICES`` via ``cuda``'s
        current-device default.
        """
        if self.device.type == "cpu" and torch.cuda.is_available():
            logger.info(
                "sample(): parameters were on CPU; moving to %s",
                torch.cuda.current_device(),
            )
            self.to(torch.device("cuda"))
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    def _build_graphs(self, sizes: torch.Tensor) -> dgl.DGLGraph:
        graphs = []
        for n in sizes.tolist():
            n = int(n)
            edges = build_edge_idxs(n).to(self.device)
            graphs.append(
                dgl.graph(
                    (edges[0], edges[1]), num_nodes=n, device=self.device
                )
            )
        return dgl.batch(graphs)

    @torch.no_grad()
    def sample(  # noqa: PLR0913
        self,
        batch_size: int | None = None,
        nodesxsample: torch.Tensor | None = None,
        num_steps: int | None = None,
        batch: dict | None = None,
        mode: str | None = None,  # noqa: ARG002 - DDIM modes out of scope
        n_frames: int = 0,  # noqa: ARG002 - trajectories out of scope
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unconditional sampling from the all-mask prior.

        Returns the platform's ``(one_hot, charges, coords, node_mask)``.
        ``one_hot`` is **``len(atom_vocab)`` wide (10)** -- the fake-atom column
        is consumed internally and never surfaces. ``charges`` carries signed
        formal charges (MiDi's and FlowMol's precedent for that slot). The
        generated bond table has no channel in the tuple, so it is stashed on
        ``self.last_bond_types`` and, when ``sdf_output_path`` is set, written as
        an ``.sdf`` sidecar beside the platform's ``.xyz``.

        Because a node the model assigns to the fake class is physically deleted
        (upstream ``molecule_builder.py:226-231``), the produced atom count is
        ``<=`` the requested one. ``node_mask`` carries the truth; a mild size
        shortfall is upstream behaviour, not a port defect.
        """
        if num_steps is None:
            num_steps = self.fm_num_timesteps

        self._ensure_accelerated()

        if nodesxsample is not None:
            sizes = torch.as_tensor(
                nodesxsample, dtype=torch.long, device=self.device
            )
        elif batch is not None and "natoms" in batch:
            sizes = batch["natoms"].to(self.device).long()
        elif batch_size is not None:
            sizes = self.node_dist_model.sample(batch_size).to(self.device)
        else:
            msg = "sample() needs nodesxsample, batch, or batch_size"
            raise ValueError(msg)

        g = self._build_graphs(sizes)
        node_batch_idx = get_batch_idxs(g)[0]
        upper_edge_mask = get_upper_edge_mask(g)
        g = self._sample_prior(g, node_batch_idx, upper_edge_mask)

        g = self.vector_field.integrate(
            g,
            node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            n_timesteps=int(num_steps),
            stochasticity=self.stochasticity,
            high_confidence_threshold=self.high_confidence_threshold,
        )

        # Carried on edata so it survives dgl.unbatch (upstream flowmol.py:564).
        g.edata["ue_mask"] = upper_edge_mask
        return self._decode(g)

    @torch.no_grad()
    def sample_guidance_conitional(
        self,
        target_function=None,
        target_value=None,
        negative_target_value=None,
        nodesxsample=None,
        cfg_scale: float = 1,
        cfg_scale_schedule: str | None = None,
        guidance_ver: str = "cfg",
        n_frames: int = 0,
        num_steps: int | None = None,
        **kwargs: Any,
    ):
        """
        Classifier-free-guidance generation. Matches the call signature
        GenerativeFactory.conditional_generation() hardcodes for
        task_type == "cfg" (runmodes/generate/tasks_generate.py), and returns
        (one_hot, charges, x, node_mask) like sample() -- "EDM compatibility".
        """
        if guidance_ver != "cfg":
            msg = (
                f"FlowMolGraph3DTask only supports guidance_ver='cfg' (got {guidance_ver!r}); "
                "gradient-guidance variants are out of scope."
            )
            raise NotImplementedError(msg)
        if n_frames:
            logger.warning("n_frames=%s is not supported for FlowMol3 CFG sampling; ignoring.", n_frames)
        if num_steps is None:
            num_steps = self.fm_num_timesteps

        self._ensure_accelerated()
        sizes = torch.as_tensor(nodesxsample, dtype=torch.long, device=self.device)

        vals = [self._normalize_target(target_value[i], key) for i, key in enumerate(self.condition)]
        context_per_mol = torch.tensor(vals, dtype=torch.float, device=self.device).unsqueeze(0).expand(len(sizes), -1)

        if negative_target_value:
            neg = [self._normalize_target(negative_target_value[i], key) for i, key in enumerate(self.condition)]
            negative_context_per_mol = torch.tensor(neg, dtype=torch.float, device=self.device).unsqueeze(0).expand(len(sizes), -1)
        else:
            if self.n_adapter_context > 0:
                null_value = torch.empty(len(self.condition), device=self.device)
                null_value[self.adapter_indices] = 0.0
                null_value[self.concat_indices] = self.mask_value
            else:
                null_value = torch.full((len(self.condition),), self.mask_value, device=self.device)
            negative_context_per_mol = null_value.unsqueeze(0).expand(len(sizes), -1)

        g = self._build_graphs(sizes)
        node_batch_idx = get_batch_idxs(g)[0]
        upper_edge_mask = get_upper_edge_mask(g)
        g = self._sample_prior(g, node_batch_idx, upper_edge_mask)

        condition = context_per_mol[node_batch_idx]
        negative_condition = negative_context_per_mol[node_batch_idx]

        g = self.vector_field.integrate(
            g,
            node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            n_timesteps=int(num_steps),
            stochasticity=self.stochasticity,
            high_confidence_threshold=self.high_confidence_threshold,
            condition=condition,
            negative_condition=negative_condition,
            cfg_scale=cfg_scale,
            cfg_scale_schedule=cfg_scale_schedule,
        )

        g.edata["ue_mask"] = upper_edge_mask
        return self._decode(g)

    def _decode(self, g: dgl.DGLGraph):
        """Integrated graph -> padded platform tuple, fake atoms removed."""
        mols = []
        for gi in dgl.unbatch(g.to("cpu")):
            n = int(gi.num_nodes())
            atom_idx = gi.ndata["a_1"].argmax(dim=-1)
            charges = gi.ndata["c_1"].argmax(dim=-1) - self.charge_offset
            coords = gi.ndata["x_1"]

            ue = gi.edata["ue_mask"].bool()
            upper = torch.triu_indices(n, n, offset=1)
            bond_labels = gi.edata["e_1"][ue].argmax(dim=-1)
            keep_bond = bond_labels > 0
            bond_index = upper[:, keep_bond]
            bond_type = bond_labels[keep_bond]

            # Drop the nodes the model declined to use, and their bonds with
            # them. remap_bonds_after_atom_removal is the existing graph3d
            # helper and preserves i < j (the remap is monotone).
            keep = atom_idx != self.fake_atom_index
            if not bool(keep.all()):
                bond_index, bond_type = remap_bonds_after_atom_removal(
                    bond_index.numpy(), bond_type.numpy(), keep.numpy()
                )
                bond_index = torch.as_tensor(bond_index).reshape(2, -1)
                bond_type = torch.as_tensor(bond_type).reshape(-1)
                atom_idx = atom_idx[keep]
                charges = charges[keep]
                coords = coords[keep]

            mols.append((atom_idx, charges, coords, bond_index, bond_type))

        bs = len(mols)
        n_max = max((int(m[0].numel()) for m in mols), default=1)
        one_hot = torch.zeros(bs, n_max, self.n_real_atom_types)
        charges = torch.zeros(bs, n_max, dtype=torch.long)
        coords = torch.zeros(bs, n_max, 3)
        node_mask = torch.zeros(bs, n_max, dtype=torch.long)

        for i, (a_idx, chg, xyz, _bi, _bt) in enumerate(mols):
            n = int(a_idx.numel())
            if n == 0:
                continue
            one_hot[i, :n] = F.one_hot(a_idx, self.n_real_atom_types).float()
            charges[i, :n] = chg
            coords[i, :n] = xyz
            node_mask[i, :n] = 1

        self.last_bond_types = [(m[3], m[4]) for m in mols]
        if self.sdf_output_path is not None:
            self._write_sdf(mols)

        device = self.device
        return (
            one_hot.to(device),
            charges.to(device),
            coords.to(device),
            node_mask.to(device),
        )

    # -- .sdf sidecar --------------------------------------------------------

    def _write_sdf(self, mols: list) -> None:
        """Append the sampled molecules to ``sdf_output_path``.

        Opt-in (``sdf_output_path: null`` by default) because the platform's
        writer emits ``.xyz``, which has no bond channel. Reuses
        ``build_rdkit_mol`` from the graph3d dataset -- read from, never
        modified. Append mode, so multi-batch generation accumulates into one
        file; a molecule RDKit refuses to sanitize is warned about and skipped,
        never raised (one bad sample must not kill a generation run).
        """
        if Chem is None:
            logger.warning("RDKit unavailable, skipping .sdf sidecar")
            return
        from ase.data import atomic_numbers as _atomic_numbers

        path = self.sdf_output_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        z_of_vocab = [_atomic_numbers[s] for s in self.atom_vocab]

        with open(path, "a") as handle:  # noqa: PTH123
            writer = Chem.SDWriter(handle)
            for b, (
                atom_idx,
                charges,
                coords,
                bond_index,
                bond_type,
            ) in enumerate(mols):
                if int(atom_idx.numel()) == 0:
                    continue
                try:
                    mol = build_rdkit_mol(
                        [z_of_vocab[int(i)] for i in atom_idx],
                        bond_index.numpy(),
                        bond_type.numpy(),
                        formal_charge=charges.numpy(),
                        coords=coords.numpy(),
                    )
                    writer.write(mol)
                except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
                    logger.warning(
                        "Skipping unsanitizable sample %d: %s", b, exc
                    )
            writer.close()


#: Alias so ``configs/tasks/diffusion_flowmol_graph3d.yaml`` can use the same
#: ``ModelTaskFactory`` name every other bundled task config uses.
ModelTaskFactory = FlowMolGraph3DTaskFactory
