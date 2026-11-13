"""MiDi task: joint diffusion over coordinates, atom types, bonds and charges.

MiDi (Vignac et al., ECML 2023, arXiv:2302.09048) is the first model in this
platform that generates the *molecular graph itself* -- bond orders and formal
charges are diffused jointly with the 3D coordinates, so a sample arrives with
an explicit bond table instead of needing post-hoc perception.

Data path: ``data_type: graph3d`` with ``bond_collate: dense``.
``graph3d_dense_collate`` already produces MiDi's exact dense shapes, so the
adapter below is a handful of ``F.one_hot`` calls and upstream's
``utils.to_dense`` (the PyG -> dense bridge) is not needed at all.

Bond classes are the platform's canonical five (``0=none, 1=SINGLE, 2=DOUBLE,
3=TRIPLE, 4=AROMATIC``) and MiDi uses exactly those, in the same order -- the
mapping is the identity.

Formal charges are stored raw and signed; the offset and class count are
applied here (``charge_offset``/``n_charge_classes``: QM9 -> +1/3,
GEOM -> +2/6), never baked into the dataset.

Out of scope this pass (see the integration plan): ``ExtraFeatures`` (every
released config sets ``extra_features: null``), the variational-NLL validation
path, MiDi's own molecular metrics, and the size-aware loader.

Property-conditioning + classifier-free guidance use the same config
signature as ``configs/tasks/diffusion.yaml`` (en_diffusion's formulation):
set ``condition_names`` to opt in. The injection point is ``PlaceHolder.y``,
MiDi's own per-graph global feature, already FiLM-injected into every
transformer layer -- see ``modules/models/midi/transformer_model.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from MolecularDiffusion.data.datasets.graph3d_dataset import (
    N_BOND_CLASSES,
    build_rdkit_mol,
)
from MolecularDiffusion.modules.models.midi import (
    Dims,
    DiscreteUniformTransition,
    GraphTransformer,
    MarginalUniformTransition,
    PlaceHolder,
)

# Histogram-backed size sampler already implemented for TABASCO and reused by
# FlowMol; MiDi's own DistributionNodes adds only log_prob, which serves the
# out-of-scope variational NLL.
from MolecularDiffusion.modules.tasks.flow.tabasco import (
    TabascoNodeDistribution,
)
from MolecularDiffusion.utils import compute_mean_mad_from_dataloader

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:  # only the optional .sdf sidecar needs RDKit
    Chem = None


def _charge_marginals(
    stats: Any, n_atom_types: int, charge_offset: int, n_charge_classes: int
) -> torch.Tensor:
    """Rebuild MiDi's ``charges_marginals`` from ``graph3d_stats``.

    Mirrors ``AbstractDatasetInfos.complete_infos`` (upstream
    ``abstract_dataset.py:145``): a per-atom-type charge distribution
    ``(K, C)``, row-normalized, weighted by the atom-type marginal.
    """
    charge_types = np.zeros((n_atom_types, n_charge_classes), dtype=np.float64)
    for atom_type, per_charge in stats.charge_counts.items():
        if atom_type >= n_atom_types:
            continue
        for raw_charge, count in per_charge.items():
            cls = int(raw_charge) + charge_offset
            if not 0 <= cls < n_charge_classes:
                msg = (
                    f"formal charge {raw_charge} on atom type {atom_type} maps "
                    f"to class {cls}, outside [0, {n_charge_classes}). Raise "
                    "n_charge_classes / adjust charge_offset (QM9: +1/3, "
                    "GEOM: +2/6)."
                )
                raise ValueError(msg)
            charge_types[atom_type, cls] += count

    row_sums = charge_types.sum(axis=1, keepdims=True)
    charge_types = np.divide(
        charge_types, row_sums, out=np.zeros_like(charge_types), where=row_sums > 0
    )
    atom_marginal = np.asarray(stats.atom_type_marginal(), dtype=np.float64)
    atom_marginal = atom_marginal[:n_atom_types]
    marginals = (charge_types * atom_marginal[:, None]).sum(axis=0)
    total = marginals.sum()
    if total <= 0:
        msg = "charge marginals are all zero -- graph3d_stats has no charges"
        raise ValueError(msg)
    return torch.from_numpy(marginals / total).float()


def _or_uniform(
    marginal: Optional[torch.Tensor], n_classes: int
) -> torch.Tensor:
    """Return ``marginal``, or a uniform placeholder of ``n_classes``."""
    if marginal is None:
        return torch.ones(n_classes) / n_classes
    return marginal.detach().clone().float()


class ModelTaskFactory:
    """Hydra entry point for the MiDi task (``configs/tasks/diffusion_midi``).

    Declares ``train_set`` so ``cli/train.py``'s declarative seam injects the
    training dataset: MiDi's noise model needs the atom/bond/charge marginals
    and the size histogram *at construction time*, and neither is an
    ``nn.Module`` buffer, so neither survives in a checkpoint.

    ``sdf_output_path`` is declared generation-time (docs §2.5b): the task is
    rebuilt from the checkpoint's training-time config, where it is ``null``,
    so without this declaration the generate config's value never arrives and
    the bond sidecar -- the whole 2D half of the model -- is silently dropped.
    """

    generation_time_keys = ("sdf_output_path",)

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_midi",
        n_layers: int = 12,
        hidden_mlp_dims: Optional[dict] = None,
        hidden_dims: Optional[dict] = None,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "cosine",
        transition: str = "marginal",
        nu: Optional[dict] = None,
        lambda_train: Optional[list] = None,
        charge_offset: int = 1,
        n_charge_classes: int = 3,
        sdf_output_path: Optional[str] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.n_layers = n_layers
        self.hidden_mlp_dims = dict(
            hidden_mlp_dims or {"X": 256, "E": 64, "y": 256, "pos": 64}
        )
        self.hidden_dims = dict(
            hidden_dims
            or {
                "dx": 256,
                "de": 64,
                "dy": 128,
                "n_head": 8,
                "dim_ffX": 256,
                "dim_ffE": 64,
                "dim_ffy": 256,
            }
        )
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.transition = transition
        self.nu = dict(nu or {"p": 2.5, "x": 1, "c": 1, "e": 1.5, "y": 1})
        self.lambda_train = list(lambda_train or [3, 0.4, 1, 2, 0])
        self.charge_offset = charge_offset
        self.n_charge_classes = n_charge_classes
        self.sdf_output_path = sdf_output_path
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: Optional[MidiDiffusionTask] = None

    def build(self) -> MidiDiffusionTask:
        """Construct the task, deriving marginals/size histogram from data."""
        if not self.atom_vocab:
            msg = (
                "MiDi needs atom_vocab (it sizes the atom-type head). Set "
                "data.atom_vocab, or tasks.atom_vocab explicitly."
            )
            raise ValueError(msg)
        n_atom_types = len(self.atom_vocab)

        stats = getattr(self.train_set, "graph3d_stats", None)
        x_marginals = e_marginals = charges_marginals = None
        n_atoms_hist: dict[int, int] = {}

        if stats is not None:
            atom_marginal = np.asarray(stats.atom_type_marginal())
            if atom_marginal.shape[0] < n_atom_types:
                atom_marginal = np.pad(
                    atom_marginal, (0, n_atom_types - atom_marginal.shape[0])
                )
            x_marginals = torch.from_numpy(
                np.asarray(atom_marginal[:n_atom_types], dtype=np.float64)
            ).float()
            e_marginals = torch.from_numpy(
                np.asarray(stats.bond_type_marginal(), dtype=np.float64)
            ).float()
            charges_marginals = _charge_marginals(
                stats,
                n_atom_types,
                self.charge_offset,
                self.n_charge_classes,
            )
            n_atoms_hist = {int(k): int(v) for k, v in stats.n_atoms_hist.items()}
            logger.info(
                "MiDi marginals from graph3d_stats over %d molecules: "
                "atoms=%s bonds=%s charges=%s",
                stats.n_molecules,
                np.round(x_marginals.numpy(), 4).tolist(),
                np.round(e_marginals.numpy(), 4).tolist(),
                np.round(charges_marginals.numpy(), 4).tolist(),
            )
        else:
            # No train_set: this is the generation path (cli/generate.py never
            # builds a DataModule). The marginals are registered buffers, so
            # the real ones arrive with the checkpoint's state_dict a moment
            # later and overwrite these placeholders in place.
            logger.warning(
                "No train_set.graph3d_stats available -- building MiDi with "
                "UNIFORM placeholder marginals and an empty size histogram. "
                "Expected when loading a checkpoint for generation (both are "
                "restored from it); if this appears during TRAINING, set "
                "data.data_type=graph3d with graph3d_stats: true."
            )

        self.task = MidiDiffusionTask(
            atom_vocab=self.atom_vocab,
            n_layers=self.n_layers,
            hidden_mlp_dims=self.hidden_mlp_dims,
            hidden_dims=self.hidden_dims,
            diffusion_steps=self.diffusion_steps,
            diffusion_noise_schedule=self.diffusion_noise_schedule,
            transition=self.transition,
            nu=self.nu,
            lambda_train=self.lambda_train,
            charge_offset=self.charge_offset,
            n_charge_classes=self.n_charge_classes,
            sdf_output_path=self.sdf_output_path,
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            charges_marginals=charges_marginals,
            n_atoms_hist=n_atoms_hist,
            condition_names=self.kwargs.get("condition_names", []),
            context_mask_rate=self.kwargs.get("context_mask_rate", 0.0),
            mask_value=self.kwargs.get("mask_value", 0.0),
            normalize_condition=self.kwargs.get("normalize_condition", None),
            adapter_conditions=self.kwargs.get("adapter_conditions", None),
            use_adapter_module=self.kwargs.get("use_adapter_module", False),
        )
        return self.task


class MidiDiffusionTask(nn.Module):
    """MiDi wrapped in the platform's duck-typed Task contract (§2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        atom_vocab: list,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        diffusion_steps: int,
        diffusion_noise_schedule: str,
        transition: str,
        nu: dict,
        lambda_train: list,
        charge_offset: int,
        n_charge_classes: int,
        sdf_output_path: Optional[str],
        x_marginals: Optional[torch.Tensor],
        e_marginals: Optional[torch.Tensor],
        charges_marginals: Optional[torch.Tensor],
        n_atoms_hist: dict,
        condition_names: list = [],
        context_mask_rate: float = 0.0,
        mask_value: float = 0.0,
        normalize_condition: Optional[str] = None,
        adapter_conditions: Optional[list] = None,
        use_adapter_module: bool = False,
    ) -> None:
        super().__init__()
        self.task_type = "diffusion_midi"
        self.atom_vocab = list(atom_vocab)
        self.n_atom_types = len(self.atom_vocab)
        self.charge_offset = charge_offset
        self.n_charge_classes = n_charge_classes
        self.lambda_train = list(lambda_train)
        self.sdf_output_path = sdf_output_path

        # Property-conditioning / CFG setup -- same config signature as
        # en_diffusion.py's GeomMolecularGenerative / TABASCO / FlowMol.
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
            self.adapter_indices = [
                condition_names.index(ac) for ac in adapter_conditions
            ]
            self.concat_indices = [
                i
                for i in range(len(condition_names))
                if i not in self.adapter_indices
            ]
        elif use_adapter_module:
            self.adapter_indices = list(range(len(condition_names)))
            self.concat_indices = []
        else:
            self.adapter_indices = []
            self.concat_indices = list(range(len(condition_names)))
        self.n_adapter_context = len(self.adapter_indices)
        self.n_concat_context = len(self.concat_indices)

        self.input_dims = Dims(
            X=self.n_atom_types,
            charges=n_charge_classes,
            E=N_BOND_CLASSES,
            y=1,  # the timestep; extra_features is null
            pos=3,
        )
        self.output_dims = Dims(
            X=self.n_atom_types,
            charges=n_charge_classes,
            E=N_BOND_CLASSES,
            y=0,  # unconditional
            pos=3,
        )

        self.backbone = GraphTransformer(
            input_dims=self.input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=self.output_dims,
            adapter_indices=self.adapter_indices,
            concat_indices=self.concat_indices,
        )

        # The class marginals are dataset statistics, not learned weights --
        # but they are registered as BUFFERS on purpose. cli/generate.py never
        # builds a DataModule, so the only way they can reach a generation run
        # is through the checkpoint's state_dict. NoiseModel is a plain Python
        # object holding tensors, so `_sync_marginals()` re-derives its
        # transition matrices from these buffers before every use -- covering
        # both `load_state_dict` and `.to(device)`, either of which can swap
        # the underlying tensor out from under it.
        self.register_buffer(
            "x_marginals",
            _or_uniform(x_marginals, self.n_atom_types),
        )
        self.register_buffer(
            "e_marginals",
            _or_uniform(e_marginals, N_BOND_CLASSES),
        )
        self.register_buffer(
            "charges_marginals",
            _or_uniform(charges_marginals, n_charge_classes),
        )

        if transition == "uniform":
            self.noise_model = DiscreteUniformTransition(
                output_dims=self.output_dims,
                nu=nu,
                diffusion_steps=diffusion_steps,
                noise_schedule=diffusion_noise_schedule,
            )
        elif transition == "marginal":
            self.noise_model = MarginalUniformTransition(
                x_marginals=self.x_marginals,
                e_marginals=self.e_marginals,
                charges_marginals=self.charges_marginals,
                y_classes=self.output_dims.y,
                nu=nu,
                diffusion_steps=diffusion_steps,
                noise_schedule=diffusion_noise_schedule,
            )
        else:
            msg = f"Unknown transition type '{transition}'"
            raise ValueError(msg)

        # Class index -> signed formal charge, MiDi's `collapse_charges`.
        self.register_buffer(
            "collapse_charges",
            torch.arange(n_charge_classes, dtype=torch.long) - charge_offset,
        )

        # Number of reverse steps to take. Distinct from noise_model.T (the
        # schedule length, 500): a smaller value strides the schedule, which is
        # MiDi's `general.faster_sampling`. cli/generate.py's `total_step`
        # override writes here via the `model` property.
        self.T = diffusion_steps

        self.node_dist_model = TabascoNodeDistribution(
            {"num_atoms_histogram": n_atoms_hist}
        )
        self.prop_dist_model = None  # unconditional-only
        self.last_bond_types: Optional[torch.Tensor] = None

    # -- properties required by the contract --------------------------------

    @property
    def model(self) -> MidiDiffusionTask:
        """``tasks_generate.py`` reads ``task.model.T``; self is the model."""
        return self

    @property
    def device(self) -> torch.device:
        """Device of the backbone parameters."""
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        """``{n_atoms: count}`` histogram used to clamp ``mol_size``."""
        return self.node_dist_model.n_node_dist

    def preprocess(self, train_set=None, valid_set=None, test_set=None):
        """Build self.property_norms for CFG conditioning (train-side only).

        Called generically by cli/train.py if this attribute exists. Does
        NOT touch node_dist_model/n_node_dist -- those come from
        graph3d_stats at __init__ time via ModelTaskFactory, a separate
        mechanism. Deliberately skips DistributionProperty/prop_dist_model
        (out of scope -- generation always takes an explicit target_value).
        """
        if train_set is None or len(self.condition) == 0:
            return
        from MolecularDiffusion.modules.tasks import _preprocess_cache as _ppcache

        base, subset_indices = _ppcache.resolve_dataset_and_indices(train_set)
        prop_indices = _ppcache.property_sample_indices(
            len(train_set), subset_indices
        )
        props = torch.stack(
            [
                _ppcache.get_property_subset(base, name, prop_indices)
                for name in self.condition
            ]
        )
        self.property_norms = compute_mean_mad_from_dataloader(
            props, self.condition
        )

    def _normalize_target(
        self, value: torch.Tensor, key: str
    ) -> torch.Tensor:
        if self.normalize_condition is None:
            return value
        norms = self.property_norms[key]
        if self.normalize_condition == "mad":
            return (value - norms["mean"]) / norms["mad"]
        if self.normalize_condition == "maxmin":
            return (
                2 * (value - norms["min"]) / (norms["max"] - norms["min"])
                - 1
            )
        if "value" in self.normalize_condition:
            return value / float(self.normalize_condition.split("_")[1])
        msg = f"Unknown normalization method: {self.normalize_condition}"
        raise ValueError(msg)

    def _null_condition(self, device: torch.device) -> torch.Tensor:
        """Adapter/concat-aware null vector for dropout and negative CFG."""
        d = len(self.condition)
        if self.n_adapter_context > 0:
            null_value = torch.empty(d, device=device)
            null_value[self.adapter_indices] = 0.0
            null_value[self.concat_indices] = self.mask_value
        else:
            null_value = torch.full((d,), self.mask_value, device=device)
        return null_value

    def _sync_marginals(self) -> None:
        """Point the noise model at the current marginal buffers.

        Cheap (``expand`` allocates no data) and idempotent, so it just runs
        before every forward/sample rather than trying to hook every event
        that could replace a buffer tensor.
        """
        nm = self.noise_model
        if not isinstance(nm, MarginalUniformTransition):
            return
        nm.X_marginals = self.x_marginals
        nm.E_marginals = self.e_marginals
        nm.charges_marginals = self.charges_marginals
        nm.Px = (
            self.x_marginals.unsqueeze(0).expand(nm.X_classes, -1).unsqueeze(0)
        )
        nm.Pe = (
            self.e_marginals.unsqueeze(0).expand(nm.E_classes, -1).unsqueeze(0)
        )
        nm.Pcharges = (
            self.charges_marginals.unsqueeze(0)
            .expand(nm.charges_classes, -1)
            .unsqueeze(0)
        )

    # -- adapters -----------------------------------------------------------

    def _to_placeholder(self, batch: dict) -> PlaceHolder:
        """``graph3d_dense_collate`` dict -> MiDi's dense ``PlaceHolder``.

        This replaces upstream ``utils.to_dense`` outright: the collate
        already emits ``pos (B,N,3)``, ``atom_idx (B,N)``, ``charges (B,N)``
        (raw signed), ``bond_type (B,N,N)`` (symmetric integer class ids with a
        zero diagonal) and ``node_mask (B,N)``.
        """
        pos = batch["pos"].float()
        node_mask = batch["node_mask"].bool()
        atom_idx = batch["atom_idx"].long()
        raw_charges = batch["charges"].long()
        bond = batch["bond_type"].long()

        shifted = raw_charges + self.charge_offset
        # Padded rows carry charge 0 -> class == charge_offset, in range.
        if int(shifted.min()) < 0 or int(shifted.max()) >= self.n_charge_classes:
            observed = (int(raw_charges.min()), int(raw_charges.max()))
            msg = (
                f"formal charges {observed} do not fit "
                f"{self.n_charge_classes} classes at offset "
                f"{self.charge_offset} (QM9: +1/3, GEOM: +2/6)"
            )
            raise ValueError(msg)

        x = F.one_hot(atom_idx, self.n_atom_types).float()
        charges = F.one_hot(shifted, self.n_charge_classes).float()
        e = F.one_hot(bond, N_BOND_CLASSES).float()
        y = pos.new_zeros((pos.size(0), 0))

        return PlaceHolder(
            pos=pos, X=x, charges=charges, E=e, y=y, node_mask=node_mask
        ).mask()

    def _denoise(
        self, z_t: PlaceHolder, condition: Optional[torch.Tensor] = None
    ) -> PlaceHolder:
        """Run the backbone on a noised batch, appending ``t`` to ``y``."""
        model_input = z_t.copy()
        model_input.X = z_t.X.float()
        model_input.charges = z_t.charges.float()
        model_input.E = z_t.E.float()
        model_input.y = torch.hstack((z_t.y, z_t.t)).float()
        return self.backbone(model_input, condition=condition)

    # -- training -----------------------------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One training step: noise the batch, denoise it, weight the losses."""
        self._sync_marginals()
        dense_data = self._to_placeholder(batch)
        z_t = self.noise_model.apply_noise(dense_data)

        condition = None
        if len(self.condition) > 0:
            if self.property_norms is None:
                raise RuntimeError(
                    "condition_names is set but property_norms is None -- "
                    "did preprocess() run? (cli/train.py calls it only if "
                    "hasattr(task, 'preprocess'))"
                )
            device = dense_data.pos.device
            vals = [
                self._normalize_target(batch[key].float().to(device), key)
                for key in self.condition
            ]
            condition = torch.stack(vals, dim=-1)
            if self.context_mask_rate > 0:
                drop = (
                    torch.rand(condition.size(0), device=device)
                    < self.context_mask_rate
                )
                null_value = self._null_condition(device)
                condition = torch.where(
                    drop.unsqueeze(-1), null_value.unsqueeze(0), condition
                )

        pred = self._denoise(z_t, condition=condition)
        return self._loss(pred, dense_data)

    def _loss(
        self, pred: PlaceHolder, true: PlaceHolder
    ) -> tuple[torch.Tensor, dict]:
        """MiDi's ``TrainLoss`` as plain tensor ops.

        Upstream accumulates through torchmetrics/wandb; the platform's logger
        already does that, so only the weighted scalar and a stats dict are
        produced here. The reductions match: ``MeanSquaredError`` == mean-MSE,
        ``CrossEntropyMetric`` == sum CE / n == mean CE.
        """
        node_mask = true.node_mask
        bs, n = node_mask.shape

        pos_mse = F.mse_loss(pred.pos[node_mask], true.pos[node_mask])
        x_ce = F.cross_entropy(
            pred.X[node_mask], true.X[node_mask].argmax(dim=-1)
        )
        charges_ce = F.cross_entropy(
            pred.charges[node_mask], true.charges[node_mask].argmax(dim=-1)
        )

        diag_mask = ~torch.eye(
            n, device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0).expand(bs, -1, -1)
        edge_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        e_ce = F.cross_entropy(
            pred.E[edge_mask], true.E[edge_mask].argmax(dim=-1)
        )

        loss = (
            self.lambda_train[0] * pos_mse
            + self.lambda_train[1] * x_ce
            + self.lambda_train[2] * charges_ce
            + self.lambda_train[3] * e_ce
        )
        stats = {
            "pos_mse": pos_mse.detach(),
            "x_ce": x_ce.detach(),
            "charges_ce": charges_ce.detach(),
            "e_ce": e_ce.detach(),
        }
        return loss, stats

    def predict_and_target(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure-generative stub: the loss is both prediction and target."""
        loss, _ = self.forward(batch)
        loss = loss.detach().reshape(1)
        return loss, torch.zeros_like(loss)

    def evaluate(
        self, pred: torch.Tensor, target: torch.Tensor  # noqa: ARG002
    ) -> dict:
        """Reduce accumulated losses into the logged validation metric."""
        return {"val_loss": pred.mean()}

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def sample(  # noqa: PLR0913
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[dict] = None,  # noqa: ARG002
        mode: Optional[str] = None,  # noqa: ARG002 - DDIM modes out of scope
        n_frames: int = 0,  # noqa: ARG002 - trajectories out of scope
        condition: Optional[torch.Tensor] = None,
        negative_condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        cfg_scale_schedule: Optional[str] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampling, unconditional unless ``condition``/``cfg_scale`` are set.

        Returns the platform's ``(one_hot, charges, coords, node_mask)``
        tuple. ``charges`` carries **signed formal charges** (FlowMol's
        precedent for this slot) -- the element identity is in ``one_hot``.
        The generated bond matrix has no channel in that tuple, so it is
        stashed on ``self.last_bond_types`` and, when ``sdf_output_path`` is
        set, written as an ``.sdf`` sidecar alongside the platform's ``.xyz``.
        """
        self._sync_marginals()
        device = self.device
        if nodesxsample is None:
            if batch_size is None:
                msg = "sample() needs either nodesxsample or batch_size"
                raise ValueError(msg)
            nodesxsample = self.node_dist_model.sample(batch_size)
        n_nodes = torch.as_tensor(nodesxsample, dtype=torch.long, device=device)
        bs = int(n_nodes.numel())
        n_max = int(n_nodes.max().item())

        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        z_t = self.noise_model.sample_limit_dist(node_mask=node_mask)

        schedule_t = self.noise_model.T
        requested = int(num_steps or self.T)
        stride = max(1, schedule_t // max(requested, 1))
        for s_int in reversed(range(0, schedule_t, stride)):
            s_array = s_int * torch.ones(
                (bs, 1), dtype=torch.long, device=device
            )
            if condition is not None and cfg_scale != 0.0:
                step_scale = (
                    self._scale_schedule(
                        1 - z_t.t, cfg_scale, cfg_scale_schedule
                    )
                    if cfg_scale_schedule
                    else cfg_scale
                )
                pred_cond = self._denoise(z_t, condition=condition)
                pred_uncond = self._denoise(
                    z_t, condition=negative_condition
                )
                w = step_scale
                pred = PlaceHolder(
                    pos=(1 + w) * pred_cond.pos - w * pred_uncond.pos,
                    X=(1 + w) * pred_cond.X - w * pred_uncond.X,
                    charges=(1 + w) * pred_cond.charges
                    - w * pred_uncond.charges,
                    E=(1 + w) * pred_cond.E - w * pred_uncond.E,
                    y=pred_cond.y,
                    node_mask=pred_cond.node_mask,
                )
            else:
                pred = self._denoise(z_t, condition=condition)
            z_t = self.noise_model.sample_zs_from_zt_and_pred(
                z_t=z_t, pred=pred, s_int=s_array
            )

        final = z_t.collapse(self.collapse_charges)
        # collapse marks padding out of range (X=-1, charges=1000, E=-1);
        # clamp before anything downstream indexes with it.
        atom_idx = final.X.clamp(min=0)
        one_hot = F.one_hot(atom_idx, self.n_atom_types).float()
        one_hot = one_hot * node_mask.unsqueeze(-1)
        charges = torch.where(
            node_mask, final.charges, torch.zeros_like(final.charges)
        )
        bond_types = final.E.clamp(min=0)
        coords = final.pos * node_mask.unsqueeze(-1)

        self.last_bond_types = bond_types
        if self.sdf_output_path is not None:
            self._write_sdf(atom_idx, charges, bond_types, coords, node_mask)

        return one_hot, charges, coords, node_mask.long()

    @staticmethod
    def _scale_schedule(
        t: torch.Tensor, cfg_scale: float, schedule_type: str
    ) -> float:
        x = float(t)
        schedule_type = schedule_type.lower()
        if schedule_type == "linear":
            return cfg_scale * x
        if schedule_type == "exponential":
            return cfg_scale * (x**2)
        if schedule_type == "cosine":
            import math

            return cfg_scale * (1 - math.cos(x * math.pi / 2))
        msg = f"Unknown scale schedule: {schedule_type}"
        raise ValueError(msg)

    @torch.no_grad()
    def sample_guidance_conitional(  # noqa: PLR0913
        self,
        target_function: Any = None,  # noqa: ARG002
        target_value: Optional[list] = None,
        negative_target_value: Optional[list] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        cfg_scale: float = 1,
        cfg_scale_schedule: Optional[str] = None,
        guidance_ver: str = "cfg",
        n_frames: int = 0,  # noqa: ARG002 - trajectories out of scope
        num_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Classifier-free-guidance generation.

        Matches the call signature ``GenerativeFactory.conditional_generation()``
        hardcodes for ``task_type == "cfg"``, and returns
        ``(one_hot, charges, x, node_mask)`` like ``sample()``.
        """
        if guidance_ver != "cfg":
            msg = (
                f"MidiDiffusionTask only supports guidance_ver='cfg' (got "
                f"{guidance_ver!r}); gradient-guidance variants are out of "
                "scope."
            )
            raise NotImplementedError(msg)

        device = self.device
        n_nodes = torch.as_tensor(
            nodesxsample, dtype=torch.long, device=device
        )
        bs = int(n_nodes.numel())

        vals = [
            self._normalize_target(
                torch.as_tensor(target_value[i], device=device), key
            )
            for i, key in enumerate(self.condition)
        ]
        condition = (
            torch.stack(vals).unsqueeze(0).expand(bs, -1).float()
        )

        if negative_target_value:
            neg = [
                self._normalize_target(
                    torch.as_tensor(negative_target_value[i], device=device),
                    key,
                )
                for i, key in enumerate(self.condition)
            ]
            negative_condition = (
                torch.stack(neg).unsqueeze(0).expand(bs, -1).float()
            )
        else:
            negative_condition = (
                self._null_condition(device).unsqueeze(0).expand(bs, -1)
            )

        return self.sample(
            nodesxsample=nodesxsample,
            num_steps=num_steps,
            condition=condition,
            negative_condition=negative_condition,
            cfg_scale=cfg_scale,
            cfg_scale_schedule=cfg_scale_schedule,
        )

    def _write_sdf(  # noqa: PLR0913
        self,
        atom_idx: torch.Tensor,
        charges: torch.Tensor,
        bond_types: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        """Append the sampled molecules to ``sdf_output_path``.

        Opt-in (``sdf_output_path: null`` by default) because the platform's
        writer emits ``.xyz``, which has no bond channel. Molecule building
        reuses ``build_rdkit_mol`` from the graph3d dataset -- that file is
        read from, never modified. Append mode so multi-batch generation
        accumulates into one file; a molecule RDKit refuses to sanitize is
        warned about and skipped, never raised: one bad sample must not kill a
        generation run.
        """
        if Chem is None:
            logger.warning("RDKit unavailable, skipping .sdf sidecar")
            return
        from ase.data import atomic_numbers as _atomic_numbers

        parent = os.path.dirname(os.path.abspath(self.sdf_output_path))
        os.makedirs(parent, exist_ok=True)

        z_of_vocab = [_atomic_numbers[s] for s in self.atom_vocab]
        atom_idx = atom_idx.cpu()
        charges = charges.cpu()
        bond_types = bond_types.cpu()
        coords = coords.cpu()
        node_mask = node_mask.cpu()

        with open(self.sdf_output_path, "a") as handle:  # noqa: PTH123
            writer = Chem.SDWriter(handle)
            for b in range(atom_idx.size(0)):
                n = int(node_mask[b].sum())
                if n == 0:
                    continue
                zs = [z_of_vocab[int(i)] for i in atom_idx[b, :n]]
                sub = bond_types[b, :n, :n]
                rows, cols = torch.triu_indices(n, n, offset=1)
                keep = sub[rows, cols] > 0
                bond_index = torch.stack((rows[keep], cols[keep])).numpy()
                bond_type = sub[rows, cols][keep].numpy()
                try:
                    mol = build_rdkit_mol(
                        zs,
                        bond_index,
                        bond_type,
                        formal_charge=charges[b, :n].numpy(),
                        coords=coords[b, :n].numpy(),
                    )
                    writer.write(mol)
                except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
                    logger.warning("Skipping unsanitizable sample %d: %s", b, exc)
            writer.close()
