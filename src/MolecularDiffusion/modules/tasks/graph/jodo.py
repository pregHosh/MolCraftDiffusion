"""JODO task: joint continuous diffusion over coordinates, atom types, formal
charges *and* an explicit bond tensor.

JODO (Huang et al., NeurIPS 2023, arXiv:2305.12347) diffuses the 2D molecular
graph and the 3D geometry together in one continuous VP process: a sample
arrives with real bond orders and formal charges, no geometry-based perception
needed. What separates it from MiDi -- the platform's other bond-generating
model -- is that bonds are *continuous* here, not categorical: a bond is two
(or three) float channels that get thresholded back to a class at decode time.

Data path: ``data_type: graph3d`` with ``bond_collate: dense``.
``graph3d_dense_collate`` already emits the padded ``(B,N,N)`` integer bond
matrix, so the adapter below is a handful of vectorized lines -- upstream's
``collate_edge`` / ``EdgeComCondTransform`` are not needed at all.

Bond mapping (identity on the class ids, then JODO's compressed encoding)::

    canonical   JODO class   channels (exist, order[, aromatic])
    0 = none    0            (0, 0   [, 0])
    1 = SINGLE  1            (1, 1/3 [, 0])
    2 = DOUBLE  2            (1, 2/3 [, 0])
    3 = TRIPLE  3            (1, 1   [, 0])
    4 = AROMATIC 4           (1, 0   ,  1)     -- only when edge_ch == 3

``edge_ch: 2`` (QM9) has no aromatic channel, so the data config MUST set
``kekulize: true``; an aromatic bond would otherwise silently become "no bond"
(upstream ``datasets/build_dataset.py:151``). ``edge_ch: 3`` (GEOM) keeps
aromatic and takes ``kekulize: false``.

Formal charges are the one place JODO differs from MiDi: there is no
categorical charge head. The raw signed charge goes into a single float
channel divided by ``fc_charge_norm`` and is rounded back at decode.

Conditional generation rides the platform's existing
``GenerativeFactory.conditional_generation`` seam -- ``condition: [gap]``
selects ``Cond_DGT_concat`` and this class implements ``sample_conditonal``.

Out of scope this pass (see the integration plan): multi-property
conditioning, the EGNN property classifiers (an evaluation tool, not the
model), 2D-only mode, DPM-solver fast sampling, and trajectories/inpainting.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Optional

import torch
from MolecularDiffusion.device import get_device
from torch import nn
from torch.nn import functional as F  # noqa: N812

from MolecularDiffusion.data.datasets.graph3d_dataset import build_rdkit_mol
from MolecularDiffusion.modules.models.edm.en_diffusion import DistributionProperty
from MolecularDiffusion.modules.models.jodo import (
    Cond_DGT_concat,
    DGT_concat,
    NoiseScheduleVP,
)
from MolecularDiffusion.modules.models.jodo.utils import (
    remove_mean_with_mask,
    sample_combined_position_feature_noise,
    sample_symmetric_edge_feature_noise,
)

# Reused rather than re-implemented: the histogram-backed size sampler
# (TABASCO's, already shared by FlowMol and MiDi).
from MolecularDiffusion.modules.tasks.flow.tabasco import (
    TabascoNodeDistribution,
)
from MolecularDiffusion.utils.diffusion_utils import (
    compute_mean_mad_from_dataloader,
)

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
except ImportError:  # only the optional .sdf sidecar needs RDKit
    Chem = None


# JODO's own bond-order encoding: class id -> the `order` channel value.
# class 4 (aromatic) rides the third channel instead, hence order 0.
_ORDER_OF_CLASS = (0.0, 1 / 3, 2 / 3, 1.0, 0.0)
_AROMATIC_CLASS = 4


def expand_dims(v: torch.Tensor, dims: int) -> torch.Tensor:
    """``v`` of shape ``[N]`` -> ``[N, 1, ..., 1]`` with ``dims`` dimensions."""
    return v[(...,) + (None,) * (dims - 1)]


@torch.no_grad()
def kabsch_batch(
    coords_pred: torch.Tensor, coords_tar: torch.Tensor
) -> torch.Tensor:
    """Batched Kabsch rotation, port of upstream ``losses.py:424``."""
    a = torch.einsum("...ki, ...kj -> ...ij", coords_pred, coords_tar)
    u, _, vt = torch.linalg.svd(a)
    corr_diag = torch.ones((a.size(0), u.size(-1)), device=a.device)
    corr_diag[:, -1] = torch.sign(torch.det(a))
    corr = torch.diag_embed(corr_diag)
    return torch.einsum("...ij, ...jk, ...kl -> ...il", u, corr, vt)


@torch.no_grad()
def get_align_position(
    z_t: torch.Tensor, xh: torch.Tensor
) -> torch.Tensor:
    """Rotate the clean positions onto the noisy ones (``losses.py:403``)."""
    rotations = kabsch_batch(z_t[:, :, :3], xh[:, :, :3])
    return torch.einsum("...ki, ...ji -> ...jk", rotations, xh[:, :, :3])


class ModelTaskFactory:
    """Hydra entry point for JODO (``configs/tasks/diffusion_jodo.yaml``).

    Declares ``train_set`` so ``cli/train.py``'s declarative seam injects the
    training dataset: the molecule-size histogram (and, for the conditional
    variant, the property distribution + mean/MAD normalizer) is needed at
    construction time and is not an ``nn.Module`` buffer.

    ``sdf_output_path`` is declared generation-time (docs §2.5b): the task is
    rebuilt from the checkpoint's training-time config, where it is ``null``,
    so without this declaration the generate config's value never arrives and
    the bond sidecar -- JODO's whole 2D half -- is silently dropped.
    """

    generation_time_keys = ("sdf_output_path",)

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_jodo",
        nf: int = 256,
        n_layers: int = 8,
        n_heads: int = 16,
        n_extra_heads: int = 2,
        dropout: float = 0.1,
        mlp_ratio: int = 2,
        spatial_cut_off: float = 2.0,
        edge_ch: int = 2,
        include_fc_charge: bool = True,
        cond_time: bool = True,
        dist_gbf: bool = True,
        gbf_name: str = "CondGaussianLayer",
        trans_name: str = "TransMixLayer",
        softmax_inf: bool = True,
        edge_quan_th: float = 0.0,
        com: bool = True,
        pred_data: bool = True,
        self_cond: bool = True,
        noise_align: bool = True,
        centered: bool = True,
        normalize_factors: Optional[list] = None,
        loss_weights: Optional[list] = None,
        reduce_mean: bool = False,
        noise_schedule: str = "cosine",
        beta_0: float = 0.1,
        beta_1: float = 20.0,
        sampling_steps: int = 1000,
        condition: Optional[list] = None,
        normalize_condition: str = "mad",
        sdf_output_path: Optional[str] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.nf = nf
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_extra_heads = n_extra_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.spatial_cut_off = spatial_cut_off
        self.edge_ch = edge_ch
        self.include_fc_charge = include_fc_charge
        self.cond_time = cond_time
        self.dist_gbf = dist_gbf
        self.gbf_name = gbf_name
        self.trans_name = trans_name
        self.softmax_inf = softmax_inf
        self.edge_quan_th = edge_quan_th
        self.com = com
        self.pred_data = pred_data
        self.self_cond = self_cond
        self.noise_align = noise_align
        self.centered = centered
        self.normalize_factors = list(normalize_factors or [1, 4, 4, 1])
        self.loss_weights = list(loss_weights or [1.0, 0.25, 0.1])
        self.reduce_mean = reduce_mean
        self.noise_schedule = noise_schedule
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.sampling_steps = sampling_steps
        self.condition = list(condition or [])
        self.normalize_condition = normalize_condition
        self.sdf_output_path = sdf_output_path
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: Optional[JodoDiffusionTask] = None

    def build(self) -> JodoDiffusionTask:
        """Construct the task, deriving size/property priors from the data."""
        if not self.atom_vocab:
            msg = (
                "JODO needs atom_vocab (it sizes the atom-type head). Set "
                "data.atom_vocab, or tasks.atom_vocab explicitly."
            )
            raise ValueError(msg)

        n_atoms_hist: dict[int, int] = {}
        prop_dist_model = None
        property_norms = None

        stats = getattr(self.train_set, "graph3d_stats", None)
        if stats is None:
            base = getattr(self.train_set, "dataset", None)
            stats = getattr(base, "graph3d_stats", None)
        if stats is not None:
            n_atoms_hist = {
                int(k): int(v) for k, v in stats.n_atoms_hist.items()
            }
            logger.info(
                "JODO size histogram from graph3d_stats: %d sizes over %d "
                "molecules",
                len(n_atoms_hist),
                stats.n_molecules,
            )
        else:
            # No train_set -> the generation path (cli/generate.py never builds
            # a DataModule). The size histogram and property distribution ride
            # in the checkpoint's `node_dist_model`/`prop_dist_model` entries
            # and are restored a moment after this returns.
            logger.warning(
                "No train_set.graph3d_stats available -- building JODO with an "
                "EMPTY size histogram. Expected when loading a checkpoint for "
                "generation; during TRAINING it means data.data_type is not "
                "graph3d with graph3d_stats: true."
            )

        if self.condition and self.train_set is not None:
            prop_dist_model, property_norms = self._build_property_prior()

        self.task = JodoDiffusionTask(
            atom_vocab=self.atom_vocab,
            nf=self.nf,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_extra_heads=self.n_extra_heads,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            spatial_cut_off=self.spatial_cut_off,
            edge_ch=self.edge_ch,
            include_fc_charge=self.include_fc_charge,
            cond_time=self.cond_time,
            dist_gbf=self.dist_gbf,
            gbf_name=self.gbf_name,
            trans_name=self.trans_name,
            softmax_inf=self.softmax_inf,
            edge_quan_th=self.edge_quan_th,
            com=self.com,
            pred_data=self.pred_data,
            self_cond=self.self_cond,
            noise_align=self.noise_align,
            centered=self.centered,
            normalize_factors=self.normalize_factors,
            loss_weights=self.loss_weights,
            reduce_mean=self.reduce_mean,
            noise_schedule=self.noise_schedule,
            beta_0=self.beta_0,
            beta_1=self.beta_1,
            sampling_steps=self.sampling_steps,
            condition=self.condition,
            normalize_condition=self.normalize_condition,
            sdf_output_path=self.sdf_output_path,
            n_atoms_hist=n_atoms_hist,
            prop_dist_model=prop_dist_model,
            property_norms=property_norms,
        )
        return self.task

    def _build_property_prior(self):
        """``DistributionProperty`` + mean/MAD over the training targets.

        Same construction as ``modules/tasks/diffusion.py:385-403``, reading
        the graph3d dataset's ``targets``/``num_atoms`` accessors.
        """
        base = self.train_set
        indices = None
        if hasattr(base, "dataset") and hasattr(base, "indices"):
            indices = list(base.indices)
            base = base.dataset

        num_atoms = base.num_atoms
        props = []
        for name in self.condition:
            values = base.get_property(name)
            if values is None:
                msg = (
                    f"tasks.condition wants property {name!r}, which the "
                    f"dataset does not carry (has: {list(base.targets)}). "
                    "Rebuild the db with `import-graph3d qm9 ... --targets` "
                    "and list the property in data.target_fields."
                )
                raise ValueError(msg)
            props.append(values)
        props = torch.stack(props)
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long)
            num_atoms = num_atoms[idx]
            props = props[:, idx]

        prop_dist = DistributionProperty(
            num_atoms, props, self.condition, num_bins=10
        )
        norms = compute_mean_mad_from_dataloader(props, self.condition)
        prop_dist.set_normalizer(norms)
        logger.info(
            "JODO property prior over %d molecules: %s",
            props.size(1),
            {
                k: (round(float(v["mean"]), 4), round(float(v["mad"]), 4))
                for k, v in norms.items()
            },
        )
        return prop_dist, norms


class JodoDiffusionTask(nn.Module):
    """JODO wrapped in the platform's duck-typed Task contract (§2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        atom_vocab: list,
        nf: int,
        n_layers: int,
        n_heads: int,
        n_extra_heads: int,
        dropout: float,
        mlp_ratio: int,
        spatial_cut_off: float,
        edge_ch: int,
        include_fc_charge: bool,
        cond_time: bool,
        dist_gbf: bool,
        gbf_name: str,
        trans_name: str,
        softmax_inf: bool,
        edge_quan_th: float,
        com: bool,
        pred_data: bool,
        self_cond: bool,
        noise_align: bool,
        centered: bool,
        normalize_factors: list,
        loss_weights: list,
        reduce_mean: bool,
        noise_schedule: str,
        beta_0: float,
        beta_1: float,
        sampling_steps: int,
        condition: list,
        normalize_condition: str,
        sdf_output_path: Optional[str],
        n_atoms_hist: dict,
        prop_dist_model: Optional[DistributionProperty] = None,
        property_norms: Optional[dict] = None,
    ) -> None:
        super().__init__()

        self.atom_vocab = list(atom_vocab)
        self.n_atom_types = len(self.atom_vocab)
        self.edge_ch = edge_ch
        self.include_fc_charge = include_fc_charge
        self.pred_data = pred_data
        self.self_cond = self_cond
        self.noise_align = noise_align
        self.centered = centered
        self.reduce_mean = reduce_mean
        self.loss_weights = list(loss_weights)
        self.condition = list(condition)
        self.normalize_condition = normalize_condition
        self.sdf_output_path = sdf_output_path

        (
            self.pos_norm,
            self.atom_type_norm,
            self.fc_charge_norm,
            self.edge_norm,
        ) = normalize_factors

        cfg = SimpleNamespace(
            data=SimpleNamespace(
                atom_types=self.n_atom_types, edge_ch=edge_ch, centered=centered
            ),
            model=SimpleNamespace(
                nf=nf,
                n_layers=n_layers,
                n_heads=n_heads,
                n_extra_heads=n_extra_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                spatial_cut_off=spatial_cut_off,
                edge_ch=edge_ch,
                include_fc_charge=include_fc_charge,
                cond_time=cond_time,
                dist_gbf=dist_gbf,
                gbf_name=gbf_name,
                trans_name=trans_name,
                softmax_inf=softmax_inf,
                edge_quan_th=edge_quan_th,
                CoM=com,
                pred_data=pred_data,
                normalize_factors=list(normalize_factors),
                cond_ch=max(1, len(self.condition)),
            ),
        )
        self.backbone = (
            Cond_DGT_concat(cfg) if self.condition else DGT_concat(cfg)
        )

        self.noise_scheduler = NoiseScheduleVP(
            schedule=noise_schedule,
            continuous_beta_0=beta_0,
            continuous_beta_1=beta_1,
        )
        # Number of ancestral steps. cli/generate.py's `total_step` override
        # writes here through the `model` property.
        self.T = sampling_steps

        # Leave this None when there is no histogram (the generation path,
        # where stats come from the checkpoint instead). EngineLightning's
        # on_load_checkpoint only restores the checkpoint's node_dist_model if
        # the task does not already have one -- an EMPTY-but-not-None
        # distribution here would make it "Skip node_dist_model from
        # checkpoint" and sampling would then draw molecule sizes from nothing.
        self.node_dist_model = (
            TabascoNodeDistribution({"num_atoms_histogram": n_atoms_hist})
            if n_atoms_hist
            else None
        )
        self.prop_dist_model = prop_dist_model
        self.property_norms = property_norms
        self.last_bond_types: Optional[torch.Tensor] = None

    # -- properties required by the contract --------------------------------

    @property
    def model(self) -> JodoDiffusionTask:
        """``tasks_generate.py`` reads ``task.model.T``; self is the model."""
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        # Empty (not an error) before the checkpoint's node_dist_model is
        # restored -- see the note where node_dist_model is built.
        if self.node_dist_model is None:
            return {}
        return self.node_dist_model.n_node_dist

    # -- scaler (upstream utils.get_data_scaler / get_data_inverse_scaler) ---

    def _scale(
        self,
        pos: torch.Tensor,
        atom_type: torch.Tensor,
        fc: torch.Tensor,
        edge_type: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask4: torch.Tensor,
    ):
        if self.centered:
            atom_type = atom_type * 2.0 - 1.0
            edge_type = edge_type * 2.0 - 1.0
        pos = pos / self.pos_norm * node_mask
        atom_type = atom_type / self.atom_type_norm * node_mask
        fc = fc / self.fc_charge_norm * node_mask
        edge_type = edge_type / self.edge_norm * edge_mask4
        return pos, atom_type, fc, edge_type

    def _unscale(
        self,
        pos: torch.Tensor,
        atom_type: torch.Tensor,
        fc: torch.Tensor,
        edge_type: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask4: torch.Tensor,
    ):
        pos = pos * self.pos_norm * node_mask
        atom_type = atom_type * self.atom_type_norm
        fc = fc * self.fc_charge_norm * node_mask
        edge_type = edge_type * self.edge_norm
        if self.centered:
            atom_type = (atom_type + 1.0) / 2.0 * node_mask
            edge_type = (edge_type + 1.0) / 2.0
        edge_type = edge_type * edge_mask4
        return pos, atom_type, fc, edge_type

    # -- adapters -----------------------------------------------------------

    @staticmethod
    def _masks(node_mask_bool: torch.Tensor):
        """``(B,N)`` bool -> JODO's ``(B,N,1)`` float and ``(B*N*N,1)`` float.

        The diagonal is excluded, which is what makes the symmetric edge noise
        and the transpose-averaged edge prediction well-defined -- upstream's
        ``collate_edge`` diag_mask (``build_dataset.py:414``).
        """
        node_mask = node_mask_bool.float().unsqueeze(-1)
        m = node_mask.squeeze(-1)
        b, n = m.shape
        edge = m.unsqueeze(1) * m.unsqueeze(2)
        edge = edge * (~torch.eye(n, dtype=torch.bool, device=m.device))
        return node_mask, edge.reshape(b * n * n, 1)

    def _edge_one_hot(self, bond: torch.Tensor) -> torch.Tensor:
        """``(B,N,N)`` canonical class ids -> JODO's ``(B,N,N,edge_ch)``.

        See the module docstring for the table. With ``edge_ch == 2`` there is
        no aromatic channel and class 4 collapses to "exists, order 0" -- the
        data config must kekulize so class 4 never occurs.
        """
        order_lut = torch.tensor(
            _ORDER_OF_CLASS, dtype=torch.float32, device=bond.device
        )
        exist = (bond > 0).float()
        order = order_lut[bond]
        channels = [exist, order]
        if self.edge_ch == 3:  # noqa: PLR2004 - aromatic channel
            channels.append((bond == _AROMATIC_CLASS).float())
        return torch.stack(channels, dim=-1)

    def _adapt(self, batch: dict):
        """``graph3d_dense_collate`` dict -> JODO's normalized tensors."""
        node_mask_bool = batch["node_mask"].bool()
        node_mask, edge_mask = self._masks(node_mask_bool)
        b, n = node_mask_bool.shape
        edge_mask4 = edge_mask.reshape(b, n, n, 1)

        pos = batch["pos"].float() * node_mask
        pos = remove_mean_with_mask(pos, node_mask)
        atom_type = F.one_hot(
            batch["atom_idx"].long(), self.n_atom_types
        ).float()
        fc = batch["charges"].float().unsqueeze(-1)
        if not self.include_fc_charge:
            fc = fc.new_zeros((b, n, 0))
        edge_type = self._edge_one_hot(batch["bond_type"].long())

        pos, atom_type, fc, edge_type = self._scale(
            pos, atom_type, fc, edge_type, node_mask, edge_mask4
        )
        xh = torch.cat([pos, atom_type, fc], dim=2)

        context = None
        if self.condition:
            context = self._context_from_batch(batch, node_mask.device)

        return xh, edge_type, node_mask, edge_mask, context

    def _context_from_batch(
        self, batch: dict, device: torch.device
    ) -> torch.Tensor:
        """Per-molecule property targets, mean/MAD-normalized.

        The dense collate forwards every ``target_fields`` column straight
        through as a ``(B,)`` float tensor keyed by the property name.
        """
        cols = []
        for name in self.condition:
            if name not in batch:
                msg = (
                    f"tasks.condition wants {name!r} but the batch has no such "
                    "key; add it to data.target_fields."
                )
                raise KeyError(msg)
            value = batch[name].float().to(device)
            if self.property_norms is not None:
                norm = self.property_norms[name]
                value = (value - norm["mean"].to(device)) / norm["mad"].to(
                    device
                )
            cols.append(value.reshape(-1, 1))
        return torch.cat(cols, dim=1)

    def _denoise(  # noqa: PLR0913
        self,
        t: torch.Tensor,
        z_t: torch.Tensor,
        edge_z_t: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        noise_level: torch.Tensor,
        context: Optional[torch.Tensor],
        cond_x: Optional[torch.Tensor] = None,
        cond_edge_x: Optional[torch.Tensor] = None,
    ):
        return self.backbone(
            t,
            z_t,
            node_mask,
            edge_mask,
            context=context,
            edge_x=edge_z_t,
            cond_x=cond_x,
            cond_edge_x=cond_edge_x,
            noise_level=noise_level,
        )

    # -- training -----------------------------------------------------------

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """One training step: port of ``losses.py:285-382``."""
        xh, edge_x, node_mask, edge_mask, context = self._adapt(batch)
        n_nodes = torch.sum(node_mask.squeeze(-1), dim=-1)

        t_eps = 1e-5
        t = torch.rand(xh.shape[0], device=xh.device) * (1.0 - t_eps) + t_eps
        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)

        noise = sample_combined_position_feature_noise(
            xh.shape[0], xh.shape[1], xh.shape[2] - 3, node_mask
        )
        edge_noise = sample_symmetric_edge_feature_noise(
            edge_x.shape[0], edge_x.shape[1], edge_x.shape[-1], edge_mask
        )
        z_t = expand_dims(alpha_t, xh.dim()) * xh + expand_dims(
            sigma_t, noise.dim()
        ) * noise
        edge_z_t = expand_dims(alpha_t, edge_x.dim()) * edge_x + expand_dims(
            sigma_t, edge_noise.dim()
        ) * edge_noise

        align_pos = (
            get_align_position(z_t, xh) if self.noise_align else xh[:, :, :3]
        )
        noise_level = torch.log(alpha_t**2 / sigma_t**2)

        cond_x = cond_edge_x = None
        if self.self_cond and torch.rand(1).item() < 0.5:  # noqa: PLR2004
            with torch.no_grad():
                cond_x, cond_edge_x = self._denoise(
                    t, z_t, edge_z_t, node_mask, edge_mask, noise_level, context
                )
                cond_x, cond_edge_x = cond_x.detach(), cond_edge_x.detach()
        pred, edge_pred = self._denoise(
            t,
            z_t,
            edge_z_t,
            node_mask,
            edge_mask,
            noise_level,
            context,
            cond_x,
            cond_edge_x,
        )

        losses_pos = torch.square(pred[:, :, :3] - align_pos).mean(-1).sum(-1)
        losses_atom = (
            torch.square(pred[:, :, 3:] - xh[:, :, 3:]).mean(-1).sum(-1)
        )
        losses_edge = torch.square(edge_x - edge_pred).mean(-1)
        losses_edge = losses_edge.reshape(xh.size(0), -1).sum(-1)

        if self.reduce_mean:
            losses_pos = losses_pos / n_nodes
            losses_atom = losses_atom / n_nodes
            losses_edge = losses_edge / (
                edge_mask.reshape(xh.size(0), -1).sum(-1) + 1e-8
            )

        losses = (
            self.loss_weights[0] * losses_pos
            + self.loss_weights[1] * losses_atom
            + self.loss_weights[2] * losses_edge
        )
        if self.pred_data:
            # Data-prediction reweighting (losses.py:378-380).
            losses = expand_dims(
                torch.sqrt(alpha_t / sigma_t), losses.dim()
            ) * losses

        loss = losses.mean()
        stats = {
            "loss_pos": losses_pos.mean().detach(),
            "loss_atom": losses_atom.mean().detach(),
            "loss_edge": losses_edge.mean().detach(),
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
        return {"val_loss": pred.mean()}

    # -- generation ---------------------------------------------------------

    def _place_on_accelerator(self) -> None:
        """Move self to CUDA if it is still on the CPU.

        JODO defines ``device`` as a read-only property, so it takes the
        platform's SECOND device contract: the
        ``not hasattr(task, "device")`` guard in ``cli/generate.py`` and
        ``core/engine.py`` skips it, and the task must place itself. Without
        this, ``load_model`` reads the checkpoint with ``map_location="cpu"``
        and nothing on the ``GenerativeFactory`` path ever moves it, so
        generation runs entirely on the CPU. Same approach as
        ``diffusion_equifm.py:193`` and ``diffusion_diffsbdd.py:611``.
        Training is unaffected -- Lightning moves the module itself.
        """
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.to("cuda")
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    @torch.no_grad()
    def sample(  # noqa: PLR0913
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[dict] = None,  # noqa: ARG002
        mode: Optional[str] = None,  # noqa: ARG002 - DPM-solver out of scope
        n_frames: int = 0,  # noqa: ARG002 - trajectories out of scope
        context: Optional[torch.Tensor] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ancestral sampling; port of ``sampling.py:518-597`` + ``post_process``.

        Returns the platform's ``(one_hot, charges, coords, node_mask)``.
        ``charges`` carries **signed formal charges** (FlowMol/MiDi precedent).
        The generated bond matrix has no slot in that tuple, so it is stashed on
        ``self.last_bond_types`` and, when ``sdf_output_path`` is set, written
        as an ``.sdf`` sidecar alongside the platform's ``.xyz``.
        """
        self._place_on_accelerator()
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
        node_mask, edge_mask = self._masks(arange < n_nodes.unsqueeze(1))
        edge_mask4 = edge_mask.reshape(bs, n_max, n_max, 1)

        if context is not None:
            context = context.to(device).float()
            if context.size(0) == 1 and bs > 1:
                context = context.expand(bs, -1)

        node_nf = self.n_atom_types + int(self.include_fc_charge)
        x = sample_combined_position_feature_noise(
            bs, n_max, node_nf, node_mask
        )
        edge_x = sample_symmetric_edge_feature_noise(
            bs, n_max, self.edge_ch, edge_mask
        )

        eps = 1e-3
        steps = int(num_steps or self.T)
        t_array = torch.linspace(
            self.noise_scheduler.T, eps, steps, device=device
        )
        s_array = torch.cat([t_array[1:], torch.zeros(1, device=device)])

        cond_x = cond_edge_x = None
        x_mean, edge_x_mean = x, edge_x
        for i in range(steps):
            t, s = t_array[i], s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)
            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t**2 - alpha_t_given_s**2 * sigma_s**2
            sigma = torch.sqrt(sigma2_t_given_s) * sigma_s / sigma_t

            vec_t = torch.ones(bs, device=device) * t
            noise_level = torch.ones(bs, device=device) * torch.log(
                alpha_t**2 / sigma_t**2
            )
            pred_t, edge_pred_t = self._denoise(
                vec_t,
                x,
                edge_x,
                node_mask,
                edge_mask,
                noise_level,
                context,
                cond_x,
                cond_edge_x,
            )
            if self.self_cond:
                cond_x, cond_edge_x = pred_t, edge_pred_t

            if self.pred_data:
                c_t = (alpha_t_given_s * sigma_s**2 / sigma_t**2).repeat(bs)
                c_p = (alpha_s * sigma2_t_given_s / sigma_t**2).repeat(bs)
                x_mean = expand_dims(c_t, x.dim()) * x + expand_dims(
                    c_p, pred_t.dim()
                ) * pred_t
                edge_x_mean = expand_dims(
                    c_t, edge_x.dim()
                ) * edge_x + expand_dims(c_p, edge_pred_t.dim()) * edge_pred_t
            else:
                c_a = alpha_t_given_s.repeat(bs)
                c_n = (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs)
                x_mean = x / expand_dims(c_a, x.dim()) - expand_dims(
                    c_n, pred_t.dim()
                ) * pred_t
                edge_x_mean = edge_x / expand_dims(
                    c_a, edge_x.dim()
                ) - expand_dims(c_n, edge_pred_t.dim()) * edge_pred_t

            x = x_mean + expand_dims(
                sigma.repeat(bs), x_mean.dim()
            ) * sample_combined_position_feature_noise(
                bs, n_max, node_nf, node_mask
            )
            edge_x = edge_x_mean + expand_dims(
                sigma.repeat(bs), edge_x_mean.dim()
            ) * sample_symmetric_edge_feature_noise(
                bs, n_max, self.edge_ch, edge_mask
            )

        coords, one_hot, charges, bond_types = self._post_process(
            x_mean, edge_x_mean, node_mask, edge_mask4
        )
        node_mask_long = node_mask.squeeze(-1).long()
        self.last_bond_types = bond_types
        if self.sdf_output_path is not None:
            self._write_sdf(
                one_hot.argmax(-1),
                charges,
                bond_types,
                coords,
                node_mask_long.bool(),
            )
        return one_hot, charges, coords, node_mask_long

    def _post_process(
        self,
        xh: torch.Tensor,
        edge_x: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask4: torch.Tensor,
    ):
        """Unscale and discretize; port of ``sampling.py:53-97``.

        The bond decode is JODO's compressed one: threshold the existence
        channel at 0.5, round ``order * 3`` to the nearest of {0,1,2,3}, and
        (when there are three channels) promote an otherwise-zero order with a
        firing aromatic channel to class 4.
        """
        pos = xh[:, :, :3]
        if self.include_fc_charge:
            h_int = xh[:, :, -1:]
            h_cat = xh[:, :, 3:-1]
        else:
            h_int = xh.new_zeros((xh.size(0), xh.size(1), 1))
            h_cat = xh[:, :, 3:]

        pos, h_cat, h_int, h_edge = self._unscale(
            pos, h_cat, h_int, edge_x, node_mask, edge_mask4
        )
        one_hot = (
            F.one_hot(torch.argmax(h_cat, dim=2), self.n_atom_types) * node_mask
        ).float()
        charges = (torch.round(h_int) * node_mask).long().squeeze(-1)

        exist = (h_edge[:, :, :, 0] >= 0.5).float()  # noqa: PLR2004
        order = torch.clamp(torch.round(h_edge[:, :, :, 1] * 3.0), 0.0, 3.0)
        bond = (exist * order).long()
        if h_edge.size(-1) == 3:  # noqa: PLR2004
            aromatic = (h_edge[:, :, :, 2] >= 0.5).float() * exist  # noqa: PLR2004
            bond = torch.where(
                (aromatic > 0) & (bond == 0),
                torch.full_like(bond, _AROMATIC_CLASS),
                bond,
            )
        # Symmetrize defensively: the network transpose-averages its edge
        # prediction, but rounding either triangle independently can still
        # disagree by one class at the threshold.
        bond = torch.maximum(bond, bond.transpose(1, 2))
        bond = bond * (
            ~torch.eye(bond.size(1), dtype=torch.bool, device=bond.device)
        )
        return pos, one_hot, charges, bond

    @torch.no_grad()
    def sample_conditonal(  # noqa: PLR0913 - name fixed by the platform seam
        self,
        nodesxsample: torch.Tensor = None,
        target_value: Optional[list] = None,
        n_frames: int = 0,
        mode: Optional[str] = None,
        num_steps: Optional[int] = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Property-conditional sampling.

        Called by ``GenerativeFactory.conditional_generation``
        (``runmodes/generate/tasks_generate.py:536``) -- the misspelling is the
        platform's, not a typo here. Turns raw target values into the
        normalized ``context`` the conditional backbone was trained on.
        """
        self._place_on_accelerator()
        if not self.condition:
            msg = (
                "sample_conditonal() on an unconditional JODO task; train with "
                "tasks.condition: [<property>] to use the conditional path."
            )
            raise RuntimeError(msg)
        if self.property_norms is None and self.prop_dist_model is not None:
            # cli/generate.py:401 restores `prop_dist_model` from a checkpoint
            # but has no slot for the normalizer; the distribution carries its
            # own copy, so take it from there rather than adding a core hook.
            self.property_norms = self.prop_dist_model.normalizer
        if self.property_norms is None:
            msg = (
                "no property normalizer available -- the checkpoint must carry "
                "prop_dist_model/property_norms, or a train_set must be present"
            )
            raise RuntimeError(msg)
        target_value = list(target_value or [])
        if len(target_value) != len(self.condition):
            msg = (
                f"got {len(target_value)} target values for "
                f"{len(self.condition)} conditioned properties "
                f"({self.condition})"
            )
            raise ValueError(msg)

        vals = []
        for i, name in enumerate(self.condition):
            norm = self.property_norms[name]
            if self.normalize_condition == "mad":
                val = (target_value[i] - float(norm["mean"])) / float(
                    norm["mad"]
                )
            elif self.normalize_condition == "maxmin":
                lo, hi = float(norm["min"]), float(norm["max"])
                val = 2 * (target_value[i] - lo) / (hi - lo) - 1
            else:
                val = float(target_value[i])
            vals.append(val)
        context = torch.tensor([vals], dtype=torch.float32, device=self.device)
        return self.sample(
            nodesxsample=nodesxsample,
            context=context,
            num_steps=num_steps,
            mode=mode,
            n_frames=n_frames,
        )

    # -- .sdf sidecar (bonds have no channel in the platform's .xyz) --------

    def _write_sdf(  # noqa: PLR0913
        self,
        atom_idx: torch.Tensor,
        charges: torch.Tensor,
        bond_types: torch.Tensor,
        coords: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> None:
        """Append the sampled molecules to ``sdf_output_path``.

        Same shape as MiDi's sidecar, and for the same reason: the platform's
        writer emits ``.xyz``, which has no bond channel, so JODO's whole 2D
        half would be dropped at write time. Molecule building reuses
        ``build_rdkit_mol`` from the graph3d dataset.
        """
        if Chem is None:
            logger.warning("RDKit unavailable, skipping .sdf sidecar")
            return
        from ase.data import atomic_numbers as _atomic_numbers

        parent = os.path.dirname(os.path.abspath(self.sdf_output_path))
        os.makedirs(parent, exist_ok=True)  # noqa: PTH103

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
