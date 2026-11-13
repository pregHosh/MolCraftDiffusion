"""DiffSpectra: 3D structure elucidation from IR/Raman/UV-Vis spectra.

DMT (Diffusion Molecule Transformer) jointly diffuses coordinates, atom
types, formal charges **and** a dense bond tensor -- DiffSpectra is JODO's
own architecture family (Huang et al., NeurIPS 2023, arXiv:2305.12347,
already integrated as ``diffusion_jodo``), with a
:class:`~MolecularDiffusion.modules.models.diffspectra.specformer.SpecFormer`
spectral encoder added directly into the timestep embedding in place of
JODO's scalar-property MLP. See the approved
``docs/model_integrations/diffspectra/INTEGRATION_PLAN.md`` for the full
derivation; ``_adapt``/``_scale``/``_edge_one_hot``/``_masks`` below are a
mechanical adaptation of ``diffusion_jodo.py``'s (same bond encoding, same
charge handling), and ``kabsch_batch``/``get_align_position``/``expand_dims``
are reused from there directly rather than re-derived -- they are
architecture-family math, not something specific to either task.

**Data path**: ``data_type: graph3d`` with ``bond_collate: dense`` and
``kekulize: true`` (``edge_ch: 2`` has no aromatic channel -- identical
situation to JODO's own QM9 config, see
``configs/data/graph3d_qm9s_dataset.yaml``). Coordinates/atoms/bonds ride the
platform's existing ``data/qm9_graph3d.db``; the IR/Raman/UV-Vis spectra
QM9S adds on top do not fit any existing per-molecule channel, so they ride a
SMILES-keyed sidecar (``docs/model_integrations/diffspectra/scripts/
convert_dataset.py``), joined here off ``batch["smiles"]`` -- graph3d's
identity key, the same role ``xyz`` plays for ``pointcloud`` in
``diffusion_chefnmr.py``.

**No ``sample()``.** DiffSpectra's spectrum conditioning is never dropped
during training (no CFG branch, ``supports_guidance = False`` below), so an
unconditional ``sample()`` would either fabricate a spectrum or silently
zero it -- both a lie, and both would arm ``GenerativeEvalCallback`` to
produce garbage during training. The generative entry point is
``elucidate()``, reached through
:class:`~MolecularDiffusion.modules.tasks.elucidation_generator.ElucidationGenerator`
via ``configs/interference/gen_elucidation.yaml`` -- rung-1 reuse, no new
interference config, no seam change (see the plan's *Inference Task
Decision*).

**Node-count prior.** Unlike ChefNMR (formula is always an input), DiffSpectra
generates atom types itself, so all it ever needs from ``_priors`` is a
molecule SIZE. When a record's size is unknown, ``node_dist_model``
(:class:`~MolecularDiffusion.modules.tasks.diffusion_tabasco.TabascoNodeDistribution`
off ``train_set.graph3d_stats``, exactly as ``diffusion_jodo.py`` builds it)
draws a plausible one from the training distribution instead of refusing --
the user-approved resolution to the plan's node-count-prior question.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from MolecularDiffusion.device import get_device
from torch import nn
from torch.nn import functional as F  # noqa: N812

from MolecularDiffusion.data.datasets.graph3d_dataset import build_rdkit_mol
from MolecularDiffusion.modules.models.diffspectra import DMT
from MolecularDiffusion.modules.models.jodo import NoiseScheduleVP
from MolecularDiffusion.modules.models.jodo.utils import (
    remove_mean_with_mask,
    sample_combined_position_feature_noise,
    sample_symmetric_edge_feature_noise,
)

# Architecture-family math, not task-specific: reused directly rather than
# re-derived. See the module docstring.
from MolecularDiffusion.modules.tasks.graph.jodo import (
    expand_dims,
    get_align_position,
    kabsch_batch,  # noqa: F401 - re-exported for parity/debugging, not used directly here
)
from MolecularDiffusion.modules.tasks.flow.tabasco import (
    TabascoNodeDistribution,
)
from MolecularDiffusion.modules.tasks.conditional.elucidation_generator import (
    Candidate,
    ElucidationGenerator,
)

logger = logging.getLogger(__name__)

# DMT's own (exist, order) compressed bond encoding -- identical to JODO's,
# see diffusion_jodo.py's module docstring for the derivation and
# configs/data/graph3d_qm9s_dataset.yaml for why kekulize: true is required.
_ORDER_OF_CLASS = (0.0, 1 / 3, 2 / 3, 1.0, 0.0)
_AROMATIC_CLASS = 4

_SPECTRA_KEYS = ("uv", "ir", "raman")


def _log_normalize(x: torch.Tensor) -> torch.Tensor:
    """``log10(x + 1)``, upstream's spectra normalization
    (``datasets/build_dataset.py:145-151``, ``data.use_normalize: True``).

    Applied here (model-side, at read time) rather than baked into the
    sidecar, so the sidecar keeps raw intensities -- consistent with the
    platform's "store raw, normalize model-side" convention (see
    ``fc``'s handling below and CLAUDE.md's charge-storage note).
    """
    return torch.log10(x + 1.0)


@dataclass
class _SpectraSidecar:
    """SMILES -> row index, plus the three raw (un-normalized) spectra
    arrays, loaded once from ``scripts/convert_dataset.py``'s ``.npz``.
    """

    smiles_to_row: dict[str, int]
    uv: np.ndarray
    ir: np.ndarray
    raman: np.ndarray

    @classmethod
    def load(cls, path: str) -> _SpectraSidecar:
        data = np.load(path, allow_pickle=True)
        smiles = [str(s) for s in data["smiles"]]
        return cls(
            smiles_to_row={s: i for i, s in enumerate(smiles)},
            uv=np.asarray(data["uv"], dtype=np.float32),
            ir=np.asarray(data["ir"], dtype=np.float32),
            raman=np.asarray(data["raman"], dtype=np.float32),
        )

    def row(self, smiles: str, key: str) -> np.ndarray:
        idx = self.smiles_to_row.get(smiles)
        if idx is None:
            msg = (
                f"no {key!r} spectrum for SMILES {smiles!r} in this sidecar "
                f"({len(self.smiles_to_row)} molecules covered). Rebuild it "
                "with docs/model_integrations/diffspectra/scripts/"
                "convert_dataset.py, or restrict data.ase_db_path to the "
                "molecules it covers (its own diffspectra_smoke_qm9.db)."
            )
            raise KeyError(msg)
        return getattr(self, key)[idx]


# --------------------------------------------------------------------- #
# factory                                                               #
# --------------------------------------------------------------------- #
class DiffSpectraTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_diffspectra.yaml``."""

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_diffspectra",
        atom_vocab: Sequence[str] | None = None,
        include_fc_charge: bool = True,
        nf: int = 256,
        n_layers: int = 8,
        n_heads: int = 16,
        n_extra_heads: int = 2,
        dropout: float = 0.1,
        mlp_ratio: int = 2,
        spatial_cut_off: float = 2.0,
        edge_ch: int = 2,
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
        normalize_factors: list | None = None,
        loss_weights: list | None = None,
        reduce_mean: bool = False,
        noise_schedule: str = "cosine",
        beta_0: float = 0.1,
        beta_1: float = 20.0,
        sampling_steps: int = 1000,
        spectra_version: str = "allspectra",
        patch_len: list | None = None,
        stride: list | None = None,
        specformer_kwargs: dict | None = None,
        spectra_sidecar_path: str | None = None,
        train_set: torch.utils.data.Dataset | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.include_fc_charge = include_fc_charge
        self.nf = nf
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_extra_heads = n_extra_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.spatial_cut_off = spatial_cut_off
        self.edge_ch = edge_ch
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
        self.spectra_version = spectra_version
        self.patch_len = list(patch_len) if patch_len else None
        self.stride = list(stride) if stride else None
        self.specformer_kwargs = dict(specformer_kwargs or {})
        self.spectra_sidecar_path = spectra_sidecar_path
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: DiffSpectraElucidationTask | None = None

    def build(self) -> DiffSpectraElucidationTask:
        if not self.atom_vocab:
            msg = (
                "DiffSpectra needs atom_vocab (it sizes the atom-type head). "
                "Set data.atom_vocab, or tasks.atom_vocab explicitly."
            )
            raise ValueError(msg)

        n_atoms_hist: dict[int, int] = {}
        stats = getattr(self.train_set, "graph3d_stats", None)
        if stats is None:
            base = getattr(self.train_set, "dataset", None)
            stats = getattr(base, "graph3d_stats", None)
        if stats is not None:
            n_atoms_hist = {
                int(k): int(v) for k, v in stats.n_atoms_hist.items()
            }
            logger.info(
                "DiffSpectra size histogram from graph3d_stats: %d sizes over "
                "%d molecules",
                len(n_atoms_hist),
                stats.n_molecules,
            )
        else:
            # No train_set -> the generation path. The histogram rides the
            # checkpoint's node_dist_model and is restored a moment after
            # this returns -- same seam diffusion_jodo.py's factory uses.
            logger.warning(
                "No train_set.graph3d_stats available -- building DiffSpectra "
                "with an EMPTY size histogram. Expected when loading a "
                "checkpoint for generation."
            )

        sidecar = None
        if self.spectra_sidecar_path:
            sidecar = _SpectraSidecar.load(self.spectra_sidecar_path)
            logger.info(
                "[diffspectra] sidecar: %d molecules from %s",
                len(sidecar.smiles_to_row),
                self.spectra_sidecar_path,
            )

        self.task = DiffSpectraElucidationTask(
            atom_vocab=self.atom_vocab,
            include_fc_charge=self.include_fc_charge,
            nf=self.nf,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_extra_heads=self.n_extra_heads,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            spatial_cut_off=self.spatial_cut_off,
            edge_ch=self.edge_ch,
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
            spectra_version=self.spectra_version,
            patch_len=self.patch_len,
            stride=self.stride,
            specformer_kwargs=self.specformer_kwargs,
            n_atoms_hist=n_atoms_hist,
            sidecar=sidecar,
        )
        return self.task


# --------------------------------------------------------------------- #
# task                                                                  #
# --------------------------------------------------------------------- #
class DiffSpectraElucidationTask(nn.Module):
    """Task-contract implementation for DiffSpectra.

    The class name is load-bearing:
    ``elucidation_generator._TASK_TO_GENERATOR`` keys on it to find
    :class:`DiffSpectraElucidationGenerator`. Implements the FULL contract
    (``forward``/``predict_and_target``/``evaluate`` for training, PLUS
    ``elucidate`` for generation) in one class, exactly as
    ``ChefNMRElucidationTask`` does.
    """

    def __init__(  # noqa: PLR0913
        self,
        atom_vocab: list[str],
        include_fc_charge: bool,
        nf: int,
        n_layers: int,
        n_heads: int,
        n_extra_heads: int,
        dropout: float,
        mlp_ratio: int,
        spatial_cut_off: float,
        edge_ch: int,
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
        spectra_version: str,
        patch_len: list | None,
        stride: list | None,
        specformer_kwargs: dict,
        n_atoms_hist: dict,
        sidecar: _SpectraSidecar | None,
    ) -> None:
        super().__init__()

        self.atom_vocab = list(atom_vocab)
        self.n_atom_types = len(self.atom_vocab)
        self.include_fc_charge = include_fc_charge
        self.edge_ch = edge_ch
        self.pred_data = pred_data
        self.self_cond = self_cond
        self.noise_align = noise_align
        self.centered = centered
        self.loss_weights = list(loss_weights)
        self.reduce_mean = reduce_mean
        self.spectra_version = spectra_version
        self.sampling_steps = sampling_steps

        (
            self.pos_norm,
            self.atom_type_norm,
            self.fc_charge_norm,
            self.edge_norm,
        ) = normalize_factors

        self.backbone = DMT(
            atom_types=self.n_atom_types,
            include_fc_charge=include_fc_charge,
            nf=nf,
            n_layers=n_layers,
            n_heads=n_heads,
            n_extra_heads=n_extra_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            spatial_cut_off=spatial_cut_off,
            edge_ch=edge_ch,
            cond_time=cond_time,
            dist_gbf=dist_gbf,
            gbf_name=gbf_name,
            trans_name=trans_name,
            softmax_inf=softmax_inf,
            edge_quan_th=edge_quan_th,
            com=com,
            pred_data=pred_data,
            patch_len=patch_len,
            stride=stride,
            spectra_version=spectra_version,
            specformer_kwargs=specformer_kwargs,
        )

        self.noise_scheduler = NoiseScheduleVP(
            schedule=noise_schedule,
            continuous_beta_0=beta_0,
            continuous_beta_1=beta_1,
        )

        # See diffusion_jodo.py's identical comment: leave this None (not an
        # empty distribution) when there is no histogram, so the checkpoint's
        # own node_dist_model restores on load instead of being skipped.
        self.node_dist_model = (
            TabascoNodeDistribution({"num_atoms_histogram": n_atoms_hist})
            if n_atoms_hist
            else None
        )
        self._sidecar = sidecar
        self.last_bond_types: torch.Tensor | None = None

    # -- required properties ------------------------------------------ #
    @property
    def model(self) -> DiffSpectraElucidationTask:
        return self

    @property
    def device(self) -> torch.device:
        """``cli/generate.py`` skips its own device move when a task has
        this defined, and the elucidation seam does ``task.to(device)``.
        """
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        if self.node_dist_model is None:
            return {}
        return self.node_dist_model.n_node_dist

    def _place_on_accelerator(self) -> None:
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.to("cuda")
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    # -- scaler (upstream utils.get_data_scaler / get_data_inverse_scaler) -- #
    def _scale(self, pos, atom_type, fc, edge_type, node_mask, edge_mask4):
        if self.centered:
            atom_type = atom_type * 2.0 - 1.0
            edge_type = edge_type * 2.0 - 1.0
        pos = pos / self.pos_norm * node_mask
        atom_type = atom_type / self.atom_type_norm * node_mask
        fc = fc / self.fc_charge_norm * node_mask
        edge_type = edge_type / self.edge_norm * edge_mask4
        return pos, atom_type, fc, edge_type

    def _unscale(self, pos, atom_type, fc, edge_type, node_mask, edge_mask4):
        pos = pos * self.pos_norm * node_mask
        atom_type = atom_type * self.atom_type_norm
        fc = fc * self.fc_charge_norm * node_mask
        edge_type = edge_type * self.edge_norm
        if self.centered:
            atom_type = (atom_type + 1.0) / 2.0 * node_mask
            edge_type = (edge_type + 1.0) / 2.0
        edge_type = edge_type * edge_mask4
        return pos, atom_type, fc, edge_type

    # -- bond adapter (DMT's compressed exist/order encoding, identical to
    # JODO's -- see the module docstring) ------------------------------ #
    @staticmethod
    def _masks(node_mask_bool: torch.Tensor):
        node_mask = node_mask_bool.float().unsqueeze(-1)
        m = node_mask.squeeze(-1)
        b, n = m.shape
        edge = m.unsqueeze(1) * m.unsqueeze(2)
        edge = edge * (~torch.eye(n, dtype=torch.bool, device=m.device))
        return node_mask, edge.reshape(b * n * n, 1)

    def _edge_one_hot(self, bond: torch.Tensor) -> torch.Tensor:
        order_lut = torch.tensor(
            _ORDER_OF_CLASS, dtype=torch.float32, device=bond.device
        )
        exist = (bond > 0).float()
        order = order_lut[bond]
        channels = [exist, order]
        if self.edge_ch == 3:  # noqa: PLR2004 - aromatic channel
            channels.append((bond == _AROMATIC_CLASS).float())
        return torch.stack(channels, dim=-1)

    # -- spectra adapter -------------------------------------------------- #
    def _sidecar_or_raise(self) -> _SpectraSidecar:
        if self._sidecar is None:
            msg = (
                "DiffSpectra training needs the spectra sidecar. Set "
                "tasks.spectra_sidecar_path to the .npz written by "
                "docs/model_integrations/diffspectra/scripts/"
                "convert_dataset.py."
            )
            raise ValueError(msg)
        return self._sidecar

    def _raw_spectra_from_smiles(
        self, smiles_list: Sequence[str], device: torch.device
    ) -> torch.Tensor | list[torch.Tensor]:
        """SMILES list -> raw (un-normalized) spectra tensor(s), per
        ``self.spectra_version``. Shared by training (``_adapt``) and
        generation (``_condition_tensor`` mirrors this for a single record).
        """
        side = self._sidecar_or_raise()
        keys = (
            _SPECTRA_KEYS
            if self.spectra_version == "allspectra"
            else (self.spectra_version,)
        )
        out = {}
        for key in keys:
            rows = np.stack([side.row(s, key) for s in smiles_list])
            out[key] = torch.from_numpy(rows).to(
                device=device, dtype=torch.float32
            )
        if self.spectra_version == "allspectra":
            return [out["uv"], out["ir"], out["raman"]]
        return out[self.spectra_version]

    def _normalize_spectra(
        self, raw: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(raw, (list, tuple)):
            return [_log_normalize(r) for r in raw]
        return _log_normalize(raw)

    def _adapt(self, batch: dict):
        """``graph3d_dense_collate`` dict -> DMT's normalized tensors."""
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

        raw_spectra = self._raw_spectra_from_smiles(
            batch["smiles"], node_mask.device
        )
        context = self._normalize_spectra(raw_spectra)

        return xh, edge_type, node_mask, edge_mask, context

    def _denoise(  # noqa: PLR0913
        self,
        t,
        z_t,
        edge_z_t,
        node_mask,
        edge_mask,
        noise_level,
        context,
        cond_x=None,
        cond_edge_x=None,
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

    # -- training ---------------------------------------------------------- #
    def forward(self, batch: dict) -> tuple:
        """One training step: mechanical port of upstream's
        ``losses.get_sde_graph_loss_fn`` (``self_cond_type: 'ori'`` in every
        shipped config, so the self-cond post-process step it calls is the
        identity and is not reproduced here).
        """
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
        z_t = (
            expand_dims(alpha_t, xh.dim()) * xh
            + expand_dims(sigma_t, noise.dim()) * noise
        )
        edge_z_t = (
            expand_dims(alpha_t, edge_x.dim()) * edge_x
            + expand_dims(sigma_t, edge_noise.dim()) * edge_noise
        )

        align_pos = (
            get_align_position(z_t, xh) if self.noise_align else xh[:, :, :3]
        )
        noise_level = torch.log(alpha_t**2 / sigma_t**2)

        cond_x = cond_edge_x = None
        if self.self_cond and torch.rand(1).item() < 0.5:  # noqa: PLR2004
            with torch.no_grad():
                cond_x, cond_edge_x = self._denoise(
                    t,
                    z_t,
                    edge_z_t,
                    node_mask,
                    edge_mask,
                    noise_level,
                    context,
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
            losses = (
                expand_dims(torch.sqrt(alpha_t / sigma_t), losses.dim())
                * losses
            )

        loss = losses.mean()
        stats = {
            "loss_pos": losses_pos.mean().detach(),
            "loss_atom": losses_atom.mean().detach(),
            "loss_edge": losses_edge.mean().detach(),
        }
        return loss, stats

    def predict_and_target(self, batch: dict) -> tuple:
        loss, _ = self.forward(batch)
        loss = loss.detach().reshape(1)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict:  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # -- generation (elucidate) -------------------------------------------- #
    def _post_process(self, xh, edge_x, node_mask, edge_mask4):
        """Unscale and discretize -- identical decode rule to JODO's
        ``_post_process`` (verified against upstream's own
        ``sampling.post_process`` with ``compress_edge=True``: threshold
        exist at 0.5, ``round(order * 3)`` clamped to {0,1,2,3}).
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
            F.one_hot(torch.argmax(h_cat, dim=2), self.n_atom_types)
            * node_mask
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
        bond = torch.maximum(bond, bond.transpose(1, 2))
        bond = bond * (
            ~torch.eye(bond.size(1), dtype=torch.bool, device=bond.device)
        )
        return pos, one_hot, charges, bond

    @torch.no_grad()
    def elucidate(
        self, batch: dict[str, torch.Tensor], num_steps: int | None = None
    ) -> dict[str, torch.Tensor]:
        """One tiled spectrum -> ``n`` candidate molecules.

        ``batch`` is what :meth:`DiffSpectraElucidationGenerator._repeat`
        built: ``condition`` (one raw ``(n, L)`` tensor, or a 3-list for
        ``allspectra``) and ``n_atoms`` ``(n,)`` -- the known or
        ``node_dist_model``-sampled size per candidate. Ancestral sampling,
        mechanical port of ``diffusion_jodo.py``'s ``sample()`` (same
        architecture family), always conditioned -- DiffSpectra has no
        unconditional branch.
        """
        self._place_on_accelerator()
        device = self.device
        n_atoms = batch["n_atoms"].to(device).long()
        context = self._normalize_spectra(batch["condition"])

        bs = int(n_atoms.numel())
        n_max = int(n_atoms.max().item())
        arange = torch.arange(n_max, device=device).unsqueeze(0).expand(bs, -1)
        node_mask, edge_mask = self._masks(arange < n_atoms.unsqueeze(1))
        edge_mask4 = edge_mask.reshape(bs, n_max, n_max, 1)

        node_nf = self.n_atom_types + int(self.include_fc_charge)
        x = sample_combined_position_feature_noise(
            bs, n_max, node_nf, node_mask
        )
        edge_x = sample_symmetric_edge_feature_noise(
            bs, n_max, self.edge_ch, edge_mask
        )

        eps = 1e-3
        steps = int(num_steps or self.sampling_steps)
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
                x_mean = (
                    expand_dims(c_t, x.dim()) * x
                    + expand_dims(c_p, pred_t.dim()) * pred_t
                )
                edge_x_mean = (
                    expand_dims(c_t, edge_x.dim()) * edge_x
                    + expand_dims(c_p, edge_pred_t.dim()) * edge_pred_t
                )
            else:
                c_a = alpha_t_given_s.repeat(bs)
                c_n = (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs)
                x_mean = (
                    x / expand_dims(c_a, x.dim())
                    - expand_dims(c_n, pred_t.dim()) * pred_t
                )
                edge_x_mean = (
                    edge_x / expand_dims(c_a, edge_x.dim())
                    - expand_dims(c_n, edge_pred_t.dim()) * edge_pred_t
                )

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
        return {
            "pos": coords,
            "atom_type": one_hot,
            "fc": charges,
            "bond_type": bond_types,
            "node_mask": node_mask_long,
        }


# --------------------------------------------------------------------- #
# elucidation generator                                                 #
# --------------------------------------------------------------------- #
@dataclass
class DiffSpectraRecord:
    """One spectrum to elucidate.

    Two sources fill this (see :meth:`DiffSpectraElucidationGenerator._records`):
    a converted labelled corpus (``smiles`` known, ``uv``/``ir``/``raman``
    read from the sidecar-shaped ``.npz``) or a bare unknown-spectra JSON
    (``smiles`` usually ``None``, spectra given inline).
    """

    name: str
    uv: np.ndarray | None = None
    ir: np.ndarray | None = None
    raman: np.ndarray | None = None
    smiles: str | None = None
    n_atoms: int | None = None


class DiffSpectraElucidationGenerator(ElucidationGenerator):
    """Walk a set of measured spectra; emit ranked 3D candidates per record.

    ``spectra_source`` takes either input, told apart by what the path IS
    (see :meth:`_records`), the same mechanism
    ``ChefNMRElucidationGenerator`` uses:

    * a **converted corpus** -- a ``.npz`` written by ``scripts/
      convert_dataset.py`` (or any file with that shape): SMILES-keyed, so
      top-k and Tanimoto are reported. Reproduces published numbers.
    * a **bare unknown-spectra JSON** -- per record a name, spectra on the
      standard QM9S grid (701/3501/3501 for uv/ir/raman) and OPTIONALLY a
      known size (``n_atoms``) or reference ``smiles``. No formula required
      -- unlike ChefNMR, DiffSpectra generates the atom types itself.

    Rides the shared seam unchanged: no ``run()`` loop here, no ``_rank``
    override (generation order is the correct default -- DiffSpectra has no
    scorer of its own), and ``_sample_kwargs`` is the base's, because
    ``num_steps`` is the only knob ``elucidate()`` takes
    (``guidance_scale`` is refused outright by ``supports_guidance = False``).
    """

    tag = "diffspectra"
    source_key = "spectra_source"
    source_required_msg = (
        "diffspectra needs `spectra_source`, which is either:\n"
        "  (a) a JSON file of UNKNOWN spectra -- per record a name and "
        "IR/Raman/UV-Vis intensities on the QM9S grid (701/3501/3501 "
        "points), optionally a known size (`n_atoms`) or a reference "
        "`smiles`. This is the real use case: hand it a spectrum, get back "
        "a molecule.\n"
        "  (b) a `.npz` written by "
        "docs/model_integrations/diffspectra/scripts/convert_dataset.py -- "
        "SMILES-keyed spectra with the answer attached. Use this to "
        "reproduce published numbers; it can only describe molecules whose "
        "structure you already have."
    )
    supports_guidance = False  # no CFG branch -- see the module docstring
    #: Architecturally each of uv/ir/raman COULD be zeroed independently
    #: (SpecFormer patches them separately), but upstream never trained
    #: with spectrum dropout, so declaring them maskable would misrepresent
    #: a capability that was never validated. Same honesty principle
    #: `supports_guidance = False` already enforces.
    maskable_channels = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._drawn = 0
        self._decoded = 0
        self._draws: list[tuple] = []

    # -- corpus ---------------------------------------------------------- #
    def _records(self) -> Sequence[DiffSpectraRecord]:
        source = self.spectra_source
        npz_path = source if source.endswith(".npz") else f"{source}.npz"
        if os.path.exists(npz_path):
            return self._corpus_records(npz_path)
        return self._unknown_records(source)

    def _corpus_records(self, npz_path: str) -> list[DiffSpectraRecord]:
        data = np.load(npz_path, allow_pickle=True)
        smiles = [str(s) for s in data["smiles"]]
        records = []
        for i, smi in enumerate(smiles):
            n_atoms = _heavy_and_h_count(smi)
            records.append(
                DiffSpectraRecord(
                    name=f"mol_{i}",
                    uv=np.asarray(data["uv"][i], dtype=np.float32),
                    ir=np.asarray(data["ir"][i], dtype=np.float32),
                    raman=np.asarray(data["raman"][i], dtype=np.float32),
                    smiles=smi,
                    n_atoms=n_atoms,
                )
            )
        if not records:
            msg = f"no records in {npz_path}"
            raise ValueError(msg)
        return records

    def _unknown_records(self, path: str) -> list[DiffSpectraRecord]:
        if not os.path.isfile(path):
            msg = (
                f"diffspectra found neither a converted corpus ({path}.npz "
                f"or {path} itself as .npz) nor an unknown-spectra file at "
                f"{path!r}. {self.source_required_msg}"
            )
            raise ValueError(msg)
        with open(path) as handle:
            entries = json.load(handle)
        records = []
        for entry in entries:
            name = entry["name"]
            record = DiffSpectraRecord(
                name=name,
                uv=_as_spectrum(entry.get("uv"), 701, name, "uv"),
                ir=_as_spectrum(entry.get("ir"), 3501, name, "ir"),
                raman=_as_spectrum(entry.get("raman"), 3501, name, "raman"),
                smiles=entry.get("smiles"),
                n_atoms=entry.get("n_atoms"),
            )
            records.append(record)
        labelled = sum(1 for r in records if r.smiles)
        print(
            f"[{self.tag}] {len(records)} unknown(s) from {path}; "
            f"{labelled} carry a reference SMILES"
            + (
                ""
                if labelled
                else " -- none, so no metrics.json will be written"
            )
            + "."
        )
        return records

    # -- hooks ------------------------------------------------------------- #
    def _condition(self, record: DiffSpectraRecord):
        task = self.task
        version = task.spectra_version
        if version == "allspectra":
            for key in _SPECTRA_KEYS:
                if getattr(record, key) is None:
                    msg = f"record {record.name} is missing '{key}', required by spectra_version=allspectra"
                    raise ValueError(msg)
            return [record.uv, record.ir, record.raman]
        value = getattr(record, version)
        if value is None:
            msg = f"record {record.name} is missing '{version}', required by spectra_version={version}"
            raise ValueError(msg)
        return value

    def _priors(self, record: DiffSpectraRecord) -> int | None:
        return record.n_atoms

    def _repeat(self, cond, priors: int | None, n: int) -> dict:
        if priors is not None:
            n_atoms = torch.full((n,), int(priors), dtype=torch.long)
        else:
            model = self.task.node_dist_model
            if model is None:
                msg = (
                    f"record {self._current_name} has no known size and this "
                    "checkpoint carries no node_dist_model (was it trained "
                    "without data.graph3d_stats: true?) -- cannot draw a "
                    "size prior."
                )
                raise RuntimeError(msg)
            n_atoms = model.sample(n)

        if isinstance(cond, list):
            condition = [
                torch.from_numpy(np.asarray(c, dtype=np.float32))
                .unsqueeze(0)
                .repeat(n, 1)
                for c in cond
            ]
        else:
            condition = (
                torch.from_numpy(np.asarray(cond, dtype=np.float32))
                .unsqueeze(0)
                .repeat(n, 1)
            )
        return {"condition": condition, "n_atoms": n_atoms}

    def _start(
        self, record: DiffSpectraRecord, index: int, total: int
    ) -> None:
        self._current_name = self._record_name(record, index)
        self._draws = []
        super()._start(record, index, total)

    # -- decoding ------------------------------------------------------------ #
    def _decode(self, raw: dict[str, torch.Tensor]) -> list[Candidate]:
        from ase.data import atomic_numbers as _atomic_numbers  # noqa: PLC0415
        from rdkit import Chem, RDLogger  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        decoder = list(self.task.atom_vocab)
        z_of_vocab = [_atomic_numbers[s] for s in decoder]

        pos = raw["pos"].detach().cpu().numpy()
        atom_type = raw["atom_type"].detach().cpu().numpy()
        fc = raw["fc"].detach().cpu().numpy()
        bond = raw["bond_type"].detach().cpu().numpy()
        node_mask = raw["node_mask"].detach().cpu().numpy()

        out: list[Candidate] = []
        for b in range(pos.shape[0]):
            self._drawn += 1
            n = int(node_mask[b].sum())
            type_idx = atom_type[b, :n].argmax(-1)
            zs = [z_of_vocab[int(i)] for i in type_idx]
            symbols = [decoder[int(i)] for i in type_idx]
            sub = bond[b, :n, :n]
            rows, cols = np.triu_indices(n, k=1)
            keep = sub[rows, cols] > 0
            bond_index = np.stack([rows[keep], cols[keep]])
            bond_type = sub[rows, cols][keep]
            try:
                mol = build_rdkit_mol(
                    zs,
                    bond_index,
                    bond_type,
                    formal_charge=fc[b, :n],
                    coords=pos[b, :n],
                )
                # H-suppressed on purpose, same reasoning as
                # diffusion_chefnmr.py's `_decode`: DMT generates explicit H
                # atoms, and comparing WITH them against a reference SMILES
                # (RDKit keeps H implicit by default) makes every top-k
                # number spuriously zero even when the molecule is exactly
                # right (explicit "[H]N([H])[H]" != implicit "N" as strings,
                # though both are ammonia) -- and `_write_metrics` Morgan-
                # fingerprints `cand.mol` itself, so the MOL must be
                # H-suppressed too, not just the SMILES string. The full
                # all-atom geometry (H included) survives in `coords`.
                mol_no_h = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol_no_h)
            except Exception:  # noqa: BLE001 - bad geometry/valence is expected
                self._draws.append((symbols, pos[b, :n], None))
                out.append(Candidate(smiles=""))
                continue
            self._decoded += 1
            self._draws.append((symbols, pos[b, :n], smiles))
            out.append(
                Candidate(smiles=smiles, mol=mol_no_h, coords=pos[b, :n])
            )
        return out

    def _write_record(
        self, name: str, candidates: list[Candidate], reference: Any
    ) -> None:
        """The base's writers, plus one ``.xyz`` per candidate.

        DiffSpectra is coordinate-first and its output is an all-atom
        geometry including hydrogens; ``candidates.sdf`` carries only the
        H-suppressed molecule (see ``_decode``). ``.xyz`` is also what the
        platform's `analyze` tooling and the smoke-test checker read, so
        writing it is not a nicety. Subclass-local: same precedent as
        ``ChefNMRElucidationGenerator._write_record``
        (``diffusion_chefnmr.py:787-820``); nothing about it belongs in the
        shared base a 2D-only elucidation model would also ride.
        """
        super()._write_record(name, candidates, reference)
        directory = os.path.join(self.output_path, name)
        rank = 0
        for i, (symbols, coords, smiles) in enumerate(self._draws):
            # EVERY draw is written, including the ones bond-perception/
            # valence assignment could not turn into a molecule -- the model
            # produced that geometry; only `build_rdkit_mol` failed. Rank
            # matches ranking.csv, because `_accept` preserves order and
            # `_rank` is the identity (see the module/class docstrings).
            if smiles:
                rank += 1
            label = f"rank={rank}" if smiles else "rank=-"
            with open(
                os.path.join(directory, f"draw_{i:03d}.xyz"), "w"
            ) as handle:
                handle.write(f"{len(symbols)}\n")
                handle.write(f"{name} {label} smiles={smiles or 'none'}\n")
                handle.writelines(
                    f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
                    for symbol, (x, y, z) in zip(symbols, coords)
                )

    def _reference(self, record: DiffSpectraRecord) -> str | None:
        if not record.smiles:
            return None
        from rdkit import Chem  # noqa: PLC0415

        mol = Chem.MolFromSmiles(record.smiles)
        if mol is None:
            return record.smiles
        try:
            return Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:  # noqa: BLE001
            return record.smiles

    def _summary(self, written: int, attempts: int) -> None:
        rate = self._decoded / self._drawn if self._drawn else 0.0
        print(
            f"[{self.tag}] wrote {written} records to {self.output_path}; "
            f"bond-to-mol reconstruction succeeded on {self._decoded}/"
            f"{self._drawn} draws ({rate:.1%})."
        )


def _as_spectrum(
    value: Any, length: int, name: str, key: str
) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (length,):
        msg = f"record {name!r}'s '{key}' spectrum has shape {arr.shape}, expected ({length},)"
        raise ValueError(msg)
    return arr


def _heavy_and_h_count(smiles: str) -> int | None:
    """Total atom count (heavy + H) for a corpus record's known composition."""
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.AddHs(mol).GetNumAtoms()


# --------------------------------------------------------------------- #
# registration                                                          #
# --------------------------------------------------------------------- #
# `elucidation_generator._TASK_TO_GENERATOR` is what `ElucidationGenerator.
# __new__` keys on to find the right subclass. ChefNMR's entry is a literal
# dict entry in that file because ChefNMR is the tenant that BUILT the
# shared seam (a "new-shared" rung integration). DiffSpectra is a plain
# "reuse" rung addition -- elucidation_generator.py is explicitly out of
# scope to edit (see the integration plan's Core-Change Requirement
# section) -- so this registers the second tenant from here instead, at
# import time. `cli/generate.py` always imports the task's own Hydra
# `_target_` (this module) before it builds the interference generator, so
# this entry exists by the time `ElucidationGenerator.__new__` looks it up.
from MolecularDiffusion.modules.tasks.conditional import elucidation_generator as _elucidation_generator

_elucidation_generator._TASK_TO_GENERATOR.setdefault(  # noqa: SLF001
    "DiffSpectraElucidationTask",
    ("diffusion_diffspectra", "DiffSpectraElucidationGenerator"),
)
