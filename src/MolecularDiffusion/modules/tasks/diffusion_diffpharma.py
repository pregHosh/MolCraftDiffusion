"""DiffPharma task: pocket- and pharmacophore-conditioned ligand diffusion.

Three objects live here, mirroring the platform's usual layout:

* :class:`DiffPharmaTask` -- the duck-typed Task (docs/adding_new_models.md
  Section 2.1) wrapping :class:`~...models.diffpharma.ConditionalDDPM`.
* :class:`DiffPharmaTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_diffpharma.yaml``.
* :class:`DiffPharmaPocketGenerator` -- the ``_target_`` of
  ``configs/interference/gen_diffpharma_pocket.yaml``. ``GenerativeFactory``
  cannot express "sample inside this pocket"; a generator behind its own
  ``_target_`` is the in-tree pattern for that (cf. ``DiffSMolShapeGenerator``)
  and needs no core change -- ``cli/generate.py`` only does
  ``instantiate(cfg.interference, task=task)`` then ``run()``.

The batch is NOT a PointCloud dict: DiffPharma needs four node sets
(ligand, pocket, H-bond particles, hydrophobic particles), flat-concatenated
with scatter masks. See ``data/component/diffpharma_data.py``.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.modules.models.diffpharma import (
    ConditionalDDPM,
    DistributionNodes,
    EGNNDynamics,
)
from MolecularDiffusion.modules.tasks.pocket_generator import PocketGenerator

FLOAT_TYPE = torch.float32
INT_TYPE = torch.int64

#: DiffPharma's ligand/pocket atom vocabulary (``full_hDSP_hpSSP``).
DIFFPHARMA_ATOM_VOCAB = [
    "C",
    "N",
    "O",
    "S",
    "B",
    "Br",
    "Cl",
    "P",
    "I",
    "F",
    "others",
]

NODE_SETS = ("lig", "pocket", "interh", "interhp")


def _node_set(batch: Dict[str, Any], group: str, device) -> Dict[str, torch.Tensor]:
    """Slice one node set out of a collated batch into DiffPharma's dict form."""
    return {
        "x": batch[f"{group}_coords"].to(device, FLOAT_TYPE),
        "one_hot": batch[f"{group}_one_hot"].to(device, FLOAT_TYPE),
        "size": batch[f"num_{group}"].to(device, INT_TYPE),
        "mask": batch[f"{group}_mask"].to(device, INT_TYPE),
    }


class DiffPharmaTask(nn.Module):
    """Task contract around ``ConditionalDDPM``."""

    def __init__(
        self,
        model: ConditionalDDPM,
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # attribute name must stay `model`: the checkpoint converter remaps
        # upstream's `ddpm.` prefix to `task.model.`, and cli/generate.py
        # strips the leading `task.`.
        self.model = model
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(DIFFPHARMA_ATOM_VOCAB)
        self.node_dist_model = model.size_distribution
        self.prop_dist_model = None
        self.split = "train"

    # -- contract properties -------------------------------------------- #
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> Dict[int, float]:
        """Ligand-size marginal. Not on the generation path (see module doc)."""
        return self.node_dist_model.n_node_dist

    # -- training -------------------------------------------------------- #
    def forward(self, batch):
        device = self.device
        ligand, pocket, interh, interhp = (
            _node_set(batch, g, device) for g in NODE_SETS
        )

        (
            delta_log_px,
            error_t_lig,
            error_t_pocket,
            snr_weight,
            loss_0_x_ligand,
            loss_0_x_pocket,
            loss_0_h,
            neg_log_const_0,
            kl_prior,
            log_pN,
            _t_int,
            _xh_lig_hat,
            _alpha_t,
            info,
        ) = self.model(ligand, pocket, interh, interhp, return_info=True)

        x_dims = self.model.n_dims
        # The L2 objective is used for validation too (the full VLB eval loss
        # is deliberately out of scope), so val/loss stays on the same scale
        # as train loss instead of jumping three orders of magnitude.
        if self.model.loss_type == "l2":
            denom_lig = x_dims * ligand["size"] + self.model.atom_nf * ligand["size"]
            error_t_lig = error_t_lig / denom_lig
            error_t_pocket = error_t_pocket / (
                (x_dims + self.model.residue_nf) * pocket["size"]
            )
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            loss_0_x_ligand = loss_0_x_ligand / (x_dims * ligand["size"])
            loss_0_x_pocket = loss_0_x_pocket / (x_dims * pocket["size"])
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h
            nll = loss_t + loss_0 + kl_prior
        else:
            # VLB form (upstream's eval objective); kept so validation and
            # training report comparable numbers on the same scale.
            loss_t = -self.model.T * 0.5 * snr_weight * (error_t_lig + error_t_pocket)
            loss_0 = loss_0_x_ligand + loss_0_x_pocket + loss_0_h + neg_log_const_0
            nll = loss_t + loss_0 + kl_prior - delta_log_px - log_pN

        loss = nll.mean(0)
        stats = {
            "loss": loss,
            "error_t_lig": error_t_lig.mean(0),
            "loss_0": loss_0.mean(0),
            "kl_prior": kl_prior.mean(0),
            "eps_hat_lig_x": info["eps_hat_lig_x"],
            "eps_hat_lig_h": info["eps_hat_lig_h"],
        }
        return loss, stats

    def predict_and_target(self, batch):
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred, target):  # noqa: ARG002 - target is a zeros stub
        return {"val_loss": pred.mean()}

    # -- generation ------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size=None,
        nodesxsample=None,
        num_steps=None,
        batch=None,
        **kwargs,
    ):
        """Sample ligands inside the pocket carried by ``batch``.

        ``batch`` must hold the ``pocket_*``/``interh_*``/``interhp_*`` keys
        of a collated DiffPharma batch (the generator builds it). There is no
        unconditional mode -- mirroring upstream's own hard refusal.

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)``, in the ORIGINAL pocket frame (the reverse process
        re-centres on the ligand CoM; the shift is undone here).
        """
        if batch is None:
            raise ValueError(
                "DiffPharmaTask.sample() is pocket-conditioned: pass "
                "batch=<dict with pocket_/interh_/interhp_ tensors>. "
                "Use interference: gen_diffpharma_pocket, not gen_unconditional."
            )
        device = self.device
        pocket, interh, interhp = (
            _node_set(batch, g, device) for g in ("pocket", "interh", "interhp")
        )
        n_samples = len(pocket["size"])

        if nodesxsample is None:
            nodesxsample = self.node_dist_model.sample_conditional(n2=pocket["size"])
        nodesxsample = torch.as_tensor(nodesxsample, device=device).long()
        if nodesxsample.numel() != n_samples:
            raise ValueError(
                f"nodesxsample has {nodesxsample.numel()} entries but the batch "
                f"carries {n_samples} pockets."
            )

        from torch_scatter import scatter_mean

        pocket_com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)

        xh_lig, xh_pocket, lig_mask, pocket_mask = self.model.sample_given_pocket(
            pocket, interh, interhp, nodesxsample, timesteps=num_steps
        )

        # translate back into the input pocket's frame
        x_dims = self.model.n_dims
        pocket_com_after = scatter_mean(xh_pocket[:, :x_dims], pocket_mask, dim=0)
        shift = pocket_com_before - pocket_com_after
        xh_lig[:, :x_dims] += shift[lig_mask]

        return self._pad(xh_lig, lig_mask, n_samples)

    def _pad(self, xh_lig, lig_mask, n_samples):
        """flat ``(n_total, 3 + atom_nf)`` -> padded ``(B, N, .)`` contract shape."""
        x_dims = self.model.n_dims
        counts = torch.bincount(lig_mask, minlength=n_samples)
        n_max = int(counts.max().item())
        device = xh_lig.device

        one_hot = torch.zeros(
            (n_samples, n_max, self.model.atom_nf), device=device, dtype=xh_lig.dtype
        )
        coords = torch.zeros((n_samples, n_max, x_dims), device=device)
        node_mask = torch.zeros((n_samples, n_max), device=device)

        offset = 0
        for i in range(n_samples):
            n = int(counts[i].item())
            coords[i, :n] = xh_lig[offset : offset + n, :x_dims]
            one_hot[i, :n] = xh_lig[offset : offset + n, x_dims:]
            node_mask[i, :n] = 1
            offset += n
        charges = torch.zeros((n_samples, n_max), device=device)
        return one_hot, charges, coords, node_mask


class DiffPharmaTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_diffpharma.yaml``."""

    def __init__(
        self,
        task_type: str = "diffusion_diffpharma",
        size_distribution_path: Optional[str] = None,
        atom_nf: int = 11,
        residue_nf: int = 11,
        interh_nf: int = 3,
        interhp_nf: int = 6,
        n_dims: int = 3,
        joint_nf: int = 128,
        hidden_nf: int = 256,
        n_layers: int = 8,
        attention: bool = True,
        tanh: bool = True,
        norm_constant: float = 1,
        inv_sublayers: int = 1,
        sin_embedding: bool = False,
        aggregation_method: str = "sum",
        normalization_factor: float = 100,
        edge_cutoff_ligand: Optional[float] = None,
        edge_cutoff_pocket: Optional[float] = 5.0,
        edge_cutoff_interaction: Optional[float] = 5.0,
        reflection_equivariant: bool = False,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "polynomial_2",
        diffusion_noise_precision: float = 5.0e-4,
        diffusion_loss_type: str = "l2",
        normalize_factors=(1, 1),
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        # kwargs absorbs the overrides cli/train.py injects unconditionally
        # (node_feature, node_feature_choice, node_feature_dim, ...).
        self.task_type = task_type
        self.size_distribution_path = size_distribution_path
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(DIFFPHARMA_ATOM_VOCAB)
        self.condition_names: List[str] = []
        self.egnn_kwargs = dict(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            interh_nf=interh_nf,
            interhp_nf=interhp_nf,
            n_dims=n_dims,
            joint_nf=joint_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            aggregation_method=aggregation_method,
            normalization_factor=normalization_factor,
            edge_cutoff_ligand=edge_cutoff_ligand,
            edge_cutoff_pocket=edge_cutoff_pocket,
            edge_cutoff_interaction=edge_cutoff_interaction,
            reflection_equivariant=reflection_equivariant,
            update_pocket_coords=False,
        )
        self.ddpm_kwargs = dict(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            interh_nf=interh_nf,
            interhp_nf=interhp_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=tuple(normalize_factors),
        )
        self.task: Optional[DiffPharmaTask] = None

    def _size_histogram(self) -> np.ndarray:
        if not self.size_distribution_path:
            raise ValueError(
                "tasks.size_distribution_path is required: DiffPharma's size "
                "prior is the 2D (n_ligand x n_pocket) histogram shipped with "
                "the dataset as size_distribution.npy."
            )
        hist = np.load(self.size_distribution_path)
        if hist.ndim != 2:
            raise ValueError(
                f"{self.size_distribution_path} must be a 2D "
                f"(n_ligand x n_pocket) histogram, got shape {hist.shape}."
            )
        return hist

    def build(self) -> DiffPharmaTask:
        dynamics = EGNNDynamics(**self.egnn_kwargs)
        ddpm = ConditionalDDPM(
            dynamics=dynamics,
            size_histogram=self._size_histogram(),
            **self.ddpm_kwargs,
        )
        self.task = DiffPharmaTask(ddpm, atom_vocab=self.atom_vocab)
        return self.task


class DiffPharmaPocketGenerator(PocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_diffpharma_pocket``.

    Two mutually exclusive pocket sources:

    * ``pocket_db`` + ``pocket_index`` -- a row of a converted ASE db
      (``scripts/convert_dataset.py``), read with ``center=False``.
    * ``pocket_pdb`` + ``ref_sdf`` -- a novel pocket, built on the fly by
      ``data.component.diffpharma_prep.complex_from_files``. The SDF is
      required, not optional: it is both the 8 A pocket-selection reference
      and the ligand ODDT detects the interactions against.

    Either way the downstream dict is identical.

    The sampling loop itself lives in :class:`PocketGenerator`.
    """

    tag = "diffpharma"
    # never seeded numpy; seeding it would change what this model generates.
    seed_numpy = False

    def __init__(
        self,
        task,
        pocket_db: Optional[str] = None,
        pocket_index: int = 0,
        pocket_pdb: Optional[str] = None,
        ref_sdf: Optional[str] = None,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        mol_size: Optional[list] = None,
        output_path: str = "generated_diffpharma",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task,
            pocket_db=pocket_db,
            pocket_index=pocket_index,
            num_generate=num_generate,
            batch_size=batch_size,
            num_steps=num_steps,
            mol_size=mol_size,
            output_path=output_path,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.pocket_pdb = pocket_pdb
        self.ref_sdf = ref_sdf

        if bool(pocket_db) == bool(pocket_pdb):
            raise ValueError(
                "Set exactly one pocket source: interference.pocket_db "
                "(a converted ASE db) or interference.pocket_pdb (+ ref_sdf)."
            )
        if pocket_pdb and not ref_sdf:
            raise ValueError(
                "interference.ref_sdf is required with pocket_pdb: the "
                "reference ligand defines both the 8 A pocket and the "
                "protein-ligand interactions."
            )

    def _complex(self) -> Dict[str, torch.Tensor]:
        if self.pocket_db:
            from MolecularDiffusion.data.integrations.diffpharma_data import (
                DiffPharmaDataset,
            )

            dataset = DiffPharmaDataset(self.pocket_db, center=False)
            return dataset[self.pocket_index]

        from MolecularDiffusion.data.integrations.diffpharma_prep import (
            complex_from_files,
        )

        return complex_from_files(self.pocket_pdb, self.ref_sdf)

    def _pocket(self) -> Dict[str, torch.Tensor]:
        return self._complex()

    def _repeat(self, item: Dict[str, torch.Tensor], n: int) -> Dict[str, Any]:
        """Tile one complex's context node sets ``n`` times, with fresh masks."""
        batch: Dict[str, Any] = {}
        for group in ("pocket", "interh", "interhp"):
            coords = item[f"{group}_coords"]
            one_hot = item[f"{group}_one_hot"]
            size = len(coords)
            batch[f"{group}_coords"] = coords.repeat(n, 1)
            batch[f"{group}_one_hot"] = one_hot.repeat(n, 1)
            batch[f"{group}_mask"] = torch.repeat_interleave(
                torch.arange(n, dtype=INT_TYPE), size
            )
            batch[f"num_{group}"] = torch.full((n,), size, dtype=INT_TYPE)
        return batch

    def _sizes(self, batch: Dict[str, Any], n: int) -> torch.Tensor:
        if self.mol_size:
            return torch.tensor(random.choices(self.mol_size, k=n), dtype=INT_TYPE)
        nd = self.task.node_dist_model
        # the joint histogram only covers pocket sizes seen in training
        max_pocket = len(nd.n1_given_n2) - 1
        pocket_size = batch["num_pocket"].clamp(max=max_pocket)
        return nd.sample_conditional(n2=pocket_size).clamp(min=1)

    def _pick_sizes(self, batch: Dict[str, Any], n: int) -> torch.Tensor:
        # the only prior that needs the TILED BATCH, not just the count.
        return self._sizes(batch, n)

    def _start(self, item: Dict[str, Any]) -> None:
        print(
            f"[{self.tag}] pocket={len(item['pocket_coords'])} atoms, "
            f"interh={len(item['interh_coords'])}, "
            f"interhp={len(item['interhp_coords'])} particles"
        )
