"""KGDiff task: pocket-conditioned ligand diffusion with self-guidance.

Three objects, mirroring the platform's usual layout:

* :class:`KGDiffDiffusionTask` -- the duck-typed Task
  (docs/adding_new_models.md Section 2.1) wrapping
  :class:`~...models.kgdiff.ScorePosNet3D`.
* :class:`ModelTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_kgdiff.yaml``.
* :class:`KGDiffPocketGenerator` -- the ``_target_`` of
  ``configs/interference/gen_kgdiff_pocket.yaml``. ``GenerativeFactory``'s
  ``sample(batch_size, nodesxsample, ...)`` has no channel for "which
  pocket", so pocket-conditioned models get their own generator behind their
  own ``_target_``; that is the established in-tree pattern
  (``PMDMPocketGenerator``, ``DiffPharmaPocketGenerator``) and needs no core
  change -- ``cli/generate.py`` only does
  ``instantiate(cfg.interference, task=task)`` then ``.run()``.

The batch is NOT a PointCloud dict: KGDiff needs a diffused ligand cloud, a
fixed pocket cloud, and one affinity label per complex, flat-concatenated
with scatter indices. The collate in ``data/component/kgdiff_data.py``
already emits KGDiff's own argument names, so the adapter here is a
``.to(device)`` map.

Two deviations from the generic contract, both deliberate and both shared
with the other pocket models in-tree:

* ``sample()`` requires a pocket. There is no unconditional path -- the
  network cannot run without ``protein_pos``/``protein_v``.
* Sampled coordinates come back in the **input pocket's frame**, not
  zero-CoM: ``center_pos_mode='protein'`` subtracts the pocket centroid and
  the sampler adds it back.

Out of scope this pass (see the integration plan): the ``valuenet*`` /
``target_diff`` / ``vina`` guide modes, PDBBind2020, and CFG / inpainting /
trajectory export.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from MolecularDiffusion.data.integrations.kgdiff_data import (
    KGDIFF_ATOM_VOCAB,
    KGDIFF_ELEMENTS,
    LIGAND_INDEX_TO_Z,
    NUM_LIGAND_CLASSES,
    PROTEIN_FEATURE_DIM,
    KGDiffDataset,
)
from MolecularDiffusion.modules.models.kgdiff import (
    GUIDE_MODES,
    ScorePosNet3D,
    log_sample_categorical,
)
from MolecularDiffusion.modules.models.kgdiff.atom_num import (
    get_space_size,
    marginal_size_distribution,
    sample_atom_num,
)
from MolecularDiffusion.modules.tasks.pocket_generator import PocketGenerator

INT_TYPE = torch.int64

#: Fallback pocket extent (Angstrom) when nothing has measured a real one --
#: the midpoint of ``atom_num_config``'s bin bounds. Only reached if a caller
#: asks ``node_dist_model.sample()`` without a pocket in hand; the real path
#: measures the pocket it was handed.
DEFAULT_POCKET_EXTENT = 30.0

#: Ligand class index -> row in the 8-element ``atom_vocab``. The model's 13
#: classes are ``(element, is_aromatic)`` pairs; aromaticity is dropped here
#: because ``save_xyz_file`` writes elements.
_CLASS_TO_ELEMENT_ROW = torch.tensor(
    [KGDIFF_ELEMENTS.index(z) for z in LIGAND_INDEX_TO_Z], dtype=INT_TYPE
)


class PocketSizePrior:
    """Ligand-size prior conditioned on the pocket's spatial extent.

    Wraps KGDiff's static ``atom_num_config`` table, so unlike the other
    in-tree node distributions it needs neither a ``train_set`` nor a
    checkpoint buffer -- the table ships with the code and is identical at
    train and generate time.

    Same interface as ``TabascoNodeDistribution`` etc.: ``.n_node_dist`` and
    ``.sample(n)``. :attr:`space_size` is set by whoever knows the pocket.
    """

    def __init__(self, space_size: float = DEFAULT_POCKET_EXTENT) -> None:
        self.space_size = float(space_size)
        self.n_node_dist = marginal_size_distribution()

    def sample(self, n_samples: int) -> torch.Tensor:
        return torch.tensor(
            [sample_atom_num(self.space_size) for _ in range(n_samples)],
            dtype=INT_TYPE,
        )


class KGDiffDiffusionTask(nn.Module):
    """Task contract around :class:`ScorePosNet3D`."""

    def __init__(
        self,
        model: ScorePosNet3D,
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        # attribute name must stay `model`: cli/generate.py reaches for
        # task.model when reconciling diffusion_steps.
        self.model = model
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(KGDIFF_ATOM_VOCAB)
        )
        self.prop_dist_model = None
        self.split = "train"
        self._node_dist_override: Optional[PocketSizePrior] = None

    # -- contract properties -------------------------------------------- #
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self) -> PocketSizePrior:
        if self._node_dist_override is None:
            self._node_dist_override = PocketSizePrior()
        return self._node_dist_override

    @node_dist_model.setter
    def node_dist_model(self, value) -> None:
        # cli/generate.py assigns this when an edm_stat.pkl sidecar exists.
        self._node_dist_override = value

    @property
    def n_node_dist(self) -> Dict[int, float]:
        return self.node_dist_model.n_node_dist

    # -- training -------------------------------------------------------- #
    _BATCH_KEYS = (
        "protein_pos",
        "protein_v",
        "protein_batch",
        "ligand_pos",
        "ligand_v",
        "ligand_batch",
        "affinity",
    )

    def _adapt(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = self.device
        return {k: batch[k].to(device) for k in self._BATCH_KEYS}

    def forward(self, batch):
        b = self._adapt(batch)
        out = self.model.get_diffusion_loss(
            protein_pos=b["protein_pos"],
            protein_v=b["protein_v"],
            affinity=b["affinity"],
            batch_protein=b["protein_batch"],
            ligand_pos=b["ligand_pos"],
            ligand_v=b["ligand_v"],
            batch_ligand=b["ligand_batch"],
        )
        return out["loss"], {
            "loss": out["loss"],
            "loss_pos": out["loss_pos"],
            "loss_v": out["loss_v"],
            "loss_exp": out["loss_exp"],
        }

    def predict_and_target(self, batch):
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred, target):  # noqa: ARG002 - target is a zeros stub
        return {"val_loss": pred.mean()}

    # -- generation ------------------------------------------------------ #
    def sample(
        self,
        batch_size=None,  # noqa: ARG002 - inferred from the pocket batch
        nodesxsample=None,
        num_steps=None,
        batch=None,
        guide_mode: str = "joint",
        type_grad_weight: float = 100.0,
        pos_grad_weight: float = 25.0,
        progress: bool = True,
        **kwargs,  # noqa: ARG002 - swallows mode/n_frames from generic callers
    ):
        """Sample ligands inside the pocket carried by ``batch``.

        **The signature deliberately deviates from Section 2.1**: it takes a
        pocket. ``batch`` must hold ``protein_pos`` / ``protein_v`` /
        ``protein_batch`` from a collated KGDiff batch
        (:class:`KGDiffPocketGenerator` builds it).

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)`` in the ORIGINAL pocket frame. ``one_hot`` is over the
        8-element ``atom_vocab``, not the model's 13 ``(element, aromatic)``
        classes. ``charges`` is zeros: KGDiff has no charge channel.
        """
        if batch is None:
            raise ValueError(
                "KGDiffDiffusionTask.sample() is pocket-conditioned: pass "
                "batch=<dict with protein_pos / protein_v / protein_batch>. "
                "Use interference: gen_kgdiff_pocket, not gen_unconditional."
            )
        if guide_mode not in GUIDE_MODES:
            raise ValueError(
                f"guide_mode={guide_mode!r} not supported; ported modes are "
                f"{GUIDE_MODES}."
            )
        device = self.device
        protein_pos = batch["protein_pos"].to(device, torch.float32)
        protein_v = batch["protein_v"].to(device, torch.float32)
        protein_batch = batch["protein_batch"].to(device, INT_TYPE)
        n_samples = int(protein_batch.max().item()) + 1

        if nodesxsample is None:
            prior = self.node_dist_model
            prior.space_size = get_space_size(protein_pos.detach().cpu().numpy())
            nodesxsample = prior.sample(n_samples)
        sizes = torch.as_tensor(nodesxsample).long().tolist()
        if len(sizes) != n_samples:
            raise ValueError(
                f"nodesxsample has {len(sizes)} entries but the batch carries "
                f"{n_samples} pockets."
            )

        ligand_batch = torch.repeat_interleave(
            torch.arange(n_samples, device=device),
            torch.tensor(sizes, device=device),
        )
        # init: pocket centroid + unit Gaussian jitter, uniform atom types
        # (scripts/sample_diffusion.py:66-73)
        from torch_scatter import scatter_mean

        centre = scatter_mean(protein_pos, protein_batch, dim=0)[ligand_batch]
        init_ligand_pos = centre + torch.randn_like(centre)
        init_ligand_v = log_sample_categorical(
            torch.zeros(len(ligand_batch), self.model.num_classes, device=device)
        ).argmax(dim=-1)

        out = self.model.sample_diffusion(
            guide_mode=guide_mode,
            type_grad_weight=type_grad_weight,
            pos_grad_weight=pos_grad_weight,
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=protein_batch,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=ligand_batch,
            num_steps=num_steps,
            center_pos_mode=self.model.center_pos_mode,
            progress=progress,
        )
        return self._pad(out["pos"].detach(), out["v"].detach(), sizes)

    @staticmethod
    def _pad(coords_flat, class_flat, sizes):
        """flat ``(n_total, .)`` -> padded ``(B, N, .)`` contract shape.

        The model's 13 ``(element, aromatic)`` classes are collapsed to the
        8-element vocabulary here, so ``save_xyz_file`` decodes correctly
        against ``atom_vocab``.
        """
        device = coords_flat.device
        rows = _CLASS_TO_ELEMENT_ROW.to(device)[class_flat]
        n_samples, n_max = len(sizes), max(sizes)

        one_hot = torch.zeros(
            (n_samples, n_max, len(KGDIFF_ELEMENTS)), device=device
        )
        coords = torch.zeros((n_samples, n_max, 3), device=device)
        node_mask = torch.zeros((n_samples, n_max), device=device)

        offset = 0
        for i, n in enumerate(sizes):
            sl = slice(offset, offset + n)
            coords[i, :n] = coords_flat[sl]
            one_hot[i, torch.arange(n, device=device), rows[sl]] = 1.0
            node_mask[i, :n] = 1
            offset += n
        charges = torch.zeros((n_samples, n_max), device=device)
        return one_hot, charges, coords, node_mask


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_kgdiff.yaml``.

    No ``train_set`` parameter: KGDiff's size prior is the static
    ``atom_num_config`` table, so nothing has to be measured at build time
    (see docs/adding_new_models.md Section 2.5 -- the seam is opt-in).
    """

    def __init__(
        self,
        task_type: str = "diffusion_kgdiff",
        protein_atom_feature_dim: int = PROTEIN_FEATURE_DIM,
        ligand_atom_feature_dim: int = NUM_LIGAND_CLASSES,
        model_mean_type: str = "C0",
        beta_schedule: str = "sigmoid",
        beta_start: float = 1.0e-7,
        beta_end: float = 2.0e-3,
        pos_beta_s: float = 0.01,
        v_beta_schedule: str = "cosine",
        v_beta_s: float = 0.01,
        num_diffusion_timesteps: int = 1000,
        loss_v_weight: float = 100.0,
        loss_exp_weight: float = 1.0,
        sample_time_method: str = "symmetric",
        use_classifier_guide: bool = True,
        time_emb_dim: int = 0,
        time_emb_mode: str = "simple",
        center_pos_mode: str = "protein",
        node_indicator: bool = True,
        model_type: str = "uni_o2",
        num_blocks: int = 1,
        num_layers: int = 9,
        hidden_dim: int = 128,
        n_heads: int = 16,
        edge_feat_dim: int = 4,
        num_r_gaussian: int = 20,
        knn: int = 32,
        num_node_types: int = 8,
        act_fn: str = "relu",
        norm: bool = True,
        cutoff_mode: str = "knn",
        ew_net_type: str = "global",
        num_x2h: int = 1,
        num_h2x: int = 1,
        r_max: float = 10.0,
        x2h_out_fc: bool = False,
        sync_twoup: bool = False,
        pred_exp_from_all: bool = False,
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by cli/train.py
    ) -> None:
        self.task_type = task_type
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(KGDIFF_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.model_kwargs = dict(
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
            model_mean_type=model_mean_type,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            pos_beta_s=pos_beta_s,
            v_beta_schedule=v_beta_schedule,
            v_beta_s=v_beta_s,
            num_diffusion_timesteps=num_diffusion_timesteps,
            loss_v_weight=loss_v_weight,
            loss_exp_weight=loss_exp_weight,
            sample_time_method=sample_time_method,
            use_classifier_guide=use_classifier_guide,
            time_emb_dim=time_emb_dim,
            time_emb_mode=time_emb_mode,
            center_pos_mode=center_pos_mode,
            node_indicator=node_indicator,
            model_type=model_type,
            num_blocks=num_blocks,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            edge_feat_dim=edge_feat_dim,
            num_r_gaussian=num_r_gaussian,
            knn=knn,
            num_node_types=num_node_types,
            act_fn=act_fn,
            norm=norm,
            cutoff_mode=cutoff_mode,
            ew_net_type=ew_net_type,
            num_x2h=num_x2h,
            num_h2x=num_h2x,
            r_max=r_max,
            x2h_out_fc=x2h_out_fc,
            sync_twoup=sync_twoup,
            pred_exp_from_all=pred_exp_from_all,
        )
        self.task: Optional[KGDiffDiffusionTask] = None

    def build(self) -> KGDiffDiffusionTask:
        self.task = KGDiffDiffusionTask(
            ScorePosNet3D(**self.model_kwargs), atom_vocab=self.atom_vocab
        )
        return self.task


class KGDiffPocketGenerator(PocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_kgdiff_pocket``.

    The pocket comes from one row of a converted ASE db
    (``docs/model_integrations/kgdiff/scripts/convert_dataset.py``); the
    sampled ligand comes back in that pocket's own frame.

    ``guide_mode='joint'`` is KGDiff's headline self-guided sampler (the
    model's own affinity head is the classifier guide); ``'wo'`` runs the
    same loop unguided, which is the paper's ablation and a useful control.

    The sampling loop itself lives in :class:`PocketGenerator`.
    """

    tag = "kgdiff"
    db_required_msg = (
        "interference.pocket_db is required: KGDiff has no "
        "unconditional mode. Point it at a converted ASE db "
        "(docs/model_integrations/kgdiff/scripts/convert_dataset.py)."
    )

    def __init__(
        self,
        task,
        pocket_db: Optional[str] = None,
        pocket_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        mol_size: Optional[list] = None,
        guide_mode: str = "joint",
        type_grad_weight: float = 100.0,
        pos_grad_weight: float = 25.0,
        output_path: str = "generated_kgdiff",
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
        self.guide_mode = guide_mode
        self.type_grad_weight = type_grad_weight
        self.pos_grad_weight = pos_grad_weight

    def _pocket(self) -> Dict[str, Any]:
        return KGDiffDataset(self.pocket_db)[self.pocket_index]

    @staticmethod
    def _repeat(item: Dict[str, torch.Tensor], n: int) -> Dict[str, Any]:
        """Tile one pocket ``n`` times, with fresh scatter indices."""
        size = len(item["protein_pos"])
        return {
            "protein_pos": item["protein_pos"].repeat(n, 1),
            "protein_v": item["protein_v"].repeat(n, 1),
            "protein_batch": torch.repeat_interleave(
                torch.arange(n, dtype=INT_TYPE), size
            ),
        }

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        return {
            "guide_mode": self.guide_mode,
            "type_grad_weight": self.type_grad_weight,
            "pos_grad_weight": self.pos_grad_weight,
            "progress": False,
        }

    def _start(self, item: Dict[str, Any]) -> None:
        print(
            f"[{self.tag}] pocket '{item['name']}': "
            f"{len(item['protein_pos'])} atoms, guide_mode={self.guide_mode}, "
            f"generating {self.num_generate} ligands"
        )
