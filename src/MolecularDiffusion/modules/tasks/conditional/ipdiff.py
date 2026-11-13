"""IPDiff task: pocket-conditioned ligand diffusion with an interaction prior.

Three objects, the same layout the other pocket-conditioned models use:

* :class:`IPDiffDiffusionTask` -- the duck-typed Task
  (docs/adding_new_models.md Section 2.1) wrapping
  :class:`~...models.ipdiff.IPDiffScorePosNet3D` **and** the frozen
  :class:`~...models.ipdiff.BAPNet` prior.
* :class:`ModelTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_ipdiff.yaml``.
* :class:`IPDiffPocketGenerator` -- the ``_target_`` of
  ``configs/interference/gen_ipdiff_pocket.yaml``.

**What IPDiff adds over KGDiff, in one sentence.** KGDiff steers *sampling*
by ascending its own predicted affinity (test-time gradient guidance);
IPDiff changes what the model is *trained on* -- a separately pretrained,
frozen interaction network (IPNet) supplies 128-d features that are folded
into every token embedding and that drive a learned shift of both the
forward noising process and the reverse posterior. No classifier, no CFG, no
gradient of any predictor. Consequently ``prop_dist_model`` is ``None`` and
``sample()`` can run under ``no_grad``.

**Everything about the data is KGDiff's**, verbatim and by import: the
collate (``data/component/kgdiff_data.py``), ``configs/data/
kgdiff_dataset.yaml``, the converted smoke db, the 13-class ligand
vocabulary, the 27-dim pocket features, the pocket-extent size prior and the
13->8 element collapse used when writing .xyz. IPDiff's
``utils/transforms.py`` and ``datasets/pl_data.py`` are a strict subset of
KGDiff's, so there is no new data code here at all. The ``affinity`` column
those batches carry is simply ignored -- IPDiff never reads it.

Two deviations from the generic contract, both shared with the other pocket
models in-tree:

* ``sample()`` requires a pocket; there is no unconditional path.
* Sampled coordinates come back in the **input pocket's frame**
  (``center_pos_mode='protein'``).

Out of scope this pass (see the integration plan): CFG/guidance of any kind
(IPDiff has none), unconditional generation, inpainting, trajectory export,
``pos_only`` sampling, and the ``range``/``ref`` ligand-size modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from MolecularDiffusion.data.integrations.kgdiff_data import (
    KGDIFF_ATOM_VOCAB,
    NUM_LIGAND_CLASSES,
    PROTEIN_FEATURE_DIM,
)
from MolecularDiffusion.modules.models.ipdiff import BAPNet, IPDiffScorePosNet3D
from MolecularDiffusion.modules.models.kgdiff.atom_num import get_space_size
from MolecularDiffusion.modules.models.kgdiff.score_model import (
    log_sample_categorical,
)

# Reused wholesale rather than re-derived: the ligand-size prior is the same
# static pocket-extent table, and the flat -> padded conversion (including
# the 13-class -> 8-element collapse) is the same operation on the same
# vocabulary. `IPDiffPocketGenerator` likewise only differs from KGDiff's in
# that it has no guidance knobs.
from MolecularDiffusion.modules.tasks.conditional.kgdiff import (
    KGDiffDiffusionTask,
    KGDiffPocketGenerator,
    PocketSizePrior,
)

INT_TYPE = torch.int64

#: Default location of the released IPNet weights, relative to the repo root
#: (the CWD ``MolCraftDiff`` is invoked from). Overridable via
#: ``tasks.net_cond_ckpt``.
DEFAULT_IPNET_CKPT = "docs/model_integrations/ipdiff/checkpoints/ipnet"


class IPDiffDiffusionTask(nn.Module):
    """Task contract around :class:`IPDiffScorePosNet3D` + frozen IPNet."""

    def __init__(
        self,
        model: IPDiffScorePosNet3D,
        net_cond: BAPNet,
        atom_vocab: Optional[List[str]] = None,
        pos_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        # attribute name must stay `model`: cli/generate.py reaches for
        # task.model when reconciling diffusion_steps.
        self.model = model
        # A real submodule, so the frozen prior round-trips through the
        # platform checkpoint and is restored at generate time instead of
        # being re-read from a loose file.
        self.net_cond = net_cond
        self.pos_noise_std = pos_noise_std
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
        self._node_dist_override = value

    @property
    def n_node_dist(self) -> Dict[int, float]:
        return self.node_dist_model.n_node_dist

    # -- training -------------------------------------------------------- #
    #: KGDiff's collate also emits `affinity`; IPDiff has no affinity term.
    _BATCH_KEYS = (
        "protein_pos",
        "protein_v",
        "protein_batch",
        "ligand_pos",
        "ligand_v",
        "ligand_batch",
    )

    def _adapt(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = self.device
        return {k: batch[k].to(device) for k in self._BATCH_KEYS}

    def forward(self, batch):
        b = self._adapt(batch)
        protein_pos = b["protein_pos"]
        if self.training and self.pos_noise_std > 0:
            # upstream train.py:124 -- jitter the pocket during training only
            protein_pos = protein_pos + torch.randn_like(protein_pos) * (
                self.pos_noise_std
            )
        out = self.model.get_diffusion_loss(
            net_cond=self.net_cond,
            protein_pos=protein_pos,
            protein_v=b["protein_v"],
            batch_protein=b["protein_batch"],
            ligand_pos=b["ligand_pos"],
            ligand_v=b["ligand_v"],
            batch_ligand=b["ligand_batch"],
        )
        return out["loss"], {
            "loss": out["loss"],
            "loss_pos": out["loss_pos"],
            "loss_v": out["loss_v"],
        }

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
        batch_size=None,  # noqa: ARG002 - inferred from the pocket batch
        nodesxsample=None,
        num_steps=None,
        batch=None,
        progress: bool = True,
        **kwargs,  # noqa: ARG002 - swallows mode/n_frames from generic callers
    ):
        """Sample ligands inside the pocket carried by ``batch``.

        **The signature deliberately deviates from Section 2.1**: it takes a
        pocket. ``batch`` must hold ``protein_pos`` / ``protein_v`` /
        ``protein_batch`` from a collated KGDiff-format batch
        (:class:`IPDiffPocketGenerator` builds it).

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)`` in the ORIGINAL pocket frame. ``one_hot`` is over the
        8-element ``atom_vocab``, not the model's 13 ``(element, aromatic)``
        classes; ``charges`` is zeros (IPDiff has no charge channel).
        """
        if batch is None:
            raise ValueError(
                "IPDiffDiffusionTask.sample() is pocket-conditioned: pass "
                "batch=<dict with protein_pos / protein_v / protein_batch>. "
                "Use interference: gen_ipdiff_pocket, not gen_unconditional."
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
        # (scripts/sample_diffusion.py)
        from torch_scatter import scatter_mean

        centre = scatter_mean(protein_pos, protein_batch, dim=0)[ligand_batch]
        init_ligand_pos = centre + torch.randn_like(centre)
        init_ligand_v = log_sample_categorical(
            torch.zeros(len(ligand_batch), self.model.num_classes, device=device)
        ).argmax(dim=-1)

        out = self.model.sample_diffusion(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=protein_batch,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=ligand_batch,
            net_cond=self.net_cond,
            num_steps=num_steps,
            center_pos_mode=self.model.center_pos_mode,
            progress=progress,
        )
        return KGDiffDiffusionTask._pad(  # noqa: SLF001 - same vocabulary
            out["pos"].detach(), out["v"].detach(), sizes
        )


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_ipdiff.yaml``.

    No ``train_set`` parameter: like KGDiff, the ligand-size prior is a
    static table conditioned on pocket extent, so nothing is measured at
    build time (docs/adding_new_models.md Section 2.5 -- that seam is
    opt-in).

    ``net_cond_ckpt`` is required in substance: :class:`BAPNet` refuses to
    build without pretrained weights, because its output *is* IPDiff's
    conditioning signal.
    """

    def __init__(
        self,
        task_type: str = "diffusion_ipdiff",
        net_cond_ckpt: str = DEFAULT_IPNET_CKPT,
        cond_dim: int = 128,
        pos_noise_std: float = 0.1,
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
        sample_time_method: str = "symmetric",
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
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by cli/train.py
    ) -> None:
        self.task_type = task_type
        self.net_cond_ckpt = net_cond_ckpt
        self.pos_noise_std = pos_noise_std
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(KGDIFF_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.model_kwargs = dict(
            cond_dim=cond_dim,
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
            sample_time_method=sample_time_method,
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
        )
        self.cond_dim = cond_dim
        self.task: Optional[IPDiffDiffusionTask] = None

    def build(self) -> IPDiffDiffusionTask:
        self.task = IPDiffDiffusionTask(
            IPDiffScorePosNet3D(**self.model_kwargs),
            BAPNet(ckpt_path=self.net_cond_ckpt, hidden_nf=self.cond_dim),
            atom_vocab=self.atom_vocab,
            pos_noise_std=self.pos_noise_std,
        )
        return self.task


class IPDiffPocketGenerator(KGDiffPocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_ipdiff_pocket``.

    Subclasses KGDiff's generator because the pocket source is literally the
    same object (one row of an ASE db written by
    ``docs/model_integrations/kgdiff/scripts/convert_dataset.py``) and the
    tiling / size-drawing logic is identical. The only difference is that
    IPDiff has no guidance knobs to pass through: its conditioning IS the
    pretrained prior, so ``_sample_kwargs`` drops them.
    """

    tag = "ipdiff"

    def __init__(self, task, **kwargs: Any) -> None:
        kwargs.setdefault("output_path", "generated_ipdiff")
        super().__init__(task=task, **kwargs)

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        return {"progress": False}

    def _start(self, item: Dict[str, Any]) -> None:
        print(
            f"[{self.tag}] pocket '{item['name']}': "
            f"{len(item['protein_pos'])} atoms, generating "
            f"{self.num_generate} ligands"
        )
