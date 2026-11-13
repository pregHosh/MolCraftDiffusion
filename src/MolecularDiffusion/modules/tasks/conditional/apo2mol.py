"""Apo2Mol task: apo-pocket-conditioned ligand diffusion with a co-generated pocket.

Three objects, the layout every pocket-conditioned model here uses:

* :class:`Apo2MolDiffusionTask` -- the duck-typed Task
  (docs/adding_new_models.md Section 2.1) wrapping
  :class:`~...models.apo2mol.ScorePosNet3D` **and** the frozen PMINet prior.
* :class:`ModelTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_apo2mol.yaml``.
* :class:`Apo2MolPocketGenerator` -- the ``_target_`` of
  ``configs/interference/gen_apo2mol_pocket.yaml``.
  ``GenerativeFactory``'s ``sample(batch_size, nodesxsample, ...)`` has no
  channel for "which pocket", so pocket models get their own generator behind
  their own ``_target_``; ``cli/generate.py`` only does
  ``instantiate(cfg.interference, task=task)`` then ``.run()``, so no core
  change is needed.

**What Apo2Mol adds over KGDiff / IPDiff, in one sentence.** Those two treat
the pocket as a fixed condition; Apo2Mol takes an **apo** (ligand-free) pocket
and generates the ligand *and* a holo-like pocket conformation together, by
diffusing per-residue rigid transforms and side-chain chi angles alongside the
ligand point cloud.

**Where the generated pocket goes.** The Section 2.1 ``sample()`` return
tuple has no protein channel, so it stays the platform-standard
``(one_hot, charges, coords, node_mask)`` and the co-generated pocket is
handed back out-of-band via :attr:`Apo2MolDiffusionTask.last_pocket`, which
the generator writes out as a ``.pdb`` sidecar per sample together with its
RMSD / TM-score against the input pocket.

Two deviations from the generic contract, both shared with the other pocket
models in-tree:

* ``sample()`` requires a pocket; there is no unconditional path.
* Sampled coordinates come back in the **input pocket's frame**
  (``center_pos_mode='protein'``).

Out of scope this pass (see the integration plan): guidance of every kind
(Apo2Mol has none -- ``prop_dist_model`` is ``None``), unconditional
generation, trajectory / ``n_frames`` export, the retrieval-prompt branch,
``pos_only`` sampling, and Vina docking.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from MolecularDiffusion.data.integrations.apo2mol_data import (
    APO2MOL_ATOM_VOCAB,
    Apo2MolDataset,
)
from MolecularDiffusion.data.integrations.kgdiff_data import (
    NUM_LIGAND_CLASSES,
    PROTEIN_FEATURE_DIM,
)
from MolecularDiffusion.modules.models.apo2mol import (
    BAPNet,
    ScorePosNet3D,
    log_sample_categorical,
)
from MolecularDiffusion.modules.models.kgdiff.atom_num import get_space_size

# Reused rather than re-derived: the ligand-size prior is the same static
# pocket-extent table, and the flat -> padded conversion (including the
# 13-class -> 8-element collapse) is the same operation on the same
# vocabulary.
from MolecularDiffusion.modules.tasks.conditional.kgdiff import (
    KGDiffDiffusionTask,
    PocketSizePrior,
)
from MolecularDiffusion.modules.tasks.conditional.pocket_generator import PocketGenerator

logger = logging.getLogger(__name__)

INT_TYPE = torch.int64

#: Default location of the converted PMINet prior, relative to the repo root
#: (the CWD ``MolCraftDiff`` is invoked from). Only consulted when training
#: from scratch -- when loading Apo2Mol's own checkpoint the prior arrives
#: with it under ``net_cond.*``. Overridable via ``tasks.net_cond_ckpt``.
DEFAULT_PMINET_CKPT = "docs/model_integrations/apo2mol/checkpoints/pminet"

#: Everything ``ScorePosNet3D`` reaches for through ``data.<field>``. Tensors
#: are moved to the task's device; the two string lists are passed through.
_TENSOR_KEYS = (
    "protein_pos",
    "protein_pos_holo",
    "protein_v",
    "protein_batch",
    "protein_element_batch",
    "protein_atom_to_aa_group",
    "protein_translations",
    "protein_translations_batch",
    "protein_rotations",
    "protein_chi_apo",
    "protein_chi_holo",
    "protein_chi_mask",
    "ligand_pos",
    "ligand_v",
    "ligand_batch",
)
_LIST_KEYS = ("protein_atom_name", "protein_atom_to_aa_name")


class Apo2MolDiffusionTask(nn.Module):
    """Task contract around :class:`ScorePosNet3D` + the frozen PMINet prior."""

    def __init__(
        self,
        model: ScorePosNet3D,
        net_cond: BAPNet,
        atom_vocab: Optional[List[str]] = None,
        pos_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        # attribute name must stay `model`: cli/generate.py reaches for
        # task.model when reconciling diffusion_steps.
        self.model = model
        # A real submodule, so the prior round-trips through the platform
        # checkpoint under `net_cond.*` -- which is also the prefix Apo2Mol's
        # own released checkpoint uses, so the two line up key for key.
        self.net_cond = net_cond
        self.net_cond.freeze()
        self.pos_noise_std = pos_noise_std
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(APO2MOL_ATOM_VOCAB)
        )
        self.prop_dist_model = None
        self.split = "train"
        self._node_dist_override: Optional[PocketSizePrior] = None
        #: Out-of-band channel for the co-generated pocket; see the module
        #: docstring. Set by :meth:`sample`, read by the generator immediately
        #: afterwards.
        # ponytail: a plain attribute, not a queue -- sample() is called
        # synchronously and the value is consumed before the next call.
        self.last_pocket: Optional[Dict[str, Any]] = None

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

    # -- adapter --------------------------------------------------------- #
    def _adapt(self, batch: Dict[str, Any]) -> SimpleNamespace:
        """Collated dict -> the ``data.<field>`` object the model expects.

        The ported model code indexes attributes, upstream's PyG ``Batch``
        being an attribute bag; a ``SimpleNamespace`` presents the collate's
        dict the same way with no container conversion.
        """
        device = self.device
        fields: Dict[str, Any] = {}
        for key in _TENSOR_KEYS:
            if key not in batch:
                raise KeyError(
                    f"batch is missing {key!r}. Apo2Mol needs a batch from "
                    "data/component/apo2mol_data.py:apo2mol_collate."
                )
            fields[key] = batch[key].to(device)
        for key in _LIST_KEYS:
            fields[key] = batch[key]
        return SimpleNamespace(**fields)

    # -- training -------------------------------------------------------- #
    def forward(self, batch):
        data = self._adapt(batch)
        protein_pos_apo = data.protein_pos
        if self.training and self.pos_noise_std > 0:
            # configs/training.yaml:65 pos_noise_std -- training-only jitter of
            # the APO pocket. The holo target is left clean.
            protein_pos_apo = protein_pos_apo + torch.randn_like(
                protein_pos_apo
            ) * self.pos_noise_std

        out = self.model.get_diffusion_loss(
            net_cond=self.net_cond,
            data=data,
            protein_pos_apo=protein_pos_apo,
            protein_pos_holo=data.protein_pos_holo,
            protein_v=data.protein_v.float(),
            batch_protein=data.protein_batch,
            ligand_pos=data.ligand_pos,
            ligand_v=data.ligand_v,
            batch_ligand=data.ligand_batch,
        )
        return out["loss"], {
            "loss": out["loss"],
            "loss_ligand_pos": out["loss_ligand_pos"],
            "loss_v": out["loss_v"],
            "loss_protein_tr": out["loss_protein_tr"],
            "loss_protein_rot": out["loss_protein_rot"],
            "loss_protein_chi": out["loss_protein_chi"],
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
        """Sample ligands inside the apo pocket carried by ``batch``.

        **The signature deliberately deviates from Section 2.1**: it takes a
        pocket. ``batch`` must come from ``apo2mol_collate``, or from
        :meth:`Apo2MolPocketGenerator._repeat`, which builds the same keys.

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)`` in the ORIGINAL pocket frame. ``one_hot`` is over the
        8-element ``atom_vocab``, not the model's 13 ``(element, aromatic)``
        classes; ``charges`` is zeros (Apo2Mol has no charge channel).

        The co-generated pocket lands in :attr:`last_pocket` as
        ``{"pos", "batch", "rmsd", "tmscore"}``.

        If ``batch`` carries no ``protein_pos_holo`` (pocket-only generation)
        the apo coordinates are used in its place, exactly as upstream's
        ``sample_custom_pocket.py:690`` does. The reported RMSD / TM-score is
        then **displacement from the input apo structure, not accuracy against
        a true holo structure**.
        """
        if batch is None:
            raise ValueError(
                "Apo2MolDiffusionTask.sample() is pocket-conditioned: pass "
                "batch=<a collated Apo2Mol pocket batch>. Use interference: "
                "gen_apo2mol_pocket, not gen_unconditional."
            )
        device = self.device
        protein_pos = batch["protein_pos"].to(device, torch.float32)
        protein_v = batch["protein_v"].to(device, torch.float32)
        protein_batch = batch["protein_batch"].to(device, INT_TYPE)
        # sample_custom_pocket.py:690 -- holo := apo when no holo is known.
        protein_pos_holo = batch.get("protein_pos_holo")
        protein_pos_holo = (
            protein_pos.clone()
            if protein_pos_holo is None
            else protein_pos_holo.to(device, torch.float32)
        )
        n_samples = int(protein_batch.max().item()) + 1

        data = SimpleNamespace(
            protein_atom_name=batch["protein_atom_name"],
            protein_atom_to_aa_name=batch["protein_atom_to_aa_name"],
            protein_atom_to_aa_group=batch["protein_atom_to_aa_group"].to(
                device, INT_TYPE
            ),
            protein_element_batch=batch["protein_element_batch"].to(
                device, INT_TYPE
            ),
            protein_chi_mask=batch["protein_chi_mask"].to(device, torch.float32),
            protein_translations_batch=batch["protein_translations_batch"].to(
                device, INT_TYPE
            ),
        )

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
        # (sample_custom_pocket.py:249-252)
        from torch_scatter import scatter_mean  # noqa: PLC0415

        centre = scatter_mean(protein_pos, protein_batch, dim=0)[ligand_batch]
        init_ligand_pos = centre + torch.randn_like(centre)
        init_ligand_v = log_sample_categorical(
            torch.zeros(len(ligand_batch), self.model.num_classes, device=device)
        )

        out = self.model.sample_diffusion(
            data=data,
            protein_pos_apo=protein_pos,
            protein_pos_holo=protein_pos_holo,
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

        self.last_pocket = {
            "pos": out["protein_pos"].detach().cpu(),
            "batch": protein_batch.detach().cpu(),
            "rmsd": [float(v) for v in out["protein_pos_rmsd"]],
            "tmscore": [float(v) for v in out["protein_pos_tmscore"]],
        }
        return KGDiffDiffusionTask._pad(  # noqa: SLF001 - same vocabulary
            out["ligand_pos"].detach(), out["v"].detach(), sizes
        )


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_apo2mol.yaml``.

    No ``train_set`` parameter: like KGDiff and IPDiff, the ligand-size prior
    is a static table conditioned on pocket extent, so nothing has to be
    measured at build time (docs/adding_new_models.md Section 2.5 -- that seam
    is opt-in).
    """

    def __init__(
        self,
        task_type: str = "diffusion_apo2mol",
        net_cond_ckpt: Optional[str] = DEFAULT_PMINET_CKPT,
        cond_dim: int = 128,
        topk_prompt: int = 0,
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
        lambda_schedule: str = "sigmoid",
        num_diffusion_timesteps: int = 1000,
        loss_v_weight: float = 100.0,
        loss_chi_weight: float = 5.0,
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
        edge_feat_dim: int = 5,
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
        num_protein_update_steps: int = 5,
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by cli/train.py
    ) -> None:
        self.task_type = task_type
        self.net_cond_ckpt = net_cond_ckpt
        self.cond_dim = cond_dim
        self.pos_noise_std = pos_noise_std
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(APO2MOL_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.model_kwargs = dict(
            protein_atom_feature_dim=protein_atom_feature_dim,
            ligand_atom_feature_dim=ligand_atom_feature_dim,
            cond_dim=cond_dim,
            topk_prompt=topk_prompt,
            model_mean_type=model_mean_type,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            pos_beta_s=pos_beta_s,
            v_beta_schedule=v_beta_schedule,
            v_beta_s=v_beta_s,
            lambda_schedule=lambda_schedule,
            num_diffusion_timesteps=num_diffusion_timesteps,
            loss_v_weight=loss_v_weight,
            loss_chi_weight=loss_chi_weight,
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
            num_protein_update_steps=num_protein_update_steps,
        )
        self.task: Optional[Apo2MolDiffusionTask] = None

    def build(self) -> Apo2MolDiffusionTask:
        ckpt = self.net_cond_ckpt
        if ckpt and not os.path.exists(ckpt):
            # Not fatal at generate time: Apo2Mol's own checkpoint carries the
            # prior under net_cond.*, so it will be overwritten a moment later.
            logger.warning(
                "PMINet weights not found at %s -- building the prior from "
                "random init. That is correct only if you are about to load a "
                "checkpoint that contains net_cond.* (Apo2Mol's own does). "
                "For training from scratch, convert them first with "
                "docs/model_integrations/apo2mol/scripts/convert_checkpoint.py",
                ckpt,
            )
            ckpt = None
        self.task = Apo2MolDiffusionTask(
            ScorePosNet3D(**self.model_kwargs),
            BAPNet(ckpt_path=ckpt, hidden_nf=self.cond_dim),
            atom_vocab=self.atom_vocab,
            pos_noise_std=self.pos_noise_std,
        )
        return self.task


def write_pocket_pdb(
    path: str,
    coords: torch.Tensor,
    atom_names: List[str],
    aa_names: List[str],
    aa_group: torch.Tensor,
    elements: Optional[List[str]] = None,
) -> None:
    """Write a minimal, viewer-loadable PDB for one generated pocket.

    Residue numbering is the local 0-based ``aa_group`` + 1, so the sidecar is
    self-consistent but is NOT the original PDB numbering.
    """
    with open(path, "w") as fh:
        for i, (x, y, z) in enumerate(coords.tolist()):
            name = atom_names[i]
            elem = (
                elements[i]
                if elements is not None
                else "".join(c for c in name if c.isalpha())[:1]
            )
            # PDB atom names are right-justified unless 4 characters long.
            name_field = f"{name:<4s}" if len(name) >= 4 else f" {name:<3s}"
            fh.write(
                f"ATOM  {i + 1:5d} {name_field}{aa_names[i]:>3s} A"
                f"{int(aa_group[i]) + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{elem:>2s}\n"
            )
        fh.write("END\n")


class Apo2MolPocketGenerator(PocketGenerator):
    """Pocket-conditioned generation behind ``interference/gen_apo2mol_pocket``.

    The apo pocket comes from one row of a converted ASE db
    (``docs/model_integrations/apo2mol/scripts/convert_dataset.py``); the
    sampled ligand comes back in that pocket's own frame.

    Two things it writes that no other in-tree generator does:

    * ``pocket_<i>.pdb`` -- the co-generated holo-like pocket for each sample,
      which is Apo2Mol's headline contribution;
    * ``pocket_metrics.csv`` -- per-sample RMSD and TM-score of that pocket
      against the reference the model was given.

    ``use_holo_reference`` decides what that reference is. Default ``false``
    reproduces upstream's custom-inference behaviour
    (``sample_custom_pocket.py:686-690``): holo is set to apo, so the metrics
    measure **displacement from the input apo structure**, not accuracy. Set
    it to ``true`` on a db row converted from a real apo/holo pair to get a
    genuine accuracy number instead.

    The sampling loop itself lives in :class:`PocketGenerator`.
    """

    tag = "apo2mol"
    db_required_msg = (
        "interference.pocket_db is required: Apo2Mol has no "
        "unconditional mode. Point it at a converted ASE db "
        "(docs/model_integrations/apo2mol/scripts/convert_dataset.py)."
    )

    def __init__(
        self,
        task,
        pocket_db: Optional[str] = None,
        pocket_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 2,
        num_steps: Optional[int] = None,
        mol_size: Optional[list] = None,
        use_holo_reference: bool = False,
        output_path: str = "generated_apo2mol",
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
        self.use_holo_reference = use_holo_reference
        self._rows: List[str] = []

    def _pocket(self) -> Dict[str, Any]:
        return Apo2MolDataset(self.pocket_db)[self.pocket_index]

    def _repeat(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile one pocket ``n`` times, with fresh scatter indices.

        The two string lists are tiled as NESTED lists (one entry per copy),
        and ``protein_atom_to_aa_group`` is repeated WITHOUT an offset -- both
        are the invariants ``apo2mol_collate`` maintains and the model relies
        on.
        """
        n_atom = len(item["protein_pos"])
        n_res = len(item["protein_rotations"])
        holo = (
            item["protein_pos_holo"] if self.use_holo_reference else item["protein_pos"]
        )
        protein_batch = torch.repeat_interleave(
            torch.arange(n, dtype=INT_TYPE), n_atom
        )
        return {
            "protein_pos": item["protein_pos"].repeat(n, 1),
            "protein_pos_holo": holo.repeat(n, 1),
            "protein_v": item["protein_v"].repeat(n, 1),
            "protein_batch": protein_batch,
            "protein_element_batch": protein_batch,
            "protein_atom_to_aa_group": item["protein_atom_to_aa_group"].repeat(n),
            "protein_chi_mask": item["protein_chi_mask"].repeat(n, 1),
            "protein_translations_batch": torch.repeat_interleave(
                torch.arange(n, dtype=INT_TYPE), n_res
            ),
            "protein_atom_name": [item["protein_atom_name"]] * n,
            "protein_atom_to_aa_name": [item["protein_atom_to_aa_name"]] * n,
        }

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        return {"progress": False}

    def _write_pockets(self, item, written: int, n: int, rows: List[str]) -> None:
        pocket = self.task.last_pocket
        if pocket is None:
            return
        for j in range(n):
            mask = pocket["batch"] == j
            write_pocket_pdb(
                os.path.join(self.output_path, f"pocket_{written + j}.pdb"),
                pocket["pos"][mask],
                item["protein_atom_name"],
                item["protein_atom_to_aa_name"],
                item["protein_atom_to_aa_group"],
            )
            rmsd = pocket["rmsd"][j] if j < len(pocket["rmsd"]) else float("nan")
            tm = pocket["tmscore"][j] if j < len(pocket["tmscore"]) else float("nan")
            rows.append(f"{written + j},{rmsd:.4f},{tm:.4f}\n")

    def _after_batch(self, item: Dict[str, Any], written: int, n: int) -> None:
        self._write_pockets(item, written, n, self._rows)

    def _start(self, item: Dict[str, Any]) -> None:
        self._rows = ["sample,pocket_rmsd,pocket_tmscore\n"]
        reference = "holo" if self.use_holo_reference else "apo (upstream default)"
        print(
            f"[{self.tag}] pocket '{item['name']}': {len(item['protein_pos'])} "
            f"atoms / {len(item['protein_rotations'])} residues, pocket "
            f"reference = {reference}, generating {self.num_generate} ligands"
        )

    def _summary(self, written: int, attempts: int) -> None:  # noqa: ARG002
        metrics_path = os.path.join(self.output_path, "pocket_metrics.csv")
        with open(metrics_path, "w") as fh:
            fh.writelines(self._rows)
        print(
            f"[{self.tag}] wrote {written} ligands, {written} pocket .pdb "
            f"sidecars and {metrics_path} to {self.output_path}"
        )
        if not self.use_holo_reference:
            print(
                f"[{self.tag}] NOTE: pocket_rmsd / pocket_tmscore are measured "
                "against the INPUT APO structure (use_holo_reference=false), "
                "so they report displacement, not accuracy against a true "
                "holo structure."
            )
