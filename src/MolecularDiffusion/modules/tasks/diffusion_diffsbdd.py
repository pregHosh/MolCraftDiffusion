"""DiffSBDD task: pocket-conditioned (and joint) 3D ligand diffusion.

Four objects, mirroring the platform's usual layout:

* :class:`DiffSBDDTask` -- the duck-typed Task (docs/adding_new_models.md
  Section 2.1) wrapping either :class:`ConditionalDDPM` or
  :class:`EnVariationalDiffusion`, selected by ``mode``.
* :class:`ModelTaskFactory` -- the ``_target_`` of
  ``configs/tasks/diffusion_diffsbdd.yaml`` (and of its 4-line
  ``diffusion_diffsbdd_joint.yaml`` sibling, which only overrides ``mode``).
* :class:`DiffSBDDPocketGenerator` -- the ``_target_`` of BOTH
  ``configs/interference/gen_diffsbdd_pocket.yaml`` and
  ``gen_diffsbdd_inpaint.yaml``; they differ only in whether
  ``fixed_atom_indices`` is null. ``GenerativeFactory``'s
  ``sample(batch_size, nodesxsample, ...)`` has no channel for "which
  pocket", so pocket-conditioned models get their own generator behind their
  own ``_target_`` -- the established in-tree pattern
  (``KGDiffPocketGenerator``, ``PMDMPocketGenerator``), needing no core
  change: ``cli/generate.py`` only does
  ``instantiate(cfg.interference, task=task)`` then ``.run()``.
* :class:`LigandPocketSizePrior` -- the ligand-size marginal of the model's
  2D joint ``(n_lig, n_pocket)`` histogram, wrapped to satisfy the
  ``node_dist_model`` contract.

Deviations from the generic contract, both shared with the other pocket
models in-tree:

* ``sample()`` requires a pocket. ``ConditionalDDPM.sample()`` itself raises
  without one, so ``interference: gen_unconditional`` does not apply here.
* Sampled coordinates come back in the **input pocket's frame** -- the
  generator restores the pocket CoM exactly as
  ``lightning_modules.py:845-852`` does.

## The `mode` guard (read this before touching `mode_id`)

Both modes share ``task_type: diffusion_diffsbdd`` AND -- verified against the
released weights -- **identical state-dict keys and shapes** (122 tensors
each; ``update_pocket_coords`` is a flag, not a parameter). The integration
plan expected a shape mismatch to catch a joint checkpoint loaded under a
``pocket_conditioning`` config; there is none, and ``cli/generate.py:322``
loads with ``strict=False``, so such a mix-up would run silently on the wrong
sampler. The ``mode_id`` buffer below is the fix: it round-trips through the
checkpoint, and :meth:`DiffSBDDTask._check_mode` refuses to train or sample
when the loaded value disagrees with the configured ``mode``.

Out of scope this pass (see the integration plan): ``optimize.py`` /
``diversify()``, ``SimpleConditionalDDPM``, CA pocket representation, virtual
nodes, the auxiliary Lennard-Jones loss, trajectory export, Vina docking, and
``EnVariationalDiffusion.sample()`` (the fully unconditional path that
hallucinates a pocket).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_scatter import scatter_mean

from MolecularDiffusion.data.integrations.diffsbdd_data import (
    DIFFSBDD_ATOM_VOCAB,
    NUM_ATOM_CLASSES,
    DiffSBDDDataset,
)
from MolecularDiffusion.modules.models.diffsbdd import DDPM_MODES, EGNNDynamics
from MolecularDiffusion.modules.tasks.pocket_generator import PocketGenerator

INT_TYPE = torch.int64
FLOAT_TYPE = torch.float32

#: ``mode`` string -> the integer stored in the ``mode_id`` buffer. Order is
#: part of the checkpoint format; append, never reorder.
MODES: Tuple[str, ...] = ("pocket_conditioning", "joint")


class LigandPocketSizePrior:
    """Ligand-size prior over the model's 2D joint histogram.

    ``.sample(n)`` returns the ligand-axis **marginal**, which is what the
    Section 2.1 ``node_dist_model`` contract asks for. The pocket generator
    does not use it: it calls ``sample_conditional(n2=pocket_sizes)`` on the
    model's own :class:`DistributionNodes`, which is the native behaviour and
    the only one that respects pocket size.
    """

    def __init__(self, histogram: torch.Tensor) -> None:
        marginal = torch.as_tensor(histogram).float().sum(dim=1)
        self.marginal = marginal
        self.n_node_dist = {
            i: float(c) for i, c in enumerate(marginal.tolist()) if c > 0
        }

    def sample(self, n_samples: int) -> torch.Tensor:
        if not self.n_node_dist:
            raise ValueError(
                "the ligand-size histogram is empty: this task was built "
                "without a train_set and the checkpoint carried no "
                "size_histogram buffer. Set interference.mol_size to an "
                "explicit list of ligand sizes."
            )
        return torch.multinomial(
            self.marginal, n_samples, replacement=True
        ).to(INT_TYPE)


class DiffSBDDTask(nn.Module):
    """Task contract around :class:`ConditionalDDPM` / :class:`EnVariationalDiffusion`."""

    def __init__(
        self,
        model: nn.Module,
        mode: str = "pocket_conditioning",
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        # attribute name must stay `model`: cli/generate.py reaches for
        # task.model when reconciling diffusion_steps (it carries `.T`).
        self.model = model
        self.mode = mode
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(DIFFSBDD_ATOM_VOCAB)
        )
        self.prop_dist_model = None
        self.split = "train"
        self._node_dist_override: Optional[LigandPocketSizePrior] = None
        # See the module docstring: the two modes are state-dict identical, so
        # this buffer is the only thing standing between a mode mix-up and a
        # silent wrong-sampler run under strict=False.
        self.register_buffer(
            "mode_id", torch.tensor(MODES.index(mode), dtype=INT_TYPE)
        )

    def _check_mode(self) -> None:
        loaded = MODES[int(self.mode_id)]
        if loaded != self.mode:
            raise ValueError(
                f"checkpoint was trained with mode={loaded!r} but this task "
                f"was built with mode={self.mode!r}. The two modes share "
                "task_type and state-dict shapes, so strict=False cannot "
                "catch this. Use `tasks: diffusion_diffsbdd_joint` for joint "
                "checkpoints and `tasks: diffusion_diffsbdd` for conditional "
                "ones."
            )

    # -- contract properties -------------------------------------------- #
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self) -> LigandPocketSizePrior:
        if self._node_dist_override is not None:
            return self._node_dist_override
        return LigandPocketSizePrior(self.model.size_histogram)

    @node_dist_model.setter
    def node_dist_model(self, value) -> None:
        # cli/generate.py assigns this when an edm_stat.pkl sidecar exists.
        self._node_dist_override = value

    @property
    def n_node_dist(self) -> Dict[int, float]:
        return self.node_dist_model.n_node_dist

    # -- training -------------------------------------------------------- #
    def _adapt(self, batch: Dict[str, Any]):
        """Collated batch -> upstream's ``(ligand, pocket)`` dict pair.

        Mirrors ``lightning_modules.py:217-234``. Called before any mode
        branch, exactly as upstream does -- both DDPM classes take the same
        pair.
        """
        device = self.device
        ligand = {
            "x": batch["lig_coords"].to(device, FLOAT_TYPE),
            "one_hot": batch["lig_one_hot"].to(device, FLOAT_TYPE),
            "size": batch["num_lig_atoms"].to(device, INT_TYPE),
            "mask": batch["lig_mask"].to(device, INT_TYPE),
        }
        pocket = {
            "x": batch["pocket_coords"].to(device, FLOAT_TYPE),
            "one_hot": batch["pocket_one_hot"].to(device, FLOAT_TYPE),
            "size": batch["num_pocket_nodes"].to(device, INT_TYPE),
            "mask": batch["pocket_mask"].to(device, INT_TYPE),
        }
        return ligand, pocket

    def forward(self, batch):
        """Loss assembly ported from ``lightning_modules.py:236-302``.

        Mode-independent: both models return the same keys, the conditional
        one substituting a constant 0 for the pocket terms. ``l2`` with
        per-example dimension normalisation while training, the full VLB
        otherwise.
        """
        self._check_mode()
        ligand, pocket = self._adapt(batch)
        out = self.model(ligand, pocket)

        x_dims = self.model.n_dims
        error_t_lig = out["error_t_lig"]
        error_t_pocket = out["error_t_pocket"]

        if self.training:
            denom_lig = x_dims * ligand["size"] + self.model.atom_nf * ligand["size"]
            denom_pocket = (x_dims + self.model.residue_nf) * pocket["size"]
            error_t_lig = error_t_lig / denom_lig
            error_t_pocket = error_t_pocket / denom_pocket
            loss_t = 0.5 * (error_t_lig + error_t_pocket)

            loss_0 = (
                out["loss_0_x_ligand"] / (x_dims * ligand["size"])
                + out["loss_0_x_pocket"] / (x_dims * pocket["size"])
                + out["loss_0_h"]
            )
            nll = loss_t + loss_0 + out["kl_prior"]
        else:
            # Note: SNR_weight should be negative.
            loss_t = (
                -self.model.T
                * 0.5
                * out["SNR_weight"]
                * (error_t_lig + error_t_pocket)
            )
            loss_0 = (
                out["loss_0_x_ligand"]
                + out["loss_0_x_pocket"]
                + out["loss_0_h"]
                + out["neg_log_constants"]
            )
            nll = loss_t + loss_0 + out["kl_prior"]
            # Correct for normalisation on x, then turn the conditional nll
            # into a joint one: log p(x,h,N) = log p(x,h|N) + log p(N).
            nll = nll - out["delta_log_px"] - out["log_pN"]

        loss = nll.mean(0)
        return loss, {
            "loss": loss,
            "loss_t": loss_t.mean(0),
            "loss_0": loss_0.mean(0),
            "kl_prior": out["kl_prior"].mean(0),
            "error_t_lig": error_t_lig.mean(0),
            "error_t_pocket": error_t_pocket.mean(0),
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
        ligand=None,
        lig_fixed=None,
        resamplings: int = 1,
        center: str = "ligand",
        jump_length: int = 1,
        **kwargs,  # noqa: ARG002 - swallows mode/n_frames from generic callers
    ):
        """Sample ligands inside the pocket carried by ``batch``.

        **The signature deliberately deviates from Section 2.1**: it takes a
        pocket. ``batch`` must hold ``pocket_coords`` / ``pocket_one_hot`` /
        ``pocket_mask`` / ``num_pocket_nodes`` from a collated DiffSBDD batch
        (:class:`DiffSBDDPocketGenerator` builds it).

        The whole cost of joint mode + inpainting is the three-way branch
        below; the last two rows of the plan's table are literally the same
        call, which is upstream's own design
        (``lightning_modules.py:814-834``), not a shortcut:

        =====================  ===========  ==============================
        mode                   lig_fixed    call
        =====================  ===========  ==============================
        pocket_conditioning    None         ``sample_given_pocket``
        pocket_conditioning    given        ``ConditionalDDPM.inpaint``
        joint                  None->zeros  ``EnVariationalDiffusion.inpaint``
        joint                  given        *same call*
        =====================  ===========  ==============================

        NB the ``mode`` kwarg that ``GenerativeFactory`` passes (``"ddpm"`` /
        ``"ddim"``) is swallowed by ``**kwargs`` and is NOT this task's
        generative ``self.mode``.

        Returns ``(one_hot, charges, coords, node_mask)`` padded to
        ``(B, N, .)`` in the INPUT POCKET's frame. ``charges`` is zeros:
        DiffSBDD has no charge channel.
        """
        self._check_mode()
        if batch is None:
            raise ValueError(
                "DiffSBDDTask.sample() is pocket-conditioned: pass "
                "batch=<dict with pocket_coords / pocket_one_hot / "
                "pocket_mask / num_pocket_nodes>. Use "
                "interference: gen_diffsbdd_pocket (or gen_diffsbdd_inpaint), "
                "not gen_unconditional."
            )
        device = self.device
        pocket = {
            "x": batch["pocket_coords"].to(device, FLOAT_TYPE),
            "one_hot": batch["pocket_one_hot"].to(device, FLOAT_TYPE),
            "size": batch["num_pocket_nodes"].to(device, INT_TYPE),
            "mask": batch["pocket_mask"].to(device, INT_TYPE),
        }
        com_before = scatter_mean(pocket["x"], pocket["mask"], dim=0)

        if ligand is not None:
            ligand = {
                "x": ligand["x"].to(device, FLOAT_TYPE),
                "one_hot": ligand["one_hot"].to(device, FLOAT_TYPE),
                "size": ligand["size"].to(device, INT_TYPE),
                "mask": ligand["mask"].to(device, INT_TYPE),
            }

        if self.mode == "joint" or lig_fixed is not None:
            if ligand is None:
                # de-novo in joint mode: a zero ligand, nothing fixed
                sizes = self._sizes(pocket, nodesxsample)
                ligand = self._zero_ligand(sizes, device)
            fixed = (
                torch.zeros(len(ligand["mask"]), device=device)
                if lig_fixed is None
                else lig_fixed.to(device, FLOAT_TYPE)
            )
            if self.mode == "joint":
                out_lig, out_pocket, lig_mask, pocket_mask = self.model.inpaint(
                    ligand,
                    pocket,
                    fixed,
                    torch.ones(len(pocket["mask"]), device=device),
                    resamplings=resamplings,
                    jump_length=jump_length,
                    timesteps=num_steps,
                )
            else:
                out_lig, out_pocket, lig_mask, pocket_mask = self.model.inpaint(
                    ligand,
                    pocket,
                    fixed,
                    resamplings=resamplings,
                    timesteps=num_steps,
                    center=center,
                )
        else:
            sizes = self._sizes(pocket, nodesxsample)
            out_lig, out_pocket, lig_mask, pocket_mask = (
                self.model.sample_given_pocket(
                    pocket, sizes, timesteps=num_steps
                )
            )

        # Move the generated molecule back to the input pocket's frame
        # (lightning_modules.py:845-852).
        com_after = scatter_mean(
            out_pocket[:, : self.model.n_dims], pocket_mask, dim=0
        )
        shift = com_before - com_after
        out_lig[:, : self.model.n_dims] += shift[lig_mask]
        return self._pad(out_lig.detach(), lig_mask)

    def _sizes(self, pocket, nodesxsample) -> torch.Tensor:
        """Explicit sizes, else the pocket-conditioned joint prior."""
        n_samples = len(pocket["size"])
        if nodesxsample is None:
            sizes = self.model.size_distribution.sample_conditional(
                n2=pocket["size"]
            )
        else:
            sizes = torch.as_tensor(nodesxsample)
        sizes = sizes.to(pocket["x"].device).long()
        if len(sizes) != n_samples:
            raise ValueError(
                f"nodesxsample has {len(sizes)} entries but the batch carries "
                f"{n_samples} pockets."
            )
        return sizes

    def _zero_ligand(self, sizes: torch.Tensor, device) -> Dict[str, torch.Tensor]:
        mask = torch.repeat_interleave(
            torch.arange(len(sizes), device=device), sizes
        )
        return {
            "x": torch.zeros((len(mask), self.model.n_dims), device=device),
            "one_hot": torch.zeros(
                (len(mask), self.model.atom_nf), device=device
            ),
            "size": sizes,
            "mask": mask,
        }

    @staticmethod
    def _pad(out_lig: torch.Tensor, lig_mask: torch.Tensor):
        """flat ``(n_total, 3 + n_classes)`` -> the padded contract shape."""
        device = out_lig.device
        n_dims = 3
        sizes = torch.bincount(lig_mask).tolist()
        n_samples, n_max = len(sizes), max(sizes)

        one_hot = torch.zeros((n_samples, n_max, NUM_ATOM_CLASSES), device=device)
        coords = torch.zeros((n_samples, n_max, n_dims), device=device)
        node_mask = torch.zeros((n_samples, n_max), device=device)

        classes = out_lig[:, n_dims:].argmax(dim=1)
        offset = 0
        for i, n in enumerate(sizes):
            sl = slice(offset, offset + n)
            coords[i, :n] = out_lig[sl, :n_dims]
            one_hot[i, torch.arange(n, device=device), classes[sl]] = 1.0
            node_mask[i, :n] = 1
            offset += n
        charges = torch.zeros((n_samples, n_max), device=device)
        return one_hot, charges, coords, node_mask


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_diffsbdd*.yaml``.

    ``train_set`` is declared so ``cli/train.py``'s declarative seam
    (docs/adding_new_models.md Section 2.5) injects the training set: the 2D
    joint size histogram is a genuine dataset statistic and is needed at
    *train* time too, for the ``log_pN`` VLB term. It is stored as a buffer on
    the model so it survives the checkpoint and generation can rebuild with
    ``train_set=None``. The buffer's shape is fixed by ``max_n_lig`` /
    ``max_n_pocket``, so checkpoint shapes never depend on which db was used;
    the defaults are the released CrossDocked histogram's own shape.
    """

    def __init__(
        self,
        task_type: str = "diffusion_diffsbdd",
        mode: str = "pocket_conditioning",
        atom_nf: int = NUM_ATOM_CLASSES,
        residue_nf: int = NUM_ATOM_CLASSES,
        n_dims: int = 3,
        # --- EGNN ---
        joint_nf: int = 32,
        hidden_nf: int = 128,
        n_layers: int = 5,
        inv_sublayers: int = 1,
        attention: bool = True,
        tanh: bool = True,
        norm_constant: float = 1,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        reflection_equivariant: bool = False,
        edge_cutoff_ligand: Optional[float] = None,
        edge_cutoff_pocket: Optional[float] = 5.0,
        edge_cutoff_interaction: Optional[float] = 5.0,
        edge_embedding_dim: Optional[int] = None,
        # --- diffusion ---
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "polynomial_2",
        diffusion_noise_precision: float = 5.0e-4,
        normalize_factors: Sequence[float] = (1.0, 4.0),
        # --- size prior ---
        max_n_lig: int = 107,
        max_n_pocket: int = 1671,
        atom_vocab: Optional[List[str]] = None,
        train_set: Any = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by cli/train.py
    ) -> None:
        if mode not in DDPM_MODES:
            raise ValueError(
                f"mode must be one of {sorted(DDPM_MODES)}, got {mode!r}. "
                "(pocket_conditioning_simple is an unreleased ablation and is "
                "out of scope for this port.)"
            )
        self.task_type = task_type
        self.mode = mode
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(DIFFSBDD_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.train_set = train_set
        self.max_n_lig = max_n_lig
        self.max_n_pocket = max_n_pocket

        self.dynamics_kwargs = dict(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            n_dims=n_dims,
            joint_nf=joint_nf,
            hidden_nf=hidden_nf,
            n_layers=n_layers,
            inv_sublayers=inv_sublayers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            reflection_equivariant=reflection_equivariant,
            edge_cutoff_ligand=edge_cutoff_ligand,
            edge_cutoff_pocket=edge_cutoff_pocket,
            edge_cutoff_interaction=edge_cutoff_interaction,
            edge_embedding_dim=edge_embedding_dim,
            # the ONE construction-time difference between the two modes
            # (lightning_modules.py:156)
            update_pocket_coords=(mode == "joint"),
        )
        self.ddpm_kwargs = dict(
            atom_nf=atom_nf,
            residue_nf=residue_nf,
            n_dims=n_dims,
            max_n_lig=max_n_lig,
            max_n_pocket=max_n_pocket,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            norm_values=tuple(normalize_factors),
        )
        self.task: Optional[DiffSBDDTask] = None

    def _size_histogram(self) -> Optional[torch.Tensor]:
        if self.train_set is None:
            return None
        hist = torch.zeros(self.max_n_lig, self.max_n_pocket)
        for item in self.train_set:
            n_lig = len(item["lig_coords"])
            n_pocket = len(item["pocket_coords"])
            if n_lig < self.max_n_lig and n_pocket < self.max_n_pocket:
                hist[n_lig, n_pocket] += 1
        return hist

    def build(self) -> DiffSBDDTask:
        model = DDPM_MODES[self.mode](
            dynamics=EGNNDynamics(**self.dynamics_kwargs),
            size_histogram=self._size_histogram(),
            **self.ddpm_kwargs,
        )
        self.task = DiffSBDDTask(
            model, mode=self.mode, atom_vocab=self.atom_vocab
        )
        return self.task


class DiffSBDDPocketGenerator(PocketGenerator):
    """Pocket-conditioned generation and inpainting.

    One class behind both interference configs -- they differ only in whether
    ``fixed_atom_indices`` is null, and a ``DiffSBDDInpaintGenerator`` would
    be an interface with one implementation.

    The pocket (and, for inpainting, the reference ligand that defined it)
    comes from one row of the converted ASE db, read with ``center=False`` so
    sampled ligands come back in that pocket's own frame.

    Fixed atoms are selected by **index into the stored reference ligand**
    (``fixed_atom_indices: [0, 1, 2, 3]``) rather than by PDB atom name
    (upstream's ``--fix_atoms "C1 N6 C5"``): the db has no atom names, and the
    same ``lig_fixed`` tensor reaches ``inpaint()`` either way.

    The sampling loop itself lives in :class:`PocketGenerator`.
    """

    tag = "diffsbdd"
    db_required_msg = (
        "interference.pocket_db is required: DiffSBDD has no "
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
        fixed_atom_indices: Optional[list] = None,
        add_n_nodes: Optional[int] = None,
        resamplings: int = 1,
        center: str = "ligand",
        jump_length: int = 1,
        output_path: str = "generated_diffsbdd",
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
        self.fixed_atom_indices = (
            list(fixed_atom_indices) if fixed_atom_indices else []
        )
        self.add_n_nodes = add_n_nodes
        self.resamplings = resamplings
        self.center = center
        self.jump_length = jump_length

    def _item(self) -> Dict[str, Any]:
        # center=False: the sampled ligand comes back in this pocket's own
        # frame and overlays on the original structure.
        return DiffSBDDDataset(self.pocket_db, center=False)[self.pocket_index]

    def _pocket(self) -> Dict[str, Any]:
        return self._item()

    def _pocket_batch(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile one pocket ``n`` times, with fresh scatter indices."""
        size = len(item["pocket_coords"])
        return {
            "pocket_coords": item["pocket_coords"].repeat(n, 1),
            "pocket_one_hot": item["pocket_one_hot"].repeat(n, 1),
            "pocket_mask": torch.repeat_interleave(
                torch.arange(n, dtype=INT_TYPE), size
            ),
            "num_pocket_nodes": torch.full((n,), size, dtype=INT_TYPE),
        }

    def _repeat(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        return self._pocket_batch(item, n)

    def _explicit_sizes(self, n: int) -> Optional[torch.Tensor]:
        """Explicit sizes, or ``None`` for the pocket-conditioned prior."""
        return self._sizes(n)

    def _fixed_ligand(self, item: Dict[str, Any], n: int):
        """Build the zero ligand + ``lig_fixed``, as ``inpaint.py:114-141``."""
        idx = torch.tensor(self.fixed_atom_indices, dtype=INT_TYPE)
        ref_x = item["lig_coords"][idx]
        ref_h = item["lig_one_hot"][idx]
        n_fixed = len(idx)

        if self.add_n_nodes is not None:
            sizes = torch.full((n,), n_fixed + self.add_n_nodes, dtype=INT_TYPE)
        else:
            pocket_sizes = torch.full(
                (n,), len(item["pocket_coords"]), dtype=INT_TYPE
            )
            sizes = self.task.model.size_distribution.sample_conditional(
                n2=pocket_sizes
            )
            sizes = torch.clamp(sizes, min=n_fixed).to(INT_TYPE)

        mask = torch.repeat_interleave(torch.arange(n, dtype=INT_TYPE), sizes)
        x = torch.zeros((len(mask), 3))
        one_hot = torch.zeros((len(mask), ref_h.shape[1]))
        lig_fixed = torch.zeros(len(mask))
        offset = 0
        for i in range(n):
            x[offset : offset + n_fixed] = ref_x
            one_hot[offset : offset + n_fixed] = ref_h
            lig_fixed[offset : offset + n_fixed] = 1.0
            offset += int(sizes[i])

        ligand = {"x": x, "one_hot": one_hot, "size": sizes, "mask": mask}
        return ligand, lig_fixed, n_fixed

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        ligand, lig_fixed = None, None
        if self.fixed_atom_indices:
            ligand, lig_fixed, _ = self._fixed_ligand(item, n)
        return {
            "ligand": ligand,
            "lig_fixed": lig_fixed,
            "resamplings": self.resamplings,
            "center": self.center,
            "jump_length": self.jump_length,
        }

    def _start(self, item: Dict[str, Any]) -> None:
        what = (
            f"inpainting around {len(self.fixed_atom_indices)} fixed atoms"
            if self.fixed_atom_indices
            else "de novo"
        )
        print(
            f"[{self.tag}] pocket '{item['name']}': "
            f"{len(item['pocket_coords'])} atoms, mode={self.task.mode}, "
            f"{what}, generating {self.num_generate} ligands"
        )
