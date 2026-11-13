"""SynCoGen task: joint masked graph diffusion + coordinate flow matching.

SynCoGen generates molecules the way a chemist would *make* them: nodes are
catalogue building blocks, edges are reaction templates, and the 3D coordinates
are flow-matched alongside. Chemical bonds are never diffused -- they are a
*consequence* of the sampled reaction graph, recovered by RDKit assembly at the
end (``BBRxnGraph.build_rdkit`` -> ``RDKitMoleculeAssembly``).

Two things follow from that, and both shape this file.

**The batch is not a point cloud.** Training consumes upstream's own PyG
``Data``/``Batch`` end to end, produced by the task-owned
``modules/models/syncogen/datamodule.py``. No platform ``data_type`` is involved
and nothing under ``MolecularDiffusion/data/`` is touched.

**Generation is.** ``GenerativeFactory``'s ``unconditional`` loop wants
``(one_hot, charges, coords, node_mask)`` plus a dense bond matrix on
``last_bond_types``. So the *only* adapters in this integration live in
``sample()``, converting the sampled ``(BBRxnGraph, Coordinates)`` pair into
those tensors -- see ``_assemble`` and ``_to_pointcloud``.

Two traps worth naming here rather than rediscovering:

* **``charges`` carries signed formal charges, not atomic numbers.** The channel
  is overloaded in the platform: ``save_xyz_file`` would read it as atomic
  numbers, but only when the model sets ``use_unknown_fallback`` (which this one
  does not, so the value is unused there), while
  ``GenerativeFactory._write_molecule_sdf`` passes it straight through as
  ``formal_charge=`` (``runmodes/generate/tasks_generate.py:344``). Atomic
  numbers here would silently stamp every SDF atom with a +6/+7/+8 charge.
* **Atom ordering is fixed by ``set_mol_coordinates``.** Upstream assigns the
  masked coordinate rows positionally onto the assembled molecule's atoms, so
  the point-cloud order and the RDKit atom order are the same order *by
  construction*. ``one_hot``, ``charges``, ``coords`` and the bond matrix are all
  built in that one order, or the ``.xyz`` and the ``.sdf`` would describe
  different molecules.

Vocabulary loading is process-global and must happen before any other import
from the ported package -- see ``modules/models/syncogen/vocab.py``. That is why
every syncogen import in this file is deferred into a function body.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import torch
from MolecularDiffusion.device import get_device
from torch import nn

from MolecularDiffusion.modules.models.syncogen.vocab import ensure_vocabulary

logger = logging.getLogger(__name__)

#: RDKit bond type -> the platform's canonical bond vocabulary
#: (``0=none 1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC``, docs §2.2.1). Populated
#: lazily so importing this module never requires RDKit.
_BOND_CLASS: dict[Any, int] = {}


def _bond_class_map() -> dict[Any, int]:
    if not _BOND_CLASS:
        from rdkit import Chem

        _BOND_CLASS.update(
            {
                Chem.BondType.SINGLE: 1,
                Chem.BondType.DOUBLE: 2,
                Chem.BondType.TRIPLE: 3,
                Chem.BondType.AROMATIC: 4,
            }
        )
    return _BOND_CLASS


def _plain(value: Any) -> Any:
    """OmegaConf container -> plain python, so downstream code can mutate it."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except (
        ImportError
    ):  # pragma: no cover - omegaconf is a hard dep of the CLI
        pass
    return value


class SyncogenSizePrior:
    """Fragment-count prior, and the small surface ``core.Diffusion`` reads.

    Serves three roles at once, which is why it is one object and not three:

    1. ``task.node_dist_model`` -- ``.sample(n) -> LongTensor`` of per-molecule
       sizes. For SynCoGen a "size" is a **fragment count** (typically 2-5), not
       an atom count, so ``mol_size`` in a generate config must be ``[0, 0]`` or
       a range inside ``[min, max_bbs]``. ``GenerativeFactory.__init__`` clamps
       against ``max(n_node_dist)`` and raises with a clear message otherwise.
    2. ``task.n_node_dist`` -- the ``{n_fragments: count}`` histogram that clamp
       reads.
    3. The stand-in for upstream's ``SyncogenDataManager`` inside
       ``core.Diffusion``, which reaches for ``max_bbs``, ``batch_size``,
       ``eval_batch_size``, ``load_bonds``, ``load_pharmacophores`` and
       ``sample_n_nodes``. Keeping it config-owned rather than dataset-owned is
       what lets generation run with no dataset present at all -- which is the
       whole point for the released checkpoints, where the 6.3 GB SynSpace
       download is not needed.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_fragments_probs: dict,
        max_bbs: int = 5,
        batch_size: int = 4,
        eval_batch_size: int = 4,
        load_bonds: bool = True,
        load_pharmacophores: bool = False,
    ) -> None:
        probs = {
            int(k): float(v) for k, v in dict(num_fragments_probs).items()
        }
        if not probs:
            msg = "num_fragments_probs must be a non-empty {n_fragments: prob} mapping"
            raise ValueError(msg)
        if any(p < 0 for p in probs.values()):
            msg = f"num_fragments_probs has a negative probability: {probs}"
            raise ValueError(msg)
        total = sum(probs.values())
        if total <= 0:
            msg = f"num_fragments_probs sums to {total}; at least one must be > 0"
            raise ValueError(msg)
        if max(probs) > max_bbs:
            msg = (
                f"num_fragments_probs covers {max(probs)} fragments but max_bbs="
                f"{max_bbs}. The backbone's positional embeddings are sized by "
                "max_bbs, so a larger fragment count cannot be represented."
            )
            raise ValueError(msg)

        values = sorted(probs)
        self.train_length_values = torch.tensor(values, dtype=torch.long)
        self.train_length_probs = torch.tensor(
            [probs[v] / total for v in values], dtype=torch.float
        )
        self.max_bbs = int(max_bbs)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.load_bonds = bool(load_bonds)
        self.load_pharmacophores = bool(load_pharmacophores)
        #: Set for the duration of one ``sample()`` call so a generate config's
        #: ``mol_size`` wins over the prior; ``None`` restores the histogram.
        self.forced_n_nodes: torch.Tensor | None = None

    def sample_n_nodes(self, batch_size: int) -> torch.Tensor:
        if self.forced_n_nodes is not None:
            forced = self.forced_n_nodes.reshape(-1).long()
            if forced.numel() < batch_size:  # repeat to fill a larger batch
                reps = -(-batch_size // max(1, forced.numel()))
                forced = forced.repeat(reps)
            return forced[:batch_size].clamp(
                min=int(self.train_length_values.min()), max=self.max_bbs
            )
        idx = torch.multinomial(
            self.train_length_probs, num_samples=batch_size, replacement=True
        )
        return self.train_length_values[idx]

    # ``node_dist_model`` contract
    def sample(self, n: int) -> torch.Tensor:
        return self.sample_n_nodes(int(n))

    @property
    def n_node_dist(self) -> dict:
        """``{n_fragments: count}``. Counts are a scaled histogram; only the keys
        and their relative sizes matter to ``GenerativeFactory``."""
        return {
            int(v): max(1, int(round(float(p) * 10_000)))
            for v, p in zip(
                self.train_length_values.tolist(),
                self.train_length_probs.tolist(),
            )
        }


class ModelTaskFactory:
    """Instantiated by ``cli/train.py`` / ``cli/generate.py`` from Hydra.

    No ``train_set`` parameter is declared, so the §2.5 injection seam stays
    inert: SynCoGen's only construction-time statistic is the fragment-count
    histogram, and that is config-owned (``num_fragments_probs``) precisely so
    generation needs no dataset.
    """

    #: Config keys the CALLER owns at generation time (docs §2.5b,
    #: ``cli/generate.py::_declared_generation_time_keys``). Every one is "which
    #: files / which conditioning", never architecture -- a key that changed a
    #: tensor shape here would silently mismatch the weights. ``reference_ligand``
    #: is the pharmacophore reference a user points at their own ``.sdf``;
    #: ``vocab_dir`` and ``sdf_output_path`` are paths baked in on the training
    #: machine that must be re-pointable elsewhere.
    generation_time_keys = (
        "vocab_dir",
        "reference_ligand",
        "sdf_output_path",
        "num_fragments_probs",
        "eval_batch_size",
        "max_sample_attempts",
    )

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_syncogen",
        vocab_dir: str | None = None,
        backbone: str = "semla",
        # --- SEMLA / SEMLA-Pharm backbone (configs/model/semla.gin) ---
        d_model: int = 384,
        d_message: int = 128,
        n_coord_sets: int = 64,
        n_layers: int = 12,
        n_head: int = 32,
        d_message_hidden: int = 128,
        d_edge: int = 128,
        coord_norm: str = "length",
        size_emb: int = 64,
        pos_emb: int = 64,
        max_bbs: int = 5,
        self_conditioning: bool = True,
        pharm_subset: int = 7,
        # --- diffusion / flow objective ---
        noise_sigma_min: float = 1e-3,
        noise_sigma_max: float = 7.0,
        num_sample_steps: int = 100,
        sampling_eps: float = 1e-3,
        antithetic_sampling: bool = True,
        importance_sampling: bool = False,
        sampling_noise_removal: bool = True,
        use_compat: bool = True,
        time_conditioning: bool = True,
        train_rot_align: bool = True,
        augmentations: Sequence[str] = (
            "center",
            "random_rotate",
            "normalize",
        ),
        scale_noise: bool = True,
        scale_noise_factor: float = 0.2,
        inference_annealing: bool = True,
        annealing_coef: float = 10.0,
        ema_decay: float = 0.0,
        # --- loss weights (identical in both shipped experiments) ---
        nll_coef: float = 1.0,
        mse_coef: float = 1.0,
        bond_length_coef: float = 0.2,
        pairwise_distance_coef: float = 0.4,
        pairwise_distance_threshold: float = 5.0,
        smooth_lddt_coef: float = 0.4,
        loss_t_threshold: float = 0.25,
        # --- data-shape flags, must agree with configs/data/syncogen_dataset.yaml ---
        load_bonds: bool = True,
        load_pharmacophores: bool = False,
        # --- run configuration ---
        num_fragments_probs: dict | None = None,
        eval_batch_size: int = 4,
        reference_ligand: str | None = None,
        sdf_output_path: str | None = None,
        max_sample_attempts: int = 5,
        atom_vocab: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.vocab_dir = vocab_dir
        self.backbone = backbone
        self.backbone_kwargs = {
            "d_model": int(d_model),
            "d_message": int(d_message),
            "n_coord_sets": int(n_coord_sets),
            "n_layers": int(n_layers),
            "n_head": int(n_head),
            "d_message_hidden": int(d_message_hidden),
            "d_edge": int(d_edge),
            "coord_norm": coord_norm,
            "size_emb": int(size_emb),
            "pos_emb": int(pos_emb),
            "length": int(max_bbs),
        }
        self.max_bbs = int(max_bbs)
        self.self_conditioning = bool(self_conditioning)
        self.pharm_subset = int(pharm_subset)
        self.noise_sigma_min = float(noise_sigma_min)
        self.noise_sigma_max = float(noise_sigma_max)
        self.num_sample_steps = int(num_sample_steps)
        self.sampling_eps = float(sampling_eps)
        self.antithetic_sampling = bool(antithetic_sampling)
        self.importance_sampling = bool(importance_sampling)
        self.sampling_noise_removal = bool(sampling_noise_removal)
        self.use_compat = bool(use_compat)
        self.time_conditioning = bool(time_conditioning)
        self.train_rot_align = bool(train_rot_align)
        self.augmentations = list(_plain(augmentations))
        self.scale_noise = bool(scale_noise)
        self.scale_noise_factor = float(scale_noise_factor)
        self.inference_annealing = bool(inference_annealing)
        self.annealing_coef = float(annealing_coef)
        self.ema_decay = float(ema_decay)
        self.loss_coefs = {
            "nll": float(nll_coef),
            "mse": float(mse_coef),
            "bond_length": float(bond_length_coef),
            "pairwise_distance": float(pairwise_distance_coef),
            "pairwise_distance_threshold": float(pairwise_distance_threshold),
            "smooth_lddt": float(smooth_lddt_coef),
            "t_threshold": float(loss_t_threshold),
        }
        self.load_bonds = bool(load_bonds)
        self.load_pharmacophores = bool(load_pharmacophores)
        self.num_fragments_probs = _plain(num_fragments_probs)
        self.eval_batch_size = int(eval_batch_size)
        self.reference_ligand = reference_ligand
        self.sdf_output_path = sdf_output_path
        self.max_sample_attempts = int(max_sample_attempts)
        self.atom_vocab = list(_plain(atom_vocab)) if atom_vocab else None
        self.task: SyncogenDiffusionTask | None = None
        if kwargs:
            # cli/train.py injects node_feature/node_feature_dim/extra_norm_values
            # from its n_dim probe. SynCoGen has no per-atom feature channel to
            # size, so they are accepted and ignored (integration plan, "accepted
            # warts").
            logger.debug(
                "SynCoGen factory ignoring injected keys: %s", sorted(kwargs)
            )

    def build(self) -> "SyncogenDiffusionTask":
        if self.vocab_dir is None:
            msg = (
                "tasks.vocab_dir is required: SynCoGen's building-block and reaction "
                "vocabulary is process-global state that must be loaded before the "
                "model is constructed. Point it at a directory holding "
                "building_blocks.json, reactions.json, compatibility.pt, "
                "fragment_features.pt and meta.json."
            )
            raise ValueError(msg)

        # MUST precede every import below -- see modules/models/syncogen/vocab.py.
        ensure_vocabulary(self.vocab_dir)

        from MolecularDiffusion.modules.models.syncogen.constants import (
            constants as C,
        )
        from MolecularDiffusion.modules.models.syncogen.core import Diffusion
        from MolecularDiffusion.modules.models.syncogen.diffusion.interpolation import (
            LinearInterpolator,
        )
        from MolecularDiffusion.modules.models.syncogen.diffusion.loss import (
            BondLengthLoss,
            MSELoss,
            NLLLoss,
            PairwiseDistanceLoss,
            SmoothLDDTLoss,
        )
        from MolecularDiffusion.modules.models.syncogen.diffusion.noise import (
            LinearNoise,
        )
        from MolecularDiffusion.modules.models.syncogen.diffusion.sampling.discrete_strategies import (  # noqa: E501
            MDLM,
        )
        from MolecularDiffusion.modules.models.syncogen.diffusion.sampling.integrators import (  # noqa: E501
            EulerIntegrator,
        )

        vocab_atoms = list(C.ATOM_TYPES)
        if (
            self.atom_vocab is not None
            and list(self.atom_vocab) != vocab_atoms
        ):
            msg = (
                f"atom_vocab mismatch: config says {list(self.atom_vocab)} but the "
                f"vocabulary at {self.vocab_dir} declares {vocab_atoms} "
                "(meta.json:atom_types). Both save_xyz_file and _write_molecule_sdf "
                "index the vocabulary positionally, so a mismatch mislabels every "
                "element rather than raising."
            )
            raise ValueError(msg)
        atom_vocab = self.atom_vocab or vocab_atoms

        cf = self.loss_coefs
        losses = [
            NLLLoss(coef=cf["nll"]),
            MSELoss(coef=cf["mse"], time_weighted=True),
            PairwiseDistanceLoss(
                distance_threshold=cf["pairwise_distance_threshold"],
                sqrd=True,
                coef=cf["pairwise_distance"],
                time_weighted=True,
                t_threshold=cf["t_threshold"],
            ),
            BondLengthLoss(
                sqrd=True,
                coef=cf["bond_length"],
                time_weighted=True,
                t_threshold=cf["t_threshold"],
            ),
            SmoothLDDTLoss(
                sqrd=True,
                coef=cf["smooth_lddt"],
                time_weighted=True,
                t_threshold=cf["t_threshold"],
            ),
        ]

        size_prior = SyncogenSizePrior(
            num_fragments_probs=self.num_fragments_probs
            or {3: 1.0, 4: 1.0, 5: 1.0},
            max_bbs=self.max_bbs,
            batch_size=self.eval_batch_size,
            eval_batch_size=self.eval_batch_size,
            load_bonds=self.load_bonds,
            load_pharmacophores=self.load_pharmacophores,
        )

        core = Diffusion(
            data_manager=size_prior,
            losses=losses,
            augmentations=self.augmentations,
            normalization_scale=1.0 / C.COORDS_STD,
            discrete_noise=LinearNoise(
                sigma_min=self.noise_sigma_min, sigma_max=self.noise_sigma_max
            ),
            interpolator=LinearInterpolator(),
            discrete_strategy=MDLM(),
            integrator=EulerIntegrator(
                inference_annealing=self.inference_annealing,
                annealing_coef=self.annealing_coef,
            ),
            train_rot_align=self.train_rot_align,
            self_conditioning=self.self_conditioning,
            time_conditioning=self.time_conditioning,
            backbone=self.backbone,
            use_compat=self.use_compat,
            sampling_eps=self.sampling_eps,
            importance_sampling=self.importance_sampling,
            antithetic_sampling=self.antithetic_sampling,
            sampling_noise_removal=self.sampling_noise_removal,
            generate_eval_samples=False,
            num_sample_steps=self.num_sample_steps,
            ema_decay=self.ema_decay,
            pharm_subset=self.pharm_subset,
            scale_noise=self.scale_noise,
            scale_noise_factor=self.scale_noise_factor,
            # The prior is owned by `size_prior` above; passing it here too would
            # only overwrite the same two tensors.
            num_fragments_probs=None,
            backbone_kwargs=self.backbone_kwargs,
        )

        self.task = SyncogenDiffusionTask(
            core=core,
            size_prior=size_prior,
            atom_vocab=atom_vocab,
            task_type=self.task_type,
            num_sample_steps=self.num_sample_steps,
            pharm_subset=self.pharm_subset,
            max_bbs=self.max_bbs,
            reference_ligand=self.reference_ligand,
            sdf_output_path=self.sdf_output_path,
            max_sample_attempts=self.max_sample_attempts,
        )
        return self.task


class SyncogenDiffusionTask(nn.Module):
    """The duck-typed ``Task`` object (docs/adding_new_models.md §2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        core: nn.Module,
        size_prior: SyncogenSizePrior,
        atom_vocab: list,
        task_type: str,
        num_sample_steps: int,
        pharm_subset: int,
        max_bbs: int,
        reference_ligand: str | None = None,
        sdf_output_path: str | None = None,
        max_sample_attempts: int = 5,
    ) -> None:
        super().__init__()
        self.core = core
        self.size_prior = size_prior
        self.atom_vocab = list(atom_vocab)
        self.task_type = task_type
        self.pharm_subset = int(pharm_subset)
        self.max_bbs = int(max_bbs)
        self.reference_ligand = reference_ligand
        self.sdf_output_path = sdf_output_path
        self.max_sample_attempts = int(max_sample_attempts)
        #: Read by tasks_generate.py to pick a default step count.
        self.T = int(num_sample_steps)
        #: Dense (B,N,N) bond matrix from the last sample(); None until then, so
        #: _warn_if_bonds_dropped's getattr is honest.
        self.last_bond_types: torch.Tensor | None = None
        self.prop_dist_model = None
        self._ref_mol = None

    # -- platform properties -------------------------------------------------

    @property
    def model(self) -> "SyncogenDiffusionTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _place_on_accelerator(self) -> None:
        """Move the task to the GPU for generation. Its half of a platform bargain.

        Two device-placement contracts exist. A task with NO ``device`` attribute
        is moved by ``recursive_module_to_device``, which assigns
        ``module.device``. A task that defines ``device`` as a read-only property
        -- which docs/adding_new_models.md §2.1 asks for, and which every task in
        this repo does -- is skipped by the ``if not hasattr(task, "device")``
        guard at ``cli/generate.py:685`` and ``core/engine.py:180``, and must
        place itself.

        Without this, generation runs on the CPU: ``load_model`` reads the
        checkpoint with ``map_location="cpu"`` and nothing on the
        ``GenerativeFactory`` path ever moves it. Same approach as
        ``diffusion_equifm.py::_place_on_accelerator`` and
        ``diffusion_diffsbdd.py:611``. Training is unaffected -- Lightning moves
        the module itself.
        """
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.to("cuda")
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    @property
    def node_dist_model(self) -> SyncogenSizePrior:
        return self.size_prior

    @node_dist_model.setter
    def node_dist_model(self, value: SyncogenSizePrior) -> None:
        """Writable because the platform restores it from a checkpoint.

        ``EngineLightning.on_save_checkpoint`` pickles ``task.node_dist_model``
        into every checkpoint, and ``cli/generate.py``'s manual reconstruction
        path assigns it straight back (``cli/generate.py:401``) with no
        ``hasattr`` guard -- so a read-only property here would turn every
        generation-from-a-trained-checkpoint into an ``AttributeError``.

        The restored object also has to replace the one ``core.Diffusion`` holds
        as its ``data_manager``, or ``sample()`` would draw fragment counts from
        the config prior while ``sample_n_nodes`` mutations landed on a detached
        copy. One object, two names.
        """
        self.size_prior = value
        self.core.data_manager = value

    @property
    def n_node_dist(self) -> dict:
        return self.size_prior.n_node_dist

    @n_node_dist.setter
    def n_node_dist(self, value: dict) -> None:  # noqa: ARG002
        # Derived from node_dist_model, so a restored value is redundant rather
        # than authoritative. Accepted (the platform assigns it) and ignored.
        logger.debug(
            "SynCoGen: ignoring restored n_node_dist; derived from the prior"
        )

    # -- training ------------------------------------------------------------

    def forward(self, batch) -> tuple[torch.Tensor, dict]:
        prefix = getattr(self, "split", "train")
        prefix = "train" if prefix not in {"train", "valid", "val"} else prefix
        raw, total = self.core._compute_step_loss(batch, prefix=prefix)  # noqa: SLF001
        stats = {}
        for key, value in raw.items():
            name = key.split("/", 1)[-1]
            stats[name] = (
                value.detach()
                if torch.is_tensor(value)
                else torch.as_tensor(float(value))
            )
        stats["loss"] = total.detach()
        return total, stats

    def predict_and_target(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure-generative stub: the loss is both prediction and target."""
        loss, _ = self.forward(batch)
        loss = loss.detach().reshape(1)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict:  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def sample(  # noqa: PLR0913
        self,
        batch_size: int | None = None,
        nodesxsample: torch.Tensor | None = None,
        num_steps: int | None = None,
        mode: str | None = None,  # noqa: ARG002 - sampling_mode is DDPM-specific
        n_frames: int = 0,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(one_hot, charges, coords, node_mask)`` for ``batch_size`` molecules.

        ``nodesxsample`` is a **fragment** count per molecule, not an atom count
        (see ``SyncogenSizePrior``). ``charges`` carries signed formal charges.

        A sampled graph can come back with a reaction the assembler rejects, so a
        round is retried up to ``max_sample_attempts`` times rather than emitting
        an empty ``.xyz`` for the failed rows. All attempts exhausted -> raise, so
        the generation loop counts the batch as failed instead of writing
        nothing and exiting 0.
        """
        if n_frames:
            msg = (
                "SynCoGen cannot produce denoising trajectories: "
                "restore_model_and_sample returns only the endpoint and the sampler "
                "has no frame-saving hook, so there is nothing to write per frame. "
                "Set interference.n_frames: 0."
            )
            raise ValueError(msg)

        self._place_on_accelerator()

        if nodesxsample is not None:
            nodesxsample = torch.as_tensor(nodesxsample).reshape(-1).long()
            batch_size = int(nodesxsample.numel())
        batch_size = int(batch_size or self.size_prior.eval_batch_size)
        steps = int(num_steps or self.T)

        collected: list[dict] = []
        previous_forced = self.size_prior.forced_n_nodes
        previous_eval_bs = self.size_prior.eval_batch_size
        self.size_prior.forced_n_nodes = nodesxsample
        self.size_prior.eval_batch_size = batch_size
        try:
            for attempt in range(self.max_sample_attempts):
                cond = self._pharmacophore_cond(batch_size)
                graphs, coords = self.core.restore_model_and_sample(
                    num_steps=steps, cond=cond
                )
                collected.extend(self._assemble(graphs, coords))
                if len(collected) >= batch_size:
                    break
                logger.info(
                    "SynCoGen: %d/%d assemblable molecules after attempt %d/%d",
                    len(collected),
                    batch_size,
                    attempt + 1,
                    self.max_sample_attempts,
                )
        finally:
            self.size_prior.forced_n_nodes = previous_forced
            self.size_prior.eval_batch_size = previous_eval_bs

        if len(collected) < batch_size:
            msg = (
                f"SynCoGen produced {len(collected)}/{batch_size} assemblable "
                f"molecules in {self.max_sample_attempts} attempts. Either the "
                "weights are untrained (a smoke checkpoint samples mostly invalid "
                "reaction graphs) or tasks.max_sample_attempts is too low."
            )
            raise RuntimeError(msg)

        collected = collected[:batch_size]
        if self.sdf_output_path:
            self._append_sdf(collected)
        return self._to_pointcloud(collected)

    # -- sample-time adapters ------------------------------------------------

    def _pharmacophore_cond(self, batch_size: int):
        """``(types_onehot, pos, mask)`` from the reference ligand, or ``None``.

        Constant for a whole run, which is exactly why ``reference_ligand`` is a
        ``generation_time_key`` and not a batch-varying platform seam: the
        generation loop genuinely does nothing different for the conditioned run.
        """
        if not self.reference_ligand:
            return None
        from rdkit import Chem

        from MolecularDiffusion.modules.models.syncogen.utils.rdkit import (
            mol_to_pharm_cond,
        )

        if self._ref_mol is None:
            mol = Chem.MolFromMolFile(self.reference_ligand)
            if mol is None:
                msg = (
                    f"Could not read reference ligand: {self.reference_ligand}"
                )
                raise ValueError(msg)
            self._ref_mol = mol

        # The tuple's leading dim is the batch, and mol_to_pharm_cond redraws its
        # random n_subset per batch element, so it is rebuilt per call rather than
        # cached. Cheap: one RDKit feature pass over a single small ligand.
        types, pos, mask = mol_to_pharm_cond(
            Chem.Mol(self._ref_mol),
            batch_size=batch_size,
            n_subset=self.pharm_subset,
            center=True,
            normalize=True,
        )
        device = self.device
        return types.to(device), pos.to(device), mask.to(device)

    def _assemble(self, graphs, coords) -> list[dict]:
        """Sampled ``(BBRxnGraph, Coordinates)`` -> per-molecule RDKit mols.

        Mirrors upstream ``sample.py`` exactly: skip graphs still carrying masked
        nodes or reactions (``build_rdkit`` asserts on them), assemble, then
        assign the masked coordinate rows positionally.
        """
        from MolecularDiffusion.modules.models.syncogen.utils.rdkit import (
            set_mol_coordinates,
        )

        out: list[dict] = []
        for i in range(graphs.batch_size):
            graph_i = graphs[i]
            n = int(graph_i.lengths.item())
            if not (
                graph_i.unmasked_bbs[:n].all()
                and graph_i.unmasked_rxns[:n, :n].all()
            ):
                continue
            try:
                mol = graph_i.build_rdkit(return_smiles=False)
            except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
                logger.debug("SynCoGen: build_rdkit failed: %s", exc)
                continue
            if mol is None:
                continue

            atom_coords = coords[i].atom_coords.reshape(-1, 3)
            atom_mask = graph_i.ground_truth_atom_mask.reshape(-1).bool()
            valid = atom_coords[: atom_mask.shape[0], :][atom_mask]
            if valid.shape[0] != mol.GetNumAtoms():
                logger.debug(
                    "SynCoGen: %d valid coordinate rows vs %d assembled atoms",
                    valid.shape[0],
                    mol.GetNumAtoms(),
                )
                continue
            try:
                # THIS is what fixes the atom ordering for everything downstream.
                mol = set_mol_coordinates(mol, valid.cpu())
            except Exception as exc:  # noqa: BLE001
                logger.debug("SynCoGen: set_mol_coordinates failed: %s", exc)
                continue
            out.append({"mol": mol, "coords": valid.detach().cpu()})
        return out

    @staticmethod
    def _bond_source(mol):
        """The molecule the dense bond matrix is read from: a Kekule copy.

        DELIBERATE, AND MEASURED. The bond matrix exists to be re-read by
        ``GenerativeFactory._write_molecule_sdf``, which rebuilds through
        ``data/component/graph3d_dataset.py::build_rdkit_mol`` -- and that builder
        creates bare ``Chem.Atom(z)`` with no aromatic flag, so a matrix carrying
        class 4 (AROMATIC) hands it a molecule it often cannot kekulize. Measured
        on 30 molecules from the released unconditional weights: the aromatic
        projection rebuilds **19/30**, the Kekule projection **30/30**.

        Nothing is lost. ``build_rdkit_mol`` calls ``Chem.SanitizeMol``, which
        re-perceives aromaticity from the Kekule structure -- and all 30 rebuilt
        molecules are identical to upstream's own after
        ``RemoveStereochemistry`` (the residual stereo difference is
        ``build_rdkit_mol``'s ``AssignStereochemistryFrom3D`` reading the
        *generated geometry*, where upstream carries the building block's declared
        chirality; that is a property of the platform's rebuild, not of this
        projection).

        The combined ``sdf_output_path`` sidecar is written from the ORIGINAL
        aromatic molecule, so upstream's own perception survives there verbatim.

        A molecule RDKit refuses to kekulize falls back to the aromatic classes
        rather than being dropped.
        """
        from rdkit import Chem

        kekule = Chem.Mol(mol)
        try:
            Chem.Kekulize(kekule, clearAromaticFlags=True)
        except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
            logger.debug(
                "SynCoGen: kekulization failed, keeping class 4: %s", exc
            )
            return mol
        return kekule

    def _to_pointcloud(
        self, items: list[dict]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assembled molecules -> the platform's generation tensors + bond matrix."""
        from MolecularDiffusion.modules.models.syncogen.constants.constants import (
            MAX_ATOMS_PER_BB,
        )

        bond_class = _bond_class_map()
        index_of = {symbol: i for i, symbol in enumerate(self.atom_vocab)}
        batch = len(items)
        n_slots = self.max_bbs * MAX_ATOMS_PER_BB

        one_hot = torch.zeros(batch, n_slots, len(self.atom_vocab))
        charges = torch.zeros(batch, n_slots, dtype=torch.long)
        coords = torch.zeros(batch, n_slots, 3)
        node_mask = torch.zeros(batch, n_slots, dtype=torch.long)
        bonds = torch.zeros(batch, n_slots, n_slots, dtype=torch.long)

        for j, item in enumerate(items):
            mol = item["mol"]
            n_atoms = mol.GetNumAtoms()
            for a, atom in enumerate(mol.GetAtoms()):
                symbol = atom.GetSymbol()
                if symbol not in index_of:
                    msg = (
                        f"assembled molecule contains {symbol!r}, which is outside "
                        f"atom_vocab {self.atom_vocab}. The vocabulary's building "
                        "blocks should not be able to produce it."
                    )
                    raise ValueError(msg)
                one_hot[j, a, index_of[symbol]] = 1.0
                # Signed formal charge -- see this module's docstring.
                charges[j, a] = atom.GetFormalCharge()
            coords[j, :n_atoms] = item["coords"]
            node_mask[j, :n_atoms] = 1
            # Kekule copy -- atom order and formal charges are untouched by
            # kekulization, so the indices below still line up with the rows above.
            for bond in self._bond_source(mol).GetBonds():
                cls = bond_class.get(bond.GetBondType())
                if cls is None:
                    msg = (
                        f"unmapped RDKit bond type {bond.GetBondType()!r}; the "
                        "canonical vocabulary is 0=none 1=SINGLE 2=DOUBLE "
                        "3=TRIPLE 4=AROMATIC"
                    )
                    raise ValueError(msg)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bonds[j, u, v] = cls
                bonds[j, v, u] = cls

        self.last_bond_types = bonds
        return one_hot, charges, coords, node_mask

    def _append_sdf(self, items: list[dict]) -> None:
        """Append the *actual* assembled molecules to the combined sidecar.

        Deliberately written from ``build_rdkit``'s own output rather than from a
        rebuild, so upstream's aromaticity perception and sanitization survive
        verbatim. The per-molecule ``molecule_XXXX.sdf`` files that
        ``interference.sdf_per_molecule`` writes go through the platform's
        ``build_rdkit_mol`` instead -- the two agreeing on every molecule is a
        free consistency check on the bond projection.
        """
        from rdkit import Chem

        parent = os.path.dirname(os.path.abspath(self.sdf_output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.sdf_output_path, "a") as handle:  # noqa: PTH123
            writer = Chem.SDWriter(handle)
            for item in items:
                try:
                    writer.write(item["mol"])
                except Exception as exc:  # noqa: BLE001 - chemistry, not a bug
                    logger.warning(
                        "SynCoGen: skipping unwritable molecule: %s", exc
                    )
            writer.close()
