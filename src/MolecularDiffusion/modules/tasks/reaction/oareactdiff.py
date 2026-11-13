"""OA-ReactDiff: transition-state generation given a reactant and a product.

Duan, Du, Jia & Kulik, *Accurate transition state generation with an
object-aware equivariant elementary reaction diffusion model*, Nature
Computational Science 3, 1045-1055 (2023). Ported from
https://github.com/chenruduan/OAReactDiff at commit 543aaa8 (MIT).

Three pieces live here:

:class:`OAReactDiffTask`
    The platform task contract around ``EnVariationalDiffusion``. Its
    ``forward`` reproduces upstream's ``DDPMModule.compute_loss``
    (``trainer/pl_trainer.py:208-282``) -- the diffusion module returns raw
    loss *terms*, and the ``pos_only`` denominators plus the per-object
    ``scales`` weighting are applied here, where upstream applies them.

:class:`ModelTaskFactory`
    Hydra entry point for ``configs/tasks/diffusion_oareactdiff.yaml``.

:class:`OAReactDiffTSGenerator`
    A thin :class:`~MolecularDiffusion.modules.tasks.ts_generator.
    TSGenerator` subclass. **It writes no ``run()`` loop and no reaction
    plumbing.** "Load one fixed structural context, tile it to a batch,
    sample the unknown part into it, write .xyz" is the shared pocket loop
    (``pocket_generator.py``); reading one
    :class:`~MolecularDiffusion.data.component.reaction_data.Reaction`,
    sizing the transition state from it and writing the reference structures
    beside the samples is the shared TS layer on top of it
    (``ts_generator.py``). What is left here is OA-ReactDiff's own: which
    corpus and filters, which collate, and the RePaint knobs.

Scope of this integration
-------------------------

**Transition state given a reactant and a product, and nothing else.** Not
in scope, each for a stated reason: unconditional whole-reaction generation
(``pos_only=True`` means the network cannot invent atom identities, and no
generation seam carries a chemical formula); the confidence / recommender
model that ranks 5 samples per reaction (no checkpoint for it ships);
React-OT (a separate repository and paper); energy-difference conditioning
(``condition_nf=1`` was fed constant zeros, so no trained conditional
variant exists); validation-time inpainting RMSD (deep-copies the model and
runs a 150-step inpaint per validation epoch); and trajectory frames (the
frame-saving lines are commented out inside upstream's own ``inpaint``).

Two facts from the released checkpoint that constrain everything
----------------------------------------------------------------

* It is ``pos_only=True``: coordinates only. Atom identities are supplied by
  the input reaction and copied straight through, never generated.
* ``ddpm.schedule.gamma_module.gamma`` is a non-learned ``nn.Parameter`` of
  shape ``(5001,)`` living *in the state dict*, so training ``timesteps`` is
  pinned at 5000 on load. Sampling does not use it: :meth:`OAReactDiffTask.
  sample` builds a **fresh** ``polynomial_2`` schedule at ``num_steps`` (250
  by default) and swaps it in for the duration of the call, which is exactly
  what upstream's ``evaluate/utils.py:14-32 set_new_schedule`` does.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from MolecularDiffusion.data.integrations.oareactdiff_data import (
    N_ELEMENT,
    OAREACTDIFF_ATOM_VOCAB,
    TS_INDEX,
    OAReactDiffTS1xDataset,
    oareactdiff_collate,
)
from MolecularDiffusion.data.datasets.reaction_data import Reaction
from MolecularDiffusion.modules.models.oareactdiff import (
    FEATURE_MAPPING,
    DiffSchedule,
    EGNNDynamics,
    EnVariationalDiffusion,
    LEFTNet,
    Normalizer,
    PredefinedNoiseSchedule,
)
from MolecularDiffusion.modules.tasks.reaction.ts_generator import (
    ReactionSource,
    TSGenerator,
)

#: Objects held at their true geometry while the transition state is
#: generated: 0 = reactant, 2 = product (``FRAGMENT_ORDER``).
FRAG_FIXED = [0, 2]


def _pad_ts(
    flat: Tensor, mask: Tensor, n_samples: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flat ``(sum n, 9)`` rows -> the platform's padded 4-tuple.

    ``inpaint`` returns one long tensor per object plus a per-row sample
    index, while ``PocketGenerator.run`` and ``save_xyz_file`` want
    ``(B, N, .)``. Column layout is ``FEATURE_MAPPING``:
    ``[pos (3) | one_hot (5) | charge (1)]``.

    ``N`` is constant across a ``PocketGenerator`` batch (every tile is the
    same reaction) and ``node_mask`` is therefore all ones -- but both are
    computed from the actual counts, so a mixed batch would still be right.

    Args:
        flat: ``(sum n, 9)`` sampled rows for one object.
        mask: ``(sum n,)`` sample index per row, ascending and grouped.
        n_samples: batch size.

    Returns:
        ``(one_hot, charges, coords, node_mask)``.
    """
    device = flat.device
    counts = torch.bincount(mask, minlength=n_samples)
    n_max = int(counts.max())
    # Row position within its own sample, without a Python loop.
    offsets = torch.cumsum(counts, dim=0) - counts
    within = torch.arange(mask.numel(), device=device) - offsets[mask]

    coords = torch.zeros(n_samples, n_max, 3, device=device)
    one_hot = torch.zeros(n_samples, n_max, N_ELEMENT, device=device)
    charges = torch.zeros(n_samples, n_max, device=device)
    node_mask = torch.zeros(n_samples, n_max, device=device)

    coords[mask, within] = flat[:, :3].float()
    one_hot[mask, within] = flat[:, 3 : 3 + N_ELEMENT].float()
    charges[mask, within] = flat[:, -1].float()
    node_mask[mask, within] = 1.0
    return one_hot, charges, coords, node_mask


class OAReactDiffTask(nn.Module):
    """Task contract around :class:`EnVariationalDiffusion`."""

    def __init__(
        self,
        ddpm: EnVariationalDiffusion,
        scales: Sequence[float] = (1.0, 2.0, 1.0),
        loss_type: str = "l2",
        pos_only: bool = True,
        precision: float = 1.0e-5,
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.ddpm = ddpm
        self.scales = [float(s) for s in scales]
        self.loss_type = loss_type
        self.pos_only = pos_only
        self.precision = precision
        self.n_fragments = len(ddpm.fragment_names)
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(OAREACTDIFF_ATOM_VOCAB)
        )
        #: Read by ``cli/generate.py`` when reconciling ``diffusion_steps``.
        #: 5000, the training schedule -- never the sampling step count.
        self.T = ddpm.T
        #: There is no size prior to build: the transition state has exactly
        #: as many atoms as the input reaction. Plain attributes so
        #: ``cli/generate.py`` and ``engine_lightning`` can stamp on them
        #: harmlessly.
        self.node_dist_model: Any = None
        self.prop_dist_model: Any = None
        self.split = "train"

    # -- contract properties -------------------------------------------- #
    @property
    def model(self) -> "OAReactDiffTask":
        """There is no separate inner module the generation code needs."""
        return self

    @property
    def device(self) -> torch.device:
        """Device of the first parameter."""
        return next(self.parameters()).device

    # -- training -------------------------------------------------------- #
    def forward(
        self, batch: Tuple[List[Dict[str, Tensor]], Tensor]
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """One training/validation step.

        Reproduces ``DDPMModule.compute_loss`` (``pl_trainer.py:208-282``).
        The ``l2`` branch is taken only while training; validation always
        takes the VLB branch, which is upstream's behaviour and is why the
        reported validation loss is on a different scale from the training
        loss.

        Args:
            batch: ``(representations, conditions)`` from
                :func:`~MolecularDiffusion.data.component.oareactdiff_data.
                oareactdiff_collate`.

        Returns:
            ``(loss, stats)``; ``stats`` carries upstream's logged keys.
        """
        representations, conditions = batch
        loss_terms = self.ddpm.forward(representations, conditions)

        if self.pos_only:
            denoms = [
                self.ddpm.pos_dim * representations[ii]["size"]
                for ii in range(self.n_fragments)
            ]
        else:
            denoms = [
                (self.ddpm.pos_dim + self.ddpm.node_nfs[ii])
                * representations[ii]["size"]
                for ii in range(self.n_fragments)
            ]
        error_t_normalized = [
            loss_terms["error_t"][ii] / denoms[ii] * self.scales[ii]
            for ii in range(self.n_fragments)
        ]

        if self.loss_type == "l2" and self.training:
            loss_t = torch.stack(error_t_normalized, dim=0).sum(dim=0)
            loss_0_x = torch.stack(
                [
                    loss_terms["loss_0_x"][ii]
                    * self.scales[ii]
                    / (self.ddpm.pos_dim * representations[ii]["size"])
                    for ii in range(self.n_fragments)
                ],
                dim=0,
            ).sum(dim=0)
            loss_0 = (
                loss_0_x
                + torch.stack(loss_terms["loss_0_cat"], dim=0).sum(dim=0)
                + torch.stack(loss_terms["loss_0_charge"], dim=0).sum(dim=0)
            )
        else:
            # VLB objective / evaluation. SNR_weight is negative.
            error_t = [
                -self.ddpm.T * 0.5 * loss_terms["SNR_weight"] * _error_t
                for _error_t in loss_terms["error_t"]
            ]
            loss_t = torch.stack(error_t, dim=0).sum(dim=0)
            loss_0 = (
                torch.stack(loss_terms["loss_0_x"], dim=0).sum(dim=0)
                + torch.stack(loss_terms["loss_0_cat"], dim=0).sum(dim=0)
                + torch.stack(loss_terms["loss_0_charge"], dim=0).sum(dim=0)
                + loss_terms["neg_log_constants"]
            )

        nll = loss_t + loss_0 + loss_terms["kl_prior"]
        if not (self.loss_type == "l2" and self.training):
            # Correct for normalisation on x, then turn the conditional nll
            # into a joint one.
            nll = nll - loss_terms["delta_log_px"] - loss_terms["log_pN"]

        stats: Dict[str, Any] = {"loss": nll.mean(0)}
        for ii in range(self.n_fragments):
            stats[f"error_t_{ii}"] = error_t_normalized[ii].mean().item() / (
                self.scales[ii] + 1e-4
            )
            stats[f"unorm_error_t_{ii}"] = (
                loss_terms["error_t"][ii].mean().item()
            )
        return nll.mean(0), stats

    def predict_and_target(
        self, batch: Tuple[List[Dict[str, Tensor]], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Pure-generative stub: the loss is the prediction, target is zero."""
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:  # noqa: ARG002
        """Validation metric.

        Upstream's ``eval_inplaint_batch`` RMSD is deliberately not here: it
        deep-copies the whole model and runs a 150-step inpaint every
        ``eval_epochs``, which is far too heavy for a training loop.
        """
        return {"val_loss": pred.mean()}

    # -- generation ------------------------------------------------------ #
    @contextlib.contextmanager
    def _sampling_schedule(
        self, timesteps: int, noise_schedule: str
    ) -> Iterator[None]:
        """Swap in a fresh sampling schedule, then put the trained one back.

        Upstream mutates the module permanently (``set_new_schedule``); this
        restores it, so a task can be sampled from and then trained or saved
        without carrying a 250-step schedule where its 5000-step one was.
        """
        gamma_module = PredefinedNoiseSchedule(
            noise_schedule=noise_schedule,
            timesteps=timesteps,
            precision=self.precision,
        )
        schedule = DiffSchedule(
            gamma_module=gamma_module, norm_values=self.ddpm.norm_values
        ).to(self.device)
        old_schedule, old_t = self.ddpm.schedule, self.ddpm.T
        self.ddpm.schedule, self.ddpm.T = schedule, timesteps
        try:
            yield
        finally:
            self.ddpm.schedule, self.ddpm.T = old_schedule, old_t

    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,  # noqa: ARG002 - from the batch
        nodesxsample: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, Any]] = None,
        resamplings: int = 5,
        jump_length: int = 5,
        noise_schedule: str = "polynomial_2",
        **kwargs: Any,  # noqa: ARG002 - swallows mode/n_frames
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate transition states for the reactions in ``batch``.

        **The signature deviates from Section 2.1 the same way every
        pocket-conditioned task's does**: the conditioning structure arrives
        as ``batch``, because there is no channel for "which reaction" in
        ``sample(batch_size, nodesxsample, ...)``.

        Args:
            batch_size: ignored; taken from ``batch``.
            nodesxsample: accepted and *checked*, not used to choose. The
                transition state has exactly as many atoms as the reaction,
                so a value disagreeing with it is a caller bug worth
                surfacing rather than silently overriding.
            num_steps: reverse-process steps. ``None`` falls back to
                ``self.T`` (5000) -- both shipped generate configs set 250,
                which is upstream's evaluation setting.
            batch: ``{"representations": [...], "conditions": tensor}``.
            resamplings: RePaint re-noise rounds per jump.
            jump_length: steps to jump back on each resampling.
            noise_schedule: the *sampling* schedule to build fresh.

        Returns:
            ``(one_hot, charges, coords, node_mask)`` for the transition
            state only, padded to ``(B, N, .)``.

        Raises:
            ValueError: if ``batch`` is missing, or ``nodesxsample``
                disagrees with the reaction's atom count.
        """
        if batch is None:
            raise ValueError(
                "OAReactDiffTask.sample needs `batch`: a reactant and a "
                "product are the conditioning, and there is no "
                "unconditional mode in this integration."
            )
        device = self.device
        representations = [
            {k: v.to(device) for k, v in rep.items()}
            for rep in batch["representations"]
        ]
        conditions = batch["conditions"].to(device)

        n_samples = int(representations[0]["size"].size(0))
        fragments_nodes = [rep["size"] for rep in representations]
        if nodesxsample is not None:
            want = torch.as_tensor(nodesxsample).view(-1).to(device).long()
            got = fragments_nodes[TS_INDEX].long()
            if not torch.equal(want, got):
                raise ValueError(
                    f"nodesxsample={want.tolist()} disagrees with the "
                    f"reaction's atom count {got.tolist()}. A transition "
                    "state has the same atoms as its reactant and product; "
                    "it cannot be resized."
                )

        xh_fixed = [
            torch.cat([rep[key] for key in FEATURE_MAPPING], dim=1)
            for rep in representations
        ]
        steps = int(num_steps) if num_steps else int(self.T)

        with self._sampling_schedule(steps, noise_schedule):
            out_samples, fragments_masks = self.ddpm.inpaint(
                n_samples=n_samples,
                fragments_nodes=fragments_nodes,
                conditions=conditions,
                return_frames=1,
                resamplings=resamplings,
                jump_length=jump_length,
                timesteps=steps,
                xh_fixed=xh_fixed,
                frag_fixed=FRAG_FIXED,
            )
        return _pad_ts(
            out_samples[0][TS_INDEX], fragments_masks[TS_INDEX], n_samples
        )


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_oareactdiff.yaml``.

    No ``train_set`` parameter: the only construction-time statistic
    upstream has a slot for is ``size_histogram``, which is passed ``None``
    and never read (``en_diffusion.py:31,42``; ``pl_trainer.py:117``). The
    Section 2.5 seam is therefore not used and ``cli/train.py`` is untouched.
    """

    #: Belt-and-braces. ``cli/generate.py:173-202`` reads this off the
    #: factory so a generation-time key in the *tasks* config survives being
    #: rebuilt from a checkpoint. ``reaction_pkl`` actually lives under
    #: ``interference:``, which is instantiated unfiltered, so today this is
    #: a no-op -- it matters only if someone moves the key.
    generation_time_keys = ("reaction_pkl",)

    def __init__(
        self,
        task_type: str = "diffusion_oareactdiff",
        model_config: Optional[Dict[str, Any]] = None,
        node_nfs: Sequence[int] = (9, 9, 9),
        edge_nf: int = 0,
        condition_nf: int = 1,
        fragment_names: Sequence[str] = ("R", "TS", "P"),
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        norm_values: Sequence[float] = (1.0, 1.0, 1.0),
        norm_biases: Sequence[float] = (0.0, 0.0, 0.0),
        noise_schedule: str = "cosine",
        timesteps: int = 5000,
        precision: float = 1.0e-5,
        loss_type: str = "l2",
        pos_only: bool = True,
        scales: Sequence[float] = (1.0, 2.0, 1.0),
        fixed_idx: Optional[List[int]] = None,
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by train.py
    ) -> None:
        if model_config is None:
            raise ValueError(
                "tasks.model_config is required: it is LEFTNet's own "
                "architecture block and the released checkpoint will not "
                "load against anything else."
            )
        self.task_type = task_type
        # dict(), not the DictConfig: BaseDynamics mutates model_config
        # in place (`_base.py:47-50`) and LEFTNet is called with **it.
        self.model_config = dict(model_config)
        self.node_nfs = [int(n) for n in node_nfs]
        self.edge_nf = edge_nf
        self.condition_nf = condition_nf
        self.fragment_names = [str(n) for n in fragment_names]
        self.pos_dim = pos_dim
        self.update_pocket_coords = update_pocket_coords
        self.condition_time = condition_time
        self.edge_cutoff = edge_cutoff
        self.norm_values = tuple(float(v) for v in norm_values)
        self.norm_biases = tuple(float(v) for v in norm_biases)
        self.noise_schedule = noise_schedule
        self.timesteps = int(timesteps)
        self.precision = float(precision)
        self.loss_type = loss_type
        self.pos_only = bool(pos_only)
        self.scales = [float(s) for s in scales]
        self.fixed_idx = list(fixed_idx) if fixed_idx else None
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(OAREACTDIFF_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.task: Optional[OAReactDiffTask] = None

    def build(self) -> OAReactDiffTask:
        """Assemble dynamics + schedule + normaliser into the task."""
        dynamics = EGNNDynamics(
            model_config=self.model_config,
            fragment_names=self.fragment_names,
            node_nfs=self.node_nfs,
            edge_nf=self.edge_nf,
            condition_nf=self.condition_nf,
            pos_dim=self.pos_dim,
            update_pocket_coords=self.update_pocket_coords,
            condition_time=self.condition_time,
            edge_cutoff=self.edge_cutoff,
            model=LEFTNet,
        )
        schedule = DiffSchedule(
            gamma_module=PredefinedNoiseSchedule(
                noise_schedule=self.noise_schedule,
                timesteps=self.timesteps,
                precision=self.precision,
            ),
            norm_values=self.norm_values,
        )
        ddpm = EnVariationalDiffusion(
            dynamics=dynamics,
            schdule=schedule,  # upstream's spelling; keep it
            normalizer=Normalizer(
                norm_values=self.norm_values,
                norm_biases=self.norm_biases,
                pos_dim=self.pos_dim,
            ),
            size_histogram=None,
            loss_type=self.loss_type,
            pos_only=self.pos_only,
            fixed_idx=self.fixed_idx,
        )
        self.task = OAReactDiffTask(
            ddpm,
            scales=self.scales,
            loss_type=self.loss_type,
            pos_only=self.pos_only,
            precision=self.precision,
            atom_vocab=self.atom_vocab,
        )
        return self.task


class OAReactDiffTSGenerator(TSGenerator):
    """Transition-state generation behind ``interference/gen_oareactdiff_ts``.

    The shape of the request is identical to every other TS run -- *load one
    reaction, tile it to a batch, sample the transition state into it, write
    .xyz* -- so the loop is
    :meth:`~MolecularDiffusion.modules.tasks.pocket_generator.
    PocketGenerator.run`, the reaction plumbing is
    :class:`~MolecularDiffusion.modules.tasks.ts_generator.TSGenerator`, and
    what is left here is only what is OA-ReactDiff's: which corpus, which
    filters, which collate, and the RePaint knobs.

    The reaction comes from one row of an upstream Transition1x pickle, read
    with the released training filters (``single_frag_only``,
    ``use_by_ind``) but **without** ``swapping_react_prod``: a generator
    should not silently hand back the reverse reaction. With
    ``valid_addprop.pkl`` that makes ``reaction_index`` an index into the
    783 held-out single-fragment reactions.

    The output folder is self-describing: ``reactant.xyz``, ``product.xyz``
    and ``reference_ts.xyz`` are written once alongside the generated
    ``molecule_NNN.xyz`` files, so a sample can be judged against its own
    inputs and against the DFT transition state without going back to the
    pickle.

    The corpus key is ``reaction_pkl`` (declared as
    :attr:`~MolecularDiffusion.modules.tasks.ts_generator.TSGenerator.
    source_key`, not as a constructor parameter, so there is exactly one of
    it); the shared ``interference/gen_ts.yaml`` reaches the same slot as
    ``reaction_source``. Either works, both together raise.
    """

    tag = "oareactdiff"
    #: Upstream's scripts seed through ``seed_everything``, which seeds numpy.
    seed_numpy = True
    #: ``_accept`` is not overridden, so the loop always progresses.
    max_retries: Optional[int] = None
    #: This model's corpus really is a pickle, so its own config says so.
    #: The base accepts both this and the shared ``reaction_source``.
    source_key = "reaction_pkl"
    db_required_msg = (
        "interference.reaction_pkl (or reaction_source, on the shared "
        "gen_ts.yaml) is required: OA-ReactDiff has no unconditional mode "
        "in this integration -- it needs a reactant and a product. Point it "
        "at a Transition1x pickle, e.g. docs/model_integrations/oareactdiff/"
        "data/t1x_source_valid.pkl."
    )

    def __init__(
        self,
        task: Any,
        reaction_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = 250,
        resamplings: int = 5,
        jump_length: int = 5,
        noise_schedule: str = "polynomial_2",
        output_path: str = "generated_oareactdiff",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            task,
            reaction_index=reaction_index,
            num_generate=num_generate,
            batch_size=batch_size,
            num_steps=num_steps,
            output_path=output_path,
            seed=seed,
            device=device,
            **kwargs,
        )
        self.resamplings = resamplings
        self.jump_length = jump_length
        self.noise_schedule = noise_schedule

    # -- hooks ----------------------------------------------------------- #
    def _reactions(self) -> ReactionSource:
        """The held-out Transition1x rows, with the released filters.

        ``swapping_react_prod`` is off on purpose: a generator should not
        silently hand back the reverse reaction.
        """
        return OAReactDiffTS1xDataset(
            str(self.pocket_db),
            center=True,
            zero_charge=False,
            single_frag_only=True,
            swapping_react_prod=False,
            use_by_ind=True,
        )

    def _collate(self, reaction: Reaction, n: int) -> Dict[str, Any]:
        """Tile the reaction ``n`` times, with fresh scatter indices.

        The tiling *is* the collate: reusing it rather than hand-rolling one
        guarantees the sampler sees exactly the layout training saw --
        including the ``int64`` per-object scatter indices ``torch_scatter``
        needs, which the collate rebuilds from scratch for each batch.
        """
        representations, conditions = oareactdiff_collate(
            [reaction] * n, zero_charge=False
        )
        return {"representations": representations, "conditions": conditions}

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:  # noqa: ARG002
        return {
            "resamplings": self.resamplings,
            "jump_length": self.jump_length,
            "noise_schedule": self.noise_schedule,
        }

    def _label(self, reaction: Reaction) -> str:
        """The reaction id, both SMILES, and the energy difference."""
        meta = reaction.meta
        return (
            f"{meta['rxn']} ({'.'.join(meta['smi_reactant'])} >> "
            f"{'.'.join(meta['smi_product'])}), dE = {meta['ediff']:.2f}"
        )

    def _settings_note(self) -> str:
        """RePaint's knobs are the quality dial, so they go in the header."""
        return (
            f" (resamplings={self.resamplings}, "
            f"jump_length={self.jump_length})"
        )
