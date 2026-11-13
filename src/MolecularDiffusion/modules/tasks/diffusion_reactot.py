"""React-OT: transition states in ten deterministic network evaluations.

Duan, C.; Liu, G.-H.; Du, Y.; ...; Kulik, H. J., *Optimal transport for
generating transition states in chemical reactions*, Nature Machine
Intelligence (2025), doi:10.1038/s42256-025-01010-0. Preprint, under a
**different title**: arXiv:2404.13430, *React-OT: Optimal Transport for
Generating Transition State in Chemical Reactions*. Ported from
https://github.com/deepprinciple/react-ot at commit 6dfccd0.

Three pieces live here:

:class:`ReactOTTask`
    The platform task contract around :class:`~MolecularDiffusion.modules.
    models.reactot.en_sb.EnSB`. Its ``forward`` reproduces upstream's
    ``SBModule.compute_loss`` (``pl_trainer.py:993-1004``) -- which is a
    two-line unpacking, because the bridge returns a finished scalar. There
    is no ELBO to reduce, no ``loss_0``, no ``kl_prior`` and no per-object
    ``scales`` weighting, which is why this is a fraction of the size of
    :class:`~MolecularDiffusion.modules.tasks.diffusion_oareactdiff.
    OAReactDiffTask`'s.

:class:`ModelTaskFactory`
    Hydra entry point for ``configs/tasks/diffusion_reactot.yaml``.

:class:`ReactOTTSGenerator`
    A thin :class:`~MolecularDiffusion.modules.tasks.ts_generator.
    TSGenerator` subclass. **No ``run()`` loop, and no ``PocketGenerator``
    hook overridden.** The loop is the shared pocket loop, the reaction
    plumbing is the shared TS layer, and what is left here is only React-OT's
    own: which corpus and filters, which collate, and the solver knobs.

Relationship to OA-ReactDiff
----------------------------

Same authors, same backbone, same corpus, same reactant/TS/product triple,
same batch layout -- so this integration **imports** rather than duplicates:
``LEFTNet`` / ``EGNNDynamics`` / ``Normalizer`` / the graph helpers from
``modules/models/oareactdiff/``, ``OAReactDiffTS1xDataset`` /
``oareactdiff_collate`` / ``OAReactDiffDataModule`` from
``data/component/oareactdiff_data.py``, and :func:`~MolecularDiffusion.
modules.tasks.diffusion_oareactdiff._pad_ts` from its task module. None of
those files is edited.

What differs is the *process*: OA-ReactDiff runs a stochastic reverse
diffusion with RePaint re-noising and ranks five samples afterwards;
React-OT integrates a Schrodinger bridge from the reactant/product midpoint
to the transition state, deterministically, in ``nfe`` network evaluations
(10 in the README's published command). Run it twice and you get the same
structure -- which is why ``gen_reactot_ts.yaml`` defaults ``num_generate:
1``.

The transition-state leak this class closes on purpose
------------------------------------------------------

``EnSB.sample`` builds its node features from ``representations[1]``, whose
position columns are the **reference** transition state when the batch comes
from a corpus. Upstream gets away with it because ``net_out_fn`` overwrites
those columns before every network call -- i.e. it does not leak today, but
only by accident, and it means upstream's sampler cannot be pointed at an
R/P pair with no known TS at all. :meth:`ReactOTTask.sample` substitutes the
midpoint into that slot explicitly, exactly as upstream's own R/P-only
deployment path does (``run_model.py:122``), so leak-freedom is structural.

Verified: on CPU, replacing ``representations[1]["pos"]`` with random noise
before sampling changes the output by **exactly 0.0 A**, and two identical
calls agree to **exactly 0.0 A**.

**On CUDA both of those become ~5e-3 A, and that is not a leak.** It is
``torch_scatter``'s atomic-add nondeterminism (``remove_mean_batch`` and
LEFTNet's aggregations) amplified through ten ODE steps: the *repeat-run*
spread is the same size as the corrupted-slot spread, which is exactly what
noise looks like and exactly what information would not. Run the leak check
on CPU, where the answer is a clean zero. This is pre-existing platform
behaviour shared with OA-ReactDiff, not something this integration
introduced.

Scope of this integration
-------------------------

**Transition state given a reactant and a product, and nothing else.** Not
in scope, each for a stated reason: the ``ei`` exponential-integrator solver
(unpublished research variant needing a Monte-Carlo quadrature per
coefficient); the other three ``mapping`` modes and five other
``mapping_initial`` values (no released weights, no data path); ``ts_guess``
conditioning (unreachable -- the checkpoint records ``ts_guess=None``); PBC /
zeolite support (a different corpus, and the part of the backbone the shared
vendored LEFTNet omits); validation-time sampling RMSD (runs a full sampler
inside the validation loop); the size-aware ``DynamicBatchSampler``; and the
GFN2-xTB-pretrained variant the NMI paper reports (no second checkpoint
ships, and its pretraining corpus is not in the Zenodo record).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from MolecularDiffusion.data.integrations.oareactdiff_data import (
    OAREACTDIFF_ATOM_VOCAB,
    TS_INDEX,
    OAReactDiffTS1xDataset,
    oareactdiff_collate,
)
from MolecularDiffusion.data.datasets.reaction_data import Reaction
from MolecularDiffusion.modules.models.oareactdiff import (
    FEATURE_MAPPING,
    EGNNDynamics,
    LEFTNet,
    Normalizer,
)
from MolecularDiffusion.modules.models.oareactdiff.graph_tools import (
    remove_mean_batch,
)
from MolecularDiffusion.modules.models.reactot import EnSB, SBSchedule
from MolecularDiffusion.modules.tasks.diffusion_oareactdiff import _pad_ts
from MolecularDiffusion.modules.tasks.ts_generator import (
    ReactionSource,
    TSGenerator,
)

#: Objects held at their true geometry while the transition state is
#: generated: 0 = reactant, 2 = product. **Inert** in this model -- upstream
#: stores ``fixed_idx`` and never reads it, because freezing R and P is
#: structural (only object 1's position columns are ever overwritten).
#: Recorded for checkpoint fidelity.
FRAG_FIXED = [0, 2]


class ReactOTTask(nn.Module):
    """Task contract around :class:`EnSB`."""

    def __init__(
        self,
        ddpm: EnSB,
        nfe: int = 10,
        ot_ode: bool = True,
        solver: str = "ode",
        method: str = "midpoint",
        atol: float = 1e-2,
        rtol: float = 1e-2,
        clip_denoise: bool = True,
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        """Bind the bridge to the platform's training/generation contract.

        Args:
            ddpm: the built bridge. Named ``ddpm`` for the same reason
                upstream does (``SBModule.ddpm``): it is what the released
                checkpoint's tensor prefix says, and renaming it would mean
                remapping 246 keys for nothing. React-OT is not a diffusion
                model.
            nfe: default network evaluations per sample. The interference
                config's ``num_steps`` overrides it.
            ot_ode: ``True`` => fully deterministic, the released setting.
            solver: ``"ode"`` (published default) or ``"ddpm"``.
            method: ODE method; only ``"midpoint"`` is implemented.
            atol: dead for a fixed-grid method; carried for fidelity.
            rtol: as ``atol``.
            clip_denoise: clamp each predicted ``x0`` into +/-10 A.
            atom_vocab: output vocabulary; defaults to ``[H, C, N, O, F]``.
        """
        super().__init__()
        self.ddpm = ddpm
        self.nfe = int(nfe)
        self.ot_ode = bool(ot_ode)
        self.solver = solver
        self.method = method
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.clip_denoise = bool(clip_denoise)
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(OAREACTDIFF_ATOM_VOCAB)
        )
        #: Read by ``cli/generate.py`` when reconciling ``diffusion_steps``.
        #: **A trap if it is ever used**: 3000 is the bridge grid, and
        #: running 3000 network evaluations is 300x the intended cost. The
        #: interference config must always set ``num_steps``.
        self.T = ddpm.T
        #: There is no size prior: a transition state has exactly the atoms
        #: of its reaction. Plain attributes so ``cli/generate.py`` and
        #: ``engine_lightning`` can stamp on them harmlessly.
        self.node_dist_model: Any = None
        self.prop_dist_model: Any = None
        self.split = "train"

    # -- contract properties --------------------------------------------- #
    @property
    def model(self) -> "ReactOTTask":
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

        This is the whole of upstream's ``SBModule.compute_loss``
        (``pl_trainer.py:993-1004``): unpack, call, report. ``loss`` is
        already a 0-dim scalar (``F.mse_loss`` reduces by default), so
        nothing is reduced again here.

        Args:
            batch: ``(representations, conditions)`` from
                :func:`~MolecularDiffusion.data.component.oareactdiff_data.
                oareactdiff_collate`.

        Returns:
            ``(loss, stats)``. ``scaled_err`` is reported because it is what
            upstream's checkpoint callback actually monitors
            (``val_ep_scaled_err``).
        """
        representations, conditions = batch
        loss_terms = self.ddpm.forward(
            representations, conditions, ot_ode=self.ot_ode
        )
        loss = loss_terms["loss"]
        return loss, {
            "loss": loss,
            "scaled_err": loss_terms["scaled_err"].detach(),
        }

    def predict_and_target(
        self, batch: Tuple[List[Dict[str, Tensor]], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Pure-generative stub: the loss is the prediction, target is zero.

        Args:
            batch: as :meth:`forward`.

        Returns:
            ``(pred, target)``, each ``(1,)``.
        """
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:  # noqa: ARG002
        """Validation metric.

        Upstream's validation-time sampling RMSD is deliberately not here:
        ``eval_sample_batch`` runs a full sampler inside the validation loop
        every epoch, and ``eval_rmsd`` sweeps the whole no-swap loader every
        ten (``pl_trainer.py:1087-1140``). Far too heavy for a training loop.

        Args:
            pred: what :meth:`predict_and_target` returned.
            target: ignored.

        Returns:
            ``{"val_loss": ...}``.
        """
        return {"val_loss": pred.mean()}

    # -- generation ------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,  # noqa: ARG002 - from the batch
        nodesxsample: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, Any]] = None,
        solver: Optional[str] = None,
        method: Optional[str] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        **kwargs: Any,  # noqa: ARG002 - swallows mode/n_frames
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate transition states for the reactions in ``batch``.

        **The signature deviates from Section 2.1 the same way every
        pocket-conditioned task's does**: the conditioning structure arrives
        as ``batch``, because ``sample(batch_size, nodesxsample, ...)`` has
        no channel for "which reaction".

        **``num_steps`` IS upstream's ``--nfe``** -- the number of network
        evaluations, not a fraction of the 3000-step bridge grid. There is
        deliberately no second key spelled ``nfe``; the family key carries
        it and :meth:`ReactOTTSGenerator._settings_note` prints it as
        ``nfe=`` so the run log speaks upstream's vocabulary.

        Args:
            batch_size: ignored; taken from ``batch``.
            nodesxsample: accepted and *checked*, not used to choose. A
                transition state has exactly as many atoms as its reaction,
                so a disagreeing value is a caller bug worth surfacing.
            num_steps: network evaluations. ``None`` => the task's ``nfe``.
            batch: ``{"representations": [...], "conditions": tensor}``.
            solver: ``"ode"`` / ``"ddpm"``; ``None`` => the task's default.
            method: ODE method; ``None`` => the task's default.
            atol: ``None`` => the task's default. Dead for ``midpoint``.
            rtol: as ``atol``.
            **kwargs: swallowed, as every other task's ``sample`` does.

        Returns:
            ``(one_hot, charges, coords, node_mask)`` for the transition
            state only, padded to ``(B, N, .)``.

        Raises:
            ValueError: if ``batch`` is missing, or ``nodesxsample``
                disagrees with the reaction's atom count.
        """
        if batch is None:
            raise ValueError(
                "ReactOTTask.sample needs `batch`: a reactant and a product "
                "are the conditioning, and there is no unconditional mode "
                "in this integration."
            )
        device = self.device
        representations = [
            {k: v.to(device) for k, v in rep.items()}
            for rep in batch["representations"]
        ]
        conditions = batch["conditions"].to(device)

        n_samples = int(representations[0]["size"].size(0))
        if nodesxsample is not None:
            want = torch.as_tensor(nodesxsample).view(-1).to(device).long()
            got = representations[TS_INDEX]["size"].long()
            if not torch.equal(want, got):
                raise ValueError(
                    f"nodesxsample={want.tolist()} disagrees with the "
                    f"reaction's atom count {got.tolist()}. A transition "
                    "state has the same atoms as its reactant and product; "
                    "it cannot be resized."
                )

        # THE MIDPOINT SUBSTITUTION. See the module docstring: object 1's
        # positions are the *reference* transition state when the batch
        # comes from a corpus, and although EnSB overwrites them before
        # every network call, relying on that is relying on an accident.
        # This is upstream's own R/P-only path (run_model.py:122), and it
        # is also what makes the sampler usable on a reaction whose
        # transition state is unknown. CoM-removed per sample so the slot
        # holds exactly the `x1` EnSB.sample_batch computes.
        representations[TS_INDEX]["pos"] = remove_mean_batch(
            (representations[0]["pos"] + representations[2]["pos"]) / 2,
            representations[TS_INDEX]["mask"],
        )

        xs, _pred_x0 = self.ddpm.sample(
            representations,
            conditions,
            clip_denoise=self.clip_denoise,
            nfe=int(num_steps) if num_steps else self.nfe,
            ot_ode=self.ot_ode,
            solver=solver if solver is not None else self.solver,
            method=method if method is not None else self.method,
            atol=atol if atol is not None else self.atol,
            rtol=rtol if rtol is not None else self.rtol,
        )
        # EnSB returns POSITIONS only, (sum n, n_logged, 3), on the CPU;
        # index 0 is the final structure (pl_trainer.py:1029). _pad_ts wants
        # the 9-wide FEATURE_MAPPING row, so the TS object's own one_hot and
        # charge columns -- which this model never generates -- go back on.
        ts = representations[TS_INDEX]
        flat = torch.cat(
            [
                xs[:, 0, :].to(device).float(),
                *(ts[key].float() for key in FEATURE_MAPPING[1:]),
            ],
            dim=1,
        )
        return _pad_ts(flat, ts["mask"], n_samples)


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_reactot.yaml``.

    No ``train_set`` parameter: the only construction-time statistic
    upstream has a slot for is ``size_histogram``, which is passed ``None``
    (``pl_trainer.py:813``) and never read (``en_sb.py:42,58``). The Section
    2.5 seam is therefore not used and ``cli/train.py`` is untouched.
    """

    #: Belt-and-braces, mirroring ``diffusion_oareactdiff``.
    #: ``cli/generate.py:173-202`` reads this off the factory so a
    #: generation-time key in the *tasks* config survives being rebuilt from
    #: a checkpoint. ``reaction_pkl`` actually lives under ``interference:``,
    #: which is instantiated unfiltered, so today this is a no-op.
    generation_time_keys = ("reaction_pkl",)

    def __init__(
        self,
        task_type: str = "diffusion_reactot",
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
        timesteps: int = 3000,
        beta_max: float = 0.3,
        power: float = 0.5,
        inv_power: float = 1,
        noise_schedule: str = "cosine",
        precision: float = 1.0e-5,
        loss_type: str = "l2",
        pos_only: bool = True,
        scales: Sequence[float] = (1.0, 2.0, 1.0),
        fixed_idx: Optional[List[int]] = None,
        mapping: str = "R+P->TS",
        mapping_initial: str = "RP",
        sigma: float = 0.0,
        ts_guess: Optional[Any] = None,
        idx: int = 1,
        nfe: int = 10,
        ot_ode: bool = True,
        solver: str = "ode",
        method: str = "midpoint",
        atol: float = 1e-2,
        rtol: float = 1e-2,
        clip_denoise: bool = True,
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by train.py
    ) -> None:
        """Record the released checkpoint's own hyperparameters.

        Every default here is the released ``reactot-pretrained.ckpt``'s
        ``hyper_parameters`` block, not the repo's ``train_rpsb_ts1x.py``;
        where the two disagree the checkpoint wins. See the plan's
        Hyperparameter Provenance table.

        Args:
            task_type: ``"diffusion_reactot"``.
            model_config: LEFTNet's own architecture block. Required.
            node_nfs: per-object input widths, ``[9, 9, 9]``.
            edge_nf: 0; no edge features anywhere in this model.
            condition_nf: 1, and **inert** -- fed constant zeros.
            fragment_names: ``["R", "TS", "P"]``; the order binds each
                object to its encoder/decoder pair in the state dict.
            pos_dim: 3.
            update_pocket_coords: ``False`` raises ``NotImplementedError``.
            condition_time: ``True``.
            edge_cutoff: ``None`` => fully connected within a reaction.
            norm_values: ``(1,1,1)``: no normalisation.
            norm_biases: ``(0,0,0)``.
            timesteps: the bridge grid. **3000; do not change** -- nothing
                in the state dict pins it, so a wrong value loads silently
                and samples wrong, and it is what makes the ODE solver's
                constant-beta assertion hold.
            beta_max: **0.3; do not change**, same assertion.
            power: 0.5.
            inv_power: 1.
            noise_schedule: recorded by the checkpoint and marked
                ``# not used`` in upstream's train script. Carried for
                fidelity; nothing reads it.
            precision: as ``noise_schedule``.
            loss_type: asserted in ``{"vlb", "l2"}``, then unread by
                ``EnSB``.
            pos_only: coordinates only; atom identities are supplied by the
                input reaction and copied through.
            scales: per-object loss weights -- **inert in ``EnSB``**. They
                reach only upstream's dead ``DDPMModule`` path. Recorded.
            fixed_idx: ``[0, 2]`` -- **inert**; see :data:`FRAG_FIXED`.
            mapping: only ``"R+P->TS"``.
            mapping_initial: only ``"RP"``.
            sigma: 0.0; no endpoint jitter.
            ts_guess: must stay ``None``; out of scope.
            idx: 1, the transition state.
            nfe: default network evaluations at sampling time.
            ot_ode: ``True`` -- **this is what makes it deterministic**.
            solver: ``"ode"``, the README's published default.
            method: ``"midpoint"``.
            atol: dead for a fixed-grid method.
            rtol: as ``atol``.
            clip_denoise: clamp predicted structures into +/-10 A.
            atom_vocab: defaults to ``[H, C, N, O, F]``.
            **kwargs: swallowed; ``cli/train.py`` injects ``node_feature*``.

        Raises:
            ValueError: if ``model_config`` is missing.
        """
        if model_config is None:
            raise ValueError(
                "tasks.model_config is required: it is LEFTNet's own "
                "architecture block and the released checkpoint will not "
                "load against anything else."
            )
        self.task_type = task_type
        # dict(), not the DictConfig: BaseDynamics mutates model_config in
        # place and LEFTNet is called with **it.
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
        self.timesteps = int(timesteps)
        self.beta_max = float(beta_max)
        self.power = float(power)
        self.inv_power = float(inv_power)
        # Recorded for checkpoint fidelity; both are marked "# not used" in
        # upstream's own train script and nothing in EnSB reads them.
        self.noise_schedule = noise_schedule
        self.precision = float(precision)
        self.loss_type = loss_type
        self.pos_only = bool(pos_only)
        self.scales = [float(s) for s in scales]
        self.fixed_idx = list(fixed_idx) if fixed_idx else None
        self.mapping = mapping
        self.mapping_initial = mapping_initial
        self.sigma = float(sigma)
        self.ts_guess = ts_guess
        self.idx = int(idx)
        self.nfe = int(nfe)
        self.ot_ode = bool(ot_ode)
        self.solver = solver
        self.method = method
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.clip_denoise = bool(clip_denoise)
        self.atom_vocab = (
            list(atom_vocab) if atom_vocab else list(OAREACTDIFF_ATOM_VOCAB)
        )
        self.condition_names: List[str] = []
        self.task: Optional[ReactOTTask] = None

    def build(self) -> ReactOTTask:
        """Assemble dynamics + bridge schedule + normaliser into the task.

        Returns:
            The built :class:`ReactOTTask`.
        """
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
        ddpm = EnSB(
            dynamics=dynamics,
            schedule=SBSchedule(
                timesteps=self.timesteps,
                beta_max=self.beta_max,
                power=self.power,
                inv_power=self.inv_power,
            ),
            normalizer=Normalizer(
                norm_values=self.norm_values,
                norm_biases=self.norm_biases,
                pos_dim=self.pos_dim,
            ),
            size_histogram=None,
            loss_type=self.loss_type,
            pos_only=self.pos_only,
            fixed_idx=self.fixed_idx,
            mapping=self.mapping,
            mapping_initial=self.mapping_initial,
            sigma=self.sigma,
            ts_guess=self.ts_guess,
            idx=self.idx,
        )
        self.task = ReactOTTask(
            ddpm,
            nfe=self.nfe,
            ot_ode=self.ot_ode,
            solver=self.solver,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
            clip_denoise=self.clip_denoise,
            atom_vocab=self.atom_vocab,
        )
        return self.task


class ReactOTTSGenerator(TSGenerator):
    """Transition-state generation behind ``interference/gen_reactot_ts``.

    The shape of the request is identical to every other TS run -- *load one
    reaction, tile it to a batch, sample the transition state into it, write
    .xyz* -- so the loop is
    :meth:`~MolecularDiffusion.modules.tasks.pocket_generator.
    PocketGenerator.run`, the reaction plumbing is
    :class:`~MolecularDiffusion.modules.tasks.ts_generator.TSGenerator`, and
    what is left here is only React-OT's: which corpus, which filters, which
    collate, and the solver knobs. **No ``PocketGenerator`` hook is
    overridden.**

    **The sampler is deterministic** (``ot_ode=True``, ``sigma=0.0``), so
    ``num_generate: N`` yields N identical structures. That is honest but
    wasteful, which is why ``gen_reactot_ts.yaml`` defaults ``num_generate:
    1`` / ``batch_size: 1`` and why :meth:`_settings_note` says so in the
    header. It is a config default and a print statement, not a missing
    hook.

    The corpus filters differ from
    :class:`~MolecularDiffusion.modules.tasks.diffusion_oareactdiff.
    OAReactDiffTSGenerator` in exactly one value: ``single_frag_only=False``.
    React-OT's checkpoint was trained on multi-fragment reactions too -- that
    is its capability claim -- so its generator must be able to index them.
    ``swapping_react_prod`` stays off for the same reason as OA-ReactDiff's:
    a generator should not silently hand back the reverse reaction.
    """

    tag = "reactot"
    #: Upstream seeds through ``seed_everything``, which seeds numpy.
    seed_numpy = True
    #: ``_accept`` is not overridden, so the loop always progresses.
    max_retries: Optional[int] = None
    #: This model's corpus really is a pickle, so its own config says so.
    #: The base accepts both this and the shared ``reaction_source``.
    source_key = "reaction_pkl"
    db_required_msg = (
        "interference.reaction_pkl (or reaction_source, on the shared "
        "gen_ts.yaml) is required: React-OT has no unconditional mode -- it "
        "needs a reactant and a product. Point it at a Transition1x pickle, "
        "e.g. docs/model_integrations/reactot/data/reactot_ts1x_valid.pkl."
    )

    def __init__(
        self,
        task: Any,
        reaction_index: int = 0,
        num_generate: int = 1,
        batch_size: int = 1,
        num_steps: Optional[int] = 10,
        solver: str = "ode",
        method: str = "midpoint",
        atol: float = 1e-2,
        rtol: float = 1e-2,
        output_path: str = "generated_reactot",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the run.

        Args:
            task: the loaded :class:`ReactOTTask`.
            reaction_index: which reaction in the corpus to generate for.
            num_generate: how many transition states to sample. Defaults to
                1 because the sampler is deterministic.
            batch_size: how many to sample at once.
            num_steps: **this is upstream's ``--nfe``**: the number of
                network evaluations, 10 in the README's published command.
            solver: ``"ode"`` (published) or ``"ddpm"`` (upstream's train
                script default, and what their FastAPI service runs).
            method: ODE method; only ``"midpoint"`` is implemented.
            atol: dead for a fixed-grid method; carried for fidelity.
            rtol: as ``atol``.
            output_path: directory for the .xyz files.
            seed: torch/random/numpy seed. Does not affect the structure.
            device: ``None`` => cuda if available.
            **kwargs: rejected by the base, on purpose.
        """
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
        self.solver = solver
        self.method = method
        self.atol = atol
        self.rtol = rtol

    # -- hooks ----------------------------------------------------------- #
    def _reactions(self) -> ReactionSource:
        """The held-out Transition1x rows, with React-OT's own filters.

        ``single_frag_only=False`` is the one value that differs from
        OA-ReactDiff's: this checkpoint was trained on multi-fragment
        reactions, so the generator must be able to reach them.

        Returns:
            The filtered corpus.
        """
        return OAReactDiffTS1xDataset(
            str(self.pocket_db),
            center=True,
            zero_charge=False,
            single_frag_only=False,
            swapping_react_prod=False,
            use_by_ind=True,
        )

    def _collate(self, reaction: Reaction, n: int) -> Dict[str, Any]:
        """Tile the reaction ``n`` times, with fresh scatter indices.

        Reusing the training collate rather than hand-rolling one is what
        guarantees the sampler sees exactly the layout training saw --
        including the ``int64`` per-object scatter indices
        ``torch_scatter`` needs.

        Args:
            reaction: the one reaction this run conditions on.
            n: how many copies.

        Returns:
            ``{"representations": [...], "conditions": tensor}``.
        """
        representations, conditions = oareactdiff_collate(
            [reaction] * n, zero_charge=False
        )
        return {"representations": representations, "conditions": conditions}

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:  # noqa: ARG002
        """The solver knobs, threaded to :meth:`ReactOTTask.sample`.

        Args:
            item: the loaded reaction; unused.
            n: batch size; unused.

        Returns:
            The extra kwargs of ``task.sample()``.
        """
        return {
            "solver": self.solver,
            "method": self.method,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    def _label(self, reaction: Reaction) -> str:
        """The reaction id, both SMILES, and the energy difference.

        Args:
            reaction: the reaction being generated for.

        Returns:
            A one-line description for the run header.
        """
        meta = reaction.meta
        return (
            f"{meta['rxn']} ({'.'.join(meta['smi_reactant'])} >> "
            f"{'.'.join(meta['smi_product'])}), dE = {meta['ediff']:.2f}"
        )

    def _settings_note(self) -> str:
        """Solver settings, in upstream's vocabulary.

        The header says ``nfe=`` even though the key is ``num_steps``: one
        key, one meaning, and a log a React-OT user recognises.

        Returns:
            The text appended to the run header.
        """
        nfe = self.num_steps or getattr(self.task, "nfe", "?")
        return (
            f" (solver={self.solver}, nfe={nfe}, method={self.method}; "
            "DETERMINISTIC -- num_generate > 1 yields identical copies)"
        )
