"""GoFlow: transition-state geometry from a reaction's condensed graph of
reaction (CGR), via conditional flow matching and GotenNet.

Galustian, Mark, Karwounopoulos, Kovar & Heid, *GoFlow: Efficient
Transition State Geometry Prediction with Flow Matching and E(3)-Equivariant
Neural Networks*, ChemRxiv (2025), doi:10.26434/chemrxiv-2025-bk2rh. Ported
from the repo checked out at ``others/nice/goflow`` (commit
``3ec00a09d9b283e3258ae01fe5d3e35bb3812bff``).

Three pieces live here:

:class:`GoFlowTask`
    The platform task contract around :class:`~MolecularDiffusion.modules.
    models.goflow.gotennet.GotenNet`. ``forward`` reproduces upstream's
    ``FlowModule.train_val_step`` (``flow_matching/flow_module.py:94-99``);
    ``sample`` reproduces ``FlowModule.test_step``'s ODE integration
    (``:116-134``) but returns every draw independently, unranked -- see
    Scope below.

:class:`ModelTaskFactory`
    Hydra entry point for ``configs/tasks/diffusion_goflow.yaml``.
    ``representation:`` is a nested Hydra ``_target_`` block that Hydra
    instantiates into a real :class:`GotenNet` *before* this factory is
    constructed (mirroring upstream's own ``configs/model/flow.yaml``
    nesting), so ``build()`` is a two-line assembly, not a second-level
    dict-to-object conversion.

:class:`GoFlowTSGenerator`
    A thin :class:`~MolecularDiffusion.modules.tasks.ts_generator.
    TSGenerator` subclass. **It writes no ``run()`` loop.** "Load one
    reaction, tile it to a batch, sample the transition state into it,
    write .xyz" is the shared TS layer (``ts_generator.py``) on top of the
    shared pocket loop (``pocket_generator.py``); what is left here is only
    GoFlow's own: which corpus (the converted RDB7 pickle + its
    ``feat_dict.pkl`` + split files), which collate, and nothing else --
    GoFlow has no RePaint knobs, no solver choice, just a fixed-step Euler
    integrator with one knob (``num_steps``) the base already carries.

Scope of this integration
--------------------------

**Transition-state geometry given a reaction's connectivity (reactant +
product bonds and per-atom RDKit descriptors), and nothing else.** Not
ported, each for a stated reason (see ``INTEGRATION_PLAN.md``, Explicitly
out of scope, for the full list): upstream's GT-anchored Kabsch-rotate-then-
median-consensus ensembling and RDKit-substructure permutation matching
(both require a *known* transition state, so cannot run on a real, blind
reaction); the vendored ``tsdiff/`` baseline (a separate, complete model);
``schedule_free`` optimizer mode; the model-size ablation configs; trajectory
frames. **``sample()`` returns every ``num_generate`` draw independently,
unranked and unpermuted** -- same "no RMSD scoring, no best-of-N ranking"
design as the two existing TS generators
(``modules/tasks/ts_generator.py``'s own stated principle).

Two facts that constrain everything
------------------------------------

* Atom identity is **supplied by the input reaction and copied straight
  through** -- only coordinates are generated. ``sample()``'s one-hot
  encoding reuses :data:`~MolecularDiffusion.modules.tasks.
  diffusion_oareactdiff.N_ELEMENT` and
  :func:`~MolecularDiffusion.modules.tasks.diffusion_oareactdiff._pad_ts`
  by import (do not duplicate them): the five-wide ``[H, C, N, O, F]``
  layout is architecture-agnostic padding logic, not something specific to
  OA-ReactDiff. RDB7 never populates the "F" column (it is a C/H/N/O
  corpus), but the column costs nothing.
* GoFlow has **no charge concept at all** (unlike OA-ReactDiff/React-OT,
  whose ninth column is the atomic number standing in for a charge slot):
  the charge column in ``sample()``'s output is zeros, full stop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from MolecularDiffusion.data.integrations.goflow_data import (
    GoFlowRDB7Dataset,
    goflow_collate,
)
from MolecularDiffusion.data.integrations.oareactdiff_data import ATOM_MAPPING
from MolecularDiffusion.data.datasets.reaction_data import Reaction
from MolecularDiffusion.modules.models.goflow import (
    Atomwise3DOut,
    GotenNet,
    euler_integrate,
    get_perturbed_flow_point_and_time,
    rmsd_loss,
)
from MolecularDiffusion.modules.tasks.reaction.oareactdiff import N_ELEMENT, _pad_ts
from MolecularDiffusion.modules.tasks.reaction.ts_generator import ReactionSource, TSGenerator


class GoFlowTask(nn.Module):
    """Task contract around :class:`GotenNet` + :class:`Atomwise3DOut`."""

    def __init__(
        self,
        representation: GotenNet,
        output_n_hidden: int = 64,
        num_steps: int = 25,
        atom_vocab: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.output_layer = Atomwise3DOut(
            n_in=representation.hidden_dim, n_hidden=output_n_hidden, activation=F.silu
        )
        #: Default Euler steps for `sample()`; the actual headline run's
        #: value (see INTEGRATION_PLAN.md's Hyperparameter Provenance).
        self.num_steps = int(num_steps)
        self.atom_vocab = list(atom_vocab) if atom_vocab else ["H", "C", "N", "O", "F"]
        #: No size prior: a transition state has exactly the atoms of its
        #: reaction. Plain attributes so cli/generate.py and
        #: engine_lightning can stamp on them harmlessly.
        self.node_dist_model: Any = None
        self.prop_dist_model: Any = None
        self.split = "train"

    # -- contract properties -------------------------------------------- #
    @property
    def model(self) -> "GoFlowTask":
        """There is no separate inner module the generation code needs."""
        return self

    @property
    def device(self) -> torch.device:
        """Device of the first parameter."""
        return next(self.parameters()).device

    # -- shared forward pass ---------------------------------------------- #
    def model_output(self, x_t_N_3: Tensor, batch: Any, t_G: Tensor) -> Tensor:
        """One network call: ``(x_t, t, batch) -> predicted velocity``.

        Reproduces ``FlowModule.model_output`` (``flow_module.py:101-104``).
        """
        h_N_D, x_n_l_d = self.representation(x_t_N_3, t_G, batch)
        return self.output_layer(h_N_D, x_n_l_d[:, :3, :])

    # -- training ---------------------------------------------------------- #
    def forward(self, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """One training/validation step.

        Reproduces ``FlowModule.train_val_step``
        (``flow_module.py:94-99``): draw ``x_0``, Kabsch-align the ground
        truth onto it, interpolate to a random time, and regress the
        straight-line velocity under ``rmsd_loss``.

        Args:
            batch: ``{"batch": <PyG Batch>}`` from
                :func:`~MolecularDiffusion.data.component.goflow_data.
                goflow_collate`; the batch must carry ``ts_pos``.

        Returns:
            ``(loss, stats)``.

        Raises:
            ValueError: if the batch carries no ``ts_pos`` (this model
                cannot train without a known transition state).
        """
        pyg_batch = batch["batch"].to(self.device)
        if not hasattr(pyg_batch, "ts_pos"):
            raise ValueError(
                "GoFlowTask.forward needs batch['batch'].ts_pos: training "
                "requires a known transition state (goflow_collate only "
                "omits it for a blind, corpus-free R/P query)."
            )
        x_t_n_3, dx_dt_n_3, t_g = get_perturbed_flow_point_and_time(pyg_batch, self.device)
        pred_n_3 = self.model_output(x_t_n_3, pyg_batch, t_g)
        loss = rmsd_loss(pred_n_3, dx_dt_n_3)
        return loss, {"loss": loss}

    def predict_and_target(self, batch: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """Pure-generative stub: the loss is the prediction, target is zero."""
        loss, _ = self.forward(batch)
        pred = loss.detach().reshape(1)
        return pred, torch.zeros_like(pred)

    def evaluate(self, pred: Tensor, target: Tensor) -> Dict[str, Tensor]:  # noqa: ARG002
        """Validation metric.

        Upstream's substructure-matched DMAE/RMSD validation
        (``evaluate_geometry``, ``callbacks.py:182-239``) is deliberately
        not here -- same reasoning as oareactdiff/reactot leaving their own
        heavier upstream validation-time sampling out of scope.
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
        num_generate: Optional[int] = None,  # noqa: ARG002 - sample count is len(batch)
        **kwargs: Any,  # noqa: ARG002 - swallows mode/n_frames
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate transition states for the reactions in ``batch``.

        **The signature deviates from Section 2.1 the same way every
        pocket-conditioned task's does**: the conditioning structure arrives
        as ``batch``, because there is no channel for "which reaction" in
        ``sample(batch_size, nodesxsample, ...)``.

        One fresh Gaussian draw per call, integrated with the vendored
        fixed-step Euler stepper (:func:`~MolecularDiffusion.modules.models.
        goflow.flow.euler_integrate`) over ``linspace(0, 1, num_steps)`` --
        reproduces ``FlowModule.test_step``'s per-sample loop
        (``flow_module.py:128-134``) MINUS the GT-anchored substructure
        matching + Kabsch-to-reference + median-consensus collapsing
        (structurally requires a known TS; see ``INTEGRATION_PLAN.md``'s
        Repo Inspection caveat and Explicitly out of scope). Every sample in
        ``batch`` is returned independently and unranked.

        Args:
            batch_size: ignored; taken from ``batch``.
            nodesxsample: accepted and *checked*, not used to choose. The
                transition state has exactly as many atoms as the reaction,
                so a value disagreeing with it is a caller bug worth
                surfacing rather than silently overriding.
            num_steps: Euler steps. ``None`` falls back to ``self.num_steps``
                (25, the actual headline run's value).
            batch: ``{"batch": <PyG Batch>}``, one graph per requested
                sample (built by tiling one reaction ``n`` times).
            num_generate: unused; accepted for signature parity with the
                platform contract. The number of samples is however many
                graphs ``batch`` carries.

        Returns:
            ``(one_hot, charges, coords, node_mask)`` for the transition
            state, padded to ``(B, N, .)``. ``charges`` is all zeros:
            GoFlow has no charge concept, unlike OA-ReactDiff/React-OT's
            atomic-number-as-charge column.

        Raises:
            ValueError: if ``batch`` is missing, or ``nodesxsample``
                disagrees with the reaction's atom count.
        """
        if batch is None:
            raise ValueError(
                "GoFlowTask.sample needs `batch`: a reactant and a product "
                "are the conditioning, and there is no unconditional mode "
                "in this integration."
            )
        device = self.device
        pyg_batch = batch["batch"].to(device)
        n_samples = int(pyg_batch.num_graphs)

        if nodesxsample is not None:
            want = torch.as_tensor(nodesxsample).view(-1).to(device).long()
            got = torch.bincount(pyg_batch.batch, minlength=n_samples).long()
            if not torch.equal(want, got):
                raise ValueError(
                    f"nodesxsample={want.tolist()} disagrees with the "
                    f"reaction's atom count {got.tolist()}. A transition "
                    "state has the same atoms as its reactant and product; "
                    "it cannot be resized."
                )

        steps = int(num_steps) if num_steps else int(self.num_steps)
        t_grid = torch.linspace(0, 1, steps, device=device)
        x_init_n_3 = torch.randn(pyg_batch.num_nodes, 3, device=device)

        def ode_func(t: float, x_t_n_3: Tensor) -> Tensor:
            t_g = torch.full((n_samples, 1), float(t), device=device)
            return self.model_output(x_t_n_3, pyg_batch, t_g)

        final_pos_n_3 = euler_integrate(ode_func, x_init_n_3, t_grid)

        atom_idx = torch.tensor(
            [ATOM_MAPPING[int(z)] for z in pyg_batch.atom_type.tolist()],
            dtype=torch.long, device=device,
        )
        one_hot_n_5 = F.one_hot(atom_idx, num_classes=N_ELEMENT).float()
        charge_n_1 = torch.zeros(pyg_batch.num_nodes, 1, device=device)
        flat = torch.cat([final_pos_n_3, one_hot_n_5, charge_n_1], dim=1)
        return _pad_ts(flat, pyg_batch.batch, n_samples)


class ModelTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_goflow.yaml``.

    No ``train_set`` parameter: the only construction-time statistic
    (``n_atom_rdkit_feats``) is a fixed config int, verified against the
    shipped ``feat_dict.pkl`` by :class:`~MolecularDiffusion.data.component.
    goflow_data.GoFlowRDB7Dataset` at data-load time, not injected via the
    Section 2.5 ``train_set`` seam. ``cli/train.py`` is untouched.
    """

    #: Belt-and-braces, mirroring the two existing TS tasks.
    #: ``cli/generate.py:173-202`` reads this off the factory so a
    #: generation-time key in the *tasks* config survives being rebuilt
    #: from a checkpoint. ``reaction_source`` actually lives under
    #: ``interference:``, which is instantiated unfiltered, so today this
    #: is a no-op -- it matters only if someone moves the key.
    generation_time_keys = ("reaction_source",)

    def __init__(
        self,
        task_type: str = "diffusion_goflow",
        representation: Optional[GotenNet] = None,
        output_n_hidden: int = 64,
        num_steps: int = 25,
        atom_vocab: Optional[List[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - node_feature* injected by train.py
    ) -> None:
        """Record GoFlow's own hyperparameters.

        Args:
            task_type: ``"diffusion_goflow"``.
            representation: the (already Hydra-instantiated) ``GotenNet``
                backbone -- required; ``configs/tasks/diffusion_goflow.yaml``
                nests it as ``representation: {_target_: ...GotenNet, ...}``,
                so Hydra builds it before this factory is constructed.
            output_n_hidden: ``Atomwise3DOut``'s hidden width (upstream's
                ``configs/model/flow.yaml``'s ``output.n_hidden: 64``).
            num_steps: default Euler steps for ``sample()`` (the actual
                headline run's value, 25 -- see the plan's Hyperparameter
                Provenance table for the config-default-vs-actual-run
                conflict this resolves).
            atom_vocab: output vocabulary; defaults to
                ``[H, C, N, O, F]`` (see the module docstring for why "F"
                is present but never populated by RDB7).
            **kwargs: swallowed; ``cli/train.py`` injects ``node_feature*``.

        Raises:
            ValueError: if ``representation`` is missing.
        """
        if representation is None:
            raise ValueError(
                "tasks.representation is required: it is GotenNet's own "
                "architecture block (configs/tasks/diffusion_goflow.yaml)."
            )
        self.task_type = task_type
        self.representation = representation
        self.output_n_hidden = int(output_n_hidden)
        self.num_steps = int(num_steps)
        self.atom_vocab = list(atom_vocab) if atom_vocab else ["H", "C", "N", "O", "F"]
        self.condition_names: List[str] = []
        self.task: Optional[GoFlowTask] = None

    def build(self) -> GoFlowTask:
        """Assemble the representation + output head into the task."""
        self.task = GoFlowTask(
            representation=self.representation,
            output_n_hidden=self.output_n_hidden,
            num_steps=self.num_steps,
            atom_vocab=self.atom_vocab,
        )
        return self.task


class GoFlowTSGenerator(TSGenerator):
    """Transition-state generation behind ``interference/gen_goflow_ts``.

    The shape of the request is identical to every other TS run -- *load one
    reaction, tile it to a batch, sample the transition state into it, write
    .xyz* -- so the loop is
    :meth:`~MolecularDiffusion.modules.tasks.pocket_generator.
    PocketGenerator.run`, the reaction plumbing is
    :class:`~MolecularDiffusion.modules.tasks.ts_generator.TSGenerator`, and
    what is left here is only GoFlow's own: which corpus (the converted
    RDB7 pickle, plus the ``feat_dict.pkl`` and split files it needs
    alongside it) and which collate. **No ``PocketGenerator`` hook beyond
    what ``TSGenerator`` already fills is overridden.**

    Unlike OA-ReactDiff/React-OT, GoFlow's corpus is not one self-contained
    pickle: :class:`~MolecularDiffusion.data.component.goflow_data.
    GoFlowRDB7Dataset` also needs the frozen ``feat_dict.pkl`` (the one-hot
    vocabulary the converted ``ReactionSide.feat`` columns were built
    against) and the split directory/file, so this generator adds those
    three as its own constructor keys rather than folding them into
    ``reaction_source`` -- exactly the same "a subclass may add its own
    extra kwargs beyond the shared base" pattern as OA-ReactDiff's
    RePaint knobs or React-OT's solver knobs.

    ``source_key`` stays the shared default, ``"reaction_source"``: GoFlow's
    corpus really is "a reaction source", a converted pickle, no more
    honest model-specific name exists (mirrors ``INTEGRATION_PLAN.md``'s
    Inference Task Decision table verbatim).
    """

    tag = "goflow"
    #: Upstream seeds through ``seed_everything``, which seeds numpy.
    seed_numpy = True
    #: ``_accept`` is not overridden, so the loop always progresses.
    max_retries: Optional[int] = None
    source_key = "reaction_source"
    db_required_msg = (
        "interference.reaction_source is required: GoFlow has no "
        "unconditional mode -- it needs a reactant and a product. Point it "
        "at the converted Reaction pickle, e.g. "
        "docs/model_integrations/goflow/data/goflow_rdb7_reactions.pkl."
    )

    def __init__(
        self,
        task: Any,
        reaction_index: int = 0,
        feat_dict_file: str = "",
        split_path: str = "",
        split_file: str = "random_split.pkl",
        split: str = "test",
        num_generate: int = 25,
        batch_size: int = 4,
        num_steps: Optional[int] = 25,
        output_path: str = "generated_goflow",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the run.

        Args:
            task: the loaded :class:`GoFlowTask`; its ``sample()`` is what
                gets called.
            reaction_index: which reaction in the split to generate for.
            feat_dict_file: the frozen ``feat_dict.pkl`` this corpus's
                one-hot columns were built against.
            split_path: directory holding the split pickle.
            split_file: which split pickle, e.g. ``random_split.pkl``.
            split: ``"train"``, ``"val"`` or ``"test"``; defaults to the
                held-out ``"test"`` split.
            num_generate: how many transition states to sample. 25 mirrors
                the actual headline run's ensemble size
                (``train_test_all_splits.sh``'s ``model.num_samples=25``).
            batch_size: how many of those to sample at once.
            num_steps: Euler steps; ``None`` => the task's default (25).
            output_path: directory for the .xyz files.
            seed: torch/random/numpy seed.
            device: ``None`` => cuda if available.
            **kwargs: forwarded to :class:`TSGenerator` (``reaction_source``
                arrives this way), then rejected by the base if unknown.

        Raises:
            ValueError: if ``feat_dict_file`` or ``split_path`` is not set.
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
        if not feat_dict_file or not split_path:
            raise ValueError(
                "interference.feat_dict_file and interference.split_path "
                "are required alongside interference.reaction_source: "
                "GoFlow's corpus needs the frozen feat_dict.pkl and the "
                "split directory the converted reaction pickle was built "
                "against (see configs/interference/gen_goflow_ts.yaml)."
            )
        self.feat_dict_file = feat_dict_file
        self.split_path = split_path
        self.split_file = split_file
        self.split = split

    # -- hooks ----------------------------------------------------------- #
    def _reactions(self) -> ReactionSource:
        """The RDB7 reactions in :attr:`split`, from the converted pickle."""
        return GoFlowRDB7Dataset(
            str(self.pocket_db), self.feat_dict_file, self.split_path,
            split_file=self.split_file, split=self.split,
        )

    def _collate(self, reaction: Reaction, n: int) -> Dict[str, Any]:
        """Tile the reaction ``n`` times via the shared training collate."""
        return goflow_collate([reaction], n=n)

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:  # noqa: ARG002
        """GoFlow's ``sample()`` needs no extra kwargs beyond what
        ``PocketGenerator.run`` already passes (``num_steps`` included)."""
        return {}

    def _label(self, reaction: Reaction) -> str:
        """The reaction id and its atom-mapped SMARTS."""
        meta = reaction.meta
        return f"{meta.get('rxn', '?')} ({meta.get('smiles', '?')})"

    def _settings_note(self) -> str:
        """The Euler step count is the whole quality dial here."""
        return f" (num_steps={self.num_steps or 25}, euler)"
