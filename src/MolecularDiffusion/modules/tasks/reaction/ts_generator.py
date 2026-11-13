"""The one ``run()`` loop behind every transition-state generator.

A transition-state model asks for the same thing every pocket-conditioned
model asks for -- *load one fixed structural context, tile it to a batch,
sample the unknown part into it, write .xyz* -- so this is not a second
loop. :class:`TSGenerator` subclasses
:class:`~MolecularDiffusion.modules.tasks.pocket_generator.PocketGenerator`
and fills its hooks once, for the whole family, in terms of the shared
:class:`~MolecularDiffusion.data.component.reaction_data.Reaction`
container. A TS model then supplies two things and nothing else:

==============  ================================================
hook            what it decides
==============  ================================================
``_reactions``  where reactions come from (which Dataset, which
                filters), as anything with ``__len__`` /
                ``__getitem__``
``_collate``    how one reaction becomes a batch of ``n`` copies,
                in whatever layout that model's ``sample()`` reads
==============  ================================================

and may override three more, all defaulted:
:meth:`~TSGenerator._label` (how the reaction prints),
:meth:`~TSGenerator._settings_note` (extra sampler knobs in the header) and
:meth:`~PocketGenerator._sample_kwargs` (the model-specific kwargs of
``task.sample()``). Everything else -- seeding, the retry loop, size
selection, writing the samples, writing the reference structures beside
them -- is inherited.

**Why the context is a ``Reaction``, not a geometry.** Of the three TS
models surveyed when the container was designed, only OA-ReactDiff conditions
on reactant and product *coordinates*; GoFlow and RitS are connectivity-only
condensed graphs of reaction and never read an endpoint geometry at sampling
time. So this class must never assume a side has 3D. It asks
:attr:`~MolecularDiffusion.data.component.reaction_data.ReactionSide.
has_geometry` and writes ``reactant.xyz`` / ``product.xyz`` only for the
sides that actually have coordinates -- a connectivity-only model gets
``reference_ts.xyz`` alone, and no lie on disk.

**What is deliberately NOT here.** No RMSD scoring and no best-of-N ranking.
Both siblings do rank samples, but offline, in their own analysis scripts and
against metrics they disagree about (GoFlow: permutation-matched, mirror-
allowed, median-consensus; RitS: none at all in the sampling path). Writing
``reference_ts.xyz`` next to the samples is what a shared loop can honestly
do; the ranking belongs to whoever knows the metric.

**Interference-key naming.** The base takes ``reaction_source`` /
``reaction_index``, not ``pocket_db`` / ``pocket_index``: a config should
never say "pocket" for a reaction. A subclass whose corpus has a more honest
name declares :attr:`TSGenerator.source_key` (OA-ReactDiff:
``reaction_pkl``); both that name and ``reaction_source`` then work, so the
shared config drives every model and each model's own config keeps its own
vocabulary. Anything that is neither still raises.

**Two ways to reach a generator**, exactly as for pockets:

* ``interference/gen_<model>_ts.yaml`` names the concrete subclass and
  carries that model's own knobs (OA-ReactDiff: RePaint's ``resamplings`` /
  ``jump_length`` / ``noise_schedule``);
* ``interference/gen_ts.yaml`` names :class:`TSGenerator` itself and carries
  only the keys every TS model shares -- the concrete subclass is then picked
  off the loaded task (see :func:`_for_task`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol

import torch
from ase.data import chemical_symbols
from torch import Tensor

from MolecularDiffusion.data.datasets.reaction_data import Reaction
from MolecularDiffusion.modules.tasks.conditional.pocket_generator import (
    INT_TYPE,
    PocketGenerator,
)

#: Task class name -> the TS generator that knows how to feed it. Keyed on
#: the task rather than a registry string for the same reason
#: ``pocket_generator._TASK_TO_GENERATOR`` is: the task object is all
#: ``cli/generate.py`` has at instantiate time (``task_type`` lives on the
#: *factory*, not the built task).
_TASK_TO_TS_GENERATOR = {
    "OAReactDiffTask": ("diffusion_oareactdiff", "OAReactDiffTSGenerator"),
    "ReactOTTask": ("diffusion_reactot", "ReactOTTSGenerator"),
    "GoFlowTask": ("diffusion_goflow", "GoFlowTSGenerator"),
}


def _for_task(task: Any) -> type:
    """The concrete TS generator matching ``task``, imported lazily."""
    import importlib  # noqa: PLC0415

    entry = _TASK_TO_TS_GENERATOR.get(type(task).__name__)
    if entry is None:
        raise ValueError(
            f"interference/gen_ts.yaml cannot generate for task "
            f"{type(task).__name__}: it is not a transition-state task. "
            f"Known: {sorted(_TASK_TO_TS_GENERATOR)}. Point `interference` "
            f"at that model's own gen_*.yaml instead."
        )
    module, name = entry
    mod = importlib.import_module(f"MolecularDiffusion.modules.tasks.{module}")
    return getattr(mod, name)


class ReactionSource(Protocol):
    """Anything indexable that yields :class:`Reaction` objects.

    A ``torch.utils.data.Dataset`` of reactions satisfies this, which is what
    every TS corpus in-tree already is. Declared structurally so a model whose
    reactions come from somewhere else -- a list, an HDF5 view, a parsed
    reaction SMARTS -- needs no base class.
    """

    def __len__(self) -> int:
        """Number of reactions available."""
        ...

    def __getitem__(self, index: int, /) -> Reaction:
        """The reaction at ``index``."""
        ...


class TSGenerator(PocketGenerator):
    """Load one reaction, sample ``num_generate`` transition states.

    Not instantiable from a config on its own -- :meth:`_reactions` and
    :meth:`_collate` are the two things a model must supply. See the module
    docstring for the full hook table.
    """

    tag = "ts"
    db_required_msg = (
        "interference.reaction_source is required: a transition-state "
        "generator conditions on a reaction, and there is no unconditional "
        "mode. Point it at this model's reaction corpus."
    )
    #: What *this model's own* config calls ``reaction_source``. The shared
    #: ``gen_ts.yaml`` always says ``reaction_source``; a subclass free to
    #: expose an honest, model-specific name (OA-ReactDiff's corpus really is
    #: a pickle, and ``reaction_pkl`` says so) declares it here instead of
    #: adding a second constructor parameter, and then **both keys work** --
    #: the shared config on every model, the model's own name on its own
    #: config. A typo is still rejected, because anything that is neither is
    #: left in ``kwargs`` for the base's strict check.
    source_key = "reaction_source"

    def __new__(cls, task: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
        # Mirrors PocketGenerator.__new__: `gen_ts.yaml` names this class,
        # and the concrete subclass is picked off the loaded task.
        return object.__new__(_for_task(task) if cls is TSGenerator else cls)

    def __init__(
        self,
        task: Any,
        reaction_source: Optional[str] = None,
        reaction_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        output_path: str = "generated_ts",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the run.

        Args:
            task: the loaded TS task; its ``sample()`` is what gets called.
            reaction_source: where reactions are read from, forwarded to the
                base as ``pocket_db``.
            reaction_index: which reaction in that source to generate for.
            num_generate: how many transition states to sample for it.
            batch_size: how many of those to sample at once.
            num_steps: reverse-process steps; ``None`` => the model's default.
            output_path: directory for the .xyz files.
            seed: torch/random/numpy seed.
            device: ``None`` => cuda if available.
            **kwargs: rejected by the base, on purpose -- an unknown
                interference key is a typo, not a no-op.

        Note:
            There is no ``mol_size``. A transition state has exactly the atoms
            of its reaction, so a size prior would be a lie; :meth:`_sizes`
            reads the count off the reaction instead.
        """
        alias = type(self).source_key
        if alias != "reaction_source" and alias in kwargs:
            aliased = kwargs.pop(alias)
            if reaction_source is not None and aliased is not None:
                raise ValueError(
                    f"set either reaction_source or {alias}, not both: they "
                    f"are the same key. {alias} is what "
                    f"{type(self).__name__} calls it; reaction_source is "
                    "what the shared interference/gen_ts.yaml calls it."
                )
            reaction_source = reaction_source or aliased
        if "mol_size" in kwargs:
            # The base declares mol_size, so it would be swallowed here and
            # then ignored by _sizes -- the silent no-op the base's own key
            # checking exists to prevent.
            raise ValueError(
                "mol_size does not apply to a transition state: it has "
                "exactly the atoms of its reaction, so a size prior would "
                "be a lie. Remove the key."
            )
        super().__init__(
            task,
            pocket_db=reaction_source,
            pocket_index=reaction_index,
            num_generate=num_generate,
            batch_size=batch_size,
            num_steps=num_steps,
            output_path=output_path,
            seed=seed,
            device=device,
            **kwargs,
        )
        #: Filled by :meth:`_pocket`. Underscored so ``log_hyperparameters``
        #: does not print a whole reaction's tensors into the run log.
        self._reaction: Optional[Reaction] = None
        self._references: List[str] = []

    # --- hooks a TS model fills ------------------------------------------

    def _reactions(self) -> ReactionSource:
        """The corpus this run indexes into, already filtered."""
        raise NotImplementedError

    def _collate(self, reaction: Reaction, n: int) -> Dict[str, Any]:
        """``reaction`` tiled ``n`` times, in this model's batch layout.

        Reuse the model's own training collate rather than hand-rolling one:
        that is what guarantees the sampler sees the layout training saw,
        including per-object ``int64`` scatter indices.

        Returns a dict because ``PocketGenerator._repeat`` is annotated
        ``-> Dict[str, Any]``; a model whose batch is a PyG ``Batch`` wraps it
        (``{"batch": batch}``) the same way :meth:`_pocket` wraps a
        :class:`Reaction`.
        """
        raise NotImplementedError

    def _load_reaction(self) -> Reaction:
        """The one reaction this run conditions on.

        Default: index :meth:`_reactions` at ``reaction_index``. Override
        this instead of :meth:`_reactions` for a model that builds its
        reaction from something unindexable -- a reaction SMARTS on the
        command line, say.
        """
        source = self._reactions()
        if not 0 <= self.pocket_index < len(source):
            raise IndexError(
                f"reaction_index {self.pocket_index} is out of range: "
                f"{os.path.basename(str(self.pocket_db))} yields "
                f"{len(source)} reactions under this model's filters."
            )
        return source[self.pocket_index]

    def _label(self, reaction: Reaction) -> str:
        """How the reaction identifies itself in the header line."""
        return str(reaction.meta.get("rxn", "?"))

    def _settings_note(self) -> str:
        """Extra sampler settings to append to the header line."""
        return ""

    # --- PocketGenerator hooks, filled once for the family ----------------

    def _pocket(self) -> Dict[str, Any]:
        """The fixed context: one reaction, wrapped in a dict.

        The wrapper is not decoration -- ``PocketGenerator._pocket`` is
        annotated ``-> Dict[str, Any]`` and ``_repeat`` takes the same type,
        so returning a bare :class:`Reaction` would be a Liskov violation
        this repo's mypy settings reject.
        """
        self._reaction = self._load_reaction()
        return {"reaction": self._reaction}

    def _repeat(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile the reaction ``n`` times via the model's own collate."""
        return self._collate(item["reaction"], n)

    def _sizes(self, n: int) -> Optional[Tensor]:
        """The TS size comes from the reaction, never from a prior."""
        if self._reaction is None:
            raise RuntimeError(
                "_sizes called before _pocket loaded a reaction."
            )
        return torch.full((n,), len(self._reaction), dtype=INT_TYPE)

    def _start(self, item: Dict[str, Any]) -> None:
        """Print the header and write the reference structures."""
        reaction: Reaction = item["reaction"]
        steps = self.num_steps if self.num_steps else "the model's default"
        print(
            f"[{self.tag}] reaction {self._label(reaction)}, "
            f"{len(reaction)} atoms; generating {self.num_generate} "
            f"transition states in {steps} steps{self._settings_note()}"
        )
        self._write_references(reaction)

    def _summary(self, written: int, attempts: int) -> None:  # noqa: ARG002
        alongside = (
            f" (alongside {', '.join(self._references)})"
            if self._references
            else ""
        )
        print(
            f"[{self.tag}] wrote {written} transition states to "
            f"{self.output_path}{alongside}"
        )

    # --- reference structures --------------------------------------------

    def _write_references(self, reaction: Reaction) -> None:
        """Write whichever of the run's known structures actually exist.

        A connectivity-only reaction has no endpoint geometry, so only
        ``reference_ts.xyz`` is written -- and only when the corpus carries a
        reference transition state at all.
        """
        self._references = []
        for name, pos in (
            ("reactant", reaction.reactant.pos),
            ("product", reaction.product.pos),
            ("reference_ts", reaction.ts_pos),
        ):
            if pos is not None:
                self._write_xyz(name, reaction, pos)
                self._references.append(f"{name}.xyz")

    def _write_xyz(self, name: str, reaction: Reaction, pos: Tensor) -> None:
        """Write one reference structure beside the generated ones.

        Symbols come from the reaction's own atomic numbers rather than the
        task's ``atom_vocab``: ``z`` is the container's shared ordering, and a
        model conditioning on connectivity alone may have no coordinate
        vocabulary to decode against.

        The comment line carries the bare reaction id, not :meth:`_label` --
        these files are read back by scoring scripts, and the header stays as
        it was before this loop was shared.
        """
        path = os.path.join(self.output_path, f"{name}.xyz")
        rxn = reaction.meta.get("rxn", "?")
        with open(path, "w") as handle:
            handle.write(f"{len(reaction)}\n{name} {rxn}\n")
            for z, (x, y, zc) in zip(reaction.z.tolist(), pos.tolist()):
                handle.write(
                    f"{chemical_symbols[int(z)]} {x:.6f} {y:.6f} {zc:.6f}\n"
                )
