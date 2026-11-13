"""The one ``run()`` loop behind every pocket-conditioned generator.

Seven models (DiffSBDD, DiffInt, KGDiff, IPDiff, PMDM, Apo2Mol, DiffPharma)
each need "load one pocket, tile it, sample ligands into it, write .xyz".
That loop used to be copy-pasted seven times. It lives here once; what
genuinely differs per model lives in a handful of hooks:

===========================  ====================================================
hook                         what it decides
===========================  ====================================================
:attr:`tag`                  log prefix
:attr:`seed_numpy`           whether ``np.random.seed`` is called (see below)
:attr:`max_retries`          bound on resampling rounds; ``None`` => unbounded
:attr:`db_required_msg`      the "pocket_db is required" error, or ``None``
:meth:`_pocket`              read one pocket item (which Dataset, which centering)
:meth:`_repeat`              tile that item ``n`` times, with fresh scatter indices
:meth:`_sizes` /
:meth:`_pick_sizes`          how many atoms each sampled ligand gets
:meth:`_sample_kwargs`       the model-specific kwargs of ``task.sample()``
:meth:`_accept`              which of a batch's molecules are written out
:meth:`_after_batch`         side outputs per batch (Apo2Mol's pocket .pdb)
:meth:`_start` /
:meth:`_summary`             the header / footer the run prints
===========================  ====================================================

**``seed_numpy`` is not a style knob.** PMDM and DiffPharma have never seeded
numpy, the other five always have. Seeding it everywhere would change what
those two generate; that is a behaviour change, not a fix.

The loop is not pocket-only. A second family rides it through
:class:`~MolecularDiffusion.modules.tasks.ts_generator.TSGenerator`, which
fills these hooks once for transition-state models -- the fixed structural
context there is a reaction rather than a protein pocket. Add a new TS model
by subclassing *that*, not this.

Two ways to reach a generator:

* ``interference/gen_<model>_pocket.yaml`` names the concrete subclass, and
  carries that model's own knobs (guidance weights, sampler settings,
  inpainting);
* ``interference/gen_pocket.yaml`` names :class:`PocketGenerator` itself and
  carries only the nine keys all seven share -- the concrete subclass is then
  picked from the loaded task (see :func:`_for_task`).

Because that second path bypasses per-model Hydra key checking, the base
``__init__`` rejects any key that no generator in the resolved chain declares,
instead of ignoring it: a typo'd ``guide_mdoe`` must not silently do nothing.
Note this is *not* a claim that per-model keys are refused on the shared
config -- dispatch reaches the concrete subclass first, so a key that subclass
declares is consumed and honoured there.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from MolecularDiffusion.device import get_device
from tqdm import tqdm

INT_TYPE = torch.int64

#: Task class name -> the generator that knows how to feed it. Keyed on the
#: task rather than a registry string because the task object is all
#: ``cli/generate.py`` has at instantiate time (``task_type`` lives on the
#: *factory*, not the built task).
#:
#: A value is either a plain ``(module, name)`` tuple (the task has exactly
#: one generator) or a ``{mode: (module, name)}`` dict, for a task whose
#: generator depends on ``interference.mode`` -- e.g. PMDM's de novo path
#: (``mode=None``) needs a converted ``pocket_db`` row, while its constrained
#: modes (``mode="lead_opt"``/``"linker"``) read a standalone ``pocket_file``/
#: ``mol_file`` pair instead: a genuinely different input source, not a value
#: on the same one, so it is a different generator class even though it is
#: the SAME task class (``PMDMDiffusionTask`` throughout).
#:
#: DiffInt is deliberately absent: it reuses ``DiffSBDDTask`` verbatim, so the
#: task cannot distinguish the two -- unlike PMDM's modes, there is no kwarg
#: DiffInt's runs set that DiffSBDD's don't, so a ``mode`` axis cannot resolve
#: this one. DiffInt runs keep using ``gen_diffint_pocket.yaml``.
_TASK_TO_GENERATOR: Dict[str, Any] = {
    "DiffSBDDTask": ("diffusion_diffsbdd", "DiffSBDDPocketGenerator"),
    "KGDiffDiffusionTask": ("diffusion_kgdiff", "KGDiffPocketGenerator"),
    "IPDiffDiffusionTask": ("diffusion_ipdiff", "IPDiffPocketGenerator"),
    "PMDMDiffusionTask": {
        None: ("diffusion_pmdm", "PMDMPocketGenerator"),
        "lead_opt": ("diffusion_pmdm", "PMDMConstrainedGenerator"),
        "linker": ("diffusion_pmdm", "PMDMConstrainedGenerator"),
    },
    "Apo2MolDiffusionTask": ("diffusion_apo2mol", "Apo2MolPocketGenerator"),
    "DiffPharmaTask": ("diffusion_diffpharma", "DiffPharmaPocketGenerator"),
}


def _for_task(task: Any, mode: Optional[str] = None) -> type:
    """The concrete generator matching ``task`` (and ``mode``, for
    multi-mode tasks), imported lazily."""
    import importlib  # noqa: PLC0415

    entry = _TASK_TO_GENERATOR.get(type(task).__name__)
    if entry is None:
        raise ValueError(
            f"interference/gen_pocket.yaml cannot generate for task "
            f"{type(task).__name__}: it is not a pocket-conditioned task. "
            f"Known: {sorted(_TASK_TO_GENERATOR)}. Point `interference` at "
            f"that model's own gen_*.yaml instead."
        )
    if isinstance(entry, dict):
        target = entry.get(mode)
        if target is None:
            known = sorted(k for k in entry if k is not None)
            raise ValueError(
                f"interference/gen_pocket.yaml: task {type(task).__name__} has "
                f"no generator for interference.mode={mode!r}. Known modes: "
                f"{known} (omit `mode`, or set it null, for the default)."
            )
    else:
        if mode is not None:
            raise ValueError(
                f"interference/gen_pocket.yaml: task {type(task).__name__} "
                f"does not support interference.mode (got {mode!r}) -- only "
                f"multi-mode tasks like PMDM's do. Remove `mode` from this run."
            )
        target = entry
    module, name = target
    mod = importlib.import_module(f"MolecularDiffusion.modules.tasks.{module}")
    return getattr(mod, name)


class PocketGenerator:
    """Load one pocket, sample ``num_generate`` ligands into it, write .xyz.

    Instantiating this class directly (what ``gen_pocket.yaml`` does) returns
    the concrete subclass for the task that was passed in.
    """

    #: Log prefix, e.g. ``[kgdiff] wrote 20 ligands``.
    tag = "pocket"
    #: See the module docstring -- per-model, never uniform.
    seed_numpy = True
    #: Bound on sampling rounds, for generators that may reject a batch.
    max_retries: Optional[int] = None
    #: Raised when ``pocket_db`` is missing; ``None`` => the subclass checks.
    db_required_msg: Optional[str] = None

    def __new__(cls, task: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
        # CPython calls ``type(obj).__init__``, so the subclass' own
        # __init__ still runs with the config's keys (`mode` included --
        # it is a real constructor param on multi-mode subclasses like
        # PMDMConstrainedGenerator, not consumed here).
        return object.__new__(
            _for_task(task, kwargs.get("mode")) if cls is PocketGenerator else cls
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
        output_path: str = "generated_pocket",
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(
                f"{type(self).__name__} does not accept interference key(s) "
                f"{sorted(kwargs)}. Either the key is a typo, or it belongs to "
                f"a different model -- silently ignoring it would make the run "
                f"look configured when it is not. See "
                f"configs/interference/gen_pocket.yaml for the shared keys."
            )
        if self.db_required_msg and not pocket_db:
            raise ValueError(self.db_required_msg)
        self.task = task
        self.pocket_db = pocket_db
        self.pocket_index = pocket_index
        self.num_generate = num_generate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.mol_size = list(mol_size) if mol_size else []
        self.output_path = output_path
        self.seed = seed
        self.device = device or (
            str(get_device())
        )

    # --- hooks -----------------------------------------------------------

    def _pocket(self) -> Dict[str, Any]:
        """Read the one pocket item this run conditions on."""
        raise NotImplementedError

    def _repeat(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile ``item`` ``n`` times, with fresh scatter indices."""
        raise NotImplementedError

    def _sizes(self, n: int) -> Optional[torch.Tensor]:
        """Explicit sizes, or ``None`` to let the model's own prior decide."""
        if self.mol_size:
            return torch.tensor(
                random.choices(self.mol_size, k=n), dtype=INT_TYPE
            )
        return None

    def _pick_sizes(
        self, batch: Dict[str, Any], n: int
    ) -> Optional[torch.Tensor]:
        """``_sizes`` for everyone whose prior does not need the tiled batch."""
        return self._sizes(n)

    def _sample_kwargs(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Model-specific kwargs of ``task.sample()``."""
        return {}

    def _accept(self, one_hot, coords, node_mask) -> List[int]:
        """Indices of the batch's molecules to write out."""
        return list(range(one_hot.size(0)))

    def _after_batch(
        self, item: Dict[str, Any], written: int, n: int
    ) -> None:
        """Side outputs for the batch just written."""

    def _start(self, item: Dict[str, Any]) -> None:
        print(
            f"[{self.tag}] pocket '{item.get('name', '?')}', generating "
            f"{self.num_generate} ligands"
        )

    def _summary(self, written: int, attempts: int) -> None:  # noqa: ARG002
        print(f"[{self.tag}] wrote {written} ligands to {self.output_path}")

    def _max_attempts(self) -> Optional[int]:
        # `is None`, not falsy: max_retries=0 is a user-settable value meaning
        # "one round, then give up", and must stay bounded. Conflating it with
        # None turns that into an unbounded loop.
        if self.max_retries is None:
            return None
        return max(1, self.max_retries) * max(
            1, -(-self.num_generate // self.batch_size)
        )

    # --- the loop --------------------------------------------------------

    def run(self) -> None:
        from MolecularDiffusion.utils.geometry.geom_utils import (  # noqa: PLC0415
            save_xyz_file,
        )

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if self.seed_numpy:
            np.random.seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        self.task.to(self.device)
        self.task.eval()
        item = self._pocket()
        self._start(item)

        written = attempts = 0
        max_attempts = self._max_attempts()
        bar = tqdm(total=self.num_generate, desc="Sampling ligands", leave=True)
        while written < self.num_generate and (
            max_attempts is None or attempts < max_attempts
        ):
            attempts += 1
            n = min(self.batch_size, self.num_generate - written)
            batch = self._repeat(item, n)
            one_hot, _charges, coords, node_mask = self.task.sample(
                batch=batch,
                nodesxsample=self._pick_sizes(batch, n),
                num_steps=self.num_steps,
                **self._sample_kwargs(item, n),
            )
            one_hot, coords, node_mask = (
                one_hot.cpu(),
                coords.cpu(),
                node_mask.cpu(),
            )
            keep = self._accept(one_hot, coords, node_mask)
            if not keep:
                # ponytail: `written` is not incremented here, so a subclass
                # that overrides _accept without setting max_retries loops
                # forever. Safe today because PMDM is the only override and it
                # sets max_retries; bound it there if a second one appears.
                continue
            index = torch.tensor(keep, dtype=torch.long)
            save_xyz_file(
                self.output_path,
                one_hot[index],
                coords[index],
                self.task.atom_vocab,
                id_from=written,
                name="molecule",
                node_mask=node_mask[index],
            )
            self._after_batch(item, written, n)
            written += len(keep)
            bar.update(len(keep))
        bar.close()
        self._summary(written, attempts)
