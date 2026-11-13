"""The one ``run()`` loop behind every measurement-conditioned generator.

**Structure elucidation** is a different problem from the ones the existing
seams solve. You are handed a *measurement* of an unknown compound -- a tandem
mass spectrum, an NMR pair, an IR trace -- and you return a **ranked shortlist
of candidate structures**. The molecule is the unknown; the conditioning is read
per item from a corpus of measurements; and the answer is a set, not a sample.

Every existing generator was surveyed before this one was written, and each
fails on a named mechanism rather than on taste:

* :class:`~MolecularDiffusion.runmodes.generate.tasks_generate.GenerativeFactory`
  conditions on ``target_values`` -- *a flat scalar list broadcast identically
  to every molecule in the run* -- or on one reference ``.xyz`` tiled to the
  batch. Elucidation needs a different payload per output. Its size logic is
  wrong in kind too (``nodesxsample`` from a prior), and every writer in it is
  coordinate-based.
* :class:`~MolecularDiffusion.modules.tasks.pocket_generator.PocketGenerator`
  has the right candidate-set shape but is **single-context by construction**
  (``item = self._pocket()``, called once), and its output contract is
  hard-wired to coordinates (``_accept(one_hot, coords, node_mask)``,
  ``save_xyz_file``). ``TSGenerator`` rides it precisely because a transition
  state *is* coordinates.
* ``ConformerFactory`` is closest by loop shape, but its pool items are
  *molecules* (``_to_mol(item, item.pos...)``), it writes a ``reference.sdf`` of
  its input, and its metric is RMSD against that input geometry. Here the input
  is a measurement and the molecule is the unknown.

Hence a new shared seam rather than a bandaid. What differs per model lives in
hooks:

===========================  ====================================================
hook                         what it decides
===========================  ====================================================
:attr:`tag`                  log prefix
:attr:`source_key`           this model's own name for ``spectra_source``
:attr:`source_required_msg`  the "source is required" error, or ``None``
:meth:`_records`             which reader walks the corpus, and in what order
:meth:`_condition`           the model-specific conditioning payload (opaque)
:meth:`_priors`              the known-composition slot: formula / atom multiset
:meth:`_repeat`              how one measurement is tiled to ``n`` candidates
:attr:`supports_guidance`    whether this model has a CFG branch at all
:attr:`maskable_channels`    which measurement channels ``drop_channels``
                             may blank out, by name
:meth:`_sample_kwargs`       model-specific ``elucidate()`` kwargs, on
                             top of the base's ``num_steps`` /
                             ``guidance_scale`` forwarding
:meth:`_decode`              raw output -> :class:`Candidate` (3D **nullable**)
:meth:`_accept`              validity filter
:meth:`_rank`                **defaults to generation order**
:meth:`_reference`           ground truth, **or None** -- scoring is optional
:meth:`_start` / :meth:`_summary`  header / footer
===========================  ====================================================

Four constraints keep this seam general rather than shaped around any one
model. Each was checked against two structurally different elucidation models
-- one graph-based and coordinate-free, one coordinate-first -- before being
written down:

* **``_priors`` is an explicit named slot, not a kwarg blob.** Both models are
  handed the molecular formula; composition-is-known is the norm in this
  problem, not an extra.
* **``_decode`` returns SMILES plus nullable 3D.** A graph-based model emits
  connectivity and never fills ``coords``; a coordinate-first model fills it
  and reaches SMILES through bond perception. Both slots always exist. ``_decode`` is also allowed
  to be slow and to fail per candidate.
* **``_rank`` defaults to identity.** No ranking framework is built here. A
  model that votes over repeated draws overrides it (e.g. canonical-InChI
  frequency); a model with no scorer at all takes the default and is not
  forced into a no-op override.
* **``_reference`` may return ``None``.** Scoring is a separate, optional pass.
  The core loop must run on genuinely unknown spectra -- that is the actual use
  case, and an evaluation harness that cannot run without labels is not one.

Padding is deliberately *not* in the seam: it belongs to :meth:`_repeat`,
because one model pads to a dataset-wide ``max_n_atoms`` while another is
variable-size PyG.

Outputs, per run::

    <output_path>/record_XXXX/candidates.{sdf,smi}   ranked candidates
    <output_path>/record_XXXX/ranking.csv            rank, smiles, score
    <output_path>/predictions.csv                    run-level, one row per record
    <output_path>/metrics.json                       only if _reference gave labels
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Sequence

import numpy as np
import torch
from MolecularDiffusion.device import get_device
from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Task class name -> the generator that knows how to feed it. Keyed on the
#: task rather than a registry string because the task object is all
#: ``cli/generate.py`` has at instantiate time (``task_type`` lives on the
#: *factory*, not the built task). Same mechanism as
#: ``pocket_generator._TASK_TO_GENERATOR``.
#: One entry per elucidation task, e.g.
#: ``"MyTask": ("diffusion_mymodel", "MyElucidationGenerator")``.
_TASK_TO_GENERATOR: dict[str, tuple[str, str]] = {
    "ChefNMRElucidationTask": (
        "diffusion_chefnmr",
        "ChefNMRElucidationGenerator",
    ),
}


def _for_task(task: Any) -> type:
    """The concrete generator matching ``task``, imported lazily."""
    import importlib  # noqa: PLC0415

    entry = _TASK_TO_GENERATOR.get(type(task).__name__)
    if entry is None:
        msg = (
            f"interference/gen_elucidation.yaml cannot generate for task "
            f"{type(task).__name__}: it is not a measurement-conditioned "
            f"elucidation task. Known: {sorted(_TASK_TO_GENERATOR)}. Point "
            f"`interference` at that model's own gen_*.yaml instead."
        )
        raise ValueError(msg)
    module, name = entry
    mod = importlib.import_module(f"MolecularDiffusion.modules.tasks.{module}")
    return getattr(mod, name)


@dataclass
class Candidate:
    """One proposed structure.

    Attributes:
        smiles: canonical SMILES. Always present -- it is the common currency
            between a graph-generating model and a coordinate-generating one.
        mol: the RDKit molecule, when the model produced one.
        coords: ``(n_atoms, 3)`` positions, or ``None``. **Nullable on purpose**:
            a 2D elucidation model never fills it, and forcing a fake conformer
            in would be inventing geometry the model did not predict.
        score: rank score, when the model has a scorer. ``None`` means the
            candidate is ordered by generation order.
    """

    smiles: str
    mol: Any = None
    coords: Optional[np.ndarray] = None
    score: Optional[float] = None
    meta: dict = field(default_factory=dict)


class ElucidationGenerator:
    """Walk a corpus of measurements; emit a ranked candidate set per record.

    Not instantiable from a config on its own -- :meth:`_records`,
    :meth:`_condition`, :meth:`_repeat` and :meth:`_decode` are what a model
    must supply.
    """

    #: Log prefix, e.g. ``[mymodel] record 3/10``.
    tag: ClassVar[str] = "elucidation"

    #: What *this model's own* config calls ``spectra_source``. A subclass free
    #: to expose an honest, model-specific name declares it here instead of
    #: adding a second constructor parameter, and then **both keys work** -- the
    #: shared config on every model, the model's own name on its own config. A
    #: typo is still rejected, because anything that is neither is left in
    #: ``kwargs`` for the base's strict check. Same mechanism as
    #: ``TSGenerator.source_key``.
    source_key: ClassVar[str] = "spectra_source"

    #: Raised when the source is missing; ``None`` => the subclass validates its
    #: own sources (e.g. it takes a labels file plus a subformulae folder rather
    #: than one path).
    source_required_msg: ClassVar[Optional[str]] = None

    #: Whether this model has a classifier-free-guidance branch at all.
    #: ``False`` turns ``guidance_scale`` from a silent no-op into a
    #: refusal: a model trained without conditioning dropout has no
    #: unconditional branch to interpolate against, and quietly
    #: accepting the key would make a run look configured when it is
    #: not -- the same reason unknown keys are rejected above.
    supports_guidance: ClassVar[bool] = True

    #: Named measurement channels this model can blank out at inference --
    #: two nuclei, MS1 vs MS2, an IR region. Empty (the default) means the
    #: measurement is ONE indivisible payload for this model, and a
    #: non-empty ``drop_channels`` is refused rather than silently ignored,
    #: for the same reason ``supports_guidance = False`` refuses
    #: ``guidance_scale``.
    #:
    #: The base only validates the *names*. Which slice of the payload a
    #: name zeroes is :meth:`_condition`'s business -- ``_condition`` is
    #: opaque on purpose, so there is no shared shape here to slice.
    maskable_channels: ClassVar[tuple[str, ...]] = ()

    def __new__(cls, task: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
        # `gen_elucidation.yaml` names this class and carries only the keys
        # every elucidation run shares; the concrete subclass is picked off the
        # loaded task. CPython then calls `type(obj).__init__`, so the
        # subclass' own __init__ still runs with the config's keys.
        return object.__new__(_for_task(task) if cls is ElucidationGenerator else cls)

    def __init__(  # noqa: PLR0913
        self,
        task: Any,
        spectra_source: Optional[str] = None,
        spectra_index: int = 0,
        max_records: Optional[int] = None,
        num_candidates: int = 100,
        batch_size: int = 10,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        split: Optional[str] = None,
        drop_channels: Sequence[str] = (),
        top_k: Sequence[int] = (1, 10),
        seed: int = 42,
        device: Optional[str] = None,
        output_path: str = "generated_elucidation",
        **kwargs: Any,
    ) -> None:
        """Configure the run.

        Args:
            task: the loaded elucidation task; its ``elucidate()`` is called.
            spectra_source: where measurements are read from. A subclass may
                alias this under its own name via :attr:`source_key`.
            spectra_index: index of the first record to process.
            max_records: how many records to process; ``None`` => all of them.
            num_candidates: candidates to draw per record.
            batch_size: how many of those to draw at once. ``k`` is a batch
                dimension, never a Python loop.
            num_steps: reverse-process steps; ``None`` => the model's default.
            guidance_scale: classifier-free guidance strength -- the
                ``w`` in ``(1+w)*cond - w*uncond``. ``None`` => the
                model's own default. Note ``0.0`` is a *meaningful,
                different* value: it means unconditional. Rejected when
                the model declares ``supports_guidance = False``.
            split: which fold of a labelled corpus to elucidate (e.g.
                ``test``). ``None`` => whatever the model's own reader
                defaults to. Corpora with no folds ignore it.
            drop_channels: measurement channels to blank out before the
                model is shown the record, by name. Valid names are this
                model's :attr:`maskable_channels`; anything else is
                rejected naming the valid set, and a model declaring none
                rejects any non-empty list. ``()`` => the measurement
                exactly as recorded.
            top_k: which top-k accuracies to report, when references exist.
            seed: torch/random/numpy seed.
            device: ``None`` => cuda if available.
            output_path: directory for the per-record candidate sets.
            **kwargs: rejected by the base, on purpose.

        Raises:
            ValueError: on any unrecognised interference key. A silently
                ignored key makes a run look configured when it is not -- the
                same reason ``PocketGenerator`` rejects rather than ignores.
        """
        alias = type(self).source_key
        if alias != "spectra_source" and alias in kwargs:
            aliased = kwargs.pop(alias)
            if spectra_source is not None and aliased is not None:
                msg = (
                    f"set either spectra_source or {alias}, not both: they are "
                    f"the same key. {alias} is what {type(self).__name__} calls "
                    "it; spectra_source is what the shared "
                    "interference/gen_elucidation.yaml calls it."
                )
                raise ValueError(msg)
            spectra_source = spectra_source or aliased
        if kwargs:
            msg = (
                f"{type(self).__name__} does not accept interference key(s) "
                f"{sorted(kwargs)}. Either the key is a typo, or it belongs to "
                f"a different model -- silently ignoring it would make the run "
                f"look configured when it is not. See "
                f"configs/interference/gen_elucidation.yaml for the shared keys."
            )
            raise ValueError(msg)
        if self.source_required_msg and not spectra_source:
            raise ValueError(self.source_required_msg)
        if guidance_scale is not None and not type(self).supports_guidance:
            msg = (
                f"{type(self).__name__} has no classifier-free-guidance "
                f"branch, so guidance_scale={guidance_scale} would have "
                "no effect. Leave it null in "
                "interference/gen_elucidation.yaml. (A model can only "
                "use it if it was trained with conditioning dropout.)"
            )
            raise ValueError(msg)

        channels = tuple(str(c) for c in drop_channels or ())
        known = type(self).maskable_channels
        unknown = [c for c in channels if c not in known]
        if unknown:
            detail = (
                f"It has {list(known)}."
                if known
                else (
                    "It declares none: its measurement is one indivisible "
                    "payload, so there is no channel to blank out. Leave "
                    "drop_channels empty."
                )
            )
            msg = (
                f"{type(self).__name__} has no measurement channel named "
                f"{unknown}. {detail} (drop_channels ABLATES a channel that "
                "IS present; a channel the corpus never held is already "
                "absent.)"
            )
            raise ValueError(msg)

        self.task = task
        self.spectra_source = spectra_source
        self.spectra_index = int(spectra_index)
        self.max_records = max_records
        self.num_candidates = int(num_candidates)
        self.batch_size = int(batch_size)
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.split = split
        self.drop_channels = channels
        self.top_k = tuple(int(k) for k in top_k)
        self.seed = int(seed)
        self.output_path = output_path
        self.device = device or (str(get_device()))

    # --- hooks ---------------------------------------------------------------

    def _records(self) -> Sequence[Any]:
        """The measurements this run walks, in order."""
        raise NotImplementedError

    def _condition(self, record: Any) -> Any:
        """The model-specific conditioning payload for one record.

        Deliberately opaque -- there is no universal ``Spectrum`` dataclass,
        because one model reads annotated peak formulae and another reads binned
        NMR intensities, and a lowest common denominator would serve neither.
        """
        raise NotImplementedError

    def _priors(self, record: Any) -> Any:  # noqa: ARG002
        """Known composition for this record -- formula, atom multiset, adduct.

        ``None`` when the model infers composition itself. In practice most
        elucidation models are handed the formula, which is why this is a named
        slot rather than something smuggled through ``_sample_kwargs``.
        """
        return None

    def _repeat(self, cond: Any, priors: Any, n: int) -> Any:
        """Tile one measurement into a batch of ``n`` candidates.

        Padding and collation live here, not in the base: one model pads to a
        dataset-wide maximum, another is variable-size PyG.
        """
        raise NotImplementedError

    def _sample_kwargs(self, record: Any, n: int) -> dict:  # noqa: ARG002
        """Kwargs for ``task.elucidate()``: the base's, plus the model's.

        Forwards only the knobs the config actually set. ``None`` means
        "the model's own default", and the way to express that to a
        callee is to not pass the argument at all -- passing ``None``
        through would force every ``elucidate()`` to re-implement the
        same defaulting.

        A model with extra kwargs extends rather than replaces::

            def _sample_kwargs(self, record, n):
                base = super()._sample_kwargs(record, n)
                return {**base, "adduct": record.adduct}
        """
        kwargs: dict = {}
        if self.num_steps is not None:
            kwargs["num_steps"] = self.num_steps
        if self.guidance_scale is not None:
            kwargs["guidance_scale"] = self.guidance_scale
        return kwargs

    def _decode(self, raw: Any) -> list[Candidate]:
        """Raw model output -> candidates. May be slow; may fail per candidate."""
        raise NotImplementedError

    def _accept(self, candidates: list[Candidate]) -> list[Candidate]:
        """Drop what did not decode. Override to filter harder."""
        return [c for c in candidates if c is not None and c.smiles]

    def _rank(self, candidates: list[Candidate]) -> list[Candidate]:
        """Generation order, unchanged.

        Identity by default and that is the point: a model with no scorer must
        not be forced into a no-op override, and no ranking abstraction is
        imposed on one that has its own.
        """
        return candidates

    def _reference(self, record: Any) -> Any:  # noqa: ARG002
        """Ground truth for scoring, or ``None`` on an unlabelled corpus."""
        return None

    def _record_name(self, record: Any, index: int) -> str:
        """Directory-safe name for one record's output folder."""
        name = getattr(record, "name", None) or getattr(record, "spec_name", None)
        return str(name) if name else f"record_{index:04d}"

    def _start(self, record: Any, index: int, total: int) -> None:
        print(
            f"[{self.tag}] record {index + 1}/{total} "
            f"'{self._record_name(record, index)}': "
            f"{self.num_candidates} candidates"
        )

    def _summary(self, written: int, attempts: int) -> None:  # noqa: ARG002
        print(f"[{self.tag}] wrote {written} records to {self.output_path}")

    # --- the loop ------------------------------------------------------------

    def run(self) -> None:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        # The task defines `device` as a property, so `cli/generate.py`'s
        # `if not hasattr(task, "device")` guard skips it and nothing else ever
        # moves it off the CPU that `map_location="cpu"` left it on.
        self.task.to(self.device)
        self.task.eval()

        records = list(self._records())
        start = self.spectra_index
        end = len(records) if self.max_records is None else start + int(self.max_records)
        selected = records[start:end]
        if not selected:
            msg = (
                f"no records to process: the corpus has {len(records)} "
                f"record(s) and spectra_index={self.spectra_index} selects none."
            )
            raise ValueError(msg)

        rows: list[dict] = []
        scored: list[tuple[list[Candidate], Any]] = []
        for offset, record in enumerate(
            tqdm(selected, desc=f"{self.tag}: records", leave=True)
        ):
            self._start(record, offset, len(selected))
            candidates = self._candidates_for(record)
            reference = self._reference(record)
            name = self._record_name(record, start + offset)
            self._write_record(name, candidates, reference)
            rows.append(
                {
                    "record": name,
                    "n_candidates": len(candidates),
                    "top1_smiles": candidates[0].smiles if candidates else "",
                    "reference": _as_smiles(reference) or "",
                }
            )
            if reference is not None:
                scored.append((candidates, reference))

        self._write_predictions(rows)
        if scored:
            self._write_metrics(scored)
        self._summary(len(rows), len(selected))

    def _candidates_for(self, record: Any) -> list[Candidate]:
        """Draw, decode, filter and rank ``num_candidates`` for one record."""
        cond = self._condition(record)
        priors = self._priors(record)
        collected: list[Candidate] = []
        drawn = 0
        while drawn < self.num_candidates:
            n = min(self.batch_size, self.num_candidates - drawn)
            batch = move_to_device(self._repeat(cond, priors, n), self.device)
            with torch.no_grad():
                raw = self.task.elucidate(batch, **self._sample_kwargs(record, n))
            collected.extend(self._accept(self._decode(raw)))
            drawn += n
        return self._rank(collected)

    # --- writers -------------------------------------------------------------

    def _write_record(
        self, name: str, candidates: list[Candidate], reference: Any
    ) -> None:
        directory = os.path.join(self.output_path, name)
        os.makedirs(directory, exist_ok=True)

        has_3d = any(c.coords is not None for c in candidates)
        if has_3d:
            self._write_sdf(os.path.join(directory, "candidates.sdf"), candidates)
        else:
            # .smi, not .sdf: a 2D model produced no geometry, and an SDF with a
            # fabricated all-zero conformer reads as a structure that was
            # predicted in 3D.
            with open(os.path.join(directory, "candidates.smi"), "w") as handle:
                for rank, cand in enumerate(candidates, start=1):
                    handle.write(f"{cand.smiles}\t{name}_{rank}\n")

        with open(os.path.join(directory, "ranking.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["rank", "smiles", "score"])
            for rank, cand in enumerate(candidates, start=1):
                writer.writerow(
                    [rank, cand.smiles, "" if cand.score is None else cand.score]
                )
        if reference is not None:
            with open(os.path.join(directory, "reference.smi"), "w") as handle:
                handle.write(f"{_as_smiles(reference)}\t{name}_reference\n")

    @staticmethod
    def _write_sdf(path: str, candidates: list[Candidate]) -> None:
        from rdkit import Chem  # noqa: PLC0415

        writer = Chem.SDWriter(path)
        try:
            for rank, cand in enumerate(candidates, start=1):
                mol = cand.mol
                if mol is None:
                    continue
                mol.SetProp("_Name", f"candidate_{rank}")
                if cand.score is not None:
                    mol.SetProp("score", str(cand.score))
                writer.write(mol)
        finally:
            writer.close()

    def _write_predictions(self, rows: list[dict]) -> None:
        path = os.path.join(self.output_path, "predictions.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["record", "n_candidates", "top1_smiles", "reference"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_metrics(
        self, scored: list[tuple[list[Candidate], Any]]
    ) -> None:
        """Top-k accuracy, max Tanimoto@k and validity -- only where labels exist."""
        from rdkit import Chem, DataStructs, RDLogger  # noqa: PLC0415
        from rdkit.Chem import AllChem  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        metrics: dict[str, float] = {}
        n = len(scored)
        for k in self.top_k:
            hits = 0
            tanimoto = 0.0
            for candidates, reference in scored:
                ref_smiles = _as_smiles(reference)
                ref_mol = Chem.MolFromSmiles(ref_smiles) if ref_smiles else None
                top = candidates[:k]
                if ref_smiles and any(c.smiles == ref_smiles for c in top):
                    hits += 1
                if ref_mol is None:
                    continue
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                best = 0.0
                for cand in top:
                    mol = cand.mol or Chem.MolFromSmiles(cand.smiles)
                    if mol is None:
                        continue
                    try:
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, 2, nBits=2048
                        )
                    except Exception:  # noqa: BLE001 - unsanitizable candidate
                        continue
                    best = max(best, DataStructs.TanimotoSimilarity(fp, ref_fp))
                tanimoto += best
            metrics[f"top{k}_accuracy"] = hits / n
            metrics[f"max_tanimoto_at_{k}"] = tanimoto / n

        total = sum(len(c) for c, _ in scored)
        valid = sum(
            1
            for candidates, _ in scored
            for c in candidates
            if Chem.MolFromSmiles(c.smiles) is not None
        )
        metrics["validity"] = valid / total if total else 0.0
        metrics["n_records"] = n

        path = os.path.join(self.output_path, "metrics.json")
        with open(path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[{self.tag}] metrics: {metrics}")


def move_to_device(obj: Any, device: Any) -> Any:
    """Recursively place a batch on ``device``, leaving unknown types alone.

    The conditioning payload is opaque to this seam by design, but "the batch
    must be on the same device as the model" is true for every model that will
    ever ride it -- and getting it wrong surfaces as a ``RuntimeError`` deep
    inside an encoder, not as anything that names the batch. So the base applies
    this to whatever :meth:`ElucidationGenerator._repeat` returns. It handles
    tensors, dicts, lists/tuples and anything with a ``.to()`` (a PyG ``Batch``,
    for instance); anything else is returned untouched, so a model whose payload
    needs custom placement can simply do it in ``_repeat`` -- this then finds
    nothing left to move.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    to = getattr(obj, "to", None)
    if callable(to):
        try:
            return to(device)
        except (TypeError, RuntimeError, AttributeError):
            return obj
    return obj


def _as_smiles(reference: Any) -> Optional[str]:
    """A reference may be a SMILES string or an RDKit molecule."""
    if reference is None:
        return None
    if isinstance(reference, str):
        return reference
    from rdkit import Chem  # noqa: PLC0415

    try:
        return Chem.MolToSmiles(reference)
    except Exception:  # noqa: BLE001
        return None
