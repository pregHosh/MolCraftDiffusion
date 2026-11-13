"""ChefNMR: 3D structure elucidation from a binned 1H/13C NMR pair.

Binds ``modules/models/chefnmr`` to the duck-typed Task contract and rides
the shared :class:`~MolecularDiffusion.modules.tasks.elucidation_generator.
ElucidationGenerator` seam for inference.

Approved INTEGRATION_PLAN (docs/model_integrations/chefnmr/), in one
paragraph: the routing is stock ``pointcloud`` -- coordinates and atom
types, no bonds anywhere in the forward path -- and the two payloads
``pointcloud`` has no channel for (the ``(10080,)`` condition and the
ground-truth conformer stack) ride row-aligned memmap sidecars joined
**here**, off ``batch["xyz"]``, exactly as ``diffusion_diffsmol.py`` joins
its shape latent. The data layer is untouched.

**No ``sample()``, deliberately.** ChefNMR cannot generate unconditionally:
the atom composition is an *input* (``known_atoms: True``) and the spectrum
is mandatory. A ``sample()`` that ignored both would be a lie, and would
also arm ``GenerativeEvalCallback`` during training to produce garbage. The
generative entry point is :meth:`ChefNMRElucidationTask.elucidate`.

**No ``node_dist_model`` / ``n_node_dist``**, for the same reason: the
formula fixes the molecule size, so there is no size prior to draw from.

Three traps worth knowing before editing this file:

* The condition must **never** enter the batch dict.
  ``pointcloud_collate_v0`` applies an *atom-axis* boolean mask to every
  tensor in the mapping, so a ``(B, 10080)`` condition would be silently
  sliced by an atom mask. ``batch["xyz"]`` survives untouched because both
  collates pass strings through.
* ``save_pickle(cheap_data=True)`` nulls ``xyzs``, destroying the join key.
  DiffSMol falls back to a zero latent on a miss; here a zero condition *is*
  the CFG unconditional branch, so every miss raises instead.
* ``sigma_data`` is baked into the EDM preconditioning and therefore into
  the weights. It is a task-config key, not something derived from whatever
  dataset is attached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.modules.models.chefnmr.diffusion import AtomDiffusion
from MolecularDiffusion.modules.models.chefnmr.sidecar import (
    ChefNMRSidecar,
    load_sidecar,
    parse_row_index,
)
from MolecularDiffusion.modules.models.chefnmr.unknown import (
    Channel,
    read_unknown_spectra,
)
from MolecularDiffusion.modules.tasks.conditional.elucidation_generator import (
    Candidate,
    ElucidationGenerator,
)

logger = logging.getLogger(__name__)

#: Upstream's ``atom_decoder`` (``configs/data/uspto.yaml:5``). **The order is
#: load-bearing**: the platform's ``compute_ohe`` uses position-in-vocab as the
#: one-hot column, so sorting this list silently permutes the checkpoint's
#: ``x_embedder`` input and yields a model that trains, samples, and is wrong.
CHEFNMR_ATOM_DECODER = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]


def canonical_smiles(smiles: Optional[str], remove_stereo: bool = True) -> Optional[str]:
    """Canonicalise the way upstream compares structures.

    ``RemoveHs`` -> optional ``RemoveStereochemistry`` -> ``CanonSmiles``,
    per ``src/evaluation/bond_analyzer.py:26-33``. Used for BOTH the
    reference and the candidates so top-k accuracy compares like with like.
    """
    if not smiles:
        return None
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    try:
        return Chem.CanonSmiles(Chem.MolToSmiles(mol))
    except Exception:  # noqa: BLE001 - unsanitizable input
        return None


# --------------------------------------------------------------------- #
# factory                                                               #
# --------------------------------------------------------------------- #
class ChefNMRTaskFactory:
    """Hydra entry point for ``configs/tasks/diffusion_chefnmr.yaml``."""

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_chefnmr",
        atom_decoder: Sequence[str] = tuple(CHEFNMR_ATOM_DECODER),
        max_n_atoms: int = 101,
        sigma_data: float = 2.67,
        condition_type: str = "H1C13NMRSpectrum",
        in_condition_size: Sequence[int] = (10000, 80),
        score_model_args: Optional[dict] = None,
        diffusion_process_args: Optional[dict] = None,
        diffusion_loss_args: Optional[dict] = None,
        multitask_args: Optional[dict] = None,
        max_n_conformers: int = 3,
        diffusion_multiplicity: int = 1,
        num_sampling_steps: int = 50,
        guidance_scale: float = 1.5,
        cond_path: Optional[str] = None,
        conf_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        atom_vocab: Optional[Sequence[str]] = None,
        **kwargs: Any,  # noqa: ARG002 - train.py injects node_feature_dim etc.
    ) -> None:
        self.task_type = task_type
        self.atom_decoder = list(atom_decoder)
        self.max_n_atoms = int(max_n_atoms)
        self.sigma_data = float(sigma_data)
        self.condition_type = condition_type
        self.in_condition_size = [int(v) for v in in_condition_size]
        self.score_model_args = dict(score_model_args or {})
        self.diffusion_process_args = dict(diffusion_process_args or {})
        self.diffusion_loss_args = dict(diffusion_loss_args or {})
        self.multitask_args = dict(multitask_args or {})
        self.max_n_conformers = int(max_n_conformers)
        self.diffusion_multiplicity = int(diffusion_multiplicity)
        self.num_sampling_steps = int(num_sampling_steps)
        self.guidance_scale = float(guidance_scale)
        self.cond_path = cond_path
        self.conf_path = conf_path
        self.meta_path = meta_path
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.task: Optional["ChefNMRElucidationTask"] = None

    def build(self) -> "ChefNMRElucidationTask":
        if self.atom_vocab and list(self.atom_vocab) != self.atom_decoder:
            msg = (
                "data.atom_vocab must equal the task's atom_decoder EXACTLY, "
                "including order -- the one-hot column is position-in-vocab, "
                "so a permutation trains and samples a silently wrong model.\n"
                f"  data.atom_vocab   = {list(self.atom_vocab)}\n"
                f"  tasks.atom_decoder = {self.atom_decoder}"
            )
            raise ValueError(msg)

        sidecar = load_sidecar(self.cond_path, self.conf_path, self.meta_path)
        if sidecar is not None:
            expected = sum(self.in_condition_size)
            if sidecar.cond_dim != expected:
                msg = (
                    f"sidecar condition width {sidecar.cond_dim} != "
                    f"tasks.in_condition_size sum {expected}. The converter "
                    "and the task config disagree about the spectral grids."
                )
                raise ValueError(msg)
            if sidecar.max_n_atoms < self.max_n_atoms:
                logger.warning(
                    "[chefnmr] sidecar conformers are padded to %d atoms but "
                    "tasks.max_n_atoms is %d; molecules are still bounded by "
                    "data.max_atom, so this is only wasted padding.",
                    sidecar.max_n_atoms,
                    self.max_n_atoms,
                )
            logger.info(
                "[chefnmr] sidecar: %d rows, condition width %d, up to %d "
                "conformers, split %r",
                sidecar.n_rows,
                sidecar.cond_dim,
                sidecar.conf.shape[1],
                sidecar.meta.get("split"),
            )

        # `cli/train.py` reaches for `factory.task` right after `build()`
        # (to stamp node_feature_dim), so the factory must keep it.
        self.task = ChefNMRElucidationTask(
            task_type=self.task_type,
            atom_decoder=self.atom_decoder,
            max_n_atoms=self.max_n_atoms,
            sigma_data=self.sigma_data,
            condition_type=self.condition_type,
            in_condition_size=self.in_condition_size,
            score_model_args=self.score_model_args,
            diffusion_process_args=self.diffusion_process_args,
            diffusion_loss_args=self.diffusion_loss_args,
            multitask_args=self.multitask_args,
            max_n_conformers=self.max_n_conformers,
            diffusion_multiplicity=self.diffusion_multiplicity,
            num_sampling_steps=self.num_sampling_steps,
            guidance_scale=self.guidance_scale,
            sidecar=sidecar,
        )
        return self.task


# --------------------------------------------------------------------- #
# task                                                                  #
# --------------------------------------------------------------------- #
class ChefNMRElucidationTask(nn.Module):
    """Task-contract implementation for ChefNMR.

    The class name is load-bearing: ``elucidation_generator._TASK_TO_GENERATOR``
    keys on it to find :class:`ChefNMRElucidationGenerator`.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type: str,
        atom_decoder: List[str],
        max_n_atoms: int,
        sigma_data: float,
        condition_type: str,
        in_condition_size: List[int],
        score_model_args: dict,
        diffusion_process_args: dict,
        diffusion_loss_args: dict,
        multitask_args: dict,
        max_n_conformers: int,
        diffusion_multiplicity: int,
        num_sampling_steps: int,
        guidance_scale: float,
        sidecar: Optional[ChefNMRSidecar],
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.atom_decoder = list(atom_decoder)
        # `atom_vocab` is a plain attribute the CLI stamps and reads; ours is
        # always the decoder, in decoder order.
        self.atom_vocab = list(atom_decoder)
        self.max_n_atoms = int(max_n_atoms)
        self.condition_type = condition_type
        self.in_condition_size = list(in_condition_size)
        self.hnmr_dim, self.cnmr_dim = self.in_condition_size
        self.max_n_conformers = int(max_n_conformers)
        self.diffusion_multiplicity = int(diffusion_multiplicity)
        self.num_sampling_steps = int(num_sampling_steps)
        self.guidance_scale = float(guidance_scale)

        self.p_drop_h1nmr = float(multitask_args.get("p_drop_h1nmr", 0.0))
        self.p_drop_c13nmr = float(multitask_args.get("p_drop_c13nmr", 0.0))
        self.p_drop_both = float(multitask_args.get("p_drop_both", 0.0))
        p_sum = self.p_drop_h1nmr + self.p_drop_c13nmr + self.p_drop_both
        if p_sum > 1.0:
            msg = f"modality dropout probabilities sum to {p_sum} > 1.0"
            raise ValueError(msg)
        self._p_drop = [
            self.p_drop_h1nmr,
            self.p_drop_c13nmr,
            self.p_drop_both,
            1.0 - p_sum,
        ]

        self.add_smooth_lddt_loss = bool(
            diffusion_loss_args.get("add_smooth_lddt_loss", True)
        )
        self.lddt_loss_threshold = list(
            diffusion_loss_args.get("lddt_loss_threshold", [0.5, 1.0, 2.0, 4.0])
        )

        process_args = dict(diffusion_process_args)
        edm_args = dict(process_args.pop("edm_args", None) or {})
        edm_args["sigma_data"] = float(sigma_data)
        process_args["num_sampling_steps"] = self.num_sampling_steps
        process_args["guidance_scale"] = self.guidance_scale

        model_args = dict(score_model_args)
        model_args.update(
            in_atom_feature_size=len(self.atom_decoder),
            out_atom_coords_size=3,
            condition=self.condition_type,
            in_condition_size=self.in_condition_size,
            max_n_atoms=self.max_n_atoms,
            drop_transform=multitask_args.get("drop_transform", "zero"),
        )

        # Named `model` so the state_dict keys are `model.score_model.*` --
        # byte-identical to upstream's Lightning module, which is what lets
        # scripts/convert_checkpoint.py copy keys across without a remap.
        self.model = AtomDiffusion(
            score_model_args=model_args,
            edm_args=edm_args,
            **process_args,
        )

        self._sidecar = sidecar
        self.prop_dist_model = None

    # -- required properties ------------------------------------------ #
    @property
    def device(self) -> torch.device:
        """Defined on purpose: ``cli/generate.py`` skips its own device move
        when a task has this, and the elucidation seam does ``task.to(...)``."""
        return next(self.parameters()).device

    # -- batch adaptation ---------------------------------------------- #
    def _sidecar_or_raise(self) -> ChefNMRSidecar:
        if self._sidecar is None:
            msg = (
                "ChefNMR training needs the sidecar arrays. Set "
                "tasks.cond_path / tasks.conf_path / tasks.meta_path to the "
                "files written by "
                "docs/model_integrations/chefnmr/scripts/convert_dataset.py."
            )
            raise ValueError(msg)
        return self._sidecar

    def _condition_from_batch(self, keys: Sequence[Any]) -> torch.Tensor:
        side = self._sidecar_or_raise()
        rows = [parse_row_index(k, side.n_rows) for k in keys]
        cond = np.asarray(side.cond[rows], dtype=np.float32)
        return torch.from_numpy(cond)

    def _target_coords(
        self, keys: Sequence[Any], n_atoms_axis: int
    ) -> torch.Tensor:
        """Ground-truth coordinates for this batch, from the conformer stack.

        Upstream draws a *random* conformer per epoch (``+aug_conf=conf3``,
        ``input_generator.py:255``) -- that augmentation is reproduced here.
        In eval mode conformer 0 is used instead, so validation loss is
        comparable between epochs; upstream randomises there too, which just
        adds noise to a number nobody optimises against.
        """
        side = self._sidecar_or_raise()
        rows = [parse_row_index(k, side.n_rows) for k in keys]
        max_c = min(side.conf.shape[1], self.max_n_conformers)
        picked = np.empty((len(rows), n_atoms_axis, 3), dtype=np.float32)
        for b, row in enumerate(rows):
            available = min(int(side.n_conf[row]), max_c)
            k = np.random.randint(available) if (self.training and available > 1) else 0
            picked[b] = np.asarray(
                side.conf[row, k, :n_atoms_axis], dtype=np.float32
            )
        return torch.from_numpy(picked)

    def _apply_modality_dropout(self, condition: torch.Tensor) -> torch.Tensor:
        """Zero the 1H and/or 13C slice, per ``input_generator.py:320-339``.

        This is what trains the classifier-free unconditional branch, so it
        is not optional decoration. It lives here rather than in the data
        layer because the condition first exists here.
        """
        if not self.training or self._p_drop[3] >= 1.0:
            return condition
        actions = np.random.choice(4, size=condition.shape[0], p=self._p_drop)
        drop_h = torch.from_numpy(np.isin(actions, (0, 2))).to(condition.device)
        drop_c = torch.from_numpy(np.isin(actions, (1, 2))).to(condition.device)
        condition = condition.clone()
        condition[drop_h, : self.hnmr_dim] = 0.0
        condition[drop_c, self.hnmr_dim :] = 0.0
        return condition

    def _adapt(self, batch: Dict[str, Any]):
        coords = batch["coords"]
        node_mask = batch["node_mask"]
        if node_mask.dim() == 3:  # (B, N, 1) -> (B, N)
            node_mask = node_mask.squeeze(-1)
        atom_mask = node_mask.to(coords.dtype)
        n_axis = coords.shape[1]
        if int(atom_mask.sum(dim=1).max().item()) != n_axis:
            msg = (
                "the batch's atom axis is not a tight prefix of real atoms "
                f"({n_axis} columns, max {int(atom_mask.sum(1).max())} real). "
                "Set data.data_efficient_collator: false -- the VRAM collate "
                "drops and duplicates batch elements, which breaks the "
                "row-index join to the sidecar."
            )
            raise ValueError(msg)

        keys = batch.get("xyz")
        if keys is None:
            msg = "batch has no 'xyz' key; ChefNMR joins its sidecar on it."
            raise ValueError(msg)

        condition = self._condition_from_batch(keys).to(coords.device)
        condition = self._apply_modality_dropout(condition)

        model_inputs = {
            "atom_mask": atom_mask,
            "atom_one_hot": batch["node_feature"].to(coords.dtype),
            "condition": condition,
        }
        target = self._target_coords(keys, n_axis).to(coords.device, coords.dtype)
        return model_inputs, target

    # -- task contract -------------------------------------------------- #
    def forward(self, batch: Dict[str, Any]):
        model_inputs, target = self._adapt(batch)
        dict_out = self.model(
            model_inputs=model_inputs,
            atom_coords=target,
            multiplicity=self.diffusion_multiplicity,
        )
        out = self.model.compute_loss(
            model_inputs=model_inputs,
            dict_out=dict_out,
            multiplicity=self.diffusion_multiplicity,
            add_smooth_lddt_loss=self.add_smooth_lddt_loss,
            lddt_loss_threshold=self.lddt_loss_threshold,
        )
        stats = {
            "mse_loss": out["loss_breakdown"]["mse_loss"].detach(),
            "lddt_loss": out["loss_breakdown"]["smooth_lddt_loss"].detach(),
        }
        return out["loss"], stats

    def predict_and_target(self, batch: Dict[str, Any]):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # -- generation ----------------------------------------------------- #
    @torch.no_grad()
    def elucidate(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """One tiled measurement -> ``n`` candidate geometries.

        ``batch`` is what :meth:`ChefNMRElucidationGenerator._repeat` built:
        ``atom_mask (n, N)``, ``atom_one_hot (n, N, T)``,
        ``condition (n, cond_dim)``. ``num_steps``/``guidance_scale`` of
        ``None`` mean "the model's own default", per the shared seam.
        """
        model_inputs = {
            "atom_mask": batch["atom_mask"],
            "atom_one_hot": batch["atom_one_hot"],
            "condition": batch["condition"],
        }
        coords, _chains = self.model.sample(
            model_inputs=model_inputs,
            num_sampling_steps=num_steps,
            multiplicity=1,  # `_repeat` already tiled; k is a batch dim
            n_chain_frames=1,
            guidance_scale=guidance_scale,
        )
        return {
            "coords": coords,
            "atom_mask": model_inputs["atom_mask"],
            "atom_one_hot": model_inputs["atom_one_hot"],
        }


# --------------------------------------------------------------------- #
# elucidation generator                                                 #
# --------------------------------------------------------------------- #
@dataclass
class ChefNMRRecord:
    """One measured spectrum plus the formula that goes with it.

    Two sources fill this, and only ``cond`` tells them apart. A converted
    benchmark corpus leaves it ``None`` and the condition is read from the
    memmap at ``row_index``; an unknown-spectra file carries the condition
    inline and sets ``row_index`` to -1, because there is no corpus row.
    """

    row_index: int
    name: str
    symbols: List[str]
    n_atoms: int
    smiles: Optional[str] = None
    #: The condition vector, when it came from an unknown-spectra file
    #: rather than from a sidecar memmap.
    cond: Optional[np.ndarray] = None


class ChefNMRElucidationGenerator(ElucidationGenerator):
    """Walk a set of NMR measurements; emit ranked 3D candidates per record.

    ``spectra_source`` takes either input, told apart by what the path IS
    (see :meth:`_records`), never by a mode flag:

    * a **converted benchmark corpus** -- ``<prefix>.db`` plus its four
      sidecars. Carries the answer, so top-k and Tanimoto are reported.
      This is how published numbers are reproduced.
    * a **bare unknown-spectra file** -- one small JSON holding, per
      unknown, a name, a molecular formula and the peaks. No coordinates,
      no SMILES. This is the real use case; see
      :mod:`MolecularDiffusion.modules.models.chefnmr.unknown`.

    Rides the shared seam unchanged: no ``run()`` loop here, no ``_rank``
    override (upstream's top-k is over the *first k drawn samples*
    -- ``model.py:395-398`` -- so generation order is the correct default,
    not a lazy one), and ``_sample_kwargs`` is the base's, because
    ``num_steps`` and ``guidance_scale`` are the only knobs
    ``elucidate()`` takes.
    """

    tag = "chefnmr"
    source_key = "spectra_source"
    source_required_msg = (
        "chefnmr needs `spectra_source`, which is either:\n"
        "  (a) a JSON file of UNKNOWN spectra -- per record a name, a "
        "molecular formula and the peaks. This is the real use case: you "
        "need the spectrum and the composition, and nothing else. See "
        "configs/chefnmr_unknown_example.json.\n"
        "  (b) the prefix of a converted benchmark corpus written by "
        "docs/model_integrations/chefnmr/scripts/convert_dataset.py -- point "
        "it at the .db and the _cond/_conf/_nconf/_meta sidecars are found "
        "beside it. Use this to reproduce published numbers; it can only "
        "describe molecules whose structure you already have."
    )
    supports_guidance = True
    #: The two halves of the condition vector, named for the nuclei they
    #: carry. These names are ChefNMR's, not the seam's: the shared
    #: `drop_channels` key takes whatever a model declares here, so a
    #: mass-spectrometry rider declares its own (or none) and nothing
    #: spectroscopic leaks into the base. `_condition` does the zeroing.
    maskable_channels = ("1H", "13C")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # `guidance_scale` and `drop_channels` are deliberately NOT
        # re-declared here: both are the base's, validated there.
        super().__init__(*args, **kwargs)
        self._sidecar: Optional[ChefNMRSidecar] = None
        self._drawn = 0
        self._decoded = 0
        # Every geometry drawn for the CURRENT record, bond perception or
        # not. See _write_record for why the failures are kept.
        self._draws: List[tuple] = []

    # -- corpus ---------------------------------------------------------- #
    @staticmethod
    def _paths(source: str):
        prefix = source[:-3] if source.endswith(".db") else source
        return (
            f"{prefix}.db",
            f"{prefix}_cond.npy",
            f"{prefix}_conf.npy",
            f"{prefix}_meta.json",
        )

    def _records(self) -> Sequence[ChefNMRRecord]:
        from ase.db import connect  # noqa: PLC0415

        db_path, cond_path, conf_path, meta_path = self._paths(self.spectra_source)
        if not os.path.exists(db_path):
            # Decided by what the path IS, not by a mode flag: no <prefix>.db
            # means this is not a converted corpus, so it must be the bare
            # unknown-spectra file. `load_sidecar` is never reached, which is
            # the whole point -- _cond/_conf/_nconf/_meta describe a molecule
            # you already know, and an unknown has none of them.
            return self._unknown_records(db_path)
        self._sidecar = load_sidecar(cond_path, conf_path, meta_path)
        decoder = list(getattr(self.task, "atom_decoder", CHEFNMR_ATOM_DECODER))
        stored = list(self._sidecar.meta.get("atom_decoder") or decoder)
        if stored != decoder:
            msg = (
                "the corpus was converted against a different atom_decoder "
                f"than the loaded model uses:\n  corpus = {stored}\n"
                f"  model  = {decoder}\nThe one-hot column is "
                "position-in-vocab, so this is a silent wrong answer, not a "
                "shape error."
            )
            raise ValueError(msg)

        records: List[ChefNMRRecord] = []
        with connect(db_path) as db:
            for i, row in enumerate(db.select()):
                data = getattr(row, "data", {}) or {}
                if self.split and str(data.get("split", "")) != str(self.split):
                    continue
                symbols = list(row.symbols)
                records.append(
                    ChefNMRRecord(
                        row_index=i,
                        name=f"mol_{data.get('mol_idx', i)}",
                        symbols=symbols,
                        n_atoms=len(symbols),
                        smiles=data.get("smiles"),
                    )
                )
        if not records:
            msg = (
                f"no records in {db_path}"
                + (f" for split={self.split!r}" if self.split else "")
                + ". The converter reports each fold's size; point `split` at "
                "one that exists, or leave it null."
            )
            raise ValueError(msg)
        return records

    def _channels(self) -> List[Channel]:
        """The condition vector's layout, read off the LOADED checkpoint.

        Which nuclei this model has a branch for, how wide each slice is,
        and whether a slice is a binary occupancy indicator are all
        properties of the weights, not of the input file. The `embed`
        tokenizer is ``x.long() * arange(1, L+1)`` into an
        ``nn.Embedding(L+1, D, padding_idx=0)``, so its arithmetic is only
        correct on {0, 1} -- reading it here is what stops a peak list
        being binned as intensities onto a channel that holds occupancy.
        The order matches ``_separate_spectra_components``
        (``embedders.py``): 1H first, then 13C.
        """
        embedder = self.task.model.score_model.y_embedder
        return [
            Channel(
                "1H",
                int(embedder.hnmr_dim),
                bool(embedder.use_hnmr),
                embedder.h_tokenizer == "embed",
            ),
            Channel(
                "13C",
                int(embedder.cnmr_dim),
                bool(embedder.use_cnmr),
                embedder.c_tokenizer == "embed",
            ),
        ]

    def _unknown_records(self, db_path: str) -> List[ChefNMRRecord]:
        """Read a bare unknown-spectra file: spectrum + formula, no more.

        The two inputs generation actually needs. `smiles` is optional, so
        `_reference` returns None for a genuine unknown and the shared seam
        then skips scoring and writes no metrics.json -- that path is
        already in the base and is used, not reimplemented.
        """
        source = self.spectra_source
        if not os.path.isfile(source):
            msg = (
                f"chefnmr found neither a converted corpus nor an "
                f"unknown-spectra file at {source!r}. `spectra_source` is "
                f"one of two things:\n"
                f"  (a) the PREFIX of a converted benchmark corpus -- "
                f"{db_path} plus its _cond/_conf/_nconf/_meta sidecars, "
                "written by "
                "docs/model_integrations/chefnmr/scripts/convert_dataset.py. "
                "Use this to reproduce published numbers; it needs the "
                "answer up front.\n"
                "  (b) a JSON file of UNKNOWN spectra -- per record a name, "
                "a molecular formula and the peaks, and nothing else. Use "
                "this to elucidate a real unknown. See "
                "configs/chefnmr_unknown_example.json for a worked file and "
                "MolecularDiffusion.modules.models.chefnmr.unknown for the "
                "format."
            )
            raise ValueError(msg)

        decoder = list(
            getattr(self.task, "atom_decoder", CHEFNMR_ATOM_DECODER)
        )
        entries = read_unknown_spectra(
            source,
            channels=self._channels(),
            decoder=decoder,
            max_n_atoms=int(getattr(self.task, "max_n_atoms", 101)),
        )
        labelled = sum(1 for e in entries if e["smiles"])
        print(
            f"[{self.tag}] {len(entries)} unknown(s) from {source}; "
            f"{labelled} carry a reference SMILES"
            + (
                ""
                if labelled
                else " -- none, so no metrics.json will be written"
            )
            + "."
        )
        return [
            ChefNMRRecord(
                row_index=-1,
                name=entry["name"],
                symbols=entry["symbols"],
                n_atoms=len(entry["symbols"]),
                smiles=entry["smiles"],
                cond=entry["cond"],
            )
            for entry in entries
        ]

    def _start(self, record: ChefNMRRecord, index: int, total: int) -> None:
        self._draws = []
        super()._start(record, index, total)

    def _condition(self, record: ChefNMRRecord) -> np.ndarray:
        if record.cond is not None:
            cond = np.array(record.cond, dtype=np.float32)
        else:
            cond = np.array(
                self._sidecar.cond[record.row_index], dtype=np.float32
            )
        h_dim = int(getattr(self.task, "hnmr_dim", 10000))
        if "1H" in self.drop_channels:
            cond[:h_dim] = 0.0
        if "13C" in self.drop_channels:
            cond[h_dim:] = 0.0
        if not cond.any():
            msg = (
                f"record {record.name} has an all-zero condition. That is the "
                "classifier-free UNCONDITIONAL branch, so the model would emit "
                "a plausible molecule of the right formula unrelated to any "
                "spectrum. Check `drop_channels` and the corpus."
            )
            raise ValueError(msg)
        return cond

    def _priors(self, record: ChefNMRRecord) -> np.ndarray:
        """The known chemical formula, as a per-atom one-hot ``(n_atoms, T)``."""
        decoder = list(getattr(self.task, "atom_decoder", CHEFNMR_ATOM_DECODER))
        index = {s: i for i, s in enumerate(decoder)}
        one_hot = np.zeros((record.n_atoms, len(decoder)), dtype=np.float32)
        for a, symbol in enumerate(record.symbols):
            if symbol not in index:
                msg = (
                    f"record {record.name} contains {symbol!r}, which is not "
                    f"in the model's atom_decoder {decoder}."
                )
                raise ValueError(msg)
            one_hot[a, index[symbol]] = 1.0
        return one_hot

    def _repeat(self, cond: np.ndarray, priors: np.ndarray, n: int) -> dict:
        """Tile one measurement to ``n`` candidates.

        # ponytail: the atom axis is exactly this molecule's atom count, not
        # the dataset-wide max_n_atoms upstream pads to. Padding columns are
        # provably inert -- the attention mask removes them as *keys* and the
        # `x * padded_atom_mask` at both ends removes them as outputs, and
        # there is no positional encoding on the atom axis -- so padding to
        # 101 would only cost (101/n)^2 attention. Reinstate padding only if
        # a future variant reads pos_embed.
        """
        n_atoms = priors.shape[0]
        return {
            "atom_mask": torch.ones(n, n_atoms, dtype=torch.float32),
            "atom_one_hot": torch.from_numpy(priors).unsqueeze(0).repeat(n, 1, 1),
            "condition": torch.from_numpy(cond).unsqueeze(0).repeat(n, 1),
        }

    # -- decoding -------------------------------------------------------- #
    def _decode(self, raw: Dict[str, torch.Tensor]) -> List[Candidate]:
        from rdkit import Chem, RDLogger  # noqa: PLC0415
        from rdkit.Chem import rdDetermineBonds  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        decoder = list(getattr(self.task, "atom_decoder", CHEFNMR_ATOM_DECODER))
        coords = raw["coords"].detach().cpu().numpy()
        masks = raw["atom_mask"].detach().cpu().numpy()
        one_hots = raw["atom_one_hot"].detach().cpu().numpy()

        out: List[Candidate] = []
        for b in range(coords.shape[0]):
            self._drawn += 1
            keep = masks[b] > 0
            xyz = coords[b][keep].astype(np.float64)
            types = one_hots[b][keep].argmax(axis=-1)
            symbols = [decoder[int(t)] for t in types]
            mol = self._to_rdkit(Chem, symbols, xyz)
            try:
                rdDetermineBonds.DetermineBonds(mol)
                smiles = self._smiles_from_mol(Chem, mol)
            except Exception:  # noqa: BLE001 - bad geometry is expected
                smiles = None
            self._draws.append((symbols, xyz, smiles))
            if smiles is None:
                out.append(Candidate(smiles=""))
                continue
            self._decoded += 1
            # `mol` is H-suppressed on purpose: the shared _write_metrics
            # fingerprints `cand.mol`, and a Morgan FP over explicit
            # hydrogens would not compare against the H-suppressed
            # reference. It keeps its 3D conformer, so candidates.sdf is
            # still real geometry; the full all-atom coordinates (H
            # included) survive in `coords` and in the .xyz files below.
            mol_no_h = Chem.RemoveHs(mol)
            Chem.RemoveStereochemistry(mol_no_h)
            out.append(Candidate(smiles=smiles, mol=mol_no_h, coords=xyz))
        return out

    def _write_record(
        self, name: str, candidates: List[Candidate], reference: Any
    ) -> None:
        """The base's writers, plus one ``.xyz`` per candidate.

        ChefNMR is coordinate-first and its output is an all-atom geometry
        including hydrogens; ``candidates.sdf`` carries only the H-suppressed
        molecule (see ``_decode``). ``.xyz`` is also what the platform's
        `analyze` tooling and the smoke-test checker read, so writing it is
        not a nicety. Subclass-local: nothing about it belongs in a seam a
        2D elucidation model has to ride.
        """
        super()._write_record(name, candidates, reference)
        import os  # noqa: PLC0415

        directory = os.path.join(self.output_path, name)
        rank = 0
        for i, (symbols, coords, smiles) in enumerate(self._draws):
            # EVERY draw is written, including the ones RDKit could not read
            # bonds off. The model produced that geometry; only the bond
            # perception failed, and an undertrained checkpoint fails all of
            # them -- discarding those would leave a run with no evidence it
            # sampled anything at all. Rank matches ranking.csv, because
            # _accept preserves order and _rank is the identity.
            if smiles is not None:
                rank += 1
            label = f"rank={rank}" if smiles is not None else "rank=-"
            with open(os.path.join(directory, f"draw_{i:03d}.xyz"), "w") as handle:
                handle.write(f"{len(symbols)}\n")
                handle.write(f"{name} {label} smiles={smiles or 'none'}\n")
                for symbol, (x, y, z) in zip(symbols, coords):
                    handle.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")

    @staticmethod
    def _to_rdkit(chem, symbols: List[str], xyz: np.ndarray):
        """Bondless ``RWMol`` with a 3D conformer (``src/data/utils.py:161-191``)."""
        mol = chem.RWMol()
        for symbol in symbols:
            mol.AddAtom(chem.Atom(symbol))
        conf = chem.Conformer(len(xyz))
        for i, pos in enumerate(xyz):
            conf.SetAtomPosition(i, pos)  # must be float64
        mol.AddConformer(conf)
        return mol

    @staticmethod
    def _smiles_from_mol(chem, mol) -> Optional[str]:
        """``src/evaluation/bond_analyzer.py:12-44``, verbatim in behaviour.

        A fragmented result is **not** a hit: upstream returns ``None`` for
        the full SMILES and only reports the largest fragment separately, and
        counts exactly that as invalid. Keeping the fragment here would
        inflate top-k.
        """
        try:
            mol_no_h = chem.RemoveHs(mol)
            chem.RemoveStereochemistry(mol_no_h)
            full = chem.CanonSmiles(chem.MolToSmiles(mol_no_h))
            fragments = chem.rdmolops.GetMolFrags(
                mol_no_h, asMols=True, sanitizeFrags=True
            )
            if len(fragments) > 1:
                return None
            return full
        except Exception:  # noqa: BLE001
            return None

    def _reference(self, record: ChefNMRRecord) -> Optional[str]:
        return canonical_smiles(record.smiles)

    def _summary(self, written: int, attempts: int) -> None:
        rate = self._decoded / self._drawn if self._drawn else 0.0
        print(
            f"[{self.tag}] wrote {written} records to {self.output_path}; "
            f"bond perception succeeded on {self._decoded}/{self._drawn} "
            f"draws ({rate:.1%}) -- that rate is the ceiling on every top-k "
            f"number below."
        )
