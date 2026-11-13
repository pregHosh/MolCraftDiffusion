"""DiffDec: pocket-aware scaffold decoration with an end-to-end diffusion model.

DiffDec (https://github.com/biomed-AI/DiffDec, Xie et al., *J. Chem. Inf.
Model.* 64(7) 2554-2564, 2024, doi:10.1021/acs.jcim.3c01466) grows an R-group
off a named anchor atom of a **fixed 3D scaffold**, inside a **fixed protein
pocket**. Scaffold and pocket atoms are never noised; only the R-group rows
are diffused. See docs/model_integrations/diffdec/INTEGRATION_PLAN.md for the
full integration plan (scope, data adapters, task-contract mapping).

**No new model package.** DiffDec is a fork of DiffLinker, and its
``src/edm_single.py``, ``src/noise.py`` and ``src/egnn.py`` are the ported
``MolecularDiffusion.modules.models.difflinker`` modules modulo formatting and
a ``fragment``/``linker`` -> ``scaffold``/``rgroup`` rename (verified by
normalized diff -- INTEGRATION_PLAN.md "Repo Inspection"). So this task reuses
``modules.models.difflinker`` rather than copying a fourth near-identical
EDM/EGNN into the tree. That package is **frozen ported upstream code now
shared by two tasks** (``diffusion_difflinker.py`` and this one) -- changing it
changes both.

The mask-name mapping across that seam is::

    DiffDec                     difflinker.EDM / Dynamics
    ---------------------------------------------------------
    scaffold_mask (scaf+pocket) fragment_mask   (never noised)
    rgroup_mask                 linker_mask     (diffused)
    context[..., -2] scaffold_only_mask         fragment_only_mask
    context[..., -1] pocket_only_mask           pocket_only_mask

``DynamicsWithPockets`` is used with ``graph_type="4A"``, i.e. a 4 A radius
graph rebuilt each forward pass, which is exactly DiffDec's
``get_dist_edges`` (egnn.py l. 531-539).

Generation goes through :class:`DiffDecScaffoldGenerator`, not through
``GenerativeFactory``: DiffDec has no unconditional mode, and
``sample(batch_size, nodesxsample)`` has no channel for "which scaffold, which
pocket, which anchor". That is the established pattern for every
pocket-conditioned model here (``DiffSBDDPocketGenerator``,
``gen_kgdiff_pocket.yaml``, ``gen_pmdm_pocket.yaml``).
"""

from __future__ import annotations

import logging
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from MolecularDiffusion.device import get_device
import torch.nn as nn

from MolecularDiffusion.data.integrations.diffdec_data import (
    DIFFDEC_ATOM_VOCAB,
    FAKE_ATOM_INDEX,
    DiffDecDataset,
    create_templates_for_rgroup_generation_single,
    diffdec_collate,
)
from MolecularDiffusion.modules.models.difflinker import utils as dl_utils
from MolecularDiffusion.modules.models.difflinker.edm import EDM
from MolecularDiffusion.modules.models.difflinker.egnn import DynamicsWithPockets
from MolecularDiffusion.modules.models.difflinker.linker_size import DistributionNodes

logger = logging.getLogger(__name__)


class DiffDecTaskFactory:
    """Factory matching ``cli/train.py``'s ``task_module.build()`` /
    ``task_module.task`` instantiation pattern (precedent:
    ``diffusion_difflinker.py::DiffLinkerTaskFactory``).

    Defaults follow DiffDec ``configs/single.yml`` + ``train_single.py``'s
    argparse defaults, cross-checked against the released
    ``diffdec_single.ckpt``'s own ``hyper_parameters``.
    """

    def __init__(
        self,
        task_type: str = "diffusion_diffdec",
        in_node_nf: int = 10,
        n_dims: int = 3,
        hidden_nf: int = 128,
        activation: str = "silu",
        tanh: bool = False,
        n_layers: int = 6,
        attention: bool = False,
        norm_constant: float = 1e-6,
        inv_sublayers: int = 2,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        model: str = "egnn_dynamics",
        normalization: Optional[str] = "batch_norm",
        condition_time: bool = True,
        anchors_context: bool = True,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "polynomial_2",
        diffusion_noise_precision: float = 1e-5,
        diffusion_loss_type: str = "l2",
        normalize_factors: tuple = (1, 4, 10),
        center_of_mass: str = "anchors",
        atom_vocab: Optional[list] = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.tanh = tanh
        self.n_layers = n_layers
        self.attention = attention
        self.norm_constant = norm_constant
        self.inv_sublayers = inv_sublayers
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.model = model
        self.normalization = normalization
        self.condition_time = condition_time
        self.anchors_context = anchors_context
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.diffusion_noise_precision = diffusion_noise_precision
        self.diffusion_loss_type = diffusion_loss_type
        self.normalize_factors = tuple(normalize_factors)
        self.center_of_mass = center_of_mass
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab")
        self.kwargs = kwargs
        self.task: Optional[DiffDecTask] = None

    def build(self) -> "DiffDecTask":
        self.task = DiffDecTask(
            in_node_nf=self.in_node_nf,
            n_dims=self.n_dims,
            hidden_nf=self.hidden_nf,
            activation=self.activation,
            tanh=self.tanh,
            n_layers=self.n_layers,
            attention=self.attention,
            norm_constant=self.norm_constant,
            inv_sublayers=self.inv_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
            model=self.model,
            normalization=self.normalization,
            condition_time=self.condition_time,
            anchors_context=self.anchors_context,
            diffusion_steps=self.diffusion_steps,
            diffusion_noise_schedule=self.diffusion_noise_schedule,
            diffusion_noise_precision=self.diffusion_noise_precision,
            diffusion_loss_type=self.diffusion_loss_type,
            normalize_factors=self.normalize_factors,
            center_of_mass=self.center_of_mass,
            atom_vocab=self.atom_vocab,
        )
        return self.task


class DiffDecTask(nn.Module):
    """Plain ``nn.Module`` task wrapper (TABASCO-style -- see
    docs/adding_new_models.md §2.6) implementing the §2.1 contract around
    DiffLinker's ``EDM`` driven by DiffDec's mask/context convention.

    ``self.edm`` is deliberately named to match upstream's own attribute path
    (``DDPM.edm``), so the released ``diffdec_single.ckpt``'s ``edm.*`` keys
    land without renaming -- see
    ``docs/model_integrations/diffdec/scripts/convert_checkpoint.py``.
    """

    def __init__(
        self,
        in_node_nf: int,
        n_dims: int,
        hidden_nf: int,
        activation: str,
        tanh: bool,
        n_layers: int,
        attention: bool,
        norm_constant: float,
        inv_sublayers: int,
        sin_embedding: bool,
        normalization_factor: float,
        aggregation_method: str,
        model: str,
        normalization: Optional[str],
        condition_time: bool,
        anchors_context: bool,
        diffusion_steps: int,
        diffusion_noise_schedule: str,
        diffusion_noise_precision: float,
        diffusion_loss_type: str,
        normalize_factors: tuple,
        center_of_mass: str,
        atom_vocab: Optional[list] = None,
    ) -> None:
        super().__init__()

        if activation != "silu":
            raise NotImplementedError(
                f"activation={activation!r}: upstream DiffDec supports only "
                "'silu' (model_single.py::get_activation)."
            )
        act = nn.SiLU()

        # context = [anchors, scaffold_only_mask, pocket_only_mask] -> 3, or
        # [scaffold_only_mask, pocket_only_mask] -> 2 without anchors.
        # train_single.py l. 51-54: `2 if anchors_context else 1`, then `+= 1`
        # for the pocket column (always taken on the '.full' pocket data).
        context_node_nf = 3 if anchors_context else 2

        dynamics = DynamicsWithPockets(
            n_dims=n_dims,
            in_node_nf=in_node_nf,
            context_node_nf=context_node_nf,
            hidden_nf=hidden_nf,
            device="cpu",
            activation=act,
            n_layers=n_layers,
            attention=attention,
            condition_time=condition_time,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            # upstream passes `centering=inpainting`, and single.yml /
            # train_single.py leave inpainting False.
            centering=False,
            # DiffDec's get_dist_edges is DiffLinker's 4 A radius graph.
            graph_type="4A",
        )
        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=tuple(normalize_factors),
        )

        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.anchors_context = anchors_context
        self.center_of_mass = center_of_mass
        self.loss_type = diffusion_loss_type
        self.atom_vocab = list(atom_vocab) if atom_vocab else list(DIFFDEC_ATOM_VOCAB)

        # Read generically off `task.model` by cli/generate.py and
        # runmodes/generate/tasks_generate.py (duck-typed).
        self.norm_values = tuple(normalize_factors)
        self.ndim_extra = 0
        self.prop_dist_model = None

        # ponytail: total-atom-count histogram accumulated lazily from real
        # forward() batches instead of a `train_set`-at-build seam
        # (docs/adding_new_models.md §2.5) -- copied from
        # diffusion_difflinker.py, where it is the proven pattern. It exists
        # to satisfy the §2.1 contract and the edm_stat.pkl stamping only:
        # DiffDec's R-group size is the FIXED 10-slot budget, never drawn from
        # this distribution.
        self._atom_count_counts: Counter = Counter()
        self._node_dist_cache: Optional[DistributionNodes] = None
        self._node_dist_override: Optional[DistributionNodes] = None
        self._n_node_dist_override: Optional[dict] = None

    # ---------------------------------------------------------------- state
    def get_extra_state(self) -> dict:
        return {"atom_count_counts": dict(self._atom_count_counts)}

    def set_extra_state(self, state: dict) -> None:
        counts = state.get("atom_count_counts") if isinstance(state, dict) else None
        if counts:
            self._atom_count_counts = Counter(counts)
            self._node_dist_cache = None

    # -------------------------------------------------------------- forward
    def _context_and_com(self, batch: Dict[str, Any]):
        """Build the 3-column context and pick the centre-of-mass mask.

        Port of ``model_single.py::DDPM.forward`` l. 145-168. ``pocket_only``
        is derived as ``scaffold_mask - scaffold_only_mask`` exactly as
        upstream does, rather than read from ``pocket_mask``, so the identity
        ``scaffold_only | pocket_only | rgroup == atom_mask`` that
        ``DynamicsWithPockets.forward`` asserts holds by construction.
        """
        anchors = batch["anchors"]
        scaffold_mask = batch["scaffold_mask"]
        scaffold_only_mask = batch["scaffold_only_mask"]
        pocket_only_mask = scaffold_mask - scaffold_only_mask

        if self.anchors_context:
            context = torch.cat(
                [anchors, scaffold_only_mask, pocket_only_mask], dim=-1
            )
        else:
            context = torch.cat([scaffold_only_mask, pocket_only_mask], dim=-1)

        if self.center_of_mass == "scaffold":
            com_mask = scaffold_only_mask
        elif self.center_of_mass == "anchors":
            com_mask = anchors
        else:
            raise NotImplementedError(
                f"center_of_mass={self.center_of_mass!r} (expected "
                "'scaffold' or 'anchors')"
            )
        return context, com_mask

    def forward(self, batch: Dict[str, Any]):
        x = batch["positions"]
        h = batch["one_hot"]
        node_mask = batch["atom_mask"]
        edge_mask = batch["edge_mask"]
        scaffold_mask = batch["scaffold_mask"]
        rgroup_mask = batch["rgroup_mask"]

        context, com_mask = self._context_and_com(batch)
        x = dl_utils.remove_partial_mean_with_mask(x, node_mask, com_mask)

        (
            delta_log_px,
            kl_prior,
            loss_term_t,
            loss_term_0,
            l2_loss,
            noise_t,
            noise_0,
        ) = self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=scaffold_mask,
            linker_mask=rgroup_mask,
            edge_mask=edge_mask,
            context=context,
        )

        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        if self.loss_type == "l2":
            loss = l2_loss
        elif self.loss_type == "vlb":
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        # Total atom count (scaffold + pocket + R-group), matching
        # diffusion_difflinker.py's semantics for this histogram.
        for size in node_mask.sum(dim=1).squeeze(-1).round().long().tolist():
            if size > 0:
                self._atom_count_counts[size] += 1
        self._node_dist_cache = None

        def _d(v):
            return v.detach() if torch.is_tensor(v) else v

        stats = {
            "delta_log_px": _d(delta_log_px),
            "kl_prior": _d(kl_prior),
            "loss_term_t": _d(loss_term_t),
            "loss_term_0": _d(loss_term_0),
            "l2_loss": _d(l2_loss),
            "vlb_loss": _d(vlb_loss),
            "noise_t": _d(noise_t),
            "noise_0": _d(noise_0),
        }
        return loss, stats

    def predict_and_target(self, batch: Dict[str, Any]):
        loss, _stats = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict:  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # ----------------------------------------------------------- properties
    @property
    def atom_count_histogram(self) -> Optional[dict]:
        # Same precedence rule as diffusion_difflinker.py: the live,
        # checkpoint-persisted counts win over the on_load_checkpoint
        # override, so a resume cannot be permanently shadowed by a stale
        # top-level `n_node_dist`.
        if self._atom_count_counts:
            return dict(self._atom_count_counts)
        if self._n_node_dist_override is not None:
            return dict(self._n_node_dist_override)
        return None

    @property
    def node_dist_model(self) -> Optional[DistributionNodes]:
        histogram = self.atom_count_histogram
        if histogram is not None:
            if self._node_dist_cache is None:
                self._node_dist_cache = DistributionNodes(histogram)
            return self._node_dist_cache
        return self._node_dist_override

    @node_dist_model.setter
    def node_dist_model(self, value: DistributionNodes) -> None:
        self._node_dist_override = value
        self._node_dist_cache = None

    @property
    def n_node_dist(self) -> Optional[dict]:
        return self.atom_count_histogram

    @n_node_dist.setter
    def n_node_dist(self, value: dict) -> None:
        self._n_node_dist_override = dict(value) if value else None
        self._node_dist_cache = None

    @property
    def model(self):
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -------------------------------------------------------------- sampling
    def decorate(
        self,
        batch: Dict[str, Any],
        rgroup_sizes: Optional[torch.Tensor] = None,
        keep_frames: int = 1,
    ):
        """Grow R-groups onto the scaffolds in ``batch``.

        Port of ``model_single.py::DDPM.sample_chain`` (l. 289-340). Returns
        ``(one_hot, positions, ligand_mask)`` where ``ligand_mask`` is
        ``atom_mask - pocket_mask`` -- scaffold + R-group, pocket dropped --
        matching upstream ``sample_single.py`` l. 148.

        Coordinates come back in the **centred** frame (partial COM removed);
        the caller adds the offset back, as upstream does at
        ``sample_single.py`` l. 146.

        ``rgroup_sizes`` defaults to ``rgroup_mask.sum(1)``, i.e. the fixed
        10-slot budget -- upstream's ``sample_fn = None`` path. Unused slots
        come back as the fake ``'#'`` atom.
        """
        if rgroup_sizes is None:
            rgroup_sizes = batch["rgroup_mask"].sum(1).view(-1).int()

        template = create_templates_for_rgroup_generation_single(batch, rgroup_sizes)
        device = self.device
        template = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in template.items()
        }

        context, com_mask = self._context_and_com(template)
        node_mask = template["atom_mask"]
        x = dl_utils.remove_partial_mean_with_mask(
            template["positions"], node_mask, com_mask
        )

        chain = self.edm.sample_chain(
            x=x,
            h=template["one_hot"],
            node_mask=node_mask,
            fragment_mask=template["scaffold_mask"],
            linker_mask=template["rgroup_mask"],
            edge_mask=template["edge_mask"],
            context=context,
            keep_frames=keep_frames,
        )
        final = chain[0]
        ligand_mask = (node_mask - template["pocket_mask"]).squeeze(-1)
        return final[..., 3:], final[..., :3], ligand_mask

    def sample(self, *args: Any, **kwargs: Any):
        """Not reachable through ``GenerativeFactory`` -- see the module
        docstring and INTEGRATION_PLAN.md's Task-contract mapping.

        The §2.1 signature is kept so the contract check passes, but DiffDec
        cannot generate without a scaffold, a pocket and an anchor atom, and
        ``sample(batch_size, nodesxsample, ...)`` carries none of those. Use
        :class:`DiffDecScaffoldGenerator` (``configs/interference/
        gen_diffdec_scaffold.yaml``), which calls :meth:`decorate`.
        """
        del args, kwargs
        raise NotImplementedError(
            "DiffDecTask.sample() is not supported: DiffDec has no "
            "unconditional generation mode -- it decorates a GIVEN 3D "
            "scaffold at a GIVEN anchor atom inside a GIVEN pocket, none of "
            "which fit sample(batch_size, nodesxsample). Generate with:\n"
            "  MolCraftDiff generate configs/diffdec_generate.yaml\n"
            "which routes through DiffDecScaffoldGenerator "
            "(configs/interference/gen_diffdec_scaffold.yaml)."
        )


class DiffDecScaffoldGenerator:
    """Scaffold decoration inside a fixed pocket.

    ``cli/generate.py:673`` does ``instantiate(cfg.interference, task=task)``
    then ``.run()``, so every key of the interference config lands straight in
    this ``__init__`` -- nothing needs registering. Precedent:
    ``DiffSBDDPocketGenerator``.

    The scaffold, its anchor atom and the pocket all come from ONE row of
    upstream's preprocessed ``.pt`` (``data_file`` + ``complex_index``).
    ``num_generate`` R-groups are then sampled for that one input, which is
    upstream's own protocol (``sample_single.py``: ``--n_samples`` decorations
    per complex).
    """

    def __init__(
        self,
        task: DiffDecTask,
        data_file: Optional[str] = None,
        complex_index: int = 0,
        num_generate: int = 20,
        batch_size: int = 4,
        num_steps: Optional[int] = None,
        output_path: str = "generated_diffdec",
        save_reference: bool = True,
        seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if not data_file:
            raise ValueError(
                "interference.data_file is required: DiffDec has no "
                "unconditional mode -- it needs a scaffold + anchor + pocket. "
                "Point it at upstream's preprocessed .pt (Zenodo record "
                "10527451), e.g. docs/model_integrations/diffdec/data/"
                "crossdocksingle_test_full.pt"
            )
        self.task = task
        self.data_file = data_file
        self.complex_index = complex_index
        self.num_generate = num_generate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.output_path = output_path
        self.save_reference = save_reference
        self.seed = seed
        self.device = device or (str(get_device()))

    # ------------------------------------------------------------------ util
    @staticmethod
    def _compact(
        one_hot: torch.Tensor, coords: torch.Tensor, keep: torch.Tensor
    ):
        """Gather kept atoms to the front of each row.

        ``utils/geom_utils.py::save_xyz_file`` reads the FIRST
        ``node_mask.sum()`` rows positionally -- it does not index by the mask.
        DiffDec's kept atoms are non-contiguous (scaffold at the front, pocket
        in the middle, R-group at the end), so they have to be compacted here
        or the writer silently emits pocket atoms instead of the ligand.

        Returns ``(one_hot, coords, node_mask)`` with ``node_mask`` shaped
        ``(B, width)`` -- ``save_xyz_file`` sums it along dim 1, so a bare
        per-molecule count vector is not an acceptable substitute.
        """
        bsz, _n, nf = one_hot.shape
        counts = keep.sum(dim=1).long()
        width = max(int(counts.max().item()) if bsz else 0, 1)
        out_h = one_hot.new_zeros((bsz, width, nf))
        out_x = coords.new_zeros((bsz, width, 3))
        out_mask = one_hot.new_zeros((bsz, width))
        for b in range(bsz):
            idx = keep[b].nonzero(as_tuple=True)[0]
            out_h[b, : len(idx)] = one_hot[b, idx]
            out_x[b, : len(idx)] = coords[b, idx]
            out_mask[b, : len(idx)] = 1.0
        return out_h, out_x, out_mask

    def _keep_mask(
        self, one_hot: torch.Tensor, ligand_mask: torch.Tensor
    ) -> torch.Tensor:
        """Ligand atoms that are not the fake ``'#'`` padding type.

        The R-group occupies a FIXED 10-slot budget and the model writes
        ``'#'`` into the slots it does not use, so R-group size is decided by
        counting non-``'#'`` slots (upstream ``const.ATOM2IDX['#'] == 9``).
        Dropping them here is what keeps ``'#'`` out of the ``.xyz`` and out
        of every downstream featurizer.
        """
        is_fake = one_hot.argmax(dim=-1) == FAKE_ATOM_INDEX
        return (ligand_mask > 0.5) & ~is_fake

    def _batch(self, item: Dict[str, Any], n: int) -> Dict[str, Any]:
        """Tile one complex ``n`` times through the real collate."""
        return diffdec_collate([item] * n)

    def _com_offset(self, batch: Dict[str, Any]) -> torch.Tensor:
        """The partial centre of mass ``decorate()`` removes, so the caller
        can add it back (upstream ``sample_single.py`` l. 110-125, 146)."""
        mask = (
            batch["anchors"]
            if self.task.center_of_mass == "anchors"
            else batch["scaffold_only_mask"]
        )
        return (batch["positions"] * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        )

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        from MolecularDiffusion.utils.geometry.geom_utils import save_xyz_file

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        self.task.to(self.device)
        self.task.eval()

        if self.num_steps is not None:
            # Upstream's own "fewer sampling steps" knob
            # (sample_single.py l. 73-74): it subsamples the SAME schedule the
            # weights were trained with, it does not rebuild gamma.
            self.task.edm.T = int(self.num_steps)

        dataset = DiffDecDataset(self.data_file)
        if not 0 <= self.complex_index < len(dataset):
            raise IndexError(
                f"complex_index={self.complex_index} out of range for "
                f"{self.data_file} ({len(dataset)} complexes)"
            )
        item = dataset[self.complex_index]

        n_scaffold = int(item["scaffold_only_mask"].sum())
        n_pocket = int(item["pocket_mask"].sum())
        n_slots = int(item["rgroup_mask"].sum())
        print(
            f"[diffdec] complex {self.complex_index} "
            f"({item.get('name', '?')}): scaffold={n_scaffold} atoms, "
            f"pocket={n_pocket} atoms, R-group budget={n_slots} slots, "
            f"generating {self.num_generate} decorations"
        )

        if self.save_reference:
            self._write_reference(item)

        written = 0
        while written < self.num_generate:
            n = min(self.batch_size, self.num_generate - written)
            batch = self._batch(item, n)
            offset = self._com_offset(batch).to(self.device)

            with torch.no_grad():
                one_hot, coords, ligand_mask = self.task.decorate(batch)
            coords = coords + offset

            keep = self._keep_mask(one_hot, ligand_mask)
            out_h, out_x, counts = self._compact(one_hot, coords, keep)

            save_xyz_file(
                self.output_path,
                out_h.cpu(),
                out_x.cpu(),
                self.task.atom_vocab,
                id_from=written,
                name="molecule",
                node_mask=counts.cpu(),
            )
            written += n
        print(f"[diffdec] wrote {written} decorated molecules to {self.output_path}")

    def _write_reference(self, item: Dict[str, Any]) -> None:
        """Write the input scaffold and the ground-truth molecule alongside
        the samples, as upstream ``sample_single.py`` l. 132-138 does. Without
        them there is nothing to judge a decoration against."""
        from MolecularDiffusion.utils.geometry.geom_utils import save_xyz_file

        batch = self._batch(item, 1)
        one_hot, coords = batch["one_hot"], batch["positions"]
        ligand = (batch["atom_mask"] - batch["pocket_mask"]).squeeze(-1)
        scaffold = batch["scaffold_only_mask"].squeeze(-1)

        for tag, mask in (("scaffold", scaffold), ("reference", ligand)):
            keep = self._keep_mask(one_hot, mask)
            out_h, out_x, counts = self._compact(one_hot, coords, keep)
            save_xyz_file(
                self.output_path,
                out_h,
                out_x,
                self.task.atom_vocab,
                id_from=0,
                name=tag,
                node_mask=counts,
            )


__all__: List[str] = [
    "DiffDecScaffoldGenerator",
    "DiffDecTask",
    "DiffDecTaskFactory",
]
