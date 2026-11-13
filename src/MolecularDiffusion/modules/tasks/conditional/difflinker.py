"""DiffLinker integration with the MolecularDiffusion data pipeline.

DiffLinker (https://github.com/igashov/DiffLinker) is a fragment/linker
mask-conditioned E(n)-equivariant diffusion model: given a set of fixed
"fragment" atoms, it diffuses only a "linker" subset of atoms that connects
them. See docs/model_integrations/difflinker/INTEGRATION_PLAN.md for the
full integration plan (data adapters, task-contract mapping, scope) this
module implements.

Ported model code lives under
``MolecularDiffusion.modules.models.difflinker`` (``edm.py``, ``egnn.py``,
``noise.py``, ``linker_size.py``).
"""

import random
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from ase.data import atomic_numbers as ase_atomic_numbers

from MolecularDiffusion.modules.models.difflinker import utils as dl_utils
from MolecularDiffusion.modules.models.difflinker.edm import EDM
from MolecularDiffusion.modules.models.difflinker.egnn import Dynamics, DynamicsWithPockets
from MolecularDiffusion.modules.models.difflinker.linker_size import DistributionNodes
from MolecularDiffusion.modules.models.difflinker.size_gnn import LinkerSizePredictor

# Column order the conversion script
# (docs/model_integrations/difflinker/scripts/convert_zinc_pt_to_asedb.py)
# packs into `row.data["node_features"]` -- implicit contract, not enforced
# by shared code (see INTEGRATION_PLAN.md's "Coupling risk" note). Column
# index len(atom_vocab) + 0 = linker_mask, len(atom_vocab) + 1 = anchors.
DIFFLINKER_ROW_DATA_COLUMNS = ("linker_mask", "anchors")

def dense_edge_mask(
    atom_mask: torch.Tensor, *, upstream_int8: bool = False
) -> torch.Tensor:
    """Build the flat ``(B*N*N, 1)`` edge mask for the dense path.

    ``upstream_int8`` reproduces a real integer-overflow bug in DiffLinker's
    own ``src/datasets.py:366-369``, which every released non-pocket weight
    was trained under::

        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(n, dtype=const.TORCH_INT)   # TORCH_INT = int8
        edge_mask *= diag_mask

    ``~`` on an ``int8`` tensor is *bitwise* NOT, not logical NOT, so
    ``diag_mask`` is ``-1`` off-diagonal and ``-2`` on-diagonal rather than
    ``True``/``False``. The resulting mask therefore **negates** every real
    message and passes self-loops through at ``-2``. That sign convention is
    baked into the released weights: fed the arithmetically-correct
    ``+1``/``0`` mask they diverge to NaN on the first reverse step, and fed
    this one they generate cleanly. Weights trained *by this platform* learnt
    the correct mask, so this defaults to ``False`` and only the converted
    upstream checkpoints turn it on.

    The pocket checkpoints are unaffected either way: ``DynamicsWithPockets``
    builds its own distance-cutoff edges and is called with ``edge_mask=None``.
    """
    am = atom_mask.squeeze(-1)
    edges = am.unsqueeze(2) * am.unsqueeze(1)
    n = am.shape[-1]
    if upstream_int8:
        diag = ~torch.eye(n, dtype=torch.int8, device=am.device)
        masked = edges.to(torch.int8) * diag.unsqueeze(0)
        return masked.reshape(-1, 1).to(am.dtype)
    diag = ~torch.eye(n, dtype=torch.bool, device=am.device)
    return (edges * diag.unsqueeze(0)).reshape(-1, 1)


class PointCloudToDiffLinkerBatch:
    """Converts a MolCraftDiffusion PointCloud batch dict into DiffLinker's
    native per-atom field layout.

    Two data paths share this one adapter (see INTEGRATION_PLAN.md's Data
    adapters section):

    - **Path A** (real ZINC data, converted via
      ``convert_zinc_pt_to_asedb.py`` + ``use_row_data_features: true``):
      ``linker_mask``/``anchors`` are sliced straight out of
      ``node_feature``'s trailing two columns.
    - **Path B** (synthetic fallback, any ordinary single-molecule dataset):
      when those trailing columns aren't present, a random contiguous slice
      of each molecule is fabricated as the "linker", with the fragment
      atoms bordering the cut tagged as anchors.
    """

    def __init__(
        self, atom_vocab: list, *, upstream_int8_edge_mask: bool = False
    ) -> None:
        self.atom_vocab = list(atom_vocab)
        self.n_vocab = len(self.atom_vocab)
        self.upstream_int8_edge_mask = upstream_int8_edge_mask

    def __call__(self, batch: dict) -> dict:
        node_feature = batch.get("node_feature")
        if node_feature is None:
            node_feature = batch.get("node_features")
        if node_feature is None:
            node_feature = batch["x"]

        coords = batch["coords"]
        device = coords.device
        node_mask_flat = batch["node_mask"].float().to(device)
        _bsz, _n_atoms = node_mask_flat.shape

        one_hot = node_feature[..., : self.n_vocab].to(device)

        has_row_data = node_feature.shape[-1] >= self.n_vocab + 2
        if has_row_data:
            linker_mask = node_feature[..., self.n_vocab : self.n_vocab + 1].to(device)
            anchors = node_feature[..., self.n_vocab + 1 : self.n_vocab + 2].to(device)
        else:
            linker_mask, anchors = self._synthesize_linker_split(node_mask_flat)

        atom_mask = node_mask_flat.unsqueeze(-1)
        linker_mask = linker_mask * atom_mask
        fragment_mask = (1.0 - linker_mask) * atom_mask

        edge_mask = dense_edge_mask(
            atom_mask, upstream_int8=self.upstream_int8_edge_mask
        )

        return {
            "positions": coords,
            "one_hot": one_hot,
            "charges": batch.get("charges"),
            "anchors": anchors,
            "fragment_mask": fragment_mask,
            "linker_mask": linker_mask,
            "atom_mask": atom_mask,
            "edge_mask": edge_mask,
            "num_atoms": batch.get("natoms"),
        }

    def _synthesize_linker_split(self, node_mask_flat: torch.Tensor):
        """Path B: fabricate a fragment/linker split (not chemically
        meaningful) -- see class docstring."""
        device = node_mask_flat.device
        bsz, n_atoms = node_mask_flat.shape
        linker_mask = torch.zeros(bsz, n_atoms, 1, device=device)
        anchors = torch.zeros(bsz, n_atoms, 1, device=device)

        for b in range(bsz):
            n = int(node_mask_flat[b].sum().item())
            if n < 3:
                # Too small to carve out a linker; leave the whole molecule
                # as "fragment" (linker_mask all zero for this sample).
                continue
            max_len = max(1, min(4, n - 2))
            linker_len = random.randint(1, max_len)
            start = random.randint(1, n - linker_len - 1)
            linker_mask[b, start : start + linker_len, 0] = 1.0
            anchors[b, start - 1, 0] = 1.0
            if start + linker_len < n:
                anchors[b, start + linker_len, 0] = 1.0

        return linker_mask, anchors


class DiffLinkerTaskFactory:
    """Factory matching ``train.py``'s ``task_module.build()`` /
    ``task_module.task`` instantiation pattern (see
    ``diffusion_tabasco.py::ModelTaskFactory`` for the precedent)."""

    def __init__(
        self,
        task_type: str,
        in_node_nf: int,
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
        center_of_mass: str = "fragments",
        atom_vocab: Optional[list] = None,
        pocket_conditioned: bool = False,
        context_node_nf: Optional[int] = None,
        graph_type: str = "FC",
        upstream_int8_edge_mask: bool = False,
        size_gnn_checkpoint: Optional[str] = None,
        **kwargs,
    ):
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
        # Pockets checkpoint family (INTEGRATION_PLAN.md Revision 9) --
        # additive only, every field defaults to today's non-pocket
        # behavior.
        self.pocket_conditioned = pocket_conditioned
        self.context_node_nf = context_node_nf
        self.graph_type = graph_type
        # Released-weight edge-mask sign convention -- see dense_edge_mask().
        # False (default) = this platform's own trained weights; True = a
        # converted upstream non-pocket checkpoint.
        self.upstream_int8_edge_mask = upstream_int8_edge_mask
        # Size GNN (INTEGRATION_PLAN.md Revision 8) -- additive only, a
        # converted LinkerSizePredictor checkpoint path.
        self.size_gnn_checkpoint = size_gnn_checkpoint
        self.kwargs = kwargs

    def build(self) -> "DiffLinkerTask":
        self.task = DiffLinkerTask(
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
            pocket_conditioned=self.pocket_conditioned,
            context_node_nf=self.context_node_nf,
            graph_type=self.graph_type,
            upstream_int8_edge_mask=self.upstream_int8_edge_mask,
        )
        if self.size_gnn_checkpoint:
            self.task.size_predictor = LinkerSizePredictor.from_checkpoint(
                self.size_gnn_checkpoint
            )
        return self.task


class DiffLinkerTask(nn.Module):
    """Plain ``nn.Module`` task wrapper (TABASCO-style, no ``Task``/
    ``core.Configurable`` base needed -- see docs/adding_new_models.md §2.6)
    implementing the §2.1 contract around DiffLinker's ``EDM``."""

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
        pocket_conditioned: bool = False,
        context_node_nf: Optional[int] = None,
        graph_type: str = "FC",
        upstream_int8_edge_mask: bool = False,
    ):
        super().__init__()

        act = nn.SiLU() if activation == "silu" else nn.SiLU()
        if context_node_nf is None:
            if pocket_conditioned:
                raise ValueError(
                    "DiffLinkerTask: pocket_conditioned=True requires an "
                    "explicit context_node_nf -- the Pockets checkpoints' "
                    "context shape (anchors + fragment_only_mask + "
                    "pocket_mask, or fragment_only_mask + pocket_mask) "
                    "isn't derivable from anchors_context alone. See "
                    "INTEGRATION_PLAN.md Revision 9, Phase A6's table for "
                    "the real per-checkpoint value."
                )
            context_node_nf = 2 if anchors_context else 1

        # Pockets checkpoint family (INTEGRATION_PLAN.md Revision 9):
        # DynamicsWithPockets overrides only forward()/edge construction --
        # identical parameter shapes to Dynamics for the same
        # hidden_nf/in_node_nf/context_node_nf/n_layers, so switching
        # classes here changes no state-dict key. `graph_type` was
        # previously hardcoded "FC"; now threaded through so pocket
        # checkpoints (graph_type "4A"/"FC-10A-4A") get their own dispatch,
        # unchanged default for every existing non-pocket config.
        dynamics_cls = DynamicsWithPockets if pocket_conditioned else Dynamics
        dynamics = dynamics_cls(
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
            centering=False,
            graph_type=graph_type,
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
        self.atom_vocab = atom_vocab
        self.pocket_conditioned = pocket_conditioned
        self.context_node_nf = context_node_nf
        self.graph_type = graph_type
        self.upstream_int8_edge_mask = upstream_int8_edge_mask
        # Size GNN (INTEGRATION_PLAN.md Revision 8) -- optional, additive
        # alternative to node_dist_model histogram sampling in sample()
        # below. None (default) => today's behavior, unchanged.
        self.size_predictor: Optional[LinkerSizePredictor] = None

        # tasks_generate.py::preprocess_ref_structure/structural_guidance read
        # these generically off `task.model` (duck-typed, not
        # GeomMolecularGenerative-specific -- INTEGRATION_PLAN.md's
        # Core-Change Requirement section).
        self.norm_values = tuple(normalize_factors)
        self.ndim_extra = 0
        self.prop_dist_model = None

        self._adapter: Optional[PointCloudToDiffLinkerBatch] = None
        # ponytail: lazy total-atom-count stats instead of a
        # `train_set`-at-build seam (docs/adding_new_models.md §2.5) --
        # accumulated in-process
        # from real forward() batches, persisted across checkpoint save/load
        # via get_extra_state/set_extra_state below (plain engine) and via
        # the node_dist_model/n_node_dist setters (Lightning engine's
        # on_load_checkpoint).
        self._atom_count_counts: Counter = Counter()
        self._node_dist_cache: Optional[DistributionNodes] = None
        # Set only via the node_dist_model/n_node_dist setters below -- lets
        # a fresh `generate` process restore the real stats saved into the
        # checkpoint at end of training (core/engine_lightning.py's
        # on_load_checkpoint), rather than falling back to a placeholder.
        self._node_dist_override: Optional[DistributionNodes] = None
        self._n_node_dist_override: Optional[dict] = None

    def get_extra_state(self) -> dict:
        return {"atom_count_counts": dict(self._atom_count_counts)}

    def set_extra_state(self, state: dict) -> None:
        counts = state.get("atom_count_counts") if isinstance(state, dict) else None
        if counts:
            self._atom_count_counts = Counter(counts)
            self._node_dist_cache = None

    def _get_adapter(self) -> PointCloudToDiffLinkerBatch:
        if self.atom_vocab is None:
            raise ValueError(
                "DiffLinkerTask.atom_vocab is not set -- cli/train.py/"
                "cli/generate.py should set it after build() (see "
                "docs/adding_new_models.md §2.1)."
            )
        if (
            self._adapter is None
            or self._adapter.atom_vocab != list(self.atom_vocab)
            or self._adapter.upstream_int8_edge_mask != self.upstream_int8_edge_mask
        ):
            self._adapter = PointCloudToDiffLinkerBatch(
                self.atom_vocab,
                upstream_int8_edge_mask=self.upstream_int8_edge_mask,
            )
        return self._adapter

    def forward(self, batch: dict):
        if self.pocket_conditioned:
            raise NotImplementedError(
                "DiffLinkerTask.forward(): training on pocket-conditioned "
                "(MOAD) data is out of scope this integration -- only "
                "inference from the released Pockets checkpoints is "
                "supported (see INTEGRATION_PLAN.md Revision 9)."
            )
        pc = self._get_adapter()(batch)
        x = pc["positions"]
        h = pc["one_hot"]
        node_mask = pc["atom_mask"]
        edge_mask = pc["edge_mask"]
        anchors = pc["anchors"]
        fragment_mask = pc["fragment_mask"]
        linker_mask = pc["linker_mask"]

        if self.anchors_context:
            context = torch.cat([anchors, fragment_mask], dim=-1)
        else:
            context = fragment_mask

        if self.center_of_mass == "fragments":
            center_of_mass_mask = fragment_mask
        elif self.center_of_mass == "anchors":
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)

        x = dl_utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.edm.forward(
            x=x,
            h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
        )
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px
        loss = l2_loss if self.loss_type == "l2" else vlb_loss

        # Total atom count (fragment + linker), NOT linker-only -- this must
        # match what GenerativeFactory's mol_size/max_atom clamp and
        # DiffLinkerTask.sample()'s own `nodesxsample` mean (see sample()'s
        # docstring/error message: "nodesxsample (total atom counts)").
        # Counting linker-only atoms here made max_atom a linker-size cap
        # (typically far smaller than any real fragment), which forced
        # nodesxsample to always clamp up to the fragment size -- a
        # permanent linker_size==0 regardless of mol_size (see
        # INTEGRATION_PLAN.md's Post-Completion Fix section).
        sizes = node_mask.sum(dim=1).squeeze(-1).round().long().tolist()
        for size in sizes:
            if size > 0:
                self._atom_count_counts[size] += 1
        self._node_dist_cache = None

        stats = {
            "delta_log_px": delta_log_px.detach(),
            "kl_prior": kl_prior.detach(),
            "loss_term_t": loss_term_t.detach(),
            "loss_term_0": loss_term_0.detach() if torch.is_tensor(loss_term_0) else loss_term_0,
            "l2_loss": l2_loss.detach(),
            "vlb_loss": vlb_loss.detach(),
            "noise_t": noise_t.detach() if torch.is_tensor(noise_t) else noise_t,
            "noise_0": noise_0.detach() if torch.is_tensor(noise_0) else noise_0,
        }
        return loss, stats

    def predict_and_target(self, batch: dict):
        loss, _stats = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        return {"val_loss": pred.mean()}

    @property
    def atom_count_histogram(self) -> Optional[dict]:
        # _atom_count_counts (persisted via get_extra_state/set_extra_state,
        # see __init__) is always the real, current-checkpoint-format stats
        # and must win over _n_node_dist_override -- the override exists only
        # for core/engine_lightning.py's on_load_checkpoint fine-tuning path
        # (injecting freshly-preprocessed stats before any forward() has run
        # this process), and checking it first would let a *resume* from an
        # older checkpoint's stale top-level `n_node_dist` (pre-dating this
        # histogram's current semantics) permanently shadow the correct,
        # already-restored _atom_count_counts for the rest of the run.
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
        # ponytail: settable so core/engine_lightning.py's generic
        # on_load_checkpoint restore (checkpoint['node_dist_model'] -- the
        # real linker-size stats saved at the end of training) actually
        # takes effect, instead of silently no-op'ing because this property
        # used to never return None ("has_fresh_node_dist" always true).
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

    def _load_pocket(self, pocket_db: str, pocket_index: int, device):
        """Loads one pocket's positions + one-hot atom types from a
        single-row-per-pocket ASE db (INTEGRATION_PLAN.md Revision 9's
        pocket data adapter -- see
        docs/model_integrations/difflinker/scripts/convert_pocket_pt_to_asedb.py
        for how such a db is built). `pocket_index` is 0-indexed, matching
        `PocketGenerator`'s own `pocket_index` convention (`ase.db` row ids
        are 1-based, hence the `+ 1` below).
        """
        import ase.db  # noqa: PLC0415

        with ase.db.connect(pocket_db) as db:
            atoms = db.get(id=pocket_index + 1).toatoms()
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
        vocab_z = {ase_atomic_numbers[s]: i for i, s in enumerate(self.atom_vocab)}
        one_hot = torch.zeros(len(atoms), len(self.atom_vocab), device=device)
        for i, z in enumerate(atoms.get_atomic_numbers()):
            if int(z) not in vocab_z:
                raise ValueError(
                    f"DiffLinkerTask._load_pocket(): pocket atom {i} (Z={z}) "
                    f"is not in atom_vocab {self.atom_vocab} -- rebuild "
                    f"{pocket_db} with only vocabulary elements."
                )
            one_hot[i, vocab_z[int(z)]] = 1.0
        return pos, one_hot

    def sample(
        self,
        nodesxsample: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        batch: Optional[dict] = None,
        condition_tensor: Optional[torch.Tensor] = None,
        condition_mode: Optional[str] = None,
        outpaint_cfgs: Optional[dict] = None,
        use_noised_conditioning: bool = False,
        n_frames: int = 0,
        n_retrys: int = 0,
        t_retry: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Generate molecules by diffusing a linker between fixed fragment
        atoms. Dispatched via the generic "outpaint" `GenerativeFactory`
        path (`runmodes/generate/tasks_generate.py::structural_guidance`) --
        see INTEGRATION_PLAN.md's Task-contract mapping section for the
        exact kwarg contract this method mirrors.

        `outpaint_cfgs` is read for two keys: `anchor_indices` -- a list of
        1-indexed positions into the reference fragment file, matching
        upstream's `--anchors` CLI convention
        (`others/difflinker/generate.py:129-132`). Required when this
        checkpoint's `center_of_mass == "anchors"` (see INTEGRATION_PLAN.md
        Revision 5, Point 1/2); unused when `center_of_mass == "fragments"`.
        `max_generate_attempts` -- a bounded full-restart retry count around
        `self.edm.sample_chain(...)`, catching `FoundNaNException`
        (INTEGRATION_PLAN.md Revision 6, Point 1). Mirrors upstream's own
        `generate.py` retry loop (fixed budget, full restart, no partial/
        `t_retry`-style resume -- `EDM.sample_chain` draws fresh noise
        internally every call). Defaults to `1` (today's single-shot
        behavior, unchanged for any config that doesn't set it). This is
        independent of the still-inert `n_retrys`/`t_retry` kwargs below,
        which the shared `structural_guidance` dispatch always forces to
        `0`/`None` -- do not confuse the two.

        When `self.pocket_conditioned` (the three genuinely pocket-
        conditioned Pockets checkpoints, INTEGRATION_PLAN.md Revision 9),
        `outpaint_cfgs` additionally requires `pocket_db` -- a path to a
        single-row-per-pocket ASE db (see
        `convert_pocket_pt_to_asedb.py`) -- and accepts `pocket_index`
        (0-indexed, default 0). The pocket is inserted as a fixed,
        non-diffused block between the fragment and linker regions;
        `nodesxsample` keeps its existing meaning (fragment + linker atom
        count, NOT including the pocket).

        When `self.size_predictor` is set (an optional `LinkerSizePredictor`,
        INTEGRATION_PLAN.md Revision 8), the incoming `nodesxsample` is
        overridden with a fragment-geometry-conditioned prediction instead
        of whatever `GenerativeFactory` sampled from `node_dist_model`.

        Several other kwargs accepted here are explicitly out of scope this
        pass (inert, not implemented with real behavior): `batch`,
        `num_steps`, `condition_mode`, `use_noised_conditioning`, `n_frames`
        (no trajectory export), `n_retrys`/`t_retry` (no bond-distance retry
        loop), `context` (property-conditioning value -- DiffLinker has no
        `prop_dist_model`, so this is always None here).
        """
        del batch, num_steps, condition_mode
        del use_noised_conditioning, n_retrys, t_retry, context, kwargs
        outpaint_cfgs = outpaint_cfgs or {}
        anchor_indices = outpaint_cfgs.get("anchor_indices")

        if condition_tensor is None:
            raise NotImplementedError(
                "DiffLinkerTask.sample() requires a fragment reference "
                "structure (condition_tensor) -- DiffLinker has no "
                "unconditional generation mode. Use the 'outpaint' "
                "interference dispatch with reference_structure_path set "
                "(see configs/interference/gen_outpaint.yaml)."
            )
        if nodesxsample is None:
            raise ValueError("DiffLinkerTask.sample() requires nodesxsample (total atom counts).")

        device = self.device
        nodesxsample = nodesxsample.to(device).long()
        batch_size = int(nodesxsample.shape[0])

        n_vocab = len(self.atom_vocab)
        norm_coords, norm_feats, _norm_charges = self.norm_values

        condition_tensor = condition_tensor.to(device)
        ref_natoms = condition_tensor.shape[1]
        ref_positions = condition_tensor[..., :3] * norm_coords
        ref_onehot = condition_tensor[..., 3 : 3 + n_vocab] * norm_feats
        if ref_positions.shape[0] != batch_size:
            ref_positions = ref_positions.expand(batch_size, -1, -1)
        if ref_onehot.shape[0] != batch_size:
            ref_onehot = ref_onehot.expand(batch_size, -1, -1)

        if self.size_predictor is not None:
            # Size GNN (INTEGRATION_PLAN.md Revision 8): overrides whatever
            # GenerativeFactory already sampled from node_dist_model for
            # this batch with a fragment-geometry-conditioned prediction.
            # ref_onehot/ref_positions are already exactly the fragment-only
            # tensors this predictor needs (implicit all-ones fragment mask
            # -- the reference structure *is* the fragment, by
            # construction). Predicted once, before the retry loop below
            # (documented scope simplification, see sample()'s docstring
            # discussion in INTEGRATION_PLAN.md Revision 8).
            predicted_linker_sizes = self.size_predictor.predict(ref_onehot, ref_positions)
            nodesxsample = ref_natoms + predicted_linker_sizes.to(device)

        if self.pocket_conditioned:
            pocket_db = outpaint_cfgs.get("pocket_db")
            if not pocket_db:
                raise ValueError(
                    "DiffLinkerTask.sample(): this checkpoint is "
                    "pocket_conditioned=True and requires "
                    "interference.condition_configs.outpaint_cfgs.pocket_db "
                    "(an ASE db path with one pocket per row -- see "
                    "convert_pocket_pt_to_asedb.py); outpaint_cfgs."
                    "pocket_index (0-indexed, default 0) selects the row."
                )
            pocket_index = int(outpaint_cfgs.get("pocket_index", 0))
            pocket_pos, pocket_onehot = self._load_pocket(pocket_db, pocket_index, device)
            pocket_natoms = pocket_pos.shape[0]
            pocket_end = ref_natoms + pocket_natoms

            max_atoms = max(int(nodesxsample.max().item()), ref_natoms) + pocket_natoms

            positions = torch.zeros(batch_size, max_atoms, 3, device=device)
            one_hot = torch.zeros(batch_size, max_atoms, n_vocab, device=device)
            fragment_only_mask = torch.zeros(batch_size, max_atoms, 1, device=device)
            pocket_mask = torch.zeros(batch_size, max_atoms, 1, device=device)
            linker_mask = torch.zeros(batch_size, max_atoms, 1, device=device)
            anchors = torch.zeros(batch_size, max_atoms, 1, device=device)
            atom_mask = torch.zeros(batch_size, max_atoms, 1, device=device)

            for b in range(batch_size):
                total_b = max(int(nodesxsample[b].item()), ref_natoms) + pocket_natoms
                positions[b, :ref_natoms] = ref_positions[b]
                one_hot[b, :ref_natoms] = ref_onehot[b]
                fragment_only_mask[b, :ref_natoms, 0] = 1.0
                positions[b, ref_natoms:pocket_end] = pocket_pos
                one_hot[b, ref_natoms:pocket_end] = pocket_onehot
                pocket_mask[b, ref_natoms:pocket_end, 0] = 1.0
                linker_mask[b, pocket_end:total_b, 0] = 1.0
                atom_mask[b, :total_b, 0] = 1.0
                # anchor_indices unchanged: still 1-indexed into the
                # fragment-only ordering, still placed at [0:ref_natoms) --
                # inserting the pocket after the fragment does not renumber
                # anchors (INTEGRATION_PLAN.md Revision 9).
                if anchor_indices:
                    for idx in anchor_indices:
                        if not 1 <= idx <= ref_natoms:
                            raise ValueError(
                                "DiffLinkerTask.sample(): anchor_indices "
                                f"entry {idx} is out of range -- the "
                                f"reference fragment has {ref_natoms} atoms "
                                f"(valid range 1..{ref_natoms})."
                            )
                        anchors[b, idx - 1, 0] = 1.0

            # EDM's "not diffused" mask is the union of fragment + pocket
            # (EDM.forward/sample_chain: z_t = xh * fragment_mask + z_t *
            # linker_mask) -- distinct from fragment_only_mask/pocket_mask,
            # which instead feed context so the network can tell fragment
            # and pocket apart (others/difflinker/src/lightning.py:163-164;
            # see INTEGRATION_PLAN.md Revision 9's "pocket data adapter"
            # section).
            fragment_mask = fragment_only_mask + pocket_mask
            if self.anchors_context:
                mask_context = torch.cat(
                    [anchors, fragment_only_mask, pocket_mask], dim=-1
                )
            else:
                mask_context = torch.cat([fragment_only_mask, pocket_mask], dim=-1)

            # DynamicsWithPockets.forward treats `edge_mask` as a flat
            # per-node batch-index tensor (get_dist_edges_4A/get_dist_edges),
            # not the dense pairwise mask the non-pocket path below builds.
            edge_mask = torch.arange(batch_size, device=device).repeat_interleave(max_atoms)

            if self.center_of_mass == "fragments":
                center_of_mass_mask = fragment_only_mask
            elif self.center_of_mass == "anchors":
                if not anchor_indices:
                    raise ValueError(
                        "DiffLinkerTask.sample(): this checkpoint's "
                        "center_of_mass == 'anchors' requires real anchor "
                        "atom indices -- pass them via "
                        "interference.condition_configs.outpaint_cfgs."
                        "anchor_indices (1-indexed positions into the "
                        "reference fragment file)."
                    )
                center_of_mass_mask = anchors
            else:
                raise NotImplementedError(self.center_of_mass)

            positions = dl_utils.remove_partial_mean_with_mask(
                positions, atom_mask, center_of_mass_mask
            )
        else:
            max_atoms = max(int(nodesxsample.max().item()), ref_natoms)

            positions = torch.zeros(batch_size, max_atoms, 3, device=device)
            one_hot = torch.zeros(batch_size, max_atoms, n_vocab, device=device)
            fragment_mask = torch.zeros(batch_size, max_atoms, 1, device=device)
            linker_mask = torch.zeros(batch_size, max_atoms, 1, device=device)
            anchors = torch.zeros(batch_size, max_atoms, 1, device=device)
            atom_mask = torch.zeros(batch_size, max_atoms, 1, device=device)

            for b in range(batch_size):
                total_b = max(int(nodesxsample[b].item()), ref_natoms)
                positions[b, :ref_natoms] = ref_positions[b]
                one_hot[b, :ref_natoms] = ref_onehot[b]
                fragment_mask[b, :ref_natoms, 0] = 1.0
                linker_mask[b, ref_natoms:total_b, 0] = 1.0
                atom_mask[b, :total_b, 0] = 1.0
                # anchors: real indices from outpaint_cfgs.anchor_indices when
                # supplied (upstream's --anchors convention, 1-indexed into the
                # reference fragment), else left zero -- a degraded-but-valid
                # fallback only when center_of_mass == "fragments" (anchors are
                # a soft context feature there); center_of_mass == "anchors"
                # fails fast below instead of silently centering on an
                # all-zero mask (INTEGRATION_PLAN.md Revision 5, Points 1/2).
                if anchor_indices:
                    for idx in anchor_indices:
                        if not 1 <= idx <= ref_natoms:
                            raise ValueError(
                                "DiffLinkerTask.sample(): anchor_indices entry "
                                f"{idx} is out of range -- the reference "
                                f"fragment has {ref_natoms} atoms (valid range "
                                f"1..{ref_natoms})."
                            )
                        anchors[b, idx - 1, 0] = 1.0

            if self.anchors_context:
                mask_context = torch.cat([anchors, fragment_mask], dim=-1)
            else:
                mask_context = fragment_mask

            edge_mask = dense_edge_mask(
            atom_mask, upstream_int8=self.upstream_int8_edge_mask
        )

            if self.center_of_mass == "fragments":
                center_of_mass_mask = fragment_mask
            elif self.center_of_mass == "anchors":
                if not anchor_indices:
                    raise ValueError(
                        "DiffLinkerTask.sample(): this checkpoint's "
                        "center_of_mass == 'anchors' requires real anchor atom "
                        "indices -- pass them via "
                        "interference.condition_configs.outpaint_cfgs."
                        "anchor_indices (1-indexed positions into the "
                        "reference fragment file), or use a DiffLinker "
                        "checkpoint that does not require anchor information "
                        "(center_of_mass == 'fragments') instead."
                    )
                center_of_mass_mask = anchors
            else:
                raise NotImplementedError(self.center_of_mass)

            positions = dl_utils.remove_partial_mean_with_mask(
                positions, atom_mask, center_of_mass_mask
            )

        max_attempts = max(1, int(outpaint_cfgs.get("max_generate_attempts", 1)))
        last_exc: Optional[dl_utils.FoundNaNException] = None
        chain = None
        for _attempt in range(max_attempts):
            try:
                chain = self.edm.sample_chain(
                    x=positions,
                    h=one_hot,
                    node_mask=atom_mask,
                    fragment_mask=fragment_mask,
                    linker_mask=linker_mask,
                    edge_mask=edge_mask,
                    context=mask_context,
                    keep_frames=1,
                )
                break
            except dl_utils.FoundNaNException as exc:
                last_exc = exc
        if chain is None:
            raise last_exc
        final_xh = chain[0]
        out_positions = final_xh[..., :3]
        out_one_hot = final_xh[..., 3:]

        z_lookup = torch.tensor(
            [ase_atomic_numbers[s] for s in self.atom_vocab], device=device, dtype=torch.long
        )
        out_charges = z_lookup[out_one_hot.argmax(dim=-1)] * atom_mask.squeeze(-1).long()

        return out_one_hot, out_charges, out_positions, atom_mask.squeeze(-1)
