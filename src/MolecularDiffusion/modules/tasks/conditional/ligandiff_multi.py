"""multi-LigandDiff integration with the MolecularDiffusion data pipeline.

multi-LigandDiff (https://github.com/Neon8988/multi_LigandDiff, ChemRxiv 2024)
regenerates **several** ligands at once around a fixed transition-metal
centre -- "partial to total generation", up to every ligand from a bare metal.
It is a direct extension of LigandDiff, already integrated here as
``modules/tasks/diffusion_ligandiff.py``.

**The network is not re-ported.** ``EDM`` and ``Dynamics`` are imported
unchanged from ``MolecularDiffusion.modules.models.ligandiff`` and constructed
with ``ligand_group_node_nf=7``; that reproduces the released checkpoint's
shapes exactly (``strict=True`` load, 238/238 tensors matched). See
``docs/model_integrations/ligandiff_multi/INTEGRATION_PLAN.md`` for the
experiment. Only the 20-metal geometry helpers are new, in
``modules/models/ligandiff_multi/utils.py``.

What is new relative to ``ligandiff``:

* the conditioning channel is **7** wide, ``cat([ligand_group(6),
  coord_site(1)])`` (upstream ``src/lightning.py:126``), where ``coord_site``
  is a per-atom "coordinates the metal" flag. ``ligand_group`` itself is still
  6 slots.
* the row-data contract is therefore **16** columns, not 15.
* ``normalize_factors`` is ``[10, 4, 1]``, not ``[1, 4, 10]``.
* ``sample()`` grows *several* ligands, picking an octahedral denticity
  partition from ``CN_OCT`` (upstream ``generate.py::reform_data``).

Two vocabularies, deliberately different widths:

* ``in_node_nf = 8`` -- the model's / dataset's one-hot width, upstream's
  heavy-atom vocab ``{C,N,O,S,Br,Cl,P,F}`` in ``src/const.py:11`` order
  (byte-identical to LigandDiff's). The metal carries an **all-zero** row and
  its element travels out of band.
* ``task.atom_vocab`` -- 28 entries (those 8, then the 20 metals). A
  decode-side symbol table only; it sizes no tensor. It is mandatory because
  ``runmodes/generate/tasks_generate.py:1463`` resolves reference-structure
  elements with ``onehot(..., allow_unknown=False)``.
"""

import random
from collections import Counter
from typing import List, Optional

import torch
import torch.nn as nn
from ase.data import atomic_numbers as ase_atomic_numbers

from MolecularDiffusion.modules.models.difflinker.linker_size import (
    DistributionNodes,
)
from MolecularDiffusion.modules.models.ligandiff.edm import EDM
from MolecularDiffusion.modules.models.ligandiff.egnn import Dynamics
from MolecularDiffusion.modules.models.ligandiff.utils import (
    remove_partial_mean_with_mask,
)
from MolecularDiffusion.modules.models.ligandiff_multi import utils as lm_utils

# Upstream heavy-atom one-hot order, verbatim from multi_LigandDiff's
# src/const.py:11 (ATOM2IDX) -- identical to LigandDiff's. Column order is
# load-bearing: the released checkpoint and the converted dataset use it.
LIGANDIFF_ATOM_VOCAB = ["C", "N", "O", "S", "Br", "Cl", "P", "F"]

# src/const.py:17 `metals` -- the twenty supported metals, appended at indices
# 8..27 of task.atom_vocab so the 0..7 heavy-atom ordering is preserved.
LIGANDIFF_MULTI_METAL_VOCAB = [
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr",
    "Mo", "Ru", "Rh", "Pd", "Cd", "W", "Re", "Os", "Ir", "Pt",
]

LIGANDIFF_MULTI_FULL_VOCAB = (
    LIGANDIFF_ATOM_VOCAB + LIGANDIFF_MULTI_METAL_VOCAB
)

# Column layout that docs/model_integrations/ligandiff_multi/scripts/
# convert_dataset.py packs into `row.data["node_features"]` and this module's
# adapter slices back out. Implicit contract -- no shared constant enforces
# it, so both files name the order in a comment.
LIGANDIFF_MULTI_ROW_DATA_COLUMNS = (
    *[f"one_hot_{s}" for s in LIGANDIFF_ATOM_VOCAB],
    "ligand_diff",
    *[f"ligand_group_{i}" for i in range(6)],
    "coord_site",
)

_N_ONEHOT = len(LIGANDIFF_ATOM_VOCAB)  # 8
_N_GROUP = 6  # ligand slots -- NOT the 7-wide conditioning channel
_N_SITE = 1
_N_ROW_DATA = _N_ONEHOT + 1 + _N_GROUP + _N_SITE  # 16
_N_LIGAND_SITE = _N_GROUP + _N_SITE  # 7 == ligand_group_node_nf


class PointCloudToLigandDiffMultiBatch:
    """Dense ``(B, N, ·)`` PointCloud batch -> multi-LigandDiff's flat layout.

    Identical in shape to ``ligandiff``'s adapter, one column wider. The
    dataset supplies
    ``[one_hot(8) | ligand_diff(1) | ligand_group(6) | coord_site(1)]``
    through the existing ``data.use_row_data_features`` seam
    (``data/component/dataset.py:1328``).

    The 16 columns are taken from the **right-hand end** of ``node_feature``,
    not from index 0: ``DataModule.load()``'s ``data_type: pointcloud`` branch
    (``runmodes/train/data.py:294-312``) does not forward ``use_ohe_feature``
    the way its ``pyg`` sibling at :284 does, so the 28-wide atom OHE is
    always prepended regardless of the config flag. Slicing from the end is
    correct either way and needs no change to shared code.

    There is no synthesis fallback: the Zenodo splits ship every column
    precomputed, so a missing one means a misconfigured dataset and must raise
    rather than silently train on a fabricated split.
    """

    def __init__(self) -> None:
        self._verified = False

    def __call__(self, batch: dict) -> dict:
        node_feature = batch.get("node_feature")
        if node_feature is None:
            node_feature = batch.get("node_features")
        if node_feature is None:
            node_feature = batch["x"]

        coords = batch["coords"]
        device = coords.device
        node_mask = batch["node_mask"].float().to(device)

        if node_feature.shape[-1] < _N_ROW_DATA:
            raise ValueError(
                "multi-LigandDiff needs "
                f"{_N_ROW_DATA} node feature columns "
                f"({LIGANDIFF_MULTI_ROW_DATA_COLUMNS}), got "
                f"{node_feature.shape[-1]}. The dataset must be built by "
                "docs/model_integrations/ligandiff_multi/scripts/"
                "convert_dataset.py and read with "
                "`data.use_row_data_features: true`."
            )

        node_feature = node_feature.to(device)
        keep = node_mask.reshape(-1).bool()
        bsz, n_atoms = node_mask.shape

        batch_seg = (
            torch.arange(bsz, device=device)
            .unsqueeze(1)
            .expand(bsz, n_atoms)
            .reshape(-1)[keep]
        )

        # Trailing 16 columns -- see the class docstring on why this is not
        # sliced from index 0.
        flat_feat = node_feature.reshape(-1, node_feature.shape[-1])[
            keep, -_N_ROW_DATA:
        ]
        one_hot = flat_feat[:, :_N_ONEHOT]
        ligand_diff = flat_feat[:, _N_ONEHOT : _N_ONEHOT + 1]
        ligand_group = flat_feat[:, _N_ONEHOT + 1 : _N_ONEHOT + 1 + _N_GROUP]
        coord_site = flat_feat[:, _N_ONEHOT + 1 + _N_GROUP : _N_ROW_DATA]

        # Because the slice is positional-from-the-end, a dataset built
        # without `use_row_data_features: true` yields 16 plausible-looking
        # but meaningless columns rather than an error. Check the block
        # actually is what the converter wrote (once, on the first batch). A
        # 0/1 test alone is NOT enough: the atom OHE the pointcloud path
        # prepends is also 0/1. The discriminating property is ligand_group --
        # a per-complex ligand-slot one-hot with exactly one all-zero row (the
        # metal), which no slice of the atom OHE satisfies. The check is
        # structural, not a hardcoded width.
        if not self._verified:
            group_sum = ligand_group.sum(dim=-1)
            ok = (
                bool(torch.all((flat_feat == 0) | (flat_feat == 1)))
                and bool(torch.all((group_sum == 0) | (group_sum == 1)))
                and int((group_sum == 0).sum()) == bsz
            )
            if not ok:
                raise ValueError(
                    "The trailing 16 node_feature columns are not the "
                    "[one_hot(8) | ligand_diff(1) | ligand_group(6) | "
                    "coord_site(1)] block written by "
                    "docs/model_integrations/ligandiff_multi/scripts/"
                    "convert_dataset.py (expected a 0/1 block whose "
                    "ligand_group is one-hot with exactly one all-zero row "
                    "per complex, i.e. the metal). Set "
                    "`data.use_row_data_features: true` and point "
                    "`data.ase_db_path` at a db that script produced."
                )
            if ligand_diff.sum() == 0:
                raise ValueError(
                    "Every atom is context (ligand_diff all zero) -- there "
                    "is nothing to diffuse. The dataset's ligand_diff column "
                    "is missing or in the wrong position."
                )
            if coord_site.sum() == 0:
                raise ValueError(
                    "coord_site is all zero -- no atom coordinates the "
                    "metal. The dataset's coord_site column is missing or in "
                    "the wrong position."
                )
            self._verified = True

        charges = batch.get("charges")
        if charges is not None:
            charges = charges.to(device).reshape(-1)[keep]

        return {
            "pos": coords.reshape(-1, coords.shape[-1])[keep],
            "one_hot": one_hot,
            "ligand_diff": ligand_diff,
            # src/lightning.py:126 -- the 7-wide conditioning channel.
            "ligand_site": torch.cat([ligand_group, coord_site], dim=-1),
            # context is exactly the complement of ligand_diff (verified on
            # all 404,126 complexes of the three Zenodo splits).
            "context": 1.0 - ligand_diff,
            "charges": charges,
            "batch_seg": batch_seg,
            "batch_size": bsz,
            "natoms": node_mask.sum(dim=1).round().long(),
        }


def LigandDiffMultiToPointCloud(  # noqa: N802
    coords_flat: torch.Tensor,
    charges_flat: torch.Tensor,
    batch_seg: torch.Tensor,
    batch_size: int,
    atom_vocab: list,
):
    """Flat ``(N_total, ·)`` output -> dense PointCloud tensors.

    ``charges_flat`` must already be the *final* per-atom atomic number: taken
    from the known context elements for ``context`` rows (which is what stops
    the metal's all-zero one-hot decoding to index 0, i.e. carbon) and from
    the 8-way ``one_hot.argmax`` for generated rows. The returned one-hot is
    re-expanded from those atomic numbers onto ``atom_vocab`` (28 entries).

    Returns ``(one_hot, charges, coords, node_mask)``, all dense.
    """
    device = coords_flat.device
    counts = torch.bincount(batch_seg, minlength=batch_size)
    max_atoms = int(counts.max().item())
    n_vocab = len(atom_vocab)

    z_to_idx = {ase_atomic_numbers[s]: i for i, s in enumerate(atom_vocab)}

    coords = torch.zeros(batch_size, max_atoms, 3, device=device)
    charges = torch.zeros(
        batch_size, max_atoms, device=device, dtype=torch.long
    )
    one_hot = torch.zeros(batch_size, max_atoms, n_vocab, device=device)
    node_mask = torch.zeros(batch_size, max_atoms, device=device)

    offsets = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(batch_seg.shape[0], device=device) - offsets[batch_seg]

    coords[batch_seg, slot] = coords_flat
    charges[batch_seg, slot] = charges_flat.long()
    node_mask[batch_seg, slot] = 1.0

    vocab_idx = torch.tensor(
        [z_to_idx[int(z)] for z in charges_flat.tolist()], device=device
    )
    one_hot[batch_seg, slot, vocab_idx] = 1.0

    return one_hot, charges, coords, node_mask


class LigandDiffMultiTaskFactory:
    """Factory matching ``cli/train.py``'s ``build()`` instantiation pattern.

    The trailing ``**kwargs`` is load-bearing, not decoration:
    ``cli/train.py:649-651`` passes ``node_feature_dim`` /
    ``extra_norm_values`` into every factory. Both are meaningless here (the
    16 row-data columns are consumed by the adapter, not normalised as extra
    features) and are deliberately absorbed and ignored.
    """

    def __init__(
        self,
        task_type: str,
        in_node_nf: int = 8,
        n_dims: int = 3,
        # 7 = ligand_group(6) + coord_site(1). Sizes the released
        # checkpoint's ligand_site_embedding (8 -> 192, the +1 being time)
        # and h_embedding_out (192 -> 16).
        ligand_group_node_nf: int = 7,
        hidden_nf: int = 192,
        n_layers: int = 5,
        activation: str = "silu",
        attention: bool = True,
        tanh: bool = True,
        norm_constant: float = 1,
        inv_sublayers: int = 1,
        sin_embedding: bool = False,
        normalization_factor: float = 100,
        aggregation_method: str = "sum",
        model: str = "gvp_dynamics",
        normalization: Optional[str] = "batch_norm",
        condition_time: bool = True,
        drop_rate: float = 0.2,
        diffusion_steps: int = 500,
        diffusion_noise_schedule: str = "learned",
        diffusion_noise_precision: float = 1e-5,
        diffusion_loss_type: str = "vlb",
        normalize_factors: tuple = (10, 4, 1),
        center_of_mass: str = "context",
        atom_vocab: Optional[list] = None,
        denticity_split: Optional[list] = None,
        **kwargs,
    ) -> None:
        self.task_type = task_type
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.ligand_group_node_nf = ligand_group_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.activation = activation
        self.attention = attention
        self.tanh = tanh
        self.norm_constant = norm_constant
        self.inv_sublayers = inv_sublayers
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.model = model
        self.normalization = normalization
        self.condition_time = condition_time
        self.drop_rate = drop_rate
        self.diffusion_steps = diffusion_steps
        self.diffusion_noise_schedule = diffusion_noise_schedule
        self.diffusion_noise_precision = diffusion_noise_precision
        self.diffusion_loss_type = diffusion_loss_type
        self.normalize_factors = tuple(normalize_factors)
        self.center_of_mass = center_of_mass
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.denticity_split = (
            list(denticity_split) if denticity_split else None
        )
        self.kwargs = kwargs

    def build(self) -> "LigandDiffMultiTask":
        """Instantiate the task module."""
        self.task = LigandDiffMultiTask(
            in_node_nf=self.in_node_nf,
            n_dims=self.n_dims,
            ligand_group_node_nf=self.ligand_group_node_nf,
            hidden_nf=self.hidden_nf,
            n_layers=self.n_layers,
            activation=self.activation,
            attention=self.attention,
            tanh=self.tanh,
            norm_constant=self.norm_constant,
            inv_sublayers=self.inv_sublayers,
            sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            aggregation_method=self.aggregation_method,
            model=self.model,
            normalization=self.normalization,
            condition_time=self.condition_time,
            drop_rate=self.drop_rate,
            diffusion_steps=self.diffusion_steps,
            diffusion_noise_schedule=self.diffusion_noise_schedule,
            diffusion_noise_precision=self.diffusion_noise_precision,
            diffusion_loss_type=self.diffusion_loss_type,
            normalize_factors=self.normalize_factors,
            center_of_mass=self.center_of_mass,
            atom_vocab=self.atom_vocab,
            denticity_split=self.denticity_split,
        )
        return self.task


class LigandDiffMultiTask(nn.Module):
    """Plain ``nn.Module`` task (docs/adding_new_models.md §2.6) wrapping the
    **imported, unmodified** LigandDiff ``EDM``.

    The EDM is held at ``self.edm`` because the released checkpoint's 238
    tensors are all named ``edm.*``; any other attribute name makes every key
    "unexpected" and ``cli/generate.py:322``'s ``strict=False`` load drops
    them all silently.
    """

    def __init__(
        self,
        in_node_nf: int,
        n_dims: int,
        ligand_group_node_nf: int,
        hidden_nf: int,
        n_layers: int,
        activation: str,
        attention: bool,
        tanh: bool,
        norm_constant: float,
        inv_sublayers: int,
        sin_embedding: bool,
        normalization_factor: float,
        aggregation_method: str,
        model: str,
        normalization: Optional[str],
        condition_time: bool,
        drop_rate: float,
        diffusion_steps: int,
        diffusion_noise_schedule: str,
        diffusion_noise_precision: float,
        diffusion_loss_type: str,
        normalize_factors: tuple,
        center_of_mass: str,
        atom_vocab: Optional[list] = None,
        denticity_split: Optional[list] = None,
    ) -> None:
        super().__init__()

        if ligand_group_node_nf != _N_LIGAND_SITE:
            raise ValueError(
                "multi-LigandDiff's conditioning channel is "
                f"{_N_LIGAND_SITE} wide (ligand_group(6) + coord_site(1)); "
                f"got ligand_group_node_nf={ligand_group_node_nf}. The "
                "released checkpoint's h_embedding_out is (16, 192) = "
                "8 + 7 + 1, so any other value makes the load a silent no-op "
                "under cli/generate.py:322's strict=False."
            )
        # The ONE line by which upstream's edm.py differs from the imported
        # LigandDiff edm.py is `ligand_site = (ligand_site - norm_biases[2]) /
        # norm_values[2]` (src/edm.py:50), which is the identity iff
        # norm_values[2] == 1 -- as it is at the released [10, 4, 1]. That is
        # what makes importing the unmodified EDM valid. Upstream itself has
        # the same line commented out in sample_chain (src/edm.py:131), so any
        # other value would also be a train/sample inconsistency there.
        if float(normalize_factors[2]) != 1.0:
            raise ValueError(
                "normalize_factors[2] must be 1 for multi-LigandDiff: it "
                "scales the ligand_site conditioning channel, and the "
                "imported (unmodified) LigandDiff EDM does not apply that "
                "scaling. Got "
                f"{list(normalize_factors)}."
            )

        dynamics = Dynamics(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            ligand_group_node_nf=ligand_group_node_nf,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            device="cpu",
            model=model,
            normalization=normalization,
            condition_time=condition_time,
            drop_rate=drop_rate,
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
        self.ligand_group_node_nf = ligand_group_node_nf
        self.center_of_mass = center_of_mass
        self.loss_type = diffusion_loss_type
        self.T = diffusion_steps
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.denticity_split = (
            [int(d) for d in denticity_split] if denticity_split else None
        )

        # Read generically off `task.model` by tasks_generate.py's
        # preprocess_ref_structure / structural_guidance.
        self.norm_values = tuple(normalize_factors)
        self.ndim_extra = 0
        self.prop_dist_model = None

        self._adapter = PointCloudToLigandDiffMultiBatch()
        # Lazy total-atom-count histogram instead of a `train_set`-at-build
        # seam, persisted through get_extra_state/set_extra_state.
        self._atom_count_counts: Counter = Counter()
        self._node_dist_cache: Optional[DistributionNodes] = None
        self._node_dist_override: Optional[DistributionNodes] = None
        self._n_node_dist_override: Optional[dict] = None

    # ------------------------------------------------------------------
    # checkpoint-persisted stats
    # ------------------------------------------------------------------

    def get_extra_state(self) -> dict:
        """Persist the atom-count histogram into the checkpoint."""
        return {"atom_count_counts": dict(self._atom_count_counts)}

    def set_extra_state(self, state: dict) -> None:
        """Restore the atom-count histogram from a checkpoint."""
        counts = (
            state.get("atom_count_counts") if isinstance(state, dict) else None
        )
        if counts:
            self._atom_count_counts = Counter(counts)
            self._node_dist_cache = None

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def forward(self, batch: dict):
        """Compute the diffusion loss (src/lightning.py:124-176)."""
        pc = self._adapter(batch)
        x = pc["pos"]
        h = pc["one_hot"]
        context = pc["context"]
        ligand_diff = pc["ligand_diff"]
        ligand_site = pc["ligand_site"]
        batch_seg = pc["batch_seg"]
        batch_size = pc["batch_size"]

        if self.center_of_mass == "context":
            com_mask = context
        elif self.center_of_mass == "ligand_diff":
            com_mask = ligand_diff
        else:
            raise ValueError(f"Unknown center_of_mass: {self.center_of_mass}")
        x = remove_partial_mean_with_mask(x, com_mask, batch_seg)

        (
            delta_log_px,
            error_t,
            SNR_weight,
            loss_0_x,
            loss_0_h,
            neg_log_const_0,
            kl_prior,
        ) = self.edm.forward(
            x=x,
            h=h,
            context=context,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=batch_size,
            # The imported EDM's `ligand_group` argument IS multi's
            # `ligand_site`; only the identifier differs upstream.
            ligand_group=ligand_site,
        )

        if self.loss_type == "l2" and self.training:
            normalization = (
                self.n_dims + self.in_node_nf
            ) * EDM.inflate_batch_array(ligand_diff, batch_seg)
            error_t = error_t / normalization
            loss_t = error_t
            loss_0_x = (
                loss_0_x
                / self.n_dims
                * EDM.inflate_batch_array(ligand_diff, batch_seg)
            )
            loss_0 = loss_0_x + loss_0_h
        else:
            loss_t = self.T * 0.5 * SNR_weight * error_t
            loss_0 = loss_0_x + loss_0_h + neg_log_const_0

        nll = loss_t + loss_0 + kl_prior
        if not (self.loss_type == "l2" and self.training):
            nll = nll - delta_log_px

        # Total atom count (context + all generated ligands), NOT
        # ligand-only -- a ligand-only histogram sits below every real
        # scaffold and forces a permanent clamp in
        # GenerativeFactory._enforce_scaffold_size.
        for size in pc["natoms"].tolist():
            if size > 0:
                self._atom_count_counts[int(size)] += 1
        self._node_dist_cache = None

        stats = {
            "error_t": error_t.mean(0).detach(),
            "SNR_weight": SNR_weight.mean(0).detach(),
            "loss_0": loss_0.mean(0).detach(),
            "kl_prior": kl_prior.mean(0).detach(),
            "delta_log_px": delta_log_px.mean(0).detach(),
            "neg_log_const_0": neg_log_const_0.mean(0).detach(),
        }
        return nll.mean(0), stats

    def predict_and_target(self, batch: dict):
        """Validation hook: the loss doubles as the prediction."""
        loss, _stats = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Validation metric."""
        del target
        return {"val_loss": pred.mean()}

    # ------------------------------------------------------------------
    # size distribution
    # ------------------------------------------------------------------

    @property
    def atom_count_histogram(self) -> Optional[dict]:
        """Total-atom-count histogram accumulated from real batches."""
        if self._atom_count_counts:
            return dict(self._atom_count_counts)
        if self._n_node_dist_override is not None:
            return dict(self._n_node_dist_override)
        return None

    @property
    def node_dist_model(self) -> Optional[DistributionNodes]:
        """Sampler over total molecule sizes."""
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
        """Raw histogram, as core/engine_lightning.py checkpoints it."""
        return self.atom_count_histogram

    @n_node_dist.setter
    def n_node_dist(self, value: dict) -> None:
        self._n_node_dist_override = dict(value) if value else None
        self._node_dist_cache = None

    @property
    def model(self):
        """``tasks_generate.py`` reads ``T``/``norm_values`` off this."""
        return self

    @property
    def device(self) -> torch.device:
        """Device the parameters live on."""
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------

    def _choose_partition(
        self,
        cn_context: int,
        n_free_slots: int,
        n_new: int,
        outpaint_cfgs: Optional[dict],
    ) -> List[int]:
        """Pick the denticity partition for the ligands to be generated.

        Every partition in ``CN_OCT[cn_context]`` sums to ``6 - cn_context``,
        so upstream's ``assert sum(coord_site) == 6``
        (``generate.py:210``) holds by construction. Precedence:
        ``outpaint_cfgs['denticity_split']`` (forwarded from
        ``interference.condition_configs.outpaint_cfgs`` by
        ``tasks_generate.py:991`` -- an existing seam), else the task yaml's
        ``denticity_split``, else a random feasible one.
        """
        if cn_context not in lm_utils.CN_OCT:
            raise ValueError(
                f"the reference scaffold already occupies {cn_context} of "
                "the metal's 6 octahedral coordination sites, so there is "
                "nothing left to generate. multi-LigandDiff is octahedral-"
                "only (upstream generate.py:210). Remove a ligand from the "
                "scaffold."
            )
        requested = None
        if outpaint_cfgs:
            requested = outpaint_cfgs.get("denticity_split")
        if requested is None:
            requested = self.denticity_split
        if requested is not None:
            part = [int(d) for d in requested]
            if sorted(part, reverse=True) not in [
                sorted(p, reverse=True) for p in lm_utils.CN_OCT[cn_context]
            ]:
                raise ValueError(
                    f"denticity_split {part} is not an octahedral partition "
                    f"of the {6 - cn_context} free coordination sites. Valid "
                    f"options: {lm_utils.CN_OCT[cn_context]}"
                )
            if len(part) > n_free_slots:
                raise ValueError(
                    f"denticity_split {part} needs {len(part)} ligand slots "
                    f"but only {n_free_slots} of the 6 are free."
                )
            if sum(part) > n_new:
                raise ValueError(
                    f"denticity_split {part} needs at least {sum(part)} new "
                    f"atoms but mol_size leaves only {n_new}."
                )
            return part

        feasible = [
            p
            for p in lm_utils.CN_OCT[cn_context]
            if len(p) <= n_free_slots and sum(p) <= n_new
        ]
        if not feasible:
            raise ValueError(
                f"no octahedral partition of the {6 - cn_context} free "
                f"coordination sites fits in {n_free_slots} free ligand "
                f"slots with {n_new} new atoms. Increase mol_size."
            )
        return list(random.choice(feasible))

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
        """Grow several ligands around a fixed metal-complex scaffold.

        Dispatched through the bundled ``outpaint`` GenerativeFactory path
        (``configs/interference/gen_outpaint.yaml`` ->
        ``tasks_generate.py::structural_guidance``), whose mechanism -- freeze
        a reference scaffold, draw a strictly larger total size, grow the
        difference -- is exactly upstream's ``generate.py::reform_data``. The
        scaffold may be as small as a bare metal (upstream's ``[]_[...]``
        total-generation case).

        Out of scope this pass, accepted and ignored: ``batch``,
        ``num_steps``, ``condition_mode``, ``use_noised_conditioning``,
        ``n_frames`` (no trajectory export), ``n_retrys``/``t_retry``,
        ``context`` (no ``prop_dist_model``).
        """
        del batch, num_steps, condition_mode, batch_size
        del use_noised_conditioning, n_frames, n_retrys, t_retry, context
        del kwargs

        if condition_tensor is None:
            raise NotImplementedError(
                "LigandDiffMultiTask.sample() requires a reference complex "
                "(condition_tensor) -- multi-LigandDiff has no unconditional "
                "generation mode. Use the 'outpaint' interference dispatch "
                "with reference_structure_path set (see "
                "configs/interference/gen_outpaint.yaml)."
            )
        if nodesxsample is None:
            raise ValueError(
                "LigandDiffMultiTask.sample() requires nodesxsample (TOTAL "
                "atom counts: scaffold + all new ligands)."
            )
        if self.atom_vocab is None:
            raise ValueError(
                "LigandDiffMultiTask.atom_vocab is not set -- declare it in "
                "configs/tasks/diffusion_ligandiff_multi.yaml."
            )

        device = self.device
        nodesxsample = nodesxsample.to(device).long()
        bsz = int(nodesxsample.shape[0])

        norm_coords, _norm_feats, norm_charges = self.norm_values

        condition_tensor = condition_tensor.to(device)
        ref_natoms = condition_tensor.shape[1]
        ref_pos = condition_tensor[..., :3] * norm_coords
        ref_z = (condition_tensor[..., -1] * norm_charges).round().long()
        if ref_pos.shape[0] != bsz:
            ref_pos = ref_pos.expand(bsz, -1, -1)
            ref_z = ref_z.expand(bsz, -1)

        # Rebuild the model's 8-wide h from the true elements: heavy atoms get
        # their upstream column, metals stay all-zero (generate.py:78-80).
        z_to_col = {
            ase_atomic_numbers[s]: i
            for i, s in enumerate(LIGANDIFF_ATOM_VOCAB)
        }
        ref_h = torch.zeros(bsz, ref_natoms, self.in_node_nf, device=device)
        for b in range(bsz):
            for a, z in enumerate(ref_z[b].tolist()):
                col = z_to_col.get(int(z))
                if col is not None:
                    ref_h[b, a, col] = 1.0
                elif int(z) not in lm_utils.METAL_Z:
                    raise ValueError(
                        f"Reference atom {a} has Z={z}, which is neither one "
                        f"of the heavy atoms {LIGANDIFF_ATOM_VOCAB} nor one "
                        "of the 20 supported metals."
                    )

        pos_l, h_l, ld_l, ls_l, seg_l, z_l = [], [], [], [], [], []
        for b in range(bsz):
            n_new = max(int(nodesxsample[b].item()) - ref_natoms, 1)

            # Recover the scaffold's ligand-slot decomposition and its
            # metal-coordinating atoms the way molSimplify's
            # ligand_breakdown/ligcon do upstream (generate.py:91-106).
            groups = lm_utils.ligand_groups_from_geometry(
                ref_pos[b], ref_z[b], n_slots=_N_GROUP
            ).to(device)
            sites = (
                lm_utils.coord_sites_from_geometry(ref_pos[b], ref_z[b])
                .to(device)
                .unsqueeze(-1)
            )

            free_slots = (
                (groups.sum(dim=0) == 0).nonzero(as_tuple=True)[0].tolist()
            )
            part = self._choose_partition(
                cn_context=int(sites.sum().item()),
                n_free_slots=len(free_slots),
                n_new=n_new,
                outpaint_cfgs=outpaint_cfgs,
            )
            sizes = lm_utils.distribute_atoms(n_new, part)

            new_groups = torch.zeros(n_new, _N_GROUP, device=device)
            new_sites = torch.zeros(n_new, 1, device=device)
            off = 0
            for slot, denticity, size in zip(free_slots, part, sizes):
                new_groups[off : off + size, slot] = 1.0
                new_sites[off : off + denticity] = 1.0
                off += size

            pos_l.append(
                torch.cat(
                    [ref_pos[b], torch.zeros(n_new, 3, device=device)], dim=0
                )
            )
            h_l.append(
                torch.cat(
                    [
                        ref_h[b],
                        torch.zeros(n_new, self.in_node_nf, device=device),
                    ],
                    dim=0,
                )
            )
            ld_l.append(
                torch.cat(
                    [
                        torch.zeros(ref_natoms, 1, device=device),
                        torch.ones(n_new, 1, device=device),
                    ],
                    dim=0,
                )
            )
            ls_l.append(
                torch.cat(
                    [
                        torch.cat([groups, sites], dim=-1),
                        torch.cat([new_groups, new_sites], dim=-1),
                    ],
                    dim=0,
                )
            )
            z_l.append(
                torch.cat(
                    [
                        ref_z[b],
                        torch.zeros(n_new, device=device, dtype=torch.long),
                    ],
                    dim=0,
                )
            )
            seg_l.append(
                torch.full(
                    (ref_natoms + n_new,), b, device=device, dtype=torch.long
                )
            )

        x = torch.cat(pos_l, dim=0)
        h = torch.cat(h_l, dim=0)
        ligand_diff = torch.cat(ld_l, dim=0)
        ligand_site = torch.cat(ls_l, dim=0)
        batch_seg = torch.cat(seg_l, dim=0)
        ctx = 1.0 - ligand_diff
        ref_z_flat = torch.cat(z_l, dim=0)

        # Sampling runs in the context's centre-of-mass frame; the shift is
        # added back afterwards (generate.py:231,241).
        fixed_mean = torch.zeros(bsz, 3, device=device)
        fixed_mean.index_add_(0, batch_seg, x * ctx)
        fixed_mean = fixed_mean / ctx.new_zeros(bsz, 1).index_add_(
            0, batch_seg, ctx
        )
        x = remove_partial_mean_with_mask(x, ctx, batch_seg)

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            context=ctx,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=bsz,
            ligand_group=ligand_site,
            keep_frames=1,
        )
        final = chain[0]
        out_pos = final[:, :3] + fixed_mean[batch_seg]
        out_onehot8 = final[:, 3:]

        # Context rows keep their known element (this is what stops the
        # metal's all-zero row decoding to carbon); generated rows decode by
        # argmax over the 8 heavy types, which is always one of them.
        heavy_z = torch.tensor(
            [ase_atomic_numbers[s] for s in LIGANDIFF_ATOM_VOCAB],
            device=device,
            dtype=torch.long,
        )
        gen_z = heavy_z[out_onehot8.argmax(dim=-1)]
        is_gen = ligand_diff.squeeze(-1).bool()
        out_z = torch.where(is_gen, gen_z, ref_z_flat)

        return LigandDiffMultiToPointCloud(
            out_pos, out_z, batch_seg, bsz, self.atom_vocab
        )
