"""LigandDiff integration with the MolecularDiffusion data pipeline.

LigandDiff (https://github.com/lgd6/LigandDiff, JCTC 2024) regenerates one
ligand of a 3D transition-metal complex while keeping the metal and the
remaining ligands frozen. It is an EDM-family continuous 3D DDPM with two
complementary per-atom masks (``context`` / ``ligand_diff``) plus a
categorical ligand-slot embedding (``ligand_group``), denoised by a GVP
network over a runtime fully-connected graph.

See ``docs/model_integrations/ligandiff/INTEGRATION_PLAN.md`` for the approved
plan this module implements. Ported model code lives under
``MolecularDiffusion.modules.models.ligandiff``.

Two vocabularies, deliberately different widths (INTEGRATION_PLAN.md, "Two
vocabularies, one model width"):

* ``in_node_nf = 8`` -- the model's / dataset's one-hot width, upstream's
  heavy-atom vocab ``{C,N,O,S,Br,Cl,P,F}`` in ``src/const.py:11`` order. The
  metal carries an **all-zero** row and its element travels out of band. The
  released checkpoint's ``edm.dynamics.h_embedding.weight`` is ``(192, 8)``,
  so this width must not be widened.
* ``task.atom_vocab`` -- 18 entries (those 8, then the 10 metals). A
  decode-side symbol table only; it sizes no tensor. It is mandatory because
  ``runmodes/generate/tasks_generate.py:1463`` resolves reference-structure
  elements with ``onehot(..., allow_unknown=False)``, which raises on every
  transition-metal complex against an 8-entry vocab.
"""

from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from ase.data import atomic_numbers as ase_atomic_numbers

from MolecularDiffusion.modules.models.difflinker.linker_size import (
    DistributionNodes,
)
from MolecularDiffusion.modules.models.ligandiff import utils as ld_utils
from MolecularDiffusion.modules.models.ligandiff.edm import EDM
from MolecularDiffusion.modules.models.ligandiff.egnn import Dynamics

# Upstream heavy-atom one-hot order, verbatim from LigandDiff's
# src/const.py:11 (ATOM2IDX). Column order is load-bearing: the released
# checkpoint and the converted dataset both use it.
LIGANDIFF_ATOM_VOCAB = ["C", "N", "O", "S", "Br", "Cl", "P", "F"]

# src/const.py:14 -- the ten supported metals, appended at indices 8..17 of
# task.atom_vocab so the 0..7 heavy-atom ordering above is preserved.
LIGANDIFF_METAL_VOCAB = [
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ru",
    "Pd",
    "Pt",
]

LIGANDIFF_FULL_VOCAB = LIGANDIFF_ATOM_VOCAB + LIGANDIFF_METAL_VOCAB

# Column layout that docs/model_integrations/ligandiff/scripts/
# convert_dataset.py packs into `row.data["node_features"]` and this module's
# adapter slices back out. Implicit contract -- no shared constant enforces
# it, so both files name the order in a comment.
LIGANDIFF_ROW_DATA_COLUMNS = (
    *[f"one_hot_{s}" for s in LIGANDIFF_ATOM_VOCAB],
    "ligand_diff",
    *[f"ligand_group_{i}" for i in range(6)],
)

_N_ONEHOT = len(LIGANDIFF_ATOM_VOCAB)  # 8
_N_GROUP = 6
_N_ROW_DATA = _N_ONEHOT + 1 + _N_GROUP  # 15


class PointCloudToLigandDiffBatch:
    """Dense ``(B, N, ·)`` PointCloud batch -> LigandDiff's flat layout.

    LigandDiff works on a ragged batch: every atom of every molecule
    concatenated on dim 0, membership via ``batch_seg``. This adapter selects
    the valid rows and slices the conditioning columns out of
    ``node_feature``, which the dataset supplies as
    ``[one_hot(8) | ligand_diff(1) | ligand_group(6)]`` through the existing
    ``data.use_row_data_features`` seam (``data/component/dataset.py:1328``).

    The 15 columns are taken from the **right-hand end** of ``node_feature``,
    not from index 0. That is the row-data seam's own guarantee --
    ``data/component/dataset.py:1328`` appends ``row.data["node_features"]``
    with ``torch.cat((node_features, row_feat), dim=1)`` -- and it is
    load-bearing here: ``DataModule.load()``'s ``data_type: pointcloud``
    branch (``runmodes/train/data.py:294-312``) does not forward
    ``use_ohe_feature`` to ``pointcloud_dataset`` the way its ``pyg`` sibling
    at :284 does, so the atom OHE is always prepended regardless of the
    config flag. Slicing from the end is correct either way and needs no
    change to shared code. The prepended OHE is simply unused: the model's
    8-wide ``h`` comes from these columns, so ``in_node_nf`` stays 8 and the
    released checkpoint still loads.

    There is no synthesis fallback: the Zenodo splits ship ``ligand_diff`` and
    ``ligand_group`` precomputed, so a missing column means a misconfigured
    dataset and must raise rather than silently train on a fabricated split.
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
                "LigandDiff needs "
                f"{_N_ROW_DATA} node feature columns "
                f"({LIGANDIFF_ROW_DATA_COLUMNS}), got "
                f"{node_feature.shape[-1]}. The dataset must be built by "
                "docs/model_integrations/ligandiff/scripts/"
                "convert_dataset.py and read with "
                "`data.use_ohe_feature: false` + "
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

        # Trailing 15 columns -- see the class docstring on why this is not
        # sliced from index 0.
        flat_feat = node_feature.reshape(-1, node_feature.shape[-1])[
            keep, -_N_ROW_DATA:
        ]
        one_hot = flat_feat[:, :_N_ONEHOT]
        ligand_diff = flat_feat[:, _N_ONEHOT : _N_ONEHOT + 1]
        ligand_group = flat_feat[:, _N_ONEHOT + 1 : _N_ROW_DATA]

        # Because the slice is positional-from-the-end, a dataset built
        # without `use_row_data_features: true` yields 15 plausible-looking
        # but meaningless columns rather than an error. Check the block
        # actually is the indicator matrix the converter wrote (once, on the
        # first batch). A 0/1 test alone is NOT enough: the atom OHE the
        # pointcloud path prepends is also 0/1. The discriminating property is
        # ligand_group -- a per-complex ligand-slot one-hot with exactly one
        # all-zero row (the metal), which no slice of the atom OHE satisfies.
        if not self._verified:
            block = flat_feat
            group_sum = ligand_group.sum(dim=-1)
            ok = (
                bool(torch.all((block == 0) | (block == 1)))
                and bool(torch.all((group_sum == 0) | (group_sum == 1)))
                and int((group_sum == 0).sum()) == bsz
            )
            if not ok:
                raise ValueError(
                    "The trailing 15 node_feature columns are not the "
                    "[one_hot(8) | ligand_diff(1) | ligand_group(6)] block "
                    "written by "
                    "docs/model_integrations/ligandiff/scripts/"
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
            self._verified = True

        charges = batch.get("charges")
        if charges is not None:
            charges = charges.to(device).reshape(-1)[keep]

        return {
            "pos": coords.reshape(-1, coords.shape[-1])[keep],
            "one_hot": one_hot,
            "ligand_diff": ligand_diff,
            "ligand_group": ligand_group,
            # generate.py:93 -- context is exactly the complement of
            # ligand_diff (verified on all 400 val.pt complexes).
            "context": 1.0 - ligand_diff,
            "charges": charges,
            "batch_seg": batch_seg,
            "batch_size": bsz,
            "natoms": node_mask.sum(dim=1).round().long(),
        }


def LigandDiffToPointCloud(  # noqa: N802
    coords_flat: torch.Tensor,
    charges_flat: torch.Tensor,
    batch_seg: torch.Tensor,
    batch_size: int,
    atom_vocab: list,
):
    """Flat ``(N_total, ·)`` LigandDiff output -> dense PointCloud tensors.

    ``charges_flat`` must already be the *final* per-atom atomic number: taken
    from the known context elements for ``context`` rows (which is what stops
    the metal's all-zero one-hot decoding to index 0, i.e. carbon) and from
    the 8-way ``one_hot.argmax`` for generated ``ligand_diff`` rows. The
    returned one-hot is re-expanded from those atomic numbers onto the
    ``atom_vocab`` (18 entries), so ``argmax`` decoding and charge-based
    decoding agree and ``tasks_generate.py:1049``'s width check takes its
    normal branch.

    Returns ``(one_hot, charges, coords, node_mask)``, all dense.
    """
    device = coords_flat.device
    counts = torch.bincount(batch_seg, minlength=batch_size)
    max_atoms = int(counts.max().item())
    n_vocab = len(atom_vocab)

    z_to_idx = {
        ase_atomic_numbers[s]: i for i, s in enumerate(atom_vocab)
    }

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


class LigandDiffTaskFactory:
    """Factory matching ``cli/train.py``'s ``build()`` instantiation pattern.

    The trailing ``**kwargs`` is load-bearing, not decoration: this model runs
    with ``data.use_ohe_feature: false``, so ``cli/train.py:649-651`` computes
    ``inferred_extra_dim = 15`` and passes ``node_feature_dim`` /
    ``extra_norm_values`` into every factory. Both are meaningless here (the
    15 columns are consumed by the adapter, not normalised as extra features)
    and are deliberately absorbed and ignored, exactly as
    ``DiffLinkerTaskFactory`` does.
    """

    def __init__(
        self,
        task_type: str,
        in_node_nf: int = 8,
        n_dims: int = 3,
        ligand_group_node_nf: int = 6,
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
        diffusion_noise_schedule: str = "polynomial_2",
        diffusion_noise_precision: float = 1e-5,
        diffusion_loss_type: str = "vlb",
        normalize_factors: tuple = (1, 4, 10),
        center_of_mass: str = "context",
        atom_vocab: Optional[list] = None,
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
        self.kwargs = kwargs

    def build(self) -> "LigandDiffTask":
        """Instantiate the task module."""
        self.task = LigandDiffTask(
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
        )
        return self.task


class LigandDiffTask(nn.Module):
    """Plain ``nn.Module`` task (docs/adding_new_models.md §2.6) wrapping
    LigandDiff's ``EDM``.

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
    ) -> None:
        super().__init__()

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

        # Read generically off `task.model` by tasks_generate.py's
        # preprocess_ref_structure / structural_guidance.
        self.norm_values = tuple(normalize_factors)
        self.ndim_extra = 0
        self.prop_dist_model = None

        self._adapter = PointCloudToLigandDiffBatch()
        # Lazy total-atom-count histogram instead of a `train_set`-at-build
        # seam, persisted through get_extra_state/set_extra_state -- the
        # pattern DiffLinkerTask already proves.
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
        ligand_group = pc["ligand_group"]
        batch_seg = pc["batch_seg"]
        batch_size = pc["batch_size"]

        if self.center_of_mass == "context":
            com_mask = context
        elif self.center_of_mass == "ligand_diff":
            com_mask = ligand_diff
        else:
            raise ValueError(
                f"Unknown center_of_mass: {self.center_of_mass}"
            )
        x = ld_utils.remove_partial_mean_with_mask(x, com_mask, batch_seg)

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
            ligand_group=ligand_group,
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

        # Total atom count (context + ligand), NOT ligand-only -- see
        # INTEGRATION_PLAN.md's `n_node_dist` note: a ligand-only histogram
        # sits below every real scaffold and forces a permanent clamp in
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
    # size distribution (verbatim from DiffLinkerTask)
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
        """Regenerate one ligand around a fixed metal-complex scaffold.

        Dispatched through the bundled ``outpaint`` GenerativeFactory path
        (``configs/interference/gen_outpaint.yaml`` ->
        ``tasks_generate.py::structural_guidance``), whose mechanism -- freeze
        a reference scaffold, draw a strictly larger total size, grow the
        difference -- is exactly upstream's ``sampling.py::reform_data``.

        ``condition_tensor`` is ``(B, ref_natoms, 3 + len(atom_vocab) + 1)``
        as built by ``tasks_generate.py::preprocess_ref_structure``; only the
        coordinate block and the trailing charge column are used, because the
        model's own 8-wide one-hot has to be rebuilt from the true elements
        (the metal's row must be all-zero, which no 18-wide one-hot encodes).

        Out of scope this pass, accepted and ignored: ``batch``,
        ``num_steps``, ``condition_mode``, ``outpaint_cfgs``,
        ``use_noised_conditioning``, ``n_frames`` (no trajectory export),
        ``n_retrys``/``t_retry``, ``context`` (LigandDiff has no
        ``prop_dist_model``).
        """
        del batch, num_steps, condition_mode, outpaint_cfgs, batch_size
        del use_noised_conditioning, n_frames, n_retrys, t_retry, context
        del kwargs

        if condition_tensor is None:
            raise NotImplementedError(
                "LigandDiffTask.sample() requires a reference complex "
                "(condition_tensor) -- LigandDiff has no unconditional "
                "generation mode. Use the 'outpaint' interference dispatch "
                "with reference_structure_path set (see "
                "configs/interference/gen_outpaint.yaml)."
            )
        if nodesxsample is None:
            raise ValueError(
                "LigandDiffTask.sample() requires nodesxsample (TOTAL atom "
                "counts: scaffold + new ligand)."
            )
        if self.atom_vocab is None:
            raise ValueError(
                "LigandDiffTask.atom_vocab is not set -- declare it in "
                "configs/tasks/diffusion_ligandiff.yaml."
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
        # their upstream column, metals stay all-zero (generate.py:66-68).
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
                elif int(z) not in ld_utils.METAL_Z:
                    raise ValueError(
                        f"Reference atom {a} has Z={z}, which is neither one "
                        f"of LigandDiff's heavy atoms {LIGANDIFF_ATOM_VOCAB} "
                        f"nor one of its metals."
                    )

        pos_list, h_list, ld_list, lg_list, seg_list, z_list = [], [], [], [], [], []
        for b in range(bsz):
            n_new = max(int(nodesxsample[b].item()) - ref_natoms, 1)
            # Recover the scaffold's ligand-slot decomposition the way
            # molSimplify's ligand_breakdown does upstream (generate.py:81-87),
            # then give the regenerated ligand the next free slot -- upstream
            # reuses the removed ligand's slot (sampling.py:34-37), which for a
            # scaffold that no longer contains it is the first unused one.
            groups = ld_utils.ligand_groups_from_geometry(
                ref_pos[b], ref_z[b], n_slots=self.ligand_group_node_nf
            ).to(device)
            used = int((groups.sum(dim=0) > 0).sum().item())
            new_slot = min(used, self.ligand_group_node_nf - 1)
            new_groups = torch.zeros(
                n_new, self.ligand_group_node_nf, device=device
            )
            new_groups[:, new_slot] = 1.0

            pos_list.append(
                torch.cat(
                    [ref_pos[b], torch.zeros(n_new, 3, device=device)], dim=0
                )
            )
            h_list.append(
                torch.cat(
                    [
                        ref_h[b],
                        torch.zeros(n_new, self.in_node_nf, device=device),
                    ],
                    dim=0,
                )
            )
            ld_list.append(
                torch.cat(
                    [
                        torch.zeros(ref_natoms, 1, device=device),
                        torch.ones(n_new, 1, device=device),
                    ],
                    dim=0,
                )
            )
            lg_list.append(torch.cat([groups, new_groups], dim=0))
            z_list.append(
                torch.cat(
                    [
                        ref_z[b],
                        torch.zeros(
                            n_new, device=device, dtype=torch.long
                        ),
                    ],
                    dim=0,
                )
            )
            seg_list.append(
                torch.full(
                    (ref_natoms + n_new,), b, device=device, dtype=torch.long
                )
            )

        x = torch.cat(pos_list, dim=0)
        h = torch.cat(h_list, dim=0)
        ligand_diff = torch.cat(ld_list, dim=0)
        ligand_group = torch.cat(lg_list, dim=0)
        batch_seg = torch.cat(seg_list, dim=0)
        ctx = 1.0 - ligand_diff
        ref_z_flat = torch.cat(z_list, dim=0)

        # Sampling runs in the context's centre-of-mass frame; the shift is
        # added back afterwards (generate.py:145,153).
        fixed_mean = torch.zeros(bsz, 3, device=device)
        fixed_mean.index_add_(0, batch_seg, x * ctx)
        fixed_mean = fixed_mean / ctx.new_zeros(bsz, 1).index_add_(
            0, batch_seg, ctx
        )
        x = ld_utils.remove_partial_mean_with_mask(x, ctx, batch_seg)

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            context=ctx,
            ligand_diff=ligand_diff,
            batch_seg=batch_seg,
            batch_size=bsz,
            ligand_group=ligand_group,
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

        return LigandDiffToPointCloud(
            out_pos, out_z, batch_seg, bsz, self.atom_vocab
        )
