"""Shape-conditioned DiffSMol diffusion task.

Binds ``modules/models/diffsmol`` to the duck-typed Task contract
(docs/adding_new_models.md §2.1) and adapts the platform's padded PointCloud
dict batch to DiffSMol's flat/ragged ``(N_total, ...) + batch_ligand``
layout.

Scope (approved INTEGRATION_PLAN): bonds dropped, D3PM atom-type diffusion
kept, shape conditioning + classifier-free guidance kept, heavy atoms only.

**Frame.** The shape latent is a ``(128, 3)`` SO(3)-equivariant feature that
rotates with the molecule and is *not* translation invariant. Upstream puts
coordinates in the **surface-centroid** frame -- the mean of the 512 sampled
mesh points -- and this task does the same by subtracting the precomputed
``shape_center``. It therefore deliberately does **not** call
``remove_mean_with_mask``: COM-centering would put the coordinates in a
different frame from the latent and silently break conditioning. If rotation
augmentation is ever switched on, the same ``R`` must be applied to both
``coords`` and ``shape_emb`` (the latter is a stack of 128 3-vectors, so
``shape_emb @ R.T`` is the correct action).

**Shape cache.** The mesh -> point cloud -> VN-AE chain is precomputed
offline by ``docs/model_integrations/diffsmol/scripts/precompute_shapes.py``
into a ``.pt`` keyed by each sample's ``xyz`` identifier -- the same key the
PointCloud dataset already carries through the default collate. No custom
DataModule and no collate change is needed: the platform's collate passes
unknown keys (including a list of strings) straight through.
"""

from __future__ import annotations

import bisect
import random
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from MolecularDiffusion.modules.models.diffsmol.score_model import (
    ScorePosNet3D,
)

#: Heavy-atom vocabulary. Upstream's ``MAP_ATOM_TYPE_ONLY_TO_INDEX``
#: (``ligand_atom_mode: 'basic'``) minus its unreachable H slot; aromaticity
#: is dropped with the bonds.
DIFFSMOL_ATOM_VOCAB = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

#: Upstream's 15-class ``MAP_ATOM_TYPE_AROMATIC_TO_INDEX``
#: (``source/utils/transforms.py:51``, ``ligand_atom_mode: 'add_aromatic'``),
#: flattened to (element symbol, is_aromatic) parallel lists in class order.
#: This is the vocabulary the released ``diffusion.pt`` was trained on --
#: ``ligand_atom_emb.weight`` is ``(128, 23)`` = 15 + 8 time-emb dims and
#: ``v_inference.2.weight`` is ``(15, 128)``, so trust the tensors, not that
#: checkpoint's ``cfg.model.num_node_types: 8``, which is stale and wrong.
#:
#: Note the deliberate duplicate symbols: aromatic and non-aromatic carbon are
#: two distinct classes with the same element. That *is* the decode path --
#: ``save_xyz_file`` looks up ``atom_vocab[argmax]`` and gets a plain element
#: either way, so generated molecules write valid ``.xyz`` with no extra
#: mapping table. Aromaticity is simply dropped on output, which is correct:
#: it is a bond-graph property and this port emits no bond graph.
DIFFSMOL_AROMATIC_VOCAB = [
    "H", "C", "C", "N", "N", "O", "O", "F",
    "P", "P", "S", "S", "Cl", "Br", "I",
]
DIFFSMOL_AROMATIC_FLAGS = [
    False, False, True, False, True, False, True, False,
    False, True, False, True, False, False, False,
]


# --------------------------------------------------------------------- #
# size distribution                                                     #
# --------------------------------------------------------------------- #
class DiffSMolShapeNodeDistribution:
    """Atom-count sampler, marginal and shape-volume-conditioned.

    ``sample(n)`` draws from the marginal atom-count histogram (the duck-typed
    contract every generic caller uses). ``sample_for_volume(volume, n)``
    draws from the sub-histogram of training molecules whose mesh volume is
    near ``volume``, mirroring upstream's volume-conditioned draw
    (``sample_diffusion_no_pocket.py:242-283``).

    # ponytail: keyed on trimesh mesh volume, not upstream's 0.5 A
    # occupied-voxel count -- the precompute already has ``mesh.volume`` for
    # free, both are monotone volume proxies, and the histogram is binned
    # anyway. Port get_atom_stamp/get_voxel_shape/make_grid only if the
    # generated size distribution measurably degrades.
    """

    def __init__(
        self,
        n_node_dist: Dict[int, int],
        volume_bins: Optional[List[float]] = None,
        volume_hist: Optional[Dict[int, Dict[int, int]]] = None,
        bin_window: int = 1,
    ) -> None:
        # An empty histogram is tolerated at construction and only refused
        # when a size is actually drawn. `_dataset_stats` is computed from
        # `train_set` and is NOT persisted into the checkpoint, so on the
        # generate path it is always empty -- raising here made every
        # generate-from-checkpoint run fail, even with an explicit
        # `interference.mol_size`, which needs no histogram at all.
        self.n_node_dist = {int(k): int(v) for k, v in (n_node_dist or {}).items()}
        self._sizes = sorted(self.n_node_dist)
        self._weights = [self.n_node_dist[s] for s in self._sizes]
        self.volume_bins = volume_bins or []
        self.volume_hist = volume_hist or {}
        self.bin_window = bin_window

    @property
    def max_n_nodes(self) -> int:
        # Empty on the generate path (see __init__): report 0 rather than
        # letting `max([])` raise a bare, uninformative ValueError.
        return max(self._sizes) if self._sizes else 0

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        if not self._sizes:
            raise ValueError(
                "DiffSMolShapeNodeDistribution has no size histogram, so it "
                "cannot choose molecule sizes. Either set "
                "`interference.mol_size` explicitly (e.g. [22]), or generate "
                "from a checkpoint trained by this platform with a dataset "
                "attached."
            )
        return torch.tensor(
            random.choices(self._sizes, weights=self._weights, k=n_samples),
            dtype=torch.long,
        )

    def sample_for_volume(
        self, volume: float, n_samples: int = 1
    ) -> torch.Tensor:
        """Draw sizes from molecules of comparable mesh volume.

        Falls back to the marginal histogram if the neighbourhood around
        ``volume`` is empty (or if no volume histogram was built).
        """
        if not self.volume_bins or not self.volume_hist:
            return self.sample(n_samples)

        idx = bisect.bisect_right(self.volume_bins, volume) - 1
        merged: Counter = Counter()
        for b in range(idx - self.bin_window, idx + self.bin_window + 1):
            merged.update(self.volume_hist.get(b, {}))
        if not merged:
            return self.sample(n_samples)

        sizes = sorted(merged)
        weights = [merged[s] for s in sizes]
        return torch.tensor(
            random.choices(sizes, weights=weights, k=n_samples),
            dtype=torch.long,
        )


# --------------------------------------------------------------------- #
# factory                                                               #
# --------------------------------------------------------------------- #
class DiffSMolTaskFactory:
    """Instantiated by ``cli/train.py``.

    Declares ``train_set`` so the declarative seam at ``cli/train.py:621``
    injects the training dataset; that is where the atom-count / mesh-volume
    histograms come from. No core change is involved.
    """

    def __init__(
        self,
        task_type: str = "diffusion_diffsmol",
        model_config: Optional[dict] = None,
        shape_cache_path: Optional[str] = None,
        num_diffusion_timesteps: int = 1000,
        default_num_steps: Optional[int] = None,
        guide_stren: float = 0.0,
        threshold_type: Optional[str] = None,
        threshold_p: Optional[float] = None,
        n_volume_bins: int = 20,
        atom_mode: str = "basic",
        dataset_stats: Optional[dict] = None,
        atom_vocab: Optional[list] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.model_config = dict(model_config or {})
        self.shape_cache_path = shape_cache_path
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.default_num_steps = default_num_steps or num_diffusion_timesteps
        self.guide_stren = guide_stren
        self.threshold_type = threshold_type
        self.threshold_p = threshold_p
        self.n_volume_bins = n_volume_bins
        if atom_mode not in ("basic", "add_aromatic"):
            raise ValueError(
                "atom_mode must be 'basic' or 'add_aromatic', "
                f"got {atom_mode!r}"
            )
        self.atom_mode = atom_mode
        self.dataset_stats = dict(dataset_stats or {})
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab")
        self.train_set = train_set
        self.kwargs = kwargs

    # -- dataset statistics ------------------------------------------- #
    def compute_dataset_stats(self, dataset, shape_cache: dict) -> None:
        """Build the marginal atom-count histogram and, if the shape cache
        covers the training set, the mesh-volume-conditioned one."""
        counts: List[int] = []
        volumes: List[float] = []
        keys: List[Optional[str]] = []
        for i in range(len(dataset)):
            item = dataset[i]
            n = item.get("natoms")
            n = int(n.item()) if torch.is_tensor(n) else int(n)
            counts.append(n)
            key = item.get("xyz")
            keys.append(key)
            entry = shape_cache.get(key) if key is not None else None
            volumes.append(
                float(entry["shape_volume"]) if entry else float("nan")
            )

        self.dataset_stats["num_atoms_histogram"] = {
            int(k): int(v) for k, v in Counter(counts).items()
        }

        valid = [
            (v, c) for v, c in zip(volumes, counts) if v == v and v > 0
        ]
        if len(valid) < self.n_volume_bins:
            print(
                f"[diffsmol] {len(valid)} molecules with a cached mesh "
                "volume; skipping the volume-conditioned size histogram "
                "(marginal histogram still available)."
            )
            self.dataset_stats["volume_bins"] = []
            self.dataset_stats["volume_histogram"] = {}
        else:
            vs = sorted(v for v, _ in valid)
            lo, hi = vs[0], vs[-1]
            width = (hi - lo) / self.n_volume_bins
            bins = [lo + i * width for i in range(self.n_volume_bins)]
            hist: Dict[int, Counter] = {}
            for v, c in valid:
                b = min(
                    int((v - lo) / width) if width > 0 else 0,
                    self.n_volume_bins - 1,
                )
                hist.setdefault(b, Counter())[c] += 1
            self.dataset_stats["volume_bins"] = bins
            self.dataset_stats["volume_histogram"] = {
                b: dict(c) for b, c in hist.items()
            }

        print(
            "[diffsmol] atom-count histogram: "
            f"{len(self.dataset_stats['num_atoms_histogram'])} unique sizes "
            f"from {len(counts)} molecules; "
            f"{len(valid)} with cached mesh volumes."
        )

    def build(self) -> "DiffSMolDiffusionTask":
        shape_cache = load_shape_cache(self.shape_cache_path)

        if not self.dataset_stats.get("num_atoms_histogram"):
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set, shape_cache)
            else:
                print(
                    "[diffsmol] WARNING: no train_set and no histogram; "
                    "generation will need an explicit nodesxsample."
                )

        cfg = dict(self.model_config)
        cfg.setdefault(
            "num_diffusion_timesteps", self.num_diffusion_timesteps
        )

        if self.atom_mode == "add_aromatic":
            # Checkpoint-compatibility mode: the vocabulary is fixed by the
            # pretrained weights, so any atom_vocab stamped on by the data
            # config is ignored on purpose.
            atom_vocab = DIFFSMOL_AROMATIC_VOCAB
            cfg.setdefault("aromatic_flags", DIFFSMOL_AROMATIC_FLAGS)
        else:
            atom_vocab = self.atom_vocab or DIFFSMOL_ATOM_VOCAB

        self.task = DiffSMolDiffusionTask(
            model_config=cfg,
            atom_vocab=list(atom_vocab),
            atom_mode=self.atom_mode,
            shape_cache=shape_cache,
            dataset_stats=self.dataset_stats,
            default_num_steps=self.default_num_steps,
            guide_stren=self.guide_stren,
            threshold_type=self.threshold_type,
            threshold_p=self.threshold_p,
            task_type=self.task_type,
        )
        return self.task


def load_shape_cache(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load the precomputed ``{key: {shape_emb, shape_center, shape_volume}}``
    map. An absent path yields an empty cache -- every molecule then falls
    back to the unconditional (zero-latent) branch, which is exactly what
    ``gen_unconditional`` wants and is enough to smoke-test the plumbing."""
    if not path:
        return {}
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise TypeError(
            f"shape cache at {path} must be a dict, got {type(cache)}"
        )
    return cache


# --------------------------------------------------------------------- #
# task                                                                  #
# --------------------------------------------------------------------- #
class DiffSMolDiffusionTask(nn.Module):
    """Task contract implementation for DiffSMol."""

    #: Class-level default so tasks pickled before the aromatic vocab existed
    #: still resolve it after unpickling.
    atom_mode = "basic"

    def __init__(
        self,
        model_config: dict,
        atom_vocab: List[str],
        shape_cache: Dict[str, Dict[str, Any]],
        dataset_stats: dict,
        default_num_steps: int,
        guide_stren: float = 0.0,
        threshold_type: Optional[str] = None,
        threshold_p: Optional[float] = None,
        task_type: str = "diffusion_diffsmol",
        atom_mode: str = "basic",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.atom_mode = atom_mode
        self.atom_vocab = list(atom_vocab)
        self.n_atom_types = len(self.atom_vocab)
        self.score_model = ScorePosNet3D(model_config, self.atom_vocab)
        self.shape_dim = self.score_model.shape_dim
        self.default_num_steps = default_num_steps
        self.guide_stren = guide_stren
        self.threshold_type = threshold_type
        self.threshold_p = threshold_p

        # ``tasks_generate.py`` walks ``task.model.T``; keep that terminating.
        self.T = self.score_model.num_timesteps

        self._shape_cache = shape_cache
        self._dataset_stats = dataset_stats
        self._node_dist_model: Optional[DiffSMolShapeNodeDistribution] = None
        self._warned_missing_shape = False
        self.prop_dist_model = None

    # -- batch adaptation --------------------------------------------- #
    def _lookup_shapes(self, batch: Dict[str, Any], coords, node_mask):
        """Per-molecule ``(shape_emb, shape_center)`` from the cache.

        Molecules absent from the cache get a zero latent (the CFG
        unconditional branch) and are centered on their atom centroid, so
        training still proceeds -- but they contribute no shape signal. The
        first miss is warned about once; a fully-empty cache is the intended
        unconditional-smoke-test configuration.
        """
        device = coords.device
        bsz = coords.size(0)
        keys = batch.get("xyz") or [None] * bsz

        emb = torch.zeros(bsz, self.shape_dim, 3, device=device)
        center = torch.zeros(bsz, 3, device=device)
        n_missing = 0
        for b in range(bsz):
            entry = self._shape_cache.get(keys[b]) if keys[b] else None
            if entry is None:
                n_missing += 1
                mask_b = node_mask[b].bool()
                if mask_b.any():
                    center[b] = coords[b][mask_b].mean(dim=0)
                continue
            emb[b] = entry["shape_emb"].to(device)
            center[b] = entry["shape_center"].to(device)

        if n_missing and not self._warned_missing_shape:
            self._warned_missing_shape = True
            warnings.warn(
                f"[diffsmol] {n_missing}/{bsz} molecules in this batch have "
                "no cached shape embedding; they fall back to the "
                "unconditional zero latent and atom-centroid centering. Run "
                "scripts/precompute_shapes.py over the dataset for "
                "shape-conditioned training.",
                RuntimeWarning,
                stacklevel=2,
            )
        return emb, center

    def _flatten(self, batch: Dict[str, Any]):
        """Padded PointCloud dict -> DiffSMol's flat/ragged layout.

        Returns ``(pos, v, batch_ligand, shape_emb)`` with ``pos`` already in
        the surface-centered frame.
        """
        if self.atom_mode == "add_aromatic":
            raise NotImplementedError(
                "atom_mode='add_aromatic' is a generation-only / pretrained-"
                "checkpoint-compatibility mode. Its 15 classes are "
                "(element, is_aromatic) pairs, and the PointCloud data "
                "contract carries no aromaticity channel, so a training "
                "batch cannot be encoded into it. Train with "
                "atom_mode='basic'."
            )
        coords = batch["coords"]
        node_mask = batch["node_mask"]
        if node_mask.dim() == 3:
            node_mask = node_mask.squeeze(-1)
        mask = node_mask.bool()

        shape_emb, shape_center = self._lookup_shapes(
            batch, coords, node_mask
        )
        # Surface-centered frame -- NOT remove_mean_with_mask. See module
        # docstring.
        coords = coords - shape_center.unsqueeze(1)

        feat = batch.get("node_feature")
        if feat is None:
            feat = batch["one_hot"]
        types = feat[..., : self.n_atom_types].argmax(dim=-1)

        pos = coords[mask]
        v = types[mask]
        natoms = mask.sum(dim=1)
        batch_ligand = torch.arange(
            coords.size(0), device=coords.device
        ).repeat_interleave(natoms)
        return pos, v, batch_ligand, shape_emb

    # -- training ------------------------------------------------------ #
    def forward(self, batch: Dict[str, Any]):
        pos, v, batch_ligand, shape_emb = self._flatten(batch)
        out = self.score_model.get_diffusion_loss(
            ligand_pos=pos,
            ligand_v=v,
            batch_ligand=batch_ligand,
            ligand_shape=shape_emb,
        )
        stats = {
            "loss_pos": out["loss_pos"].detach(),
            "loss_v": out["loss_v"].detach(),
            "total_loss": out["loss"].detach(),
        }
        return out["loss"], stats

    def predict_and_target(self, batch: Dict[str, Any]):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        return {"val_loss": pred.mean()}

    # -- generation ---------------------------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, Any]] = None,
        shape_emb: Optional[torch.Tensor] = None,
        guide_stren: Optional[float] = None,
        mode=None,
        n_frames: int = 0,
        **kwargs: Any,
    ):
        """Reverse-diffuse a batch of molecules.

        ``shape_emb=None`` uses the zero latent, i.e. the classifier-free
        unconditional branch -- so the bundled ``gen_unconditional`` config
        works with no reference molecule. Returns the platform's EDM 4-tuple
        ``(one_hot, charges, coords, node_mask)``.

        Coordinates come out in the surface-centered frame of the reference
        shape, which is the frame that makes generated molecules land inside
        the reference envelope.
        """
        device = self.device
        num_steps = num_steps or self.default_num_steps
        guide_stren = (
            self.guide_stren if guide_stren is None else guide_stren
        )

        if nodesxsample is not None:
            sizes = torch.as_tensor(nodesxsample, device=device).long()
        elif batch is not None:
            sizes = batch["natoms"].to(device).long()
        elif batch_size is not None:
            sizes = self.node_dist_model.sample(batch_size).to(device)
        else:
            raise ValueError(
                "sample() needs nodesxsample, batch, or batch_size."
            )
        sizes = sizes.view(-1)
        bsz = sizes.numel()

        if shape_emb is None:
            shape_emb = torch.zeros(bsz, self.shape_dim, 3, device=device)
        else:
            shape_emb = torch.as_tensor(
                shape_emb, dtype=torch.float32, device=device
            ).view(-1, self.shape_dim, 3)
            if shape_emb.size(0) == 1 and bsz > 1:
                shape_emb = shape_emb.expand(bsz, -1, -1).contiguous()

        batch_ligand = torch.arange(bsz, device=device).repeat_interleave(
            sizes
        )
        n_total = int(sizes.sum().item())
        init_pos = torch.randn(n_total, 3, device=device)
        init_v = torch.randint(
            0, self.n_atom_types, (n_total,), device=device
        )

        out = self.score_model.sample_diffusion(
            init_ligand_pos=init_pos,
            init_ligand_v=init_v,
            batch_ligand=batch_ligand,
            ligand_shape=shape_emb,
            num_steps=num_steps,
            guide_stren=guide_stren,
            threshold_type=self.threshold_type,
            threshold_p=self.threshold_p,
        )
        return self._to_padded(out["pos"], out["v"], sizes)

    def _to_padded(self, pos, v, sizes):
        """Flat/ragged -> padded ``(B, n_max, ...)`` + node_mask."""
        device = pos.device
        bsz = sizes.numel()
        n_max = int(sizes.max().item())

        one_hot = torch.zeros(
            bsz, n_max, self.n_atom_types, device=device
        )
        coords = torch.zeros(bsz, n_max, 3, device=device)
        charges = torch.zeros(bsz, n_max, dtype=torch.long, device=device)
        node_mask = torch.zeros(bsz, n_max, device=device)

        offset = 0
        for i, n in enumerate(sizes.tolist()):
            n = int(n)
            idx = v[offset : offset + n].clamp(0, self.n_atom_types - 1)
            one_hot[i, :n] = torch.nn.functional.one_hot(
                idx, self.n_atom_types
            ).float()
            coords[i, :n] = pos[offset : offset + n]
            charges[i, :n] = idx
            node_mask[i, :n] = 1
            offset += n
        return one_hot, charges, coords, node_mask

    # -- contract properties ------------------------------------------- #
    @property
    def model(self):
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self) -> DiffSMolShapeNodeDistribution:
        if self._node_dist_model is None:
            self._node_dist_model = DiffSMolShapeNodeDistribution(
                self._dataset_stats.get("num_atoms_histogram", {}),
                volume_bins=self._dataset_stats.get("volume_bins"),
                volume_hist={
                    int(k): v
                    for k, v in (
                        self._dataset_stats.get("volume_histogram") or {}
                    ).items()
                },
            )
        return self._node_dist_model

    @property
    def n_node_dist(self) -> Dict[int, int]:
        return self.node_dist_model.n_node_dist


# --------------------------------------------------------------------- #
# generation entry point                                                #
# --------------------------------------------------------------------- #
class DiffSMolShapeGenerator:
    """Shape-conditioned generator behind ``interference/gen_diffsmol_shape``.

    ``GenerativeFactory`` cannot express this conditioning -- its ``cfg``
    branch carries a run-wide list of scalar ``target_values`` validated
    against ``len(task.condition)``, which a per-sample ``(128, 3)`` latent has
    no expression in. A separate generator behind its own ``_target_`` is the
    in-tree pattern for exactly this (cf. ``PharmacophoreConditionGenerator``)
    and needs no core change: ``cli/generate.py`` only calls
    ``hydra.utils.instantiate(cfg.interference, task=task)`` then ``run()``.

    ``reference_shape`` is a ``.pt`` written by ``precompute_shapes.py`` --
    the *same* code path the training cache came from, so the reference
    latent is guaranteed to be in the same frame and produced by the same
    weights. Embedding a reference molecule live is deliberately not
    supported here.

    # ponytail: reference must be pre-embedded. Keeps skimage/trimesh out of
    # the generation path entirely; precompute is one command you already ran.
    """

    def __init__(
        self,
        task,
        reference_shape: Optional[str] = None,
        reference_key: Optional[str] = None,
        num_generate: int = 100,
        batch_size: int = 8,
        num_steps: Optional[int] = None,
        guide_stren: float = 0.0,
        mol_size: Optional[list] = None,
        output_path: str = "generated_diffsmol",
        seed: int = 42,
        save_xyzrender_figures: bool = False,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.num_generate = num_generate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.guide_stren = guide_stren
        self.mol_size = list(mol_size) if mol_size else []
        self.output_path = output_path
        self.seed = seed
        self.save_xyzrender_figures = save_xyzrender_figures

        self.shape_emb = None
        self.shape_volume = None
        if reference_shape:
            entry = torch.load(
                reference_shape, map_location="cpu", weights_only=False
            )
            if "shape_emb" not in entry:
                # a full cache, not a single reference
                if reference_key is None:
                    raise ValueError(
                        f"{reference_shape} looks like a full shape cache; "
                        "set interference.reference_key to pick a molecule."
                    )
                entry = entry[reference_key]
            self.shape_emb = entry["shape_emb"]
            self.shape_volume = float(entry.get("shape_volume", 0.0)) or None

    def _sizes(self, n: int) -> torch.Tensor:
        if self.mol_size:
            return torch.tensor(
                random.choices(self.mol_size, k=n), dtype=torch.long
            )
        nd = self.task.node_dist_model
        if self.shape_volume:
            return nd.sample_for_volume(self.shape_volume, n)
        return nd.sample(n)

    def run(self) -> None:
        import os

        from MolecularDiffusion.utils.geometry.geom_utils import save_xyz_file

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        os.makedirs(self.output_path, exist_ok=True)

        self.task.eval()
        written = 0
        progress_bar = tqdm(
            total=self.num_generate, desc="Sampling molecules", leave=True
        )
        while written < self.num_generate:
            n = min(self.batch_size, self.num_generate - written)
            one_hot, _charges, coords, node_mask = self.task.sample(
                nodesxsample=self._sizes(n),
                num_steps=self.num_steps,
                shape_emb=self.shape_emb,
                guide_stren=self.guide_stren,
            )
            save_xyz_file(
                self.output_path,
                one_hot.cpu(),
                coords.cpu(),
                self.task.atom_vocab,
                id_from=written,
                name="molecule",
                node_mask=node_mask.cpu(),
            )
            written += n
            progress_bar.update(n)
        progress_bar.close()
        print(f"[diffsmol] wrote {written} molecules to {self.output_path}")
