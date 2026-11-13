"""NExT-Mol de-novo 3D molecule generation (``task_type: diffusion_nextmol``).

Two decoupled halves, joined by nothing but a flat list of SMILES -- which is
upstream's own architecture, not a simplification of it:

1. **MoLlama** (1D) writes a molecule down as a SELFIES string, which is decoded
   to SMILES and built into a 2D molecular graph by RDKit. Inference only; the
   published ``acharkq/MoLlama`` checkpoint is sampled from a bare BOS token.
   It is **not** part of this task's ``state_dict`` -- the generator loads it at
   generation time, so the trained checkpoint stays DMT-only and does not grow
   by 2 GB.
2. **DMT** (3D) takes that graph -- atoms and bonds, no coordinates -- and
   diffuses coordinates onto it. This is the half that trains here.

So the training data path never sees a SELFIES string, and no new ``data_type``
is needed: ``graph3d`` already carries the explicit bond orders DMT conditions
on. See INTEGRATION_PLAN.md, "Representation Routing".

**Only coordinates are generated.** Atom types and bonds are fixed conditioning
on every forward and every sampling step, which is why:

* :meth:`NextMolTask.sample` **raises**. The de-novo pipeline is genuinely
  unconditional, but ``GenerativeFactory.sample(batch_size, nodesxsample, ...)``
  makes the *caller* choose the atom count -- whereas here the language model
  chooses the molecule and therefore its size -- and the
  ``(one_hot, charges, coords, node_mask)`` return contract throws away the
  bonds MoLlama decided on. De-novo generation goes through
  :class:`NextMolGenerator`, named by
  ``configs/interference/gen_nextmol_denovo.yaml``. Conformers of molecules
  you already have go through the shared
  :class:`~MolecularDiffusion.runmodes.generate.tasks_conformer.ConformerFactory`
  (``gen_conformer.yaml``), which drives :meth:`NextMolTask.generate_conformers`
  -- the same route LoQI and DiTMC take.
* ``node_dist_model`` / ``n_node_dist`` are deliberately absent -- they exist to
  let ``GenerativeFactory`` pick a size, and that route is bypassed.

Bond mapping (canonical -> NExT-Mol): ``nextmol_col = canonical_class - 1``, so
``1=SINGLE -> 0``, ``2=DOUBLE -> 1``, ``3=TRIPLE -> 2``, ``4=AROMATIC -> 3``.
Canonical class 0 ("no bond") is never an edge; it is the all-zero 4-vector
``to_dense_adj`` leaves on non-bonded pairs, so the platform's storage rule is a
pass-through. **Do not set ``kekulize: true``** -- aromatic is a real class in
the 4-wide vocabulary and kekulizing would change the conditioning distribution
the published weights were trained on.

``pos_std`` must match the checkpoint's dataset (GEOM-QM9 1.4182, QM9-2014/JODO
1.7226, GEOM-Drugs 2.4777 / 2.3860) or every structure comes back mis-scaled.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import numpy as np
import torch
from MolecularDiffusion.device import get_device
from torch import nn
from torch_geometric.utils import to_dense_batch

from MolecularDiffusion.modules.models.nextmol import (
    DGTDiffusion,
    NoiseScheduleVPV2,
    atom_types_for,
    featurize_mol,
    get_align_noise,
    remove_mean,
    sample_com_rand_pos,
)

logger = logging.getLogger(__name__)

#: NExT-Mol's bond vocabulary width. Canonical class 0 is never materialized.
N_BOND_CLASSES = 4


def _plain(cfg: Any) -> dict:
    """OmegaConf node -> plain dict (Hydra hands these over as DictConfig)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


def _graph3d_symbols():
    """Import shim for the graph3d helpers.

    ``MolecularDiffusion.data.component.graph3d_dataset`` has a latent import
    cycle with ``MolecularDiffusion.data.dataset`` that only bites when
    graph3d_dataset is the FIRST of the two imported. Pre-existing platform
    behaviour, not something to fix from here.
    """
    import MolecularDiffusion.data.dataset  # noqa: F401
    from MolecularDiffusion.data.datasets.graph3d_dataset import (
        build_rdkit_mol,
    )

    return build_rdkit_mol


# ---------------------------------------------------------------------------
# Featurization cache
# ---------------------------------------------------------------------------


class MoleculeFeatureCache:
    """SMILES-keyed cache of upstream's per-molecule RDKit featurization.

    Rebuilding the mol and running ``featurize_mol`` is a full RDKit round-trip;
    upstream does it once at dataset-build time, so doing it per batch here
    would dominate the step. Same pattern as
    ``modules/tasks/diffusion_ditmc.MoleculeFeatureCache``.
    """

    def __init__(self, dataset: str = "qm9", maxsize: int = 200_000) -> None:
        self.dataset = dataset
        self.types = atom_types_for(dataset)
        self._compute = functools.lru_cache(maxsize=maxsize)(
            self._compute_uncached
        )

    def _compute_uncached(self, key: tuple):
        _smiles, _n, z_bytes, bi_bytes, bt_bytes, fc_bytes = key
        z = np.frombuffer(z_bytes, dtype=np.int64)
        bond_index = np.frombuffer(bi_bytes, dtype=np.int64).reshape(2, -1)
        bond_type = np.frombuffer(bt_bytes, dtype=np.int64)
        fc = np.frombuffer(fc_bytes, dtype=np.int64)

        build_rdkit_mol = _graph3d_symbols()
        try:
            mol = build_rdkit_mol(z, bond_index, bond_type, formal_charge=fc)
        except Exception:  # noqa: BLE001 - fall back to an unsanitized mol
            mol = build_rdkit_mol(
                z, bond_index, bond_type, formal_charge=fc, sanitize=False
            )
        return featurize_mol(mol, self.types)

    def get(self, item):
        """``item`` is one PyG ``Data`` from the graph3d dataset."""
        z = item.z.detach().cpu().numpy().astype(np.int64)
        bond_index = (
            item.bond_index.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
            .reshape(2, -1)
        )
        bond_type = (
            item.bond_type.detach().cpu().numpy().astype(np.int64).reshape(-1)
        )
        if bond_type.size and int(bond_type.max()) > N_BOND_CLASSES:
            msg = (
                f"bond class {int(bond_type.max())} exceeds NExT-Mol's 4-class "
                f"vocabulary (1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC). A class "
                f"outside 1..4 means a vocabulary mismatch, not a rare bond."
            )
            raise ValueError(msg)
        fc = item.fc.detach().cpu().numpy().astype(np.int64)
        key = (
            getattr(item, "smiles", None) or "",
            int(z.shape[0]),
            z.tobytes(),
            bond_index.tobytes(),
            bond_type.tobytes(),
            fc.tobytes(),
        )
        return self._compute(key)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class Graph3DBatchToNextMolAdapter(nn.Module):
    """One ``graph3d`` PyG ``Batch`` -> the container ``DGTDiffusion`` wants.

    Bond symmetry: ``featurize_mol`` emits every bond in **both** directions
    with the same ``edge_attr``, which is upstream's only symmetry mechanism --
    ``to_dense_adj`` would silently produce an asymmetric matrix from a
    one-directional list. The mirroring therefore happens inside
    ``featurize_mol``, on the rebuilt RDKit mol, not on the stored
    upper-triangular ``bond_index``.
    """

    def __init__(
        self,
        dataset: str = "qm9",
        pos_std: float = 1.4182,
        noise_scheduler: NoiseScheduleVPV2 | None = None,
        aug_rotation: bool = True,
        t_cond: str = "t",
        disable_com: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pos_std = pos_std
        self.cache = MoleculeFeatureCache(dataset=dataset)
        self.noise_scheduler = noise_scheduler or NoiseScheduleVPV2("cosine")
        self.aug_rotation = aug_rotation
        self.t_cond = t_cond
        self.disable_com = disable_com

    # -- construction ------------------------------------------------------

    def data_list(self, items: list) -> list:
        """graph3d ``Data`` items -> upstream-shaped ``Data`` items."""
        from torch_geometric.data import Data

        out = []
        for item in items:
            x, z, edge_index, edge_attr = self.cache.get(item)
            out.append(
                Data(
                    x=x.clone(),
                    z=z.clone(),
                    edge_index=edge_index.clone(),
                    edge_attr=edge_attr.clone(),
                    pos=item.pos.float() / self.pos_std,
                    smiles=getattr(item, "smiles", None) or "",
                )
            )
        return out

    def collate(self, data_list: list, device=None, *, add_noise: bool = True):
        """Batch, then either noise (training) or seed from the prior (sampling)."""
        from torch_geometric.data import Batch

        # Single device hop for the whole batch (the `batch.to(device)` below).
        # `add_noise`/`seed_prior` are ports of upstream's collate and are
        # CPU-only by design: they index with `data.batch`, call `.numpy()` for
        # the rotation augmentation, and build their time/noise tensors with
        # bare `torch.rand`/`torch.linspace`. Lightning hands `forward` a batch
        # that is ALREADY on the device, so pin the container to CPU here --
        # the one funnel both the training and the sampling path go through --
        # rather than sprinkling `.to(device)` over each of those lines.
        batch = Batch.from_data_list(data_list).cpu()
        batch["max_seqlen"] = int((batch["ptr"][1:] - batch["ptr"][:-1]).max())
        batch = self.add_noise(batch) if add_noise else self.seed_prior(batch)
        batch.x = batch.x.to(torch.float)
        return batch.to(device) if device is not None else batch

    def forward(self, batch: Any, device=None):
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        return self.collate(
            self.data_list(pyg.to_data_list()),
            device=device if device is not None else pyg.pos.device,
        )

    # -- noising -----------------------------------------------------------

    def add_noise(self, data):
        """Port of ``QM9Collater.add_noise`` (``diffusion_data_module.py:50``).

        Runs on CPU tensors, as upstream's collate does, then the whole batch is
        moved to the device in one go.
        """
        from scipy.spatial.transform import Rotation

        t_eps = 1e-5
        bs = len(data["smiles"])
        # Stratified time: one random offset shared by the batch, spread evenly.
        # NOT independent uniforms -- this is variance reduction and it is what
        # upstream trained with.
        t = (torch.rand(1) + torch.linspace(0, 1, bs)) % 1
        data["t"] = t * (1.0 - t_eps) + t_eps

        alpha_t, sigma_t = self.noise_scheduler.marginal_prob(data["t"])
        data["alpha_t_batch"] = alpha_t
        data["sigma_t_batch"] = sigma_t
        data["loss_norm"] = torch.sqrt(alpha_t / sigma_t)
        noise_level = torch.log(alpha_t**2 / sigma_t**2)
        noise_level = noise_level[data.batch]
        alpha_t, sigma_t = alpha_t[data.batch], sigma_t[data.batch]

        if self.aug_rotation:
            # DMT is NOT equivariant by construction (enable_equiv=False), so
            # this augmentation is load-bearing, not optional.
            rot = Rotation.random(bs)[data.batch.numpy()]
            data["pos"] = torch.from_numpy(rot.apply(data["pos"].numpy())).to(
                torch.float
            )

        data["gt_pos"] = data["pos"].clone()
        noise = (
            torch.randn(data.pos.shape)
            if self.disable_com
            else sample_com_rand_pos(data.pos.shape, data.batch)
        )
        data["pos"] = (
            alpha_t.view(-1, 1) * data["pos"] + sigma_t.view(-1, 1) * noise
        )
        data["alpha_t"] = alpha_t.view(-1, 1)
        data["sigma_t"] = sigma_t.view(-1, 1)
        data["noise"] = noise
        if self.t_cond == "t":
            data["t_cond"] = data["t"][data.batch]
        elif self.t_cond == "noise_level":
            data["t_cond"] = noise_level
        else:
            msg = f"Unknown t_cond {self.t_cond!r}"
            raise ValueError(msg)
        return data

    def seed_prior(self, data):
        """Port of ``QM9InferCollater.__call__`` -- coordinates from the prior."""
        shape = (data.x.shape[0], 3)
        data["pos"] = (
            torch.randn(shape)
            if self.disable_com
            else sample_com_rand_pos(shape, data.batch)
        )
        return data


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class NextMolTask(nn.Module):
    """DMT in the platform's Task contract (docs/adding_new_models.md 2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        dataset: str = "qm9",
        model_kwargs: dict | None = None,
        pos_std: float = 1.4182,
        noise_schedule: str = "cosine",
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
        discrete_schedule: bool = False,
        sampling_steps: int = 100,
        t_cond: str = "t",
        aug_rotation: bool = True,
        align_loss: bool = True,
        reduce_node_mean: bool = False,
        disable_com: bool = True,
        atom_vocab: list | None = None,
        task_type: str = "diffusion_nextmol",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.dataset = dataset
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.pos_std = pos_std
        self.sampling_steps = sampling_steps
        self.t_cond = t_cond
        self.align_loss = align_loss
        self.reduce_node_mean = reduce_node_mean
        self.disable_com = disable_com

        self._check_atom_vocab()

        self.noise_scheduler = NoiseScheduleVPV2(
            schedule=noise_schedule,
            continuous_beta_0=continuous_beta_0,
            continuous_beta_1=continuous_beta_1,
            discrete_mode=discrete_schedule,
        )
        self.adapter = Graph3DBatchToNextMolAdapter(
            dataset=dataset,
            pos_std=pos_std,
            noise_scheduler=self.noise_scheduler,
            aug_rotation=aug_rotation,
            t_cond=t_cond,
            disable_com=disable_com,
        )
        # Named `net` so checkpoint conversion is a plain
        # `diffusion_model.` -> `net.` prefix swap.
        self.net = DGTDiffusion(
            disable_com=disable_com, **(model_kwargs or {})
        )

    def _check_atom_vocab(self) -> None:
        """Element sets are matched BY SYMBOL, never by position."""
        if not self.atom_vocab:
            return
        types = atom_types_for(self.dataset)
        missing = [s for s in self.atom_vocab if s not in types]
        if missing:
            msg = (
                f"atom_vocab contains {missing}, which NExT-Mol's "
                f"{self.dataset!r} element set does not cover. Use "
                f"dataset: drugs for anything beyond H/C/N/O/F."
            )
            raise ValueError(msg)

    # -- contract ----------------------------------------------------------

    @property
    def model(self) -> NextMolTask:
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: Any):
        data = self.adapter(batch, device=self.device)
        _pred_pos, pred_noise = self.net(data)
        loss = self._noise_loss(data, pred_noise)
        return loss, {"diff_loss": loss.detach()}

    def _noise_loss(self, data, pred_noise):
        """Port of ``DiffussionPL.get_noise_loss`` (``diffusion_pl.py:330``).

        Epsilon-prediction MSE against a **Kabsch-rotation-aligned** target.
        ``align_prediction`` and ``translation_correction`` are False for every
        released checkpoint and are not exposed.
        """
        bs, max_n = len(data["smiles"]), data.max_seqlen
        kw = {"batch_size": bs, "max_num_nodes": max_n}
        noise_pred, mask = to_dense_batch(pred_noise, data.batch, **kw)
        if self.align_loss:
            pos_0, _ = to_dense_batch(data.gt_pos, data.batch, **kw)
            pos_t, _ = to_dense_batch(data.pos, data.batch, **kw)
            target = get_align_noise(
                pos_t,
                pos_0,
                data.alpha_t_batch.view(-1, 1, 1).to(pos_t.device),
                data.sigma_t_batch.view(-1, 1, 1).to(pos_t.device),
            ).detach()
        else:
            target, _ = to_dense_batch(data.noise, data.batch, **kw)

        loss = torch.square(noise_pred - target).mean(dim=-1).sum(dim=-1)
        n_nodes = mask.sum(dim=1)
        if self.reduce_node_mean:
            return (loss / n_nodes).sum()
        return (loss / n_nodes).mean()

    def predict_and_target(self, batch: Any, all_loss=None, metric=None):  # noqa: ARG002
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    def sample(self, *args, **kwargs):
        msg = (
            "NExT-Mol generation cannot go through GenerativeFactory.sample(): "
            "that signature makes the CALLER choose the atom count, whereas "
            "here MoLlama chooses the molecule and therefore its size, and the "
            "(one_hot, charges, coords, node_mask) return contract throws away "
            "the bonds MoLlama decided on. Use "
            "configs/interference/gen_nextmol_denovo.yaml (de novo, via "
            "NextMolGenerator) or gen_conformer.yaml with "
            "interference.sample_input (conformers of molecules you supply, "
            "via the shared ConformerFactory)."
        )
        raise NotImplementedError(msg)

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate_conformers(
        self, data_list: list, *, num_steps: int | None = None
    ):
        """Ancestral VP sampler (``model/uncond_gen_pl.py:96``).

        ``data_list`` holds either upstream-shaped ``Data`` items (what the
        de-novo path builds from MoLlama's SMILES) or plain ``graph3d`` items
        (what ``ConformerFactory`` hands over from ``sample_input``); the
        latter are featurized here through the adapter's existing seam. No
        coordinates are needed either way -- they are seeded from the prior.
        Returns ``(pos, batch_segments)``, ``pos`` already rescaled by
        ``pos_std``.
        """
        device = self.device
        # `x` is upstream's featurized node matrix; a graph3d item has none.
        if data_list and getattr(data_list[0], "x", None) is None:
            data_list = self.adapter.data_list(data_list)
        data = self.adapter.collate(data_list, device=device, add_noise=False)
        num_nodes = data.x.shape[0]
        steps = num_steps or self.sampling_steps

        t_array = torch.linspace(
            self.noise_scheduler.T, 0.001, steps, device=device
        )
        s_array = torch.cat([t_array[1:], torch.zeros(1, device=device)])

        pos_mean = data.pos
        for t, s in zip(t_array, s_array, strict=True):
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)
            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t**2 - alpha_t_given_s**2 * sigma_s**2
            sigma = torch.sqrt(sigma2_t_given_s) * sigma_s / sigma_t

            if self.t_cond == "t":
                cond = torch.ones(num_nodes, device=device) * t
            else:
                cond = torch.ones(num_nodes, device=device) * torch.log(
                    alpha_t**2 / sigma_t**2
                )
            data["t_cond"] = cond
            data["alpha_t"] = (
                torch.ones((num_nodes, 1), device=device) * alpha_t
            )
            data["sigma_t"] = (
                torch.ones((num_nodes, 1), device=device) * sigma_t
            )

            pred_pos, _ = self.net(data)
            pos_mean = (
                alpha_t_given_s * sigma_s**2 / sigma_t**2
            ) * data.pos + (alpha_s * sigma2_t_given_s / sigma_t**2) * pred_pos
            pos_mean = remove_mean(pos_mean, data.batch)

            epsilon = (
                torch.randn(data.pos.shape, device=device)
                if self.disable_com
                else sample_com_rand_pos(data.pos.shape, data.batch)
            )
            data["pos"] = pos_mean + sigma * epsilon

        # The final step takes the MEAN, not the noised sample, then undoes the
        # coordinate normalization. A pos_std that does not match the checkpoint
        # returns every structure at the wrong scale.
        return pos_mean * self.pos_std, data.batch


class NextMolTaskFactory:
    """Factory instantiated by ``cli/train.py``.

    ``train_set`` is declared purely so the documented declarative injection
    seam (docs/adding_new_models.md 2.5) can hand over the dataset for an
    optional size histogram; DMT needs no construction-time statistics
    (``pos_std`` is a config scalar, not a computed stat).
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_nextmol",
        dataset: str = "qm9",
        model: dict | None = None,
        pos_std: float = 1.4182,
        noise_schedule: str = "cosine",
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
        discrete_schedule: bool = False,
        sampling_steps: int = 100,
        t_cond: str = "t",
        aug_rotation: bool = True,
        align_loss: bool = True,
        reduce_node_mean: bool = False,
        disable_com: bool = True,
        atom_vocab: list | None = None,
        train_set: Any = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.dataset = dataset
        self.model_kwargs = _plain(model or {})
        self.pos_std = pos_std
        self.noise_schedule = noise_schedule
        self.continuous_beta_0 = continuous_beta_0
        self.continuous_beta_1 = continuous_beta_1
        self.discrete_schedule = discrete_schedule
        self.sampling_steps = sampling_steps
        self.t_cond = t_cond
        self.aug_rotation = aug_rotation
        self.align_loss = align_loss
        self.reduce_node_mean = reduce_node_mean
        self.disable_com = disable_com
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: NextMolTask | None = None

    def build(self) -> NextMolTask:
        self.task = NextMolTask(
            dataset=self.dataset,
            model_kwargs=self.model_kwargs,
            pos_std=self.pos_std,
            noise_schedule=self.noise_schedule,
            continuous_beta_0=self.continuous_beta_0,
            continuous_beta_1=self.continuous_beta_1,
            discrete_schedule=self.discrete_schedule,
            sampling_steps=self.sampling_steps,
            t_cond=self.t_cond,
            aug_rotation=self.aug_rotation,
            align_loss=self.align_loss,
            reduce_node_mean=self.reduce_node_mean,
            disable_com=self.disable_com,
            atom_vocab=self.atom_vocab,
            task_type=self.task_type,
        )
        logger.info(
            "Built NExT-Mol DMT dataset=%s params=%d pos_std=%.4f",
            self.dataset,
            sum(p.numel() for p in self.task.parameters()),
            self.pos_std,
        )
        return self.task


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class NextMolGenerator:
    """The de-novo pipeline, and the conformer-only mode, in one class.

    ``cli/generate.py`` does ``instantiate(cfg.interference, task=task)`` then
    ``.run()``, so every key in the interference YAML lands in ``__init__``.

    Where the SMILES come from is the ONLY difference between the two modes:

    * ``mollama_model`` set -> **de novo**. Sample from the language model until
      ``num_molecules`` valid molecules accumulate. The full
      ``selfies\\tsmiles_chiral\\tsmiles`` TSV is written to
      ``<output_path>/sampled_sequences.tsv`` so the intermediate 1D list is
      inspectable, exactly as upstream saves it.
    * ``mollama_model: null`` -> **conformers** of molecules you supply, via
      ``smiles`` or ``db_path`` (+ ``indices``).

    Either way RDKit supplies atoms and bonds ONLY; the coordinates are what the
    model produces.
    """

    def __init__(  # noqa: PLR0913
        self,
        task: Any,
        mollama_model: str | None = None,
        num_molecules: int = 100,
        temperature: float = 1.0,
        num_beams: int = 1,
        max_sf_tokens: int = 30,
        lm_batch_size: int = 200,
        num_generate: int = 1,
        batch_size: int = 16,
        num_steps: int = 100,
        max_atom: int = 200,
        seed: int = 42,
        device: str | None = None,
        output_path: str = "generated_nextmol",
    ) -> None:
        self.task = getattr(task, "task", task)
        self.mollama_model = mollama_model
        self.num_molecules = num_molecules
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_sf_tokens = max_sf_tokens
        self.lm_batch_size = lm_batch_size
        self.num_generate = num_generate
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.max_atom = max_atom
        self.seed = seed
        self.output_path = output_path
        self.device = torch.device(
            device or (str(get_device()))
        )

    # -- step 1: get SMILES -------------------------------------------------

    def _smiles_from_mollama(self) -> list[str]:
        from MolecularDiffusion.modules.models.nextmol.mollama import (
            sample_smiles,
        )

        triples = sample_smiles(
            self.mollama_model,
            self.num_molecules,
            device=self.device,
            temperature=self.temperature,
            num_beams=self.num_beams,
            max_sf_tokens=self.max_sf_tokens,
            batch_size=self.lm_batch_size,
        )
        tsv = os.path.join(self.output_path, "sampled_sequences.tsv")
        with open(tsv, "w", encoding="utf8") as fh:
            fh.writelines("\t".join(row) + "\n" for row in triples)
        logger.info("MoLlama: %d molecules -> %s", len(triples), tsv)
        # Third column: the CHIRALITY-FREE SMILES. Upstream's own choice
        # (qm9_jodo_dm.py:438) -- DMT derives chirality from the geometry it
        # generates, so a chirality-tagged input would over-specify it.
        return [t[2] for t in triples]

    def _get_smiles(self) -> list[str]:
        if not self.mollama_model:
            # Conformers of molecules you supply now go through
            # ConformerFactory (configs/interference/gen_conformer.yaml),
            # which owns pool loading for every conformer model. This
            # generator is the de-novo path only: MoLlama invents the
            # molecules, so there is no pool to load.
            msg = (
                "NextMolGenerator is the de-novo path and needs "
                "`mollama_model`. For conformers of molecules you already "
                "have, use configs/interference/gen_conformer.yaml and set "
                "interference.sample_input."
            )
            raise ValueError(msg)
        return self._smiles_from_mollama()

    # -- step 2: SMILES -> graph -------------------------------------------

    def _items_from_smiles(self, smiles_list: list[str]):
        """Port of ``PredictDataset`` (``geom_drugs_jodo_dm.py:32-78``).

        ``MolFromSmiles -> SanitizeMol -> AddHs -> SanitizeMol -> featurize_mol``.
        No coordinates are produced or needed. Molecules RDKit cannot parse, or
        that contain an element outside the configured set, are COUNTED and
        reported -- never silently dropped.
        """
        from rdkit import Chem
        from torch_geometric.data import Data

        types = atom_types_for(self.task.dataset)
        unparsed, off_vocab, too_big = [], [], []
        items = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                unparsed.append(smi)
                continue
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
                Chem.SanitizeMol(mol)
            except Exception:  # noqa: BLE001
                unparsed.append(smi)
                continue
            bad = next(
                (
                    a.GetSymbol()
                    for a in mol.GetAtoms()
                    if a.GetSymbol() not in types
                ),
                None,
            )
            if bad is not None:
                off_vocab.append((smi, bad))
                continue
            if mol.GetNumAtoms() > self.max_atom:
                too_big.append(smi)
                continue
            x, z, edge_index, edge_attr = featurize_mol(mol, types)
            items.append(
                (
                    Data(
                        x=x,
                        z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=torch.zeros(mol.GetNumAtoms(), 3),
                        smiles=smi,
                    ),
                    mol,
                )
            )
        logger.info(
            "NExT-Mol inputs: %d kept, %d unparseable, %d off-vocabulary, "
            "%d over max_atom=%d",
            len(items),
            len(unparsed),
            len(off_vocab),
            len(too_big),
            self.max_atom,
        )
        for smi, sym in off_vocab[:20]:
            logger.info("  dropped %s (element %s)", smi, sym)
        return items

    # -- step 3: graph -> 3D ------------------------------------------------

    @staticmethod
    def _write_xyz(path, symbols, coords) -> None:
        with open(path, "w") as fh:
            fh.write(f"{len(symbols)}\n\n")
            fh.writelines(
                f"{s} {x:.6f} {y:.6f} {z:.6f}\n"
                for s, (x, y, z) in zip(symbols, coords, strict=True)
            )

    def run(self) -> str:
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        os.makedirs(self.output_path, exist_ok=True)
        torch.manual_seed(self.seed)
        self.task = self.task.to(self.device).eval()

        pairs = self._items_from_smiles(self._get_smiles())
        if not pairs:
            msg = "No input molecules survived featurization; nothing to generate."
            raise ValueError(msg)

        # Replicate BEFORE batching so `num_generate` conformers of the same
        # molecule ride along in the same batch.
        flat = [
            (i, d, m)
            for i, (d, m) in enumerate(pairs)
            for _ in range(self.num_generate)
        ]

        sdf_path = os.path.join(self.output_path, "generated.sdf")
        writer = Chem.SDWriter(sdf_path)
        written = 0
        for start in range(0, len(flat), self.batch_size):
            chunk = flat[start : start + self.batch_size]
            pos, segments = self.task.generate_conformers(
                [d for _, d, _ in chunk], num_steps=self.num_steps
            )
            pos, segments = pos.cpu(), segments.cpu()
            for k, (mol_i, _data, mol) in enumerate(chunk):
                xyz = pos[segments == k].numpy()
                out = Chem.Mol(mol)
                out.RemoveAllConformers()
                conf = Chem.Conformer(out.GetNumAtoms())
                for a, (x, y, z) in enumerate(xyz):
                    conf.SetAtomPosition(
                        a, Point3D(float(x), float(y), float(z))
                    )
                out.AddConformer(conf, assignId=True)
                # The bonds MoLlama chose survive into the file -- this is the
                # whole point of writing the molecule down before building it.
                writer.write(out)
                self._write_xyz(
                    os.path.join(
                        self.output_path,
                        f"mol_{mol_i:04d}_conf_{written:05d}.xyz",
                    ),
                    [a.GetSymbol() for a in out.GetAtoms()],
                    xyz,
                )
                written += 1
        writer.close()

        logger.info(
            "NExT-Mol wrote %d structures to %s", written, self.output_path
        )
        return self.output_path


ModelTaskFactory = NextMolTaskFactory
