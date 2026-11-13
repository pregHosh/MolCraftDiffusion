"""LoQI -- low-energy conformer generation on a fixed molecular graph.

One module, two task types, because upstream is one model with two configs:

=========================  =======================================  ============
``task_type``              interpolant                              weights
=========================  =======================================  ============
``diffusion_loqi``         VDM diffusion, 25 discrete steps,        ``loqi.ckpt``
                           cosine-adaptive, self-conditioned
``diffusion_loqi_flow``    continuous flow matching, velocity       ``loqi_flow.ckpt``
                           prediction, rigid (Kabsch) OT, linear
=========================  =======================================  ============

**This is not de-novo generation.** Only ``x`` (coordinates) is ever noised.
Atom types, formal charges, bond orders and stereochemistry are supplied
un-noised as network input at every step and come straight back out of
``sample()`` unchanged. Every sample therefore needs an input molecule; see
``sample_input`` below.

Data path: ``data_type: graph3d`` with ``bond_collate: raw`` (a PyG ``Batch``)
and ``kekulize: true``.

Bond representation mapping
---------------------------
LoQI's network consumes a **fully-connected directed** edge list with a 9-class
label: ``0=none 1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC 5=E 6=Z
7=chirality(sym) 8=chirality(directed)``. Classes 5-8 are not bonds -- they are
a stereochemistry encoding laid over the bond graph, and the class-8 edges are
deliberately asymmetric (the direction *is* the R/S signal), so they cannot be
stored in the platform's symmetric upper-triangular ``bond_index``/``bond_type``.

So storage keeps the canonical five, and :class:`Graph3DToLoQIAdapter`
reconstructs the rest per batch:

1. mirror the stored upper-triangular bonds into a directed list (reproducing
   upstream's ``adj.nonzero()``);
2. rebuild the molecule with ``build_rdkit_mol`` -- which calls
   ``AssignStereochemistryFrom3D``, the *same* source of truth upstream's
   ``add_stereo_bonds(from_3D=True)`` used -- and derive classes 5-8 from it;
3. run upstream's own ``make_graph_fully_connected`` to materialize class 0.

Class 4 never appears: the LoQI pipeline kekulizes (``process_chembl3d.py:216``,
``sample_conformers.py:84``), so the released weights have never seen one.

There is no per-item cache for step 2 in this first pass. If RDKit dominates
step time, the fix is a cache keyed on the dataset index, which needs the
dataset to carry one -- a separate change.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from MolecularDiffusion.device import get_device
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean

from MolecularDiffusion.data.component.graph3d_dataset import build_rdkit_mol
from MolecularDiffusion.modules.models.loqi import (
    N_EDGE_CLASSES,
    BaseSelfConditioningModule,
    MegaFNV3Conf,
    build_interpolant,
    derive_stereo_edges,
    make_graph_fully_connected,
)

logger = logging.getLogger(__name__)

#: LoQI's atom vocabulary, ``data_processing/utils_data.py:54-58``. The ORDER is
#: load-bearing: it indexes the pretrained ``atom_embedder``'s input columns.
FULL_ATOM_ENCODER: dict[str, int] = {
    "H": 0, "B": 1, "C": 2, "N": 3, "O": 4, "F": 5,
    "Al": 6, "Si": 7, "P": 8, "S": 9, "Cl": 10, "As": 11,
    "Br": 12, "I": 13, "Hg": 14, "Bi": 15, "Se": 16,
}

#: ``DEFAULT_CHARGES_DICT`` (``utils_data.py:60``): raw signed charge + 2, into
#: 6 classes, i.e. [-2, +3].
CHARGE_OFFSET = 2
N_CHARGE_CLASSES = 6


def _plain(cfg: Any) -> dict:
    """OmegaConf node (or plain mapping) -> plain nested ``dict``."""
    from omegaconf import OmegaConf

    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


# ---------------------------------------------------------------------------
# Conditioning-molecule pool
# ---------------------------------------------------------------------------

# ``_mol_to_data`` / ``_pool_from_smiles`` / ``load_conditioning_pool`` used to
# live here. They now live in ``utils/conformer_pool.py`` because all three
# conformer models share them, and ``runmodes/generate/tasks_conformer.py``
# must reach them without importing this (heavy) task module. Re-exported so
# existing imports from here keep working unchanged.
from MolecularDiffusion.utils.chem.conformer_pool import (  # noqa: E402
    _mol_to_data,
    _pool_from_smiles,
    load_conditioning_pool,
)


def _pool_from_train_set(train_set: Any, limit: int = 2000) -> list[Data]:
    """The honest default: condition on molecules from the training set."""
    graphs = getattr(train_set, "graph_data_list", None)
    if graphs is None:  # a Subset over a Graph3DDataset
        base = getattr(train_set, "dataset", None)
        indices = getattr(train_set, "indices", None)
        if base is None or indices is None:
            return []
        graphs = [base.graph_data_list[i] for i in indices]
    return list(graphs[:limit])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class Graph3DToLoQIAdapter(nn.Module):
    """``graph3d`` PyG ``Batch`` -> LoQI's flat conditioning dict.

    Output keys: ``batch`` (N,), ``x`` (N,3) COM-free, ``h`` (N,17) one-hot in
    ``FULL_ATOM_ENCODER`` order, ``charges`` (N,6) one-hot of ``fc + 2``,
    ``edge_index`` (2,E) fully connected directed, ``edge_attr`` (E,9) one-hot.
    """

    def __init__(self, atom_vocab: list[str]) -> None:
        super().__init__()
        self.atom_vocab = list(atom_vocab)
        self._vocab_to_loqi: torch.Tensor | None = None

    def vocab_map(self, device) -> torch.Tensor:
        """Platform ``atom_idx`` -> LoQI ``FULL_ATOM_ENCODER`` index.

        Built by SYMBOL, never by position: the platform's ``atom_vocab`` order
        is a data-config choice, LoQI's is fixed by the pretrained
        ``atom_embedder``'s input columns. Getting this wrong loads silently and
        mislabels every atom.
        """
        if self._vocab_to_loqi is None or self._vocab_to_loqi.device != device:
            missing = [s for s in self.atom_vocab if s not in FULL_ATOM_ENCODER]
            if missing:
                msg = (
                    f"atom_vocab contains {missing}, which LoQI's 17-element "
                    f"encoder has no column for: {sorted(FULL_ATOM_ENCODER)}"
                )
                raise ValueError(msg)
            self._vocab_to_loqi = torch.tensor(
                [FULL_ATOM_ENCODER[s] for s in self.atom_vocab],
                dtype=torch.long,
                device=device,
            )
        return self._vocab_to_loqi

    def forward(self, batch: Any) -> dict:
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        device = pyg.pos.device
        batch_idx = pyg.batch
        batch_size = int(batch_idx.max()) + 1

        x = pyg.pos.float()
        x = x - scatter_mean(x, batch_idx, dim=0, dim_size=batch_size)[batch_idx]

        h_idx = self.vocab_map(device)[pyg.atom_idx.long()]
        h = F.one_hot(h_idx, len(FULL_ATOM_ENCODER)).float()

        fc = pyg.fc.long() + CHARGE_OFFSET
        if int(fc.min()) < 0 or int(fc.max()) >= N_CHARGE_CLASSES:
            msg = (
                f"formal charges outside LoQI's [-2, +3]: raw range "
                f"[{int(pyg.fc.min())}, {int(pyg.fc.max())}]"
            )
            raise ValueError(msg)
        charges = F.one_hot(fc, N_CHARGE_CLASSES).float()

        edge_index, edge_attr = self.build_edges(pyg, device)
        return {
            "batch": batch_idx,
            "x": x,
            "h": h,
            "charges": charges,
            "edge_index": edge_index,
            "edge_attr": F.one_hot(edge_attr, N_EDGE_CLASSES).float(),
        }

    def build_edges(self, pyg: Any, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Mirror stored bonds, append derived stereo edges, then densify."""
        src_list, attr_list = [], []
        offset = 0
        for item in pyg.to_data_list():
            n = int(item.n_nodes)
            bi = item.bond_index.long().reshape(2, -1).cpu()
            bt = item.bond_type.long().reshape(-1).cpu()
            if bt.numel() and int(bt.max()) > 4:
                msg = f"bond class {int(bt.max())} > 4 in stored data"
                raise ValueError(msg)

            # 1. mirror: adj.nonzero() gives both directions upstream.
            ei = torch.cat([bi, bi.flip(0)], dim=1)
            ea = torch.cat([bt, bt], dim=0)

            # 2. stereo edges, re-derived from the geometry.
            stereo = self._stereo_edges(item)
            if stereo:
                st = torch.tensor(stereo, dtype=torch.long).T
                ei = torch.cat([ei, st[:2]], dim=1)
                ea = torch.cat([ea, st[2]], dim=0)

            src_list.append(ei + offset)
            attr_list.append(ea)
            offset += n

        edge_index = torch.cat(src_list, dim=1).to(device)
        edge_attr = torch.cat(attr_list, dim=0).to(device)
        # 3. densify to the fully-connected directed graph (materializes class 0)
        return make_graph_fully_connected(edge_index, edge_attr, pyg.batch)

    @staticmethod
    def _stereo_edges(item: Any) -> list[tuple[int, int, int]]:
        try:
            mol = build_rdkit_mol(
                item.z.cpu().numpy(),
                item.bond_index.cpu().numpy().reshape(2, -1),
                item.bond_type.cpu().numpy().reshape(-1),
                item.fc.cpu().numpy(),
                coords=item.pos.detach().cpu().numpy(),
            )
        except Exception as exc:  # noqa: BLE001 - chemistry failures are data
            logger.debug("Stereo derivation skipped (rebuild failed): %s", exc)
            return []
        return derive_stereo_edges(mol)


# ---------------------------------------------------------------------------
# Node-size distribution
# ---------------------------------------------------------------------------


class LoQINodeDistribution:
    """Histogram sampler over ``train_set.graph3d_stats.n_atoms_hist``.

    Only used so ``mol_size: [0, 0]`` works in a generate config; the actual
    molecule sizes always come from the conditioning molecules.
    """

    def __init__(self, histogram: dict) -> None:
        self.histogram = {int(k): int(v) for k, v in (histogram or {}).items()}
        self.n_node_dist = self.histogram

    def sample(self, n_samples: int) -> torch.Tensor:
        if not self.histogram:
            sizes = [random.randint(10, 50) for _ in range(n_samples)]
        else:
            sizes = random.choices(
                list(self.histogram), weights=list(self.histogram.values()), k=n_samples
            )
        return torch.tensor(sizes, dtype=torch.long)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class ModelTaskFactory:
    """Instantiated by ``cli/train.py`` / ``cli/generate.py`` from Hydra.

    ``train_set`` is a declared parameter, which is what makes the declarative
    injection seam (docs/adding_new_models.md §2.5) hand over the dataset --
    needed for the size histogram and the default conditioning pool.
    """

    #: Config keys the CALLER owns at generation time (cli/generate.py
    #: `_declared_generation_time_keys`, docs/adding_new_models.md §2.5b).
    #: Both are about WHICH molecules to make conformers of -- a user points a
    #: generate config at their own .sdf; the train-time value must not win.
    #: Architecture/interpolant keys stay checkpoint-owned.
    generation_time_keys = ("sample_input", "sample_pool_limit")

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_loqi",
        interpolant: dict | None = None,
        dynamics: dict | None = None,
        self_conditioning: dict | None = None,
        timesteps: int = 25,
        loss_scale: float = 3.0,
        loss_clamp: float | None = 10.0,
        return_step_output: bool = False,
        sample_input: str | Sequence[str] | None = None,
        sample_pool_limit: int = 2000,
        dataset_stats: dict | None = None,
        atom_vocab: list | None = None,
        train_set: torch.utils.data.Dataset | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.interpolant = _plain(interpolant)
        self.dynamics = _plain(dynamics)
        self.self_conditioning = _plain(self_conditioning) or None
        self.timesteps = int(timesteps)
        self.loss_scale = float(loss_scale)
        self.loss_clamp = loss_clamp
        self.return_step_output = bool(return_step_output)
        self.sample_input = sample_input
        self.sample_pool_limit = int(sample_pool_limit)
        self.dataset_stats = _plain(dataset_stats)
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.train_set = train_set
        self.kwargs = kwargs
        self.task: LoQIConformerTask | None = None

    def build(self) -> "LoQIConformerTask":
        if not self.atom_vocab:
            msg = (
                "LoQI needs atom_vocab (it maps the platform's atom indices onto "
                "LoQI's fixed 17-element FULL_ATOM_ENCODER by symbol). Set "
                "data.atom_vocab, or tasks.atom_vocab explicitly."
            )
            raise ValueError(msg)

        hist = dict(self.dataset_stats.get("num_atoms_histogram") or {})
        pool: list[Data] = []
        if not hist:
            stats = getattr(self.train_set, "graph3d_stats", None)
            if stats is not None:
                hist = {int(k): int(v) for k, v in stats.n_atoms_hist.items()}
                logger.info(
                    "LoQI built from graph3d_stats over %d molecules: sizes %d-%d, "
                    "bond counts %s, charge range %s",
                    stats.n_molecules,
                    min(hist) if hist else -1,
                    max(hist) if hist else -1,
                    list(stats.bond_type_counts),
                    stats.charge_range,
                )
            else:
                # Expected on the generation path: cli/generate.py builds no
                # DataModule. Sizes come from the conditioning molecules anyway.
                logger.warning(
                    "No train_set.graph3d_stats -- building LoQI with an EMPTY "
                    "size histogram. Expected when loading a checkpoint for "
                    "generation; during TRAINING set data.data_type=graph3d "
                    "with graph3d_stats: true."
                )

        if self.sample_input is None and self.train_set is not None:
            pool = _pool_from_train_set(self.train_set, self.sample_pool_limit)
            logger.info("LoQI conditioning pool: %d molecules from train_set", len(pool))

        self.task = LoQIConformerTask(
            task_type=self.task_type,
            atom_vocab=self.atom_vocab,
            interpolant_config=self.interpolant,
            dynamics_config=self.dynamics,
            self_cond_config=self.self_conditioning,
            timesteps=self.timesteps,
            loss_scale=self.loss_scale,
            loss_clamp=self.loss_clamp,
            return_step_output=self.return_step_output,
            sample_input=self.sample_input,
            sample_pool_limit=self.sample_pool_limit,
            n_atoms_hist=hist,
            conditioning_pool=pool,
        )
        return self.task


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class LoQIConformerTask(nn.Module):
    """LoQI wrapped in the platform's duck-typed Task contract (§2.1)."""

    def __init__(  # noqa: PLR0913
        self,
        task_type: str,
        atom_vocab: list[str],
        interpolant_config: dict,
        dynamics_config: dict,
        self_cond_config: dict | None,
        timesteps: int,
        loss_scale: float,
        loss_clamp: float | None,
        return_step_output: bool,
        sample_input: str | Sequence[str] | None,
        sample_pool_limit: int,
        n_atoms_hist: dict,
        conditioning_pool: list,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.atom_vocab = list(atom_vocab)
        self.timesteps = int(timesteps)
        #: read by runmodes/generate/tasks_generate.py to pick a default step count
        self.T = int(timesteps)
        self.loss_scale = float(loss_scale)
        self.loss_clamp = loss_clamp
        self.return_step_output = bool(return_step_output)
        self.sample_input = sample_input
        self.sample_pool_limit = int(sample_pool_limit)

        self.adapter = Graph3DToLoQIAdapter(self.atom_vocab)
        self.dynamics = MegaFNV3Conf(**dynamics_config)
        self.interpolant = build_interpolant(timesteps=timesteps, **interpolant_config)
        self.self_cond = (
            BaseSelfConditioningModule(
                [{**v, "inp_dim": 3} for v in self_cond_config["variables"]]
            )
            if self_cond_config
            else None
        )

        self.node_dist_model = LoQINodeDistribution(n_atoms_hist)
        self.prop_dist_model = None
        self._pool = list(conditioning_pool)

    # -- contract plumbing ---------------------------------------------------

    @property
    def model(self) -> "LoQIConformerTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def n_node_dist(self) -> dict:
        return self.node_dist_model.n_node_dist

    def _network_time(self, time: torch.Tensor) -> torch.Tensor:
        """Interpolant time -> the network's ``t``.

        Discrete time is rescaled to ``(T - t) / T`` exactly as upstream's
        ``MegaFNV3ConfWrapper.forward`` does; continuous time is passed through.
        """
        if self.interpolant.time_type == "discrete":
            return (self.timesteps - time.float()) / self.timesteps
        return time.float()

    # -- training ------------------------------------------------------------

    def forward(self, batch: Any) -> tuple[torch.Tensor, dict]:
        d = self.adapter(batch)
        batch_idx = d["batch"]
        batch_size = int(batch_idx.max()) + 1

        time = self.interpolant.sample_time(
            num_samples=batch_size,
            method="uniform",
            device=batch_idx.device,
            min_t=getattr(self.interpolant, "min_t", 0.0),
        )
        target, x_t, _x0 = self.interpolant.interpolate(batch_idx, d["x"], time)

        x_in = x_t
        if self.self_cond is not None:
            # Upstream (module.py:635-654): one no-grad pass to produce the
            # conditioning signal, fused into x_t; half the time the fused
            # value is discarded but kept in the graph so no parameter is
            # reported unused by DDP.
            with torch.no_grad():
                prior_out = self._denoise(d, x_t, time)
            fused, pre = self.self_cond(
                {"x_t": x_t}, {"x_hat": prior_out["x_hat"].detach()}
            )
            x_in = fused["x_t"]
            if torch.rand(1).item() <= 0.5:
                x_in = pre["x_t"] + 0 * x_in

        out = self._denoise(d, x_in, time)

        ws_t = self.interpolant.loss_weight_t(time)
        per_atom = F.mse_loss(out["x_hat"], target, reduction="none").mean(-1)
        per_mol = scatter_mean(per_atom, batch_idx, dim=0, dim_size=batch_size)
        if ws_t is not None:
            per_mol = per_mol * ws_t
        if self.loss_clamp is not None:
            per_mol = per_mol.clamp(0, self.loss_clamp)
        loss = self.loss_scale * per_mol.mean()
        return loss, {"loss": loss.detach(), "x_loss": loss.detach()}

    def _denoise(self, d: dict, x_t: torch.Tensor, time: torch.Tensor) -> dict:
        """Concatenate charges onto atom types (23 wide) and run the backbone."""
        H = torch.cat([d["h"], d["charges"]], dim=-1)
        return self.dynamics(
            batch=d["batch"],
            X=x_t,
            H=H,
            E_idx=d["edge_index"],
            E=d["edge_attr"],
            t=self._network_time(time),
        )

    def predict_and_target(self, batch: Any):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    # -- generation ----------------------------------------------------------

    def _ensure_accelerated(self) -> None:
        """``cli/generate.py`` skips placement for tasks exposing ``device``."""
        if self.device.type == "cpu" and torch.cuda.is_available():
            logger.info("sample(): parameters were on CPU; moving to cuda")
            self.to(torch.device("cuda"))
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    def conditioning_pool(self) -> list:
        if not self._pool:
            if self.sample_input is None:
                msg = (
                    "LoQI generates conformers OF a molecule, so it needs input "
                    "molecules. Either train with a graph3d train_set (the pool "
                    "is taken from it) or set tasks.sample_input to an .sdf, "
                    ".smi, or graph3d ASE .db path -- or to an inline "
                    "list of SMILES strings."
                )
                raise ValueError(msg)
            self._pool = load_conditioning_pool(
                self.sample_input, self.atom_vocab, self.sample_pool_limit
            )
            logger.info(
                "LoQI conditioning pool: %d molecules from %s",
                len(self._pool),
                self.sample_input,
            )
        return self._pool

    def _pick(self, sizes: torch.Tensor | None, count: int) -> list:
        """Draw ``count`` conditioning molecules, size-matched where possible."""
        pool = self.conditioning_pool()
        if sizes is None:
            return [pool[i % len(pool)] for i in range(count)]
        pool_sizes = torch.tensor([int(g.n_nodes) for g in pool])
        picked, used = [], set()
        for n in sizes.tolist()[:count]:
            # Closest available size; exact sizes are not enforced (§2.1 allows
            # it). Molecules already used are pushed to the back rather than
            # excluded, so a pool smaller than the request still works -- it
            # just starts repeating once every molecule has been used.
            cost = (pool_sizes - int(n)).abs().clone()
            if len(used) < len(pool):
                cost[list(used)] += 10**6
            j = int(torch.argmin(cost))
            used.add(j)
            picked.append(pool[j])
        return picked

    @torch.no_grad()
    def sample(  # noqa: PLR0913, C901
        self,
        batch_size: int | None = None,
        nodesxsample: torch.Tensor | None = None,
        num_steps: int | None = None,
        batch: dict | None = None,
        mode: str | None = None,  # noqa: ARG002 - DDIM modes out of scope
        n_frames: int = 0,
        mols: list | None = None,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Generate conformers for a batch of conditioning molecules.

        Returns the platform tuple ``(one_hot (B,N,V), charges (B,N),
        coords (B,N,3), node_mask (B,N))`` with ``V = len(atom_vocab)``. Atom
        types and charges are the conditioning molecule's -- inputs, not
        predictions.

        ``nodesxsample`` is honoured as a *count*, and molecules are size-matched
        to it where the pool allows. ``mode`` is accepted and ignored.

        ``mols`` names the conditioning molecules EXPLICITLY (a list of the
        pool's ``Data`` items, repeats allowed) and bypasses the size-matched
        draw entirely -- that is how the conformer mode asks for "k conformers
        of THIS molecule", which a size draw cannot express. When it is None
        (every pre-existing caller) nothing changes.
        """
        if n_frames:
            msg = "LoQI does not emit trajectories; set interference.n_frames to 0"
            raise ValueError(msg)

        self._ensure_accelerated()
        device = self.device

        if mols is not None:
            if not mols:
                msg = "sample(mols=[]) has nothing to make a conformer of"
                raise ValueError(msg)
            sizes, count = None, len(mols)
        elif nodesxsample is not None:
            sizes = torch.as_tensor(nodesxsample, dtype=torch.long)
            count = int(sizes.numel())
        elif batch is not None and "natoms" in batch:
            sizes = batch["natoms"].long()
            count = int(sizes.numel())
        elif batch_size is not None:
            sizes, count = None, int(batch_size)
        else:
            msg = "sample() needs nodesxsample, batch, or batch_size"
            raise ValueError(msg)

        if num_steps is None:
            num_steps = self.timesteps
        elif self.interpolant.time_type == "discrete" and int(num_steps) != self.timesteps:
            logger.warning(
                "The released LoQI diffusion weights were trained at %d steps and "
                "the README states other values are not expected to work; "
                "ignoring num_steps=%s.",
                self.timesteps,
                num_steps,
            )
            num_steps = self.timesteps
        num_steps = int(num_steps)

        picked = list(mols) if mols is not None else self._pick(sizes, count)
        pyg = Batch.from_data_list(picked).to(device)
        d = self.adapter(pyg)
        batch_idx = d["batch"]
        n_mol = int(batch_idx.max()) + 1

        # -- reverse loop ----------------------------------------------------
        if self.interpolant.time_type == "continuous":
            timeline = torch.linspace(
                getattr(self.interpolant, "min_t", 0.0), 1, num_steps + 1
            ).tolist()
            dts = [t1 - t0 for t0, t1 in zip(timeline[:-1], timeline[1:])]
        else:
            timeline = list(range(num_steps + 1))
            dts = [1.0 / num_steps] * num_steps

        prior = self.interpolant.prior(batch_idx, d["x"].shape, device)
        x_t = prior.clone()
        out: dict = {}
        for idx in range(num_steps):
            t = timeline[idx]
            time = torch.full(
                (n_mol,),
                t,
                device=device,
                dtype=torch.float32
                if self.interpolant.time_type == "continuous"
                else torch.long,
            )
            x_in = x_t
            if self.self_cond is not None and out:
                fused, _pre = self.self_cond({"x_t": x_t.clone()}, out)
                x_in = fused["x_t"]
            out = self._denoise(d, x_in, time)
            x_t = self.interpolant.step(
                batch_idx, xt=x_t, x_hat=out["x_hat"], x0=prior, time=time, dt=dts[idx]
            )

        coords_flat = x_t if self.return_step_output else out["x_hat"]
        return self._decode(pyg, coords_flat)

    def _decode(self, pyg: Any, coords_flat: torch.Tensor):
        """Flat PyG result -> padded ``(one_hot, charges, coords, node_mask)``."""
        device = self.device
        sizes = [int(g.n_nodes) for g in pyg.to_data_list()]
        bs, n_max = len(sizes), max(sizes)
        vocab = len(self.atom_vocab)

        one_hot = torch.zeros(bs, n_max, vocab, device=device)
        charges = torch.zeros(bs, n_max, dtype=torch.long, device=device)
        coords = torch.zeros(bs, n_max, 3, device=device)
        node_mask = torch.zeros(bs, n_max, dtype=torch.long, device=device)

        start = 0
        for i, (n, item) in enumerate(zip(sizes, pyg.to_data_list())):
            one_hot[i, :n] = F.one_hot(item.atom_idx.long().to(device), vocab).float()
            charges[i, :n] = item.fc.long().to(device)
            coords[i, :n] = coords_flat[start : start + n]
            node_mask[i, :n] = 1
            start += n

        # Bond orders have no channel in the platform tuple; keep them reachable
        # for callers that want an .sdf (same convention as diffusion_flowmol_graph3d).
        self.last_bond_types = [
            (g.bond_index.cpu(), g.bond_type.cpu()) for g in pyg.to_data_list()
        ]
        return one_hot, charges, coords, node_mask
