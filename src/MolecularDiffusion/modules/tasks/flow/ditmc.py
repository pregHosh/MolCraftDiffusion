"""DiTMC conformer-generation task (``task_type: diffusion_ditmc``).

Wraps ``modules/models/ditmc`` in the platform's duck-typed ``Task`` contract
(docs/adding_new_models.md 2.1) and adapts the ``graph3d`` PyG ``Batch``
(``bond_collate: raw``) into the three graphs DiTMC consumes.

**This model does not invent molecules.** You hand it a molecule you already
have -- its atoms and its bonds -- and it returns 3D conformers of exactly that
molecule. Nothing about the composition is generated, so:

* :meth:`DiTMCTask.sample` **raises**. There is no unconditional mode: without a
  molecular graph there is nothing to place in 3D, and the platform's
  ``(one_hot, charges, coords, node_mask)`` return contract has no meaning here.
  Generation goes through the shared
  :class:`~MolecularDiffusion.runmodes.generate.tasks_conformer.ConformerFactory`,
  pointed at by ``configs/interference/gen_conformer.yaml`` -- the same route
  ``gen_diffdec_scaffold.yaml`` and ``gen_apo2mol_pocket.yaml`` take.
* ``node_dist_model`` / ``n_node_dist`` are deliberately absent. They exist to
  let ``GenerativeFactory`` choose a molecule size; DiTMC's size is dictated by
  the input molecule.

Bond mapping (canonical -> DiTMC): ``ditmc_class = canonical_class - 1``, so
``1=SINGLE -> 0``, ``2=DOUBLE -> 1``, ``3=TRIPLE -> 2``, ``4=AROMATIC -> 3``.
Canonical class 0 ("no bond") is simply never an edge of the conditioner graph,
which is exactly the platform's storage rule -- a pass-through. **Do not set
``kekulize: true``**: upstream trains on RDKit's aromatic perception and
aromatic is a real class in its 4-wide vocabulary.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
from torch import nn

from MolecularDiffusion.modules.models.ditmc import (
    CondGraph,
    FlowMatching,
    LatentGraph,
    MoleculeFeatureCache,
    PriorGraph,
    build_prior,
    build_variant,
    center_data,
    node_attr_dim,
    rotation_augmentation,
)

logger = logging.getLogger(__name__)

#: canonical bond class -> DiTMC one-hot column. Class 0 is never materialized.
N_BOND_CLASSES = 4


class Graph3DBatchToDiTMCAdapter(nn.Module):
    """One ``graph3d`` PyG ``Batch`` -> ``(LatentGraph, CondGraph, PriorGraph)``.

    Everything derived per molecule (``node_attr``, shortest hops, the Laplacian
    eigendecomposition) is cached by SMILES, exactly as upstream caches it at
    dataset-build time.

    The one featurization column that is *derived* rather than copied is the
    chiral tag: upstream reads it off the GEOM mol, and here it is recovered by
    RDKit from the stored coordinates. It is the first thing to check if a
    converted pretrained checkpoint underperforms.
    """

    def __init__(self, dataset: str = "qm9") -> None:
        super().__init__()
        self.dataset = dataset
        self.cache = MoleculeFeatureCache(dataset=dataset)

    def forward(self, batch: Any):
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        return self.build(pyg.to_data_list(), device=pyg.pos.device)

    def build(self, items: list, device=None):  # noqa: PLR0914
        node_attr, atomic_numbers, positions, batch_segments = [], [], [], []
        lat_rec, lat_sen, hops = [], [], []
        cond_sen, cond_rec, cond_attr = [], [], []
        prior_d, prior_sen, prior_rec, prior_attr = [], [], [], []
        offset = 0

        for gid, item in enumerate(items):
            n = int(item.pos.shape[0])
            attr, hop, d, p = self.cache.get(item)

            node_attr.append(torch.as_tensor(attr, dtype=torch.float32))
            atomic_numbers.append(item.z.long())
            positions.append(item.pos.float())
            batch_segments.append(torch.full((n,), gid, dtype=torch.long))

            # All ordered pairs, C-ORDER over (i, j), receivers=i, senders=j.
            # Same order as `hop`, which is the only reason they line up.
            idx = np.arange(n)
            ii, jj = np.meshgrid(idx, idx, indexing="ij")
            mask = ii != jj
            lat_rec.append(torch.as_tensor(ii[mask] + offset, dtype=torch.long))
            lat_sen.append(torch.as_tensor(jj[mask] + offset, dtype=torch.long))
            hops.append(torch.as_tensor(hop, dtype=torch.long))

            # Bond edges, mirrored to both directions; class -> one-hot col - 1.
            bi = item.bond_index.long().reshape(2, -1)
            bt = item.bond_type.long().reshape(-1)
            if bt.numel() and int(bt.max()) > N_BOND_CLASSES:
                msg = (
                    f"bond class {int(bt.max())} exceeds DiTMC's 4-class "
                    f"vocabulary (1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC)"
                )
                raise ValueError(msg)
            both = torch.cat([bi, bi.flip(0)], dim=1)
            bt2 = torch.cat([bt, bt])
            cond_sen.append(both[0] + offset)
            cond_rec.append(both[1] + offset)
            cond_attr.append(
                torch.nn.functional.one_hot(
                    (bt2 - 1).clamp(min=0), N_BOND_CLASSES
                ).float()
            )

            # Harmonic prior: complete index grid, senders = eigen index.
            prior_d.append(torch.as_tensor(d, dtype=torch.float32))
            prior_rec.append(torch.as_tensor(ii.reshape(-1) + offset, dtype=torch.long))
            prior_sen.append(torch.as_tensor(jj.reshape(-1) + offset, dtype=torch.long))
            prior_attr.append(torch.as_tensor(p.reshape(-1), dtype=torch.float32))

            offset += n

        def cat(xs):
            return torch.cat(xs).to(device) if xs else torch.zeros(0)

        node_attr_t = torch.cat(node_attr).to(device)
        senders = cat(lat_sen)
        num_nodes = offset
        latent = LatentGraph(
            atomic_numbers=cat(atomic_numbers),
            node_attr=node_attr_t,
            positions=cat(positions),
            senders=senders,
            receivers=cat(lat_rec),
            shortest_hops=cat(hops),
            batch_segments=cat(batch_segments),
            num_graphs=len(items),
            cond_scaling_nodes=torch.ones(num_nodes, device=device),
            cond_scaling_edges=torch.ones(senders.shape[0], device=device),
            x1=cat(positions),
        )
        cond = CondGraph(
            node_attr=node_attr_t,
            senders=cat(cond_sen),
            receivers=cat(cond_rec),
            edge_attr=torch.cat(cond_attr).to(device)
            if cond_attr
            else torch.zeros(0, N_BOND_CLASSES, device=device),
        )
        prior = PriorGraph(
            node_attr=cat(prior_d),
            senders=cat(prior_sen),
            receivers=cat(prior_rec),
            edge_attr=cat(prior_attr),
        )
        return latent, cond, prior


class DiTMCTask(nn.Module):
    """DiTMC in the platform's Task contract."""

    def __init__(  # noqa: PLR0913
        self,
        variant: str,
        dataset: str,
        model_kwargs: dict,
        flow_kwargs: dict,
        prior: str = "harmonic",
        augmentation_bool: bool = True,
        atom_vocab: list | None = None,
        task_type: str = "diffusion_ditmc",
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.variant = variant
        self.dataset = dataset
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.augmentation_bool = augmentation_bool

        self.adapter = Graph3DBatchToDiTMCAdapter(dataset=dataset)
        self.net = build_variant(
            variant, node_attr_dim=node_attr_dim(dataset), **model_kwargs
        )
        self.process = FlowMatching(prior=build_prior(prior), **flow_kwargs)

    # -- contract ----------------------------------------------------------

    @property
    def model(self) -> "DiTMCTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: Any):
        latent, cond, prior = self.adapter(batch)
        if self.augmentation_bool and self.training:
            # One Haar rotation per molecule, applied to positions and x1 --
            # ON by default upstream (globals.augmentation_bool: True). Without
            # it dit_ape/dit_rpe, which are not equivariant by construction,
            # train on a different distribution than the published weights.
            rotated, _ = rotation_augmentation(
                {"positions": latent.positions, "x1": latent.x1},
                latent.batch_segments,
                latent.num_graphs,
            )
            latent = latent.replace(**rotated)
        return self.process.loss(self.net, latent, prior, cond)

    def predict_and_target(self, batch: Any, all_loss=None, metric=None):  # noqa: ARG002
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    def sample(self, *args, **kwargs):  # noqa: ARG002
        msg = (
            "DiTMC has no unconditional generation mode -- it places an EXISTING "
            "molecule in 3D, so there is nothing to sample without one. Use "
            "configs/interference/gen_conformer.yaml and set "
            "interference.sample_input to the molecules you want conformers "
            "of; it drives this task's generate_conformers()."
        )
        raise NotImplementedError(msg)

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate_conformers(  # noqa: PLR0913
        self,
        items: list,
        *,
        num_steps: int = 50,
        free_guidance_scale: float = 1.0,
        logarithmic_time_bool: bool = False,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
    ):
        latent, cond, prior = self.adapter.build(items, device=self.device)
        out = self.process.sample(
            self.net,
            latent,
            prior,
            cond,
            num_steps=num_steps,
            free_guidance_scale=free_guidance_scale,
            logarithmic_time_bool=logarithmic_time_bool,
            return_trajectory=return_trajectory,
            generator=generator,
        )
        if return_trajectory:
            graph, traj = out
            return graph.positions, latent.batch_segments, traj
        return out.positions, latent.batch_segments, None


class DiTMCTaskFactory:
    """Factory instantiated by ``cli/train.py``.

    ``train_set`` is deliberately **not** declared: DiTMC needs no dataset
    statistics at construction (no marginals, no valency table, no size
    histogram), so the declarative injection seam simply does not fire.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_ditmc",
        variant: str = "so3",
        dataset: str = "qm9",
        model: dict | None = None,
        flow: dict | None = None,
        prior: str = "harmonic",
        augmentation_bool: bool = True,
        atom_vocab: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.variant = variant
        self.dataset = dataset
        self.model_kwargs = _plain(model or {})
        self.flow_kwargs = _plain(flow or {})
        self.prior = prior
        self.augmentation_bool = augmentation_bool
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.kwargs = kwargs
        self.task: DiTMCTask | None = None

    def build(self) -> DiTMCTask:
        self.task = DiTMCTask(
            variant=self.variant,
            dataset=self.dataset,
            model_kwargs=self.model_kwargs,
            flow_kwargs=self.flow_kwargs,
            prior=self.prior,
            augmentation_bool=self.augmentation_bool,
            atom_vocab=self.atom_vocab,
            task_type=self.task_type,
        )
        n_params = sum(p.numel() for p in self.task.parameters())
        logger.info(
            "Built DiTMC variant=%s dataset=%s node_attr_dim=%d params=%d",
            self.variant,
            self.dataset,
            node_attr_dim(self.dataset),
            n_params,
        )
        return self.task


def _plain(cfg: Any) -> dict:
    """OmegaConf node -> plain dict (Hydra hands these over as DictConfig)."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


# ``DiTMCConformerGenerator`` used to live here: ~230 lines that loaded a
# molecule pool, ran `generate_conformers`, and wrote flat .xyz. Conformer
# generation is now unified behind
# ``runmodes/generate/tasks_conformer.ConformerFactory``
# (``configs/interference/gen_conformer.yaml``), which calls this task's
# ``generate_conformers`` through the shared contract and additionally emits
# the per-molecule layout, conformers.sdf/reference.sdf and the conformers.csv
# index this class never produced.


ModelTaskFactory = DiTMCTaskFactory
