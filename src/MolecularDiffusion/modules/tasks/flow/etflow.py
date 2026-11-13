"""ET-Flow conformer-generation task (``task_type: diffusion_etflow``).

Wraps ``modules/models/etflow`` in the platform's duck-typed ``Task`` contract
(docs/adding_new_models.md 2.1) and adapts a ``graph3d`` PyG ``Batch``
(``bond_collate: raw``) into the flat tensors ET-Flow's vector field consumes.

**This model does not invent molecules.** You hand it a molecule you already
have -- its atoms and its bonds -- and it returns 3D conformers of exactly that
molecule. Nothing about the composition is generated, so:

* :meth:`ETFlowTask.sample` **raises**. There is no unconditional mode: without
  a molecular graph there is no bond Laplacian, hence no harmonic prior, no
  node features and nothing to place. Generation goes through the shared
  :class:`~MolecularDiffusion.runmodes.generate.tasks_conformer.ConformerFactory`,
  pointed at by ``configs/interference/gen_conformer.yaml`` -- the same route
  DiTMC and NExT-Mol take.
* ``node_dist_model`` / ``n_node_dist`` are deliberately absent. They exist to
  let ``GenerativeFactory`` choose a molecule size; ET-Flow's size is dictated
  by the input molecule.

BOND MAPPING (canonical -> ET-Flow). ET-Flow has **no bond vocabulary**. Every
canonical class ``1=SINGLE 2=DOUBLE 3=TRIPLE 4=AROMATIC`` maps to the same
model-side value -- ``edge_type = 1``, "these two atoms are bonded"
(``models/utils.py:54-56``, reached because upstream passes ``edge_attr=None``
on both its training and inference paths). Class 0 is never an edge, which is
the platform's storage rule -- a pass-through. Radius-graph edges take
``edge_type = 0`` and are regenerated from the current coordinates at every
step. This is not a lossy adaptation on our side: it is upstream's shipped
behaviour, and bond order still reaches the model through ``node_attr``
(aromatic flag, hybridization, degree, implicit valence, H count).

**Do not set ``kekulize: true``**: it would zero the aromatic column and shift
hybridization for every ring atom, changing the conditioning distribution the
published weights were trained on.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from MolecularDiffusion.modules.models.etflow import (
    NODE_ATTR_DIM,
    ETFlowFeatureCache,
    HarmonicSampler,
    TorchMDDynamics,
    batchwise_l2_loss,
    center_of_mass,
    extend_bond_index,
    graph_key,
    rmsd_align,
    switch_parity_of_pos,
    unsqueeze_like,
)

logger = logging.getLogger(__name__)

#: Time is drawn uniform on this interval, ONE draw per molecule
#: (``models/model.py:262-268``).
TIME_LOW = 1e-4
TIME_HIGH = 0.9999


class Graph3DBatchToETFlowAdapter(nn.Module):
    """One ``graph3d`` PyG ``Batch`` (or a plain item list) -> flat tensors.

    Two honesty caveats, carried from the integration plan:

    1. **Chirality is recovered from geometry, not read from a SMILES.**
       ``build_rdkit_mol(..., coords=...)`` runs ``AssignStereochemistryFrom3D``
       (``data/component/graph3d_dataset.py:184-186``), so the chiral tags come
       from the input conformer. Upstream reads them off the GEOM mol. This is
       the first thing to check if a converted pretrained checkpoint
       underperforms.
    2. **Achiral molecules** yield ``chiral_index`` of shape ``(1, 0)``, and
       every consumer downstream must treat that as a no-op rather than an
       error.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cache = ETFlowFeatureCache()

    def forward(self, batch: Any) -> dict:
        pyg = batch["graph"] if isinstance(batch, dict) else batch
        return self.build(pyg.to_data_list(), device=pyg.pos.device)

    def build(self, items: list, device: Any = None) -> dict:
        """Flat, concatenated tensors in item-major order."""
        z, pos, node_attr, segments = [], [], [], []
        bonds, chi_idx, chi_nbr, chi_tag, keys = [], [], [], [], []
        offset = 0

        for gid, item in enumerate(items):
            n = int(item.pos.shape[0])
            attr, c_idx, c_nbr, c_tag = self.cache.get(item)

            z.append(item.z.long())
            pos.append(item.pos.float())
            node_attr.append(torch.as_tensor(attr, dtype=torch.float32))
            segments.append(torch.full((n,), gid, dtype=torch.long))

            # THE MIRRORING POINT. Storage keeps only the upper triangle;
            # ET-Flow wants every bond in BOTH directions, which is what
            # `compute_edge_index` (commons/utils.py:144-151) emits. The order
            # differs from upstream's by a permutation, which nothing here can
            # see: every consumer is a sparse-COO coalesce or a scatter.
            bi = item.bond_index.long().reshape(2, -1)
            bonds.append(torch.cat([bi, bi.flip(0)], dim=1) + offset)

            # Not going through Batch.from_data_list, so the node offsets PyG
            # would apply to these two "*index*" keys are added by hand.
            chi_idx.append(torch.as_tensor(c_idx, dtype=torch.long) + offset)
            chi_nbr.append(torch.as_tensor(c_nbr, dtype=torch.long) + offset)
            chi_tag.append(torch.as_tensor(c_tag, dtype=torch.float32))
            keys.append(graph_key(item))

            offset += n

        def cat(parts: list, dim: int = 0):
            return torch.cat(parts, dim=dim).to(device)

        return {
            "z": cat(z),
            "pos": cat(pos),
            "node_attr": cat(node_attr),
            "bond_index": cat(bonds, dim=1),
            "batch": cat(segments),
            "chiral_index": cat(chi_idx, dim=1),
            "chiral_nbr_index": cat(chi_nbr, dim=1),
            "chiral_tag": cat(chi_tag),
            # Harmonic-prior eigendecomposition cache keys, one per molecule.
            "keys": keys,
        }


class ETFlowTask(nn.Module):
    """ET-Flow in the platform's Task contract.

    ``self.network`` is named to match the released checkpoints exactly, so
    ``scripts/convert_checkpoint.py`` is an identity remap that can assert a
    strict bijection instead of guessing at a rename table.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_kwargs: dict,
        sigma: float = 0.1,
        prior_type: str = "harmonic",
        harmonic_alpha: float = 1.0,
        parity_switch: str | None = "post_hoc",
        sample_time_dist: str = "uniform",
        max_num_neighbors: int = 32,
        atom_vocab: list | None = None,
        task_type: str = "diffusion_etflow",
    ) -> None:
        super().__init__()
        if prior_type not in ("harmonic", "gaussian"):
            msg = f"prior_type must be 'harmonic' or 'gaussian', got {prior_type!r}"
            raise ValueError(msg)
        if parity_switch not in (None, "post_hoc"):
            msg = f"parity_switch must be null or 'post_hoc', got {parity_switch!r}"
            raise ValueError(msg)
        if sample_time_dist not in ("uniform", "logit_norm"):
            msg = f"unknown sample_time_dist {sample_time_dist!r}"
            raise ValueError(msg)
        if model_kwargs.get("node_attr_dim", NODE_ATTR_DIM) != NODE_ATTR_DIM:
            msg = (
                f"node_attr_dim must be {NODE_ATTR_DIM} -- it is the width of "
                "atom_to_feature_vector(), not a free hyperparameter."
            )
            raise ValueError(msg)

        self.task_type = task_type
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.sigma = sigma
        self.prior_type = prior_type
        self.parity_switch = parity_switch
        self.sample_time_dist = sample_time_dist
        self.max_num_neighbors = max_num_neighbors
        # The radius-graph cutoff IS the network's upper cutoff upstream
        # (model.py:100). Keeping them tied is not optional: edges beyond
        # cutoff_upper get a zero cosine envelope and contribute nothing.
        self.cutoff = model_kwargs.get("cutoff_upper", 10.0)

        self.adapter = Graph3DBatchToETFlowAdapter()
        self.network = TorchMDDynamics(**model_kwargs)
        self.harmonic_sampler = (
            HarmonicSampler(alpha=harmonic_alpha)
            if prior_type == "harmonic"
            else None
        )

    # -- contract ----------------------------------------------------------

    @property
    def model(self) -> "ETFlowTask":
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch: Any):
        data = self.adapter(batch)
        return self._flow_matching_loss(data)

    def predict_and_target(self, batch: Any, all_loss=None, metric=None):  # noqa: ARG002
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):  # noqa: ARG002
        return {"val_loss": pred.mean()}

    def sample(self, *args, **kwargs):  # noqa: ARG002
        msg = (
            "ET-Flow has no unconditional generation mode -- it places an "
            "EXISTING molecule in 3D, so there is nothing to sample without "
            "one. Without a bond graph there is no Laplacian, hence no "
            "harmonic prior at all. Use "
            "configs/interference/gen_conformer.yaml and set "
            "interference.sample_input to the molecules you want conformers "
            "of; it drives this task's generate_conformers()."
        )
        raise NotImplementedError(msg)

    # -- flow matching -----------------------------------------------------

    def _prior(self, data: dict) -> torch.Tensor:
        """Harmonic prior over the bond-graph Laplacian, or plain Gaussian."""
        size = (data["z"].size(0), 3)
        if self.harmonic_sampler is None:
            return torch.randn(size, device=self.device)

        x0 = self.harmonic_sampler.sample(
            size=size,
            edge_index=data["bond_index"],
            batch=data["batch"],
            keys=data["keys"],
        ).to(self.device)
        if torch.isnan(x0).any():
            # ValueError, not RuntimeError: the engine treats it as one bad
            # batch and escalates only if they keep coming.
            msg = (
                "Harmonic prior is NaN. The usual cause is a DISCONNECTED "
                "molecular graph (a salt, or anything in two pieces): its "
                "Laplacian has extra zero eigenvalues and 1/sqrt(D) diverges. "
                "ET-Flow genuinely cannot handle multi-fragment inputs."
            )
            raise ValueError(msg)
        return x0

    def _sample_time(self, num_graphs: int) -> torch.Tensor:
        if self.sample_time_dist == "logit_norm":
            return torch.sigmoid(
                torch.randn((num_graphs, 1), device=self.device)
            )
        return torch.zeros((num_graphs, 1), device=self.device).uniform_(
            TIME_LOW, TIME_HIGH
        )

    def _sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sqrt(t * (1 - t))

    def _sigma_dot_t(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * 0.5 * (1 - 2 * t) / torch.sqrt(t * (1 - t))

    def _conditional_vector_field(self, x0, x1, t, batch):
        """Gaussian-bridge interpolant and its exact velocity."""
        x0 = center_of_mass(x0, batch=batch)
        x1 = center_of_mass(x1, batch=batch)
        t = unsqueeze_like(t[batch], target=x0)

        eps = center_of_mass(torch.randn_like(x1), batch=batch)
        x_t = t * x1 + (1 - t) * x0 + self._sigma_t(t) * eps
        u_t = x1 - x0 + self._sigma_dot_t(t) * eps
        return x_t, u_t

    def _vector_field(self, data: dict, t: torch.Tensor, pos: torch.Tensor):
        """One network call. ``t`` is per-GRAPH, shape ``(B, 1)``."""
        batch = data["batch"]
        pos = center_of_mass(pos, batch=batch)
        # The edge set is rebuilt from the CURRENT coordinates every call:
        # bonds union radius_graph(pos). It is not a fixed graph.
        edge_index, edge_type = extend_bond_index(
            pos=pos,
            bond_index=data["bond_index"],
            batch=batch,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_num_neighbors,
        )
        return self.network(
            z=data["z"],
            t=t[batch],
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_type,
            node_attr=data["node_attr"],
            batch=batch,
        )

    def _flow_matching_loss(self, data: dict):
        batch = data["batch"]
        num_graphs = int(batch.max()) + 1

        x1 = data["pos"]
        x0 = self._prior(data)
        if self.prior_type == "harmonic":
            # THE "equivariant" in ET-Flow: Kabsch-aligning the prior sample to
            # the data conformer removes the global rotation from the target.
            x0 = rmsd_align(pos=x0, ref_pos=x1, batch=batch)

        t = self._sample_time(num_graphs)
        x_t, u_t = self._conditional_vector_field(x0, x1, t, batch)
        v_t = self._vector_field(data, t, x_t)

        loss = batchwise_l2_loss(v_t, u_t, batch=batch)
        if not torch.isfinite(loss):
            msg = "Flow-matching loss is not finite."
            raise ValueError(msg)
        return loss, {"loss": loss, "flow_matching_loss": loss}

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate_conformers(  # noqa: PLR0913
        self,
        items: list,
        *,
        n_timesteps: int = 50,
        s_churn: float = 1.0,
        t_min: float = TIME_LOW,
        t_max: float = TIME_HIGH,
        std: float = 1.0,
        sampler_type: str = "ode",
    ):
        """Euler-integrate the learned field; one output graph per input item.

        With ``sampler_type: ode`` (upstream's shipped setting) the
        ``t_min``/``t_max`` window is inert -- the churn branch below is only
        reachable with ``sampler_type: stochastic``.
        """
        data = self.adapter.build(items, device=self.device)
        batch = data["batch"]
        num_graphs = int(batch.max()) + 1

        schedule = torch.linspace(0.0, 1.0, n_timesteps + 1, device=self.device)
        x = center_of_mass(self._prior(data), batch=batch)
        gamma = s_churn / n_timesteps

        for i in range(n_timesteps):
            t_i = schedule[i]
            delta_t = schedule[i + 1] - t_i
            t = torch.full((num_graphs, 1), float(t_i), device=self.device)

            if sampler_type == "ode" or t_i < t_min or t_i >= t_max:
                x = x + delta_t * self._vector_field(data, t, x)
                continue

            # Stochastic churn: step back to t - delta_hat, re-noise, then take
            # a longer step forward.
            delta_hat = gamma * (1 - t_i)
            t_prev_val = t_i - delta_hat
            t_prev = torch.full(
                (num_graphs, 1), float(t_prev_val), device=self.device
            )
            noise = center_of_mass(
                torch.normal(mean=torch.zeros_like(x), std=std), batch=batch
            )
            x_prev = x + torch.sqrt(
                torch.abs(t_i**2 - t_prev_val**2)
            ) * noise * delta_hat
            x = x_prev + self._vector_field(data, t_prev, x_prev) * (
                delta_t + delta_hat
            )

        if self.parity_switch == "post_hoc":
            x = switch_parity_of_pos(
                x,
                data["chiral_index"],
                data["chiral_nbr_index"],
                data["chiral_tag"],
                batch,
            )
        return x, batch


class ETFlowTaskFactory:
    """Factory instantiated by ``cli/train.py``.

    ``train_set`` is deliberately **not** declared: ET-Flow needs no dataset
    statistics at construction (no marginals, no valency table, no size
    histogram), so the declarative injection seam simply does not fire.
    """

    def __init__(  # noqa: PLR0913
        self,
        task_type: str = "diffusion_etflow",
        model: dict | None = None,
        flow: dict | None = None,
        atom_vocab: list | None = None,
        **kwargs: Any,
    ) -> None:
        self.task_type = task_type
        self.model_kwargs = _plain(model or {})
        self.flow_kwargs = _plain(flow or {})
        self.atom_vocab = list(atom_vocab) if atom_vocab else None
        self.kwargs = kwargs
        self.task: ETFlowTask | None = None

    def build(self) -> ETFlowTask:
        flow = dict(self.flow_kwargs)
        self.task = ETFlowTask(
            model_kwargs=self.model_kwargs,
            max_num_neighbors=flow.pop("max_num_neighbors", 32),
            atom_vocab=self.atom_vocab,
            task_type=self.task_type,
            **flow,
        )
        logger.info(
            "Built ET-Flow: %d layers x %d channels, so3_equivariant=%s, "
            "output_layer_norm=%s, %d params",
            self.model_kwargs.get("num_layers", 20),
            self.model_kwargs.get("hidden_channels", 160),
            self.model_kwargs.get("so3_equivariant", False),
            self.model_kwargs.get("output_layer_norm", False),
            sum(p.numel() for p in self.task.parameters()),
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


ModelTaskFactory = ETFlowTaskFactory
