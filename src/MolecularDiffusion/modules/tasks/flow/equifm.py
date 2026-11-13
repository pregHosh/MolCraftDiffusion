"""EquiFM task -- equivariant flow matching with hybrid probability transport.

Paper: arXiv:2312.07168 (NeurIPS 2023). Target repo: github.com/AlgoMole/MolFM.

**The training objective here is a reconstruction of the paper's Algorithm 1,
not the authors' released objective.** MolFM's release is sampling-only: it
ships no loss, no EOT solver, and no data pipeline. ``args.pickle`` shows the
released QM9 weights were additionally trained with ``angle_penalty=True``,
``cat_loss='l2_masked_mean'`` and ``ode_regularization=0.001`` -- none of which
appear anywhere in the paper. A model trained with this task will therefore not
reproduce the paper's Table 1 numbers, and the gap must not be reported as a
reproduction failure of the paper. The *sampler* and the converted released
checkpoint are faithful ports and are the real test of this integration.

Scope (per the approved INTEGRATION_PLAN.md): unconditional QM9 generation only.
No property conditioning / CFG (the release has ``context_node_nf=0`` and no
conditional weights), no gradient guidance, no inpainting, no trajectory frames,
no ``sample_chain``.
"""

from collections import Counter
from typing import Dict, List, Optional

import torch
from MolecularDiffusion.device import get_device
import torch.nn as nn

from MolecularDiffusion.modules.models.equifm.cnflows import Cnflows
from MolecularDiffusion.modules.models.geoldm.networks import EGNN_dynamics_QM9
from MolecularDiffusion.utils import remove_mean_with_mask

# Histogram-backed size sampler already implemented for TABASCO and reused by
# FlowMol; do not re-port MolFM's own DistributionNodes.
from MolecularDiffusion.modules.tasks.flow.tabasco import (
    TabascoNodeDistribution,
)


class EquiFMTaskFactory:
    """Factory matching ``cli/train.py``'s ``task_factory.build()`` pattern.

    ``train_set`` is declared as a named parameter so the declarative seam in
    ``cli/train.py`` injects the training dataset (docs/adding_new_models.md
    §2.5); it is used only to build the atom-count histogram that backs
    ``node_dist_model``.
    """

    def __init__(
        self,
        task_type: str = "diffusion_equifm",
        n_dims: int = 3,
        include_charges: bool = True,
        normalize_factors=(1.0, 4.0, 10.0),
        sigma_min: float = 1e-4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        discrete_path: str = "HB_path",
        default_n_timesteps: int = 250,
        use_eot: bool = True,
        eot_max_iters: int = 20,
        dynamics: Optional[dict] = None,
        dataset_stats: Optional[dict] = None,
        atom_vocab: Optional[List] = None,
        train_set: Optional[torch.utils.data.Dataset] = None,
        **kwargs,
    ):
        self.task_type = task_type
        self.n_dims = n_dims
        self.include_charges = include_charges
        self.normalize_factors = tuple(normalize_factors)
        self.sigma_min = sigma_min
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.discrete_path = discrete_path
        self.default_n_timesteps = default_n_timesteps
        self.use_eot = use_eot
        self.eot_max_iters = eot_max_iters
        self.dynamics = dict(dynamics) if dynamics else {}
        self.dataset_stats = dict(dataset_stats) if dataset_stats else {}
        self.atom_vocab = atom_vocab or kwargs.get("atom_vocab")
        self.train_set = train_set
        self.kwargs = kwargs

    def compute_dataset_stats(self, dataset) -> None:
        """Atom-count histogram from the training set (same shape as FlowMol's)."""
        if hasattr(dataset, "n_atoms"):
            num_atoms_list = [int(n) for n in dataset.n_atoms]
        else:
            num_atoms_list = []
            for i in range(len(dataset)):
                item = dataset[i]
                if "natoms" in item:
                    n = item["natoms"]
                    num_atoms_list.append(int(n.item()) if torch.is_tensor(n) else int(n))
                elif "node_mask" in item:
                    num_atoms_list.append(int(item["node_mask"].sum().item()))
        histogram = {int(k): int(v) for k, v in Counter(num_atoms_list).items()}
        self.dataset_stats["num_atoms_histogram"] = histogram
        print(
            f"[equifm] atom-count histogram: {len(histogram)} unique sizes "
            f"from {len(num_atoms_list)} molecules."
        )

    def build(self) -> "EquiFMTask":
        if not self.atom_vocab:
            raise ValueError(
                "EquiFMTaskFactory requires atom_vocab (injected from "
                "data.atom_vocab by cli/train.py) to size the atom-type channels."
            )
        if not self.dataset_stats.get("num_atoms_histogram"):
            if self.train_set is not None:
                self.compute_dataset_stats(self.train_set)
            else:
                print(
                    "[equifm] WARNING: no train_set and no histogram; molecule-size "
                    "sampling will fall back to a uniform 5-29 range."
                )

        # in_node_nf = |atom types| + charge channel; the dynamics network gets
        # one extra input channel for the time conditioning.
        in_node_nf = len(self.atom_vocab) + int(self.include_charges)
        dyn_cfg = {
            "n_layers": 9,
            "hidden_nf": 256,
            "attention": True,
            "tanh": True,
            "norm_constant": 1,
            "inv_sublayers": 1,
            "sin_embedding": False,
            "normalization_factor": 1,
            "aggregation_method": "sum",
        }
        dyn_cfg.update(self.dynamics)
        dynamics = EGNN_dynamics_QM9(
            in_node_nf=in_node_nf + 1,
            context_node_nf=0,
            n_dims=self.n_dims,
            act_fn=torch.nn.SiLU(),
            condition_time=True,
            mode="egnn_dynamics",
            **dyn_cfg,
        )

        cnflows = Cnflows(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=self.n_dims,
            include_charges=self.include_charges,
            norm_values=self.normalize_factors,
            norm_biases=(None, 0.0, 0.0),
            discrete_path=self.discrete_path,
            sigma_min=self.sigma_min,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            use_eot=self.use_eot,
            eot_max_iters=self.eot_max_iters,
        )

        self.task = EquiFMTask(
            cnflows=cnflows,
            default_n_timesteps=self.default_n_timesteps,
            dataset_stats=self.dataset_stats,
            atom_vocab=list(self.atom_vocab),
        )
        self.task.task_type = self.task_type
        return self.task


class EquiFMTask(nn.Module):
    """Duck-typed ``Task`` (docs/adding_new_models.md §2.1) around ``Cnflows``."""

    def __init__(
        self,
        cnflows: Cnflows,
        default_n_timesteps: int,
        dataset_stats: dict,
        atom_vocab: List,
    ):
        super().__init__()
        self.cnflows = cnflows
        self.atom_vocab = atom_vocab
        self.task_type = "diffusion_equifm"
        # `T` is the number of fixed ODE steps. Named `T` because
        # cli/generate.py:355-365 overrides `task.model.T` from the generate
        # config's `total_step`, which is exactly the knob we want it to control.
        self.T = default_n_timesteps
        self.prop_dist_model = None
        self._dataset_stats = dataset_stats
        self._node_dist_model = None

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _place_on_accelerator(self) -> None:
        """Move the task to the GPU for generation.

        Two contracts exist for device placement. A task with no ``device``
        attribute is moved by ``recursive_module_to_device``, which *assigns*
        ``module.device`` (``utils/torch.py:33``); a task that defines ``device``
        as a read-only property is skipped by the ``not hasattr(task, "device")``
        guard (``cli/generate.py:627``, ``core/engine.py:180``) and must place
        itself. This task takes the second contract, so this is its half of the
        bargain -- without it ``cli/generate.py`` samples on the CPU, since
        ``load_model`` reads the checkpoint with ``map_location="cpu"`` and
        nothing on the ``GenerativeFactory`` path ever moves it. Same approach as
        ``diffusion_diffsbdd.py:611``. Training is unaffected: Lightning moves
        the module itself.
        """
        if self.device.type == "cpu" and torch.cuda.is_available():
            self.to("cuda")
        elif self.device.type == "cpu" and get_device().type == "mps":
            self.to(torch.device("mps"))

    def _sync_dynamics_device(self) -> None:
        """``EGNN_dynamics_QM9`` caches its edge index on ``self.device``, an
        attribute set only from its ctor default ``"cpu"``.
        ``core/engine.py:180-181`` normally patches this via
        ``recursive_module_to_device``, but only ``if not hasattr(task, "device")``
        -- and this task defines a ``device`` property, which suppresses that.
        So keep it in sync ourselves before every network call.
        """
        self.cnflows.dynamics.device = self.device

    @staticmethod
    def _unpack(batch: Dict[str, torch.Tensor]):
        node_mask = batch["node_mask"].unsqueeze(2)
        x = batch["coords"]
        h_cat = batch["node_feature"]
        h_int = batch["charges"].unsqueeze(2)
        bsz, n_nodes, _ = x.shape
        edge_mask = batch["edge_mask"].view(bsz, n_nodes * n_nodes)
        return x, h_cat, h_int, node_mask, edge_mask

    # ------------------------------------------------------------------ #
    # training / evaluation contract                                     #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]):
        self._sync_dynamics_device()
        x, h_cat, h_int, node_mask, edge_mask = self._unpack(batch)
        x = remove_mean_with_mask(x, node_mask)
        return self.cnflows.compute_loss(x, h_cat, h_int, node_mask, edge_mask)

    def predict_and_target(self, batch: Dict[str, torch.Tensor]):
        loss, _ = self.forward(batch)
        if loss.dim() == 0:
            loss = loss.unsqueeze(0)
        return loss.detach(), torch.zeros_like(loss)

    def evaluate(self, pred: torch.Tensor, target: torch.Tensor):
        return {"val_loss": pred.mean()}

    # ------------------------------------------------------------------ #
    # generation contract                                                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        batch_size: Optional[int] = None,
        nodesxsample: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        mode=None,
        n_frames: int = 0,
        **kwargs,
    ):
        """Unconditional sampling. Returns ``(one_hot, charges, coords, node_mask)``
        following the EDM/GeoLDM convention (``charges`` are true nuclear
        charges), which keeps ``runmodes/generate/tasks_generate.py`` on the
        ``save_xyz_file`` branch. ``mode`` and ``n_frames`` are accepted and
        ignored (trajectories are out of scope)."""
        del mode, n_frames, kwargs
        self._place_on_accelerator()
        self._sync_dynamics_device()

        if nodesxsample is None:
            if batch is not None:
                nodesxsample = batch["natoms"].long()
            else:
                if batch_size is None:
                    raise ValueError(
                        "EquiFMTask.sample() needs nodesxsample, batch, or batch_size."
                    )
                nodesxsample = self.node_dist_model.sample(batch_size)
        nodesxsample = nodesxsample.long()

        bsz = nodesxsample.size(0)
        n_nodes = int(nodesxsample.max().item())
        node_mask = torch.zeros(bsz, n_nodes)
        for i in range(bsz):
            node_mask[i, : nodesxsample[i]] = 1
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        edge_mask *= ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask.view(bsz * n_nodes * n_nodes, 1).to(self.device)
        node_mask = node_mask.unsqueeze(2).to(self.device)

        x, one_hot, charges = self.cnflows.sample(
            n_samples=bsz,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=None,
            num_steps=int(num_steps or self.T),
        )
        return one_hot, charges, x, node_mask

    @property
    def model(self):
        return self

    @property
    def norm_values(self):
        return self.cnflows.norm_values

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def node_dist_model(self):
        if self._node_dist_model is None:
            self._node_dist_model = TabascoNodeDistribution(self._dataset_stats)
        return self._node_dist_model

    @node_dist_model.setter
    def node_dist_model(self, value):
        # cli/generate.py restores a pickled sampler from the checkpoint sidecar.
        self._node_dist_model = value

    @property
    def n_node_dist(self):
        return self.node_dist_model.n_node_dist


# `cli/train.py` / `cli/generate.py` instantiate `_target_` then call `.build()`.
ModelTaskFactory = EquiFMTaskFactory
