"""
SSL3D — unified self-supervised pretraining task for 3D molecules.

Provides:
  - SSL3DObjective (abstract base)
  - CoordDenoiseObjective
  - MaskedAtomTypeObjective
  - PairwiseDistObjective
  - SSL3D (task orchestrator)

Architecture dispatch mirrors ProperyPrediction (regression.py) so that
backbone weights transfer cleanly between SSL pretraining and downstream tasks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from MolecularDiffusion import core
from MolecularDiffusion.modules.layers.common import SinusoidsEmbeddingNew

from MolecularDiffusion.modules.tasks.task import Task


# ─────────────────────────────────────────────────────────────────────────────
# Objectives — abstract base
# ─────────────────────────────────────────────────────────────────────────────

class SSL3DObjective(nn.Module):
    """Base class for a single SSL pretext objective.

    Subclasses must implement corrupt() and compute_loss().
    The weight attribute controls this objective's contribution to the total loss.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def corrupt(self, batch_work: dict, device) -> dict:
        """Perturb batch_work in-place and return aux info needed by compute_loss."""
        raise NotImplementedError

    def compute_loss(self, h, x_pred, batch_orig, batch_work, aux):
        """Compute loss from backbone outputs.

        Args:
            h: per-node hidden features (N, hidden_nf)
            x_pred: updated coords from backbone (N, 3) or None for invariant nets
            batch_orig: original uncorrupted batch
            batch_work: corrupted batch passed to backbone
            aux: dict returned by corrupt()

        Returns:
            (scalar loss tensor, metrics dict)
        """
        raise NotImplementedError

    def build_head(self, hidden_nf: int):
        """Called by SSL3D after the backbone is known. Override if needed."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate denoising
# ─────────────────────────────────────────────────────────────────────────────

@core.Registry.register("CoordDenoiseObjective")
class CoordDenoiseObjective(SSL3DObjective, core.Configurable):
    """Predict the Gaussian noise added to atom positions.

    For equivariant backbones that return updated coords (x_pred != None),
    we recover epsilon from the displacement: eps_hat = (x_noisy - x_pred) / sigma.
    For invariant backbones we use a learned per-node MLP head on h.
    """

    def __init__(
        self,
        weight: float = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        sigma_schedule: str = "uniform",
    ):
        super().__init__(weight=weight)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_schedule = sigma_schedule
        self.coord_head = None  # built lazily in build_head()

    def build_head(self, hidden_nf: int):
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3),
        )

    def _sample_sigma(self, batch_idx: torch.Tensor, device) -> torch.Tensor:
        """Sample one sigma per graph, broadcast to per-atom shape (N,1)."""
        n_graphs = batch_idx.max().item() + 1
        if self.sigma_schedule == "uniform":
            sigma_graph = (
                torch.rand(n_graphs, device=device)
                * (self.sigma_max - self.sigma_min)
                + self.sigma_min
            )
        elif self.sigma_schedule == "log_uniform":
            sigma_graph = torch.exp(
                torch.rand(n_graphs, device=device)
                * (math.log(self.sigma_max) - math.log(self.sigma_min))
                + math.log(self.sigma_min)
            )
        else:
            raise ValueError(f"Unknown sigma_schedule: {self.sigma_schedule}")
        return sigma_graph[batch_idx].unsqueeze(1)  # (N, 1)

    def corrupt(self, batch_work: dict, device) -> dict:
        graph = batch_work["graph"]
        pos_clean = graph.pos.clone()
        sigma = self._sample_sigma(graph.batch, device)
        eps = torch.randn_like(pos_clean)

        # COM-zero each molecule before adding noise
        com = scatter(pos_clean, graph.batch, dim=0, reduce="mean")
        pos_clean = pos_clean - com[graph.batch]

        graph.pos = pos_clean + sigma * eps
        return {"eps": eps, "sigma": sigma, "pos_clean": pos_clean}

    def compute_loss(self, h, x_pred, batch_orig, batch_work, aux):
        eps = aux["eps"]
        sigma = aux["sigma"]

        if x_pred is not None:
            # Equivariant backbone: recover epsilon from coordinate output
            eps_hat = (batch_work["graph"].pos - x_pred) / sigma.clamp(min=1e-8)
        else:
            # Invariant backbone: predict epsilon with MLP head
            assert self.coord_head is not None, "build_head() must be called before forward"
            eps_hat = self.coord_head(h)

        loss = F.mse_loss(eps_hat, eps)
        return loss, {"ssl/denoise_loss": loss.detach()}


# ─────────────────────────────────────────────────────────────────────────────
# Masked atom-type prediction
# ─────────────────────────────────────────────────────────────────────────────

@core.Registry.register("MaskedAtomTypeObjective")
class MaskedAtomTypeObjective(SSL3DObjective, core.Configurable):
    """Mask a fraction of atom one-hot features and predict the masked types.

    A learned [MASK] token replaces the masked rows in graph.x.
    The head predicts logits over atom_vocab_size categories.
    """

    def __init__(
        self,
        weight: float = 0.5,
        mask_rate: float = 0.15,
        atom_vocab_size: int = 5,
    ):
        super().__init__(weight=weight)
        self.mask_rate = mask_rate
        self.atom_vocab_size = atom_vocab_size
        # mask token appended as the last row; x has shape (N, atom_vocab_size + extra)
        # We only mask/restore the first atom_vocab_size columns.
        self.mask_token = nn.Parameter(torch.zeros(atom_vocab_size))
        self.type_head = None  # built in build_head()

    def build_head(self, hidden_nf: int):
        self.type_head = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, self.atom_vocab_size),
        )

    def corrupt(self, batch_work: dict, device) -> dict:
        graph = batch_work["graph"]
        N = graph.x.shape[0]
        mask = torch.rand(N, device=device) < self.mask_rate
        true_types = graph.x[:, : self.atom_vocab_size].clone()

        x_corrupted = graph.x.clone()
        x_corrupted[mask, : self.atom_vocab_size] = self.mask_token.to(device)
        graph.x = x_corrupted

        return {"mask": mask, "true_types": true_types}

    def compute_loss(self, h, x_pred, batch_orig, batch_work, aux):
        mask = aux["mask"]
        true_types = aux["true_types"]

        if mask.sum() == 0:
            zero = h.new_zeros(())
            return zero, {"ssl/mtype_loss": zero.detach()}

        assert self.type_head is not None, "build_head() must be called before forward"
        logits = self.type_head(h[mask])
        targets = true_types[mask].argmax(dim=-1)
        loss = F.cross_entropy(logits, targets)
        return loss, {"ssl/mtype_loss": loss.detach()}


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise distance prediction (optional, default weight 0)
# ─────────────────────────────────────────────────────────────────────────────

@core.Registry.register("PairwiseDistObjective")
class PairwiseDistObjective(SSL3DObjective, core.Configurable):
    """Predict clean pairwise distances from noisy node features.

    Samples k_pairs random edges per batch and regresses their clean distances.
    Default weight is 0 (disabled); enable by setting weight > 0 in config.
    """

    def __init__(
        self,
        weight: float = 0.0,
        k_pairs: int = 16,
        n_dist_basis: int = 16,
        dist_cutoff: float = 10.0,
    ):
        super().__init__(weight=weight)
        self.k_pairs = k_pairs
        self.n_dist_basis = n_dist_basis
        self.dist_cutoff = dist_cutoff
        self.dist_head = None  # built in build_head()

    def build_head(self, hidden_nf: int):
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_nf * 2 + self.n_dist_basis, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1),
        )

    def _gaussian_basis(self, d: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(0, self.dist_cutoff, self.n_dist_basis, device=d.device)
        width = (self.dist_cutoff / self.n_dist_basis) ** 2
        return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / width)

    def corrupt(self, batch_work: dict, device) -> dict:
        # No additional corruption; uses whatever positions are current in batch_work
        graph = batch_work["graph"]
        edge_index = graph.edge_index  # (2, E)
        E = edge_index.shape[1]
        k = min(self.k_pairs * (graph.batch.max().item() + 1), E)
        idx = torch.randperm(E, device=device)[:k]
        sampled_edges = edge_index[:, idx]
        return {"sampled_edges": sampled_edges}

    def compute_loss(self, h, x_pred, batch_orig, batch_work, aux):
        sampled_edges = aux["sampled_edges"]
        i, j = sampled_edges[0], sampled_edges[1]

        assert self.dist_head is not None, "build_head() must be called before forward"

        pos_clean = batch_orig["graph"].pos
        d_clean = (pos_clean[i] - pos_clean[j]).norm(dim=-1)

        pos_noisy = batch_work["graph"].pos
        d_noisy = (pos_noisy[i] - pos_noisy[j]).norm(dim=-1)
        d_emb = self._gaussian_basis(d_noisy)

        pair_feat = torch.cat([h[i], h[j], d_emb], dim=-1)
        d_pred = self.dist_head(pair_feat).squeeze(-1)

        loss = F.mse_loss(d_pred, d_clean)
        return loss, {"ssl/dist_loss": loss.detach()}


# ─────────────────────────────────────────────────────────────────────────────
# SSL3D task orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@core.Registry.register("SSL3D")
class SSL3D(Task, core.Configurable):
    """Self-supervised 3D molecular pretraining task.

    Args:
        model: backbone (EGNN, GraphTransformer, eSEN_Backbone, or
               EquiformerV2_dynamics).
        objectives: list of SSL3DObjective instances. Their corrupt() methods
                    are applied in order; losses are summed weighted by
                    obj.weight.
        include_charge: prepend atomic-number feature to node features.
        t_embedding: "sinusoidal" (default) — how sigma is embedded before
                     being appended as a time channel.
    """

    def __init__(
        self,
        model: nn.Module,
        objectives: list,
        include_charge: bool = True,
        t_embedding: str = "sinusoidal",
    ):
        super().__init__()
        self.model = model
        self.objectives = nn.ModuleList(objectives)
        self.include_charge = include_charge
        self.t_embedding = t_embedding

        # ── detect architecture for forward dispatch ──────────────────────────
        cls_name = model.__class__.__name__
        if cls_name in ("GraphTransformer", "GraphDiffTransformer"):
            self.architecture = "egt"
        elif cls_name == "EGNN":
            self.architecture = "egcn"
        elif cls_name == "EquiformerV2_dynamics":
            self.architecture = "equiformer"   # sigma passed as t, not embedded in h
        else:
            self.architecture = "egnn_extra"   # eSEN etc.

        # ── sigma → t embedding ───────────────────────────────────────────────
        # Equiformer handles time internally — no embedding needed in node feats.
        if self.architecture == "equiformer":
            self._sigma_emb = None
            self._t_dim = 0
        elif t_embedding == "sinusoidal":
            self._sigma_emb = SinusoidsEmbeddingNew()
            self._t_dim = self._sigma_emb.dim
        else:
            self._sigma_emb = None
            self._t_dim = 1

        # ── build heads once hidden_nf is known ──────────────────────────────
        hidden_nf = getattr(model, "hidden_nf", None)
        if hidden_nf is not None:
            self._build_heads(hidden_nf)

    # ── head initialisation ───────────────────────────────────────────────────

    def _build_heads(self, hidden_nf: int):
        for obj in self.objectives:
            obj.build_head(hidden_nf)
        self._heads_built = True

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def device(self):
        return next(self.parameters()).device

    def _embed_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (N, 1) → (N, t_dim)"""
        if self._sigma_emb is not None:
            return self._sigma_emb(sigma)
        return sigma

    def _get_sigma(self, batch_work: dict) -> torch.Tensor:
        """Collect the per-atom sigma stashed by CoordDenoiseObjective, or zeros."""
        for obj in self.objectives:
            # CoordDenoiseObjective stores sigma in the aux dict, but we need it
            # attached to the batch for the forward; grab it from the objectives'
            # aux stash if present.
            pass
        # Sigma is attached to batch_work["_ssl_sigma"] by forward() below.
        sigma = batch_work.get("_ssl_sigma", None)
        if sigma is None:
            N = batch_work["graph"].x.shape[0]
            sigma = torch.zeros(N, 1, device=self.device)
        return sigma

    # ── backbone forward dispatch ─────────────────────────────────────────────

    def _forward_backbone(self, batch_work: dict, sigma: torch.Tensor):
        """Run backbone on the corrupted batch.

        Returns:
            h (N, hidden_nf), x_pred (N, 3) or None
        """
        graph = batch_work["graph"]
        device = self.device
        x = graph.pos
        edge_index = graph.edge_index

        charges = graph.atomic_numbers if hasattr(graph, "atomic_numbers") else None
        h = graph.x

        if self.architecture == "equiformer":
            # Equiformer handles sigma as its time input; no embedding in node feats.
            if self.include_charge and charges is not None:
                h = torch.cat([h, charges.unsqueeze(-1).float()], dim=1)
            graph.x = h
            t = sigma.squeeze(1)
            mol_graph = {"graph": graph, "t": t, "context": None}
            out = self.model(mol_graph)   # (N, 3 + in_node_nf)
            x_pred = out[:, :3]
            h_out = out[:, 3:]
            return h_out, x_pred

        # All other architectures: embed sigma and append as extra node feature dim.
        t_feat = self._embed_sigma(sigma)  # (N, t_dim)
        if self.include_charge and charges is not None:
            h = torch.cat([h, charges.unsqueeze(-1).float()], dim=1)
        h = torch.cat([h, t_feat], dim=1)

        if self.architecture == "egcn":
            edges = [edge_index[0], edge_index[1]]
            h_out, x_out = self.model(h, x, edges, node_mask=None, edge_mask=None, use_embed=True)
            return h_out, x_out

        elif self.architecture == "egt":
            bs = graph.batch.max().item() + 1
            natoms = graph.natoms
            n_nodes = int(natoms.max().item())

            h_pad = self._pad(h, graph, h.shape[1])
            x_pad = self._pad(x, graph, 3)

            node_mask = torch.ones(bs, n_nodes, dtype=torch.bool, device=device)
            edge_mask = torch.ones(bs, n_nodes, n_nodes, 1, device=device)
            y = torch.ones(bs, 1, device=device)

            h_out, _, _, _ = self.model(h_pad, E=edge_mask, y=y, pos=x_pad, node_mask=node_mask)
            h_out = h_out.reshape(bs * n_nodes, -1)
            valid = self._valid_mask(graph)
            return h_out[valid], None

        else:
            # eSEN: model expects graph object with .x already set
            graph.x = h
            graph.num_atoms = graph.natoms
            graph.token_idx = torch.zeros(graph.x.shape[0], device=device, dtype=torch.long)
            output = self.model(graph)
            h_out = output["x"] if isinstance(output, dict) else output[0]
            return h_out, None

    # ── padding helpers (EGT path) ─────────────────────────────────────────────

    def _pad(self, array, graph, dim):
        bs = graph.batch.max().item() + 1
        natoms = graph.natoms
        n_nodes = int(natoms.max().item()) if hasattr(natoms, "max") else array.shape[0] // bs
        out = torch.zeros(bs, n_nodes, dim, device=array.device)
        cum = 0
        for i, na in enumerate(natoms):
            na = int(na)
            out[i, :na] = array[cum: cum + na]
            cum += na
        return out

    def _valid_mask(self, graph):
        bs = graph.batch.max().item() + 1
        natoms = graph.natoms
        n_nodes = int(natoms.max().item())
        mask = torch.zeros(bs * n_nodes, dtype=torch.bool, device=graph.batch.device)
        cum = 0
        for i, na in enumerate(natoms):
            na = int(na)
            for j in range(na):
                mask[i * n_nodes + j] = True
            cum += na
        return mask

    # ── main forward ──────────────────────────────────────────────────────────

    def forward(self, batch: dict):
        device = self.device

        # Build heads lazily (first forward pass after model is on device)
        if not getattr(self, "_heads_built", False):
            hidden_nf = getattr(self.model, "hidden_nf", None)
            if hidden_nf is None:
                raise RuntimeError(
                    "SSL3D: backbone has no .hidden_nf attribute. "
                    "Set model.hidden_nf = <dim> before calling forward()."
                )
            self._build_heads(hidden_nf)

        # ── 1. deep-copy to avoid mutating the original batch ─────────────────
        batch_orig = batch
        batch_work = {"graph": _shallow_clone_graph(batch["graph"])}

        # ── 2. run each objective's corruption in sequence ────────────────────
        aux_list = []
        sigma_per_atom = None
        for obj in self.objectives:
            aux = obj.corrupt(batch_work, device)
            aux_list.append(aux)
            # CoordDenoiseObjective exposes sigma for t-conditioning
            if "sigma" in aux and sigma_per_atom is None:
                sigma_per_atom = aux["sigma"]

        if sigma_per_atom is None:
            N = batch_work["graph"].x.shape[0]
            sigma_per_atom = torch.zeros(N, 1, device=device)

        batch_work["_ssl_sigma"] = sigma_per_atom

        # ── 3 & 4. backbone forward ───────────────────────────────────────────
        h, x_pred = self._forward_backbone(batch_work, sigma_per_atom)

        # ── 5. compute each objective's loss ──────────────────────────────────
        total_loss = torch.tensor(0.0, device=device)
        metrics = {}
        for obj, aux in zip(self.objectives, aux_list):
            loss_i, metrics_i = obj.compute_loss(h, x_pred, batch_orig, batch_work, aux)
            total_loss = total_loss + obj.weight * loss_i
            metrics.update(metrics_i)

        metrics["ssl/total_loss"] = total_loss.detach()
        return total_loss, metrics

    def preprocess(self, train_set=None, valid_set=None, test_set=None):
        pass

    def predict_and_target(self, batch):
        """Called by both Engine and EngineLightning during validation/test."""
        loss, _ = self.forward(batch)
        loss = loss.unsqueeze(0)
        return loss, torch.zeros_like(loss)

    def evaluate(self, pred, target):
        """Aggregate per-batch losses → metric dict for eval.py TASK_REGISTRY."""
        return {"ssl/total_loss": pred.mean()}

    def predict(self, batch, all_loss=None, metric=None):
        raise NotImplementedError("Use SSL3D.forward() for training.")

    def target(self, batch):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _shallow_clone_graph(graph):
    """Return a new object sharing the same tensors but allowing field replacement."""
    from torch_geometric.data import Data, Batch
    if isinstance(graph, (Data, Batch)):
        return graph.clone()
    # Fallback for plain-object batches used in EGCL/EGT
    new = type(graph).__new__(type(graph))
    new.__dict__.update(graph.__dict__)
    return new
