"""Flow matching with a constant-noise interpolant. Port of
``dit_mc/generative_process/flow_matching.py`` (+ the Kabsch / augmentation /
loss-reduction helpers from ``dit_mc/training/utils.py``).

The interpolant is

.. math::  x_\\tau = (1-\\tau)\\,x_0 + \\tau\\,x_1 + \\sigma z,\\qquad \\sigma = 0.5

with :math:`x_0` from the harmonic prior, **Kabsch-aligned to** :math:`x_1`
before interpolating, and every quantity centre-of-mass centred. With the
default ``regress_x1_bool=True`` the network regresses the clean :math:`x_1` and
the drift is recovered as :math:`v_t = (\\hat x_1 - x_\\tau)/(1-\\tau)`.

Three reductions that are easy to get wrong and are pinned here:

* **The loss is a mean over x/y/z and a mean over atoms**, then a per-graph
  weight, then a sum divided by the number of graphs. A sum over xyz would
  scale it by 3N (``training/utils.py:91-125``).
* The per-graph weight is :math:`1/(1-\\min(\\tau, 0.9))^2`, using the
  **per-graph** tau, not the per-node one.
* Classifier-free guidance draws **one** Bernoulli per batch, not per molecule
  (``flow_matching.py:271``).

Rotation augmentation is **on by default** upstream
(``globals.augmentation_bool: True``). Without it ``dit_ape``/``dit_rpe``, which
are not equivariant by construction, train on a different distribution.
"""

from __future__ import annotations

import torch

from MolecularDiffusion.modules.layers.e3x import random_rotation, segment_mean

from .graphs import LatentGraph


def center_data(
    x: torch.Tensor, batch_segments: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Subtract the per-molecule centre of mass."""
    mean = segment_mean(x, batch_segments, num_graphs)
    return x - mean.index_select(0, batch_segments)


def kabsch_align(
    p: torch.Tensor,
    q: torch.Tensor,
    batch_segments: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Rigidly align ``p`` onto ``q`` per molecule (proper rotations only)."""
    p_mean = segment_mean(p, batch_segments, num_graphs)
    q_mean = segment_mean(q, batch_segments, num_graphs)
    p_c = p - p_mean.index_select(0, batch_segments)
    q_c = q - q_mean.index_select(0, batch_segments)

    h_per_node = torch.einsum("bi,bj->bij", p_c, q_c)
    h = h_per_node.new_zeros((num_graphs, 3, 3)).index_add(
        0, batch_segments, h_per_node
    )

    u, _s, vh = torch.linalg.svd(h)
    d = torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2))
    flip = (d < 0.0).reshape(-1, 1, 1)
    flip_matrix = torch.diag(
        torch.tensor([1.0, 1.0, -1.0], dtype=h.dtype, device=h.device)
    )
    vh = torch.where(flip, flip_matrix @ vh, vh)

    rot = vh.transpose(-1, -2) @ u.transpose(-1, -2)  # (G, 3, 3)
    t = q_mean - torch.einsum("gij,gj->gi", rot, p_mean)
    out = torch.einsum("bi,bji->bj", p, rot.index_select(0, batch_segments))
    return out + t.index_select(0, batch_segments)


def rotation_augmentation(
    tensors: dict[str, torch.Tensor],
    batch_segments: torch.Tensor,
    num_graphs: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One Haar-uniform rotation per molecule, applied to every ``(N, 3)`` field."""
    ref = next(iter(tensors.values()))
    rot = random_rotation(
        num_graphs, generator=generator, device=ref.device, dtype=ref.dtype
    )
    per_node = rot.index_select(0, batch_segments)
    rotated = {
        k: torch.einsum("bij,bj->bi", per_node, v) for k, v in tensors.items()
    }
    return rotated, rot


def aggregate_node_error(
    node_error: torch.Tensor,
    batch_segments: torch.Tensor,
    num_graphs: int,
    graph_weight=1.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Mean over trailing dims, segment-mean per graph, weight, sum / #graphs."""
    node_mean_squared = node_error.reshape(len(node_error), -1).mean(dim=-1)
    per_graph_mse = segment_mean(node_mean_squared, batch_segments, num_graphs)
    per_graph_mse = graph_weight * per_graph_mse
    return scale * per_graph_mse.sum() / max(num_graphs, 1)


class FlowMatching:
    """The generative process. Holds no parameters -- the network is passed in."""

    def __init__(  # noqa: PLR0913
        self,
        prior,
        sigma: float = 0.5,
        align_bool: bool = True,
        conditioning_bool: bool = True,
        regress_x1_bool: bool = True,
        weighted_loss_bool: bool = True,
        mixture_tau_bool: bool = False,
        free_guidance_bool: bool = False,
        free_guidance_prob: float = 0.1,
        self_conditioning_bool: bool = False,
        self_conditioning_prob: float = 0.5,
    ) -> None:
        self.prior = prior
        self.sigma = sigma
        self.align_bool = align_bool
        self.conditioning_bool = conditioning_bool
        self.regress_x1_bool = regress_x1_bool
        self.weighted_loss_bool = weighted_loss_bool
        self.mixture_tau_bool = mixture_tau_bool
        self.free_guidance_bool = free_guidance_bool
        self.free_guidance_prob = free_guidance_prob
        self.self_conditioning_bool = self_conditioning_bool
        self.self_conditioning_prob = self_conditioning_prob

    # -- helpers -----------------------------------------------------------

    def _forward(self, net, time_latent, graph_latent, graph_cond):
        out = net(
            time_latent=time_latent,
            graph_latent=graph_latent,
            graph_cond=graph_cond if self.conditioning_bool else None,
        )
        # 'drift_and_noise' returns a pair; the drift head is what is used.
        return out[0] if isinstance(out, tuple) else out

    def clean_prediction(self, nn_out, graph_latent: LatentGraph, time_latent):
        """``(x1_pred, vt)`` from the raw network output."""
        nn_out = center_data(
            nn_out, graph_latent.batch_segments, graph_latent.num_graphs
        )
        if self.regress_x1_bool:
            x1_pred = nn_out
            vt = (nn_out - graph_latent.positions) / (1.0 - time_latent).unsqueeze(-1)
        else:
            x1_pred = graph_latent.positions + (1.0 - time_latent).unsqueeze(
                -1
            ) * nn_out
            vt = nn_out
        return x1_pred, vt

    @staticmethod
    def _sample_mixture_beta_uniform(
        shape, device, generator, p1=1.9, p2=1.0, uniform_prob=0.02
    ):
        beta = torch.distributions.Beta(p1, p2).sample(shape).to(device)
        uniform = torch.rand(shape, generator=generator, device=device)
        u = torch.rand(shape, generator=generator, device=device)
        return torch.where(u < uniform_prob, uniform, beta)

    # -- training ----------------------------------------------------------

    def loss(  # noqa: PLR0913, PLR0912
        self,
        net,
        graph_latent: LatentGraph,
        graph_prior,
        graph_cond,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """One training step. ``graph_latent.x1`` must be set."""
        if graph_latent.x1 is None:
            msg = "FlowMatching.loss needs graph_latent.x1 (the target coordinates)"
            raise ValueError(msg)

        segments = graph_latent.batch_segments
        num_graphs = graph_latent.num_graphs
        device = graph_latent.x1.device
        dtype = graph_latent.x1.dtype

        x1 = center_data(graph_latent.x1, segments, num_graphs)
        x0 = self.prior.sample(
            x1.shape,
            graph_prior,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        x0 = center_data(x0, segments, num_graphs)
        if self.align_bool:
            x0 = kabsch_align(x0, x1, segments, num_graphs)
            x0 = center_data(x0, segments, num_graphs)

        if self.mixture_tau_bool:
            tau_per_graph = self._sample_mixture_beta_uniform(
                (num_graphs,), device, generator
            )
        else:
            tau_per_graph = torch.rand(
                num_graphs, generator=generator, device=device, dtype=dtype
            )
        tau = tau_per_graph.index_select(0, segments)

        z = torch.randn(x1.shape, generator=generator, device=device, dtype=dtype)
        z = center_data(z, segments, num_graphs)

        alpha = (1.0 - tau).unsqueeze(-1)
        beta = tau.unsqueeze(-1)
        if self.regress_x1_bool:
            gamma = self.sigma * torch.ones_like(beta)
        else:
            gamma = self.sigma * torch.sqrt(beta * (1 - beta))
        xtau = center_data(alpha * x0 + beta * x1 + gamma * z, segments, num_graphs)

        graph_latent = graph_latent.replace(
            positions=xtau,
            cond_scaling_nodes=torch.ones_like(tau),
            cond_scaling_edges=torch.ones(
                graph_latent.senders.shape[0], device=device, dtype=dtype
            ),
        )

        if self.free_guidance_bool:
            # ONE Bernoulli per batch, not per molecule -- upstream behaviour.
            drop = (
                torch.rand(1, generator=generator, device=device).item()
                < self.free_guidance_prob
            )
            if drop:
                graph_latent = graph_latent.replace(
                    cond_scaling_nodes=torch.zeros_like(
                        graph_latent.cond_scaling_nodes
                    ),
                    cond_scaling_edges=torch.zeros_like(
                        graph_latent.cond_scaling_edges
                    ),
                )

        if self.self_conditioning_bool:
            graph_latent = graph_latent.replace(self_cond=torch.zeros_like(x1))
            use_sc = (
                torch.rand(1, generator=generator, device=device).item()
                < self.self_conditioning_prob
            )
            if use_sc:
                with torch.no_grad():
                    nn_out = self._forward(net, tau, graph_latent, graph_cond)
                    sc, _ = self.clean_prediction(nn_out, graph_latent, tau)
                graph_latent = graph_latent.replace(self_cond=sc.detach())
            else:
                graph_latent = graph_latent.replace(self_cond=torch.zeros_like(x1))

        nn_out = self._forward(net, tau, graph_latent, graph_cond)
        x1_pred, vt = self.clean_prediction(nn_out, graph_latent, tau)

        if self.regress_x1_bool:
            ut_hat, ut = x1_pred, x1
        else:
            i_grad = x1 - x0
            gamma_grad = (
                0.5 * self.sigma * (1.0 - 2 * tau) / torch.sqrt(tau * (1.0 - tau))
            )
            ut_hat = vt
            ut = i_grad + gamma_grad.unsqueeze(-1) * z

        per_node_error = (ut_hat - ut) ** 2
        graph_weight = 1.0
        if self.weighted_loss_bool:
            graph_weight = 1.0 / ((1.0 - torch.clamp(tau_per_graph, max=0.9)) ** 2)

        loss = aggregate_node_error(
            per_node_error, segments, num_graphs, graph_weight=graph_weight
        )
        return loss, {"loss": loss.detach(), "tau": tau_per_graph.mean().detach()}

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(  # noqa: PLR0913, PLR0912
        self,
        net,
        graph_latent: LatentGraph,
        graph_prior,
        graph_cond,
        *,
        num_steps: int = 50,
        free_guidance_scale: float = 1.0,
        logarithmic_time_bool: bool = False,
        return_trajectory: bool = False,
        generator: torch.Generator | None = None,
    ):
        """Explicit Euler integration of ``v_t`` from ``tau=0`` to ``tau=1``."""
        segments = graph_latent.batch_segments
        num_graphs = graph_latent.num_graphs
        num_nodes = graph_latent.num_nodes
        num_edges = graph_latent.senders.shape[0]
        device = graph_latent.positions.device
        dtype = graph_latent.positions.dtype

        xtau = self.prior.sample(
            (num_nodes, 3),
            graph_prior,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        xtau = center_data(xtau, segments, num_graphs)

        ones_n = torch.ones(num_nodes, device=device, dtype=dtype)
        graph_latent = graph_latent.replace(
            positions=xtau,
            cond_scaling_nodes=ones_n,
            cond_scaling_edges=torch.ones(num_edges, device=device, dtype=dtype),
        )
        if self.self_conditioning_bool:
            graph_latent = graph_latent.replace(self_cond=torch.zeros_like(xtau))

        if logarithmic_time_bool:
            base = 10.0
            taus = torch.logspace(
                0.0, 1.0, num_steps + 1, base=base, device=device, dtype=dtype
            )
            taus = (taus - 1) / (base - 1)
            taus = torch.flip(1 - taus, dims=(0,))
        else:
            taus = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)

        trajectory = [xtau]
        x1_pred = None
        for n in range(num_steps):
            tau = taus[n]
            dtau = taus[n + 1] - taus[n]
            time_latent = ones_n * tau

            if n > 0 and self.self_conditioning_bool:
                graph_latent = graph_latent.replace(self_cond=x1_pred)

            nn_out = self._forward(net, time_latent, graph_latent, graph_cond)
            x1_pred, vt = self.clean_prediction(nn_out, graph_latent, time_latent)

            if self.free_guidance_bool:
                uncond = graph_latent.replace(
                    cond_scaling_nodes=torch.zeros_like(ones_n),
                    cond_scaling_edges=torch.zeros(
                        num_edges, device=device, dtype=dtype
                    ),
                )
                nn_out_u = self._forward(net, time_latent, uncond, graph_cond)
                x1_u, vt_u = self.clean_prediction(nn_out_u, uncond, time_latent)
                x1_pred = x1_u + free_guidance_scale * (x1_pred - x1_u)
                vt = vt_u + free_guidance_scale * (vt - vt_u)

            graph_latent = graph_latent.replace(
                positions=graph_latent.positions + dtau * vt
            )
            trajectory.append(graph_latent.positions)

        if return_trajectory:
            return graph_latent, trajectory
        return graph_latent
