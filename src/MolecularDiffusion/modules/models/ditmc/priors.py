"""Priors for the flow-matching interpolant. Port of ``generative_process/priors.py``.

The harmonic prior is the default and the interesting one: it draws
:math:`x_0 = P\\,\\mathrm{diag}(1/\\sqrt\\lambda)\\,z` where :math:`P, \\lambda` are the
eigenvectors/values of the **bond-graph Laplacian**, so the prior sample is
already a plausible-looking chain rather than a Gaussian blob.

Upstream evaluates this as a segment sum over a complete per-molecule index grid
(``prior_senders = j``, ``prior_receivers = i``, ``edge_attr = P.flatten()``):

    sample[i] = sum_j P[i][j] * D[j] * z[j]

which is exactly the matrix product above. The sparse form is kept here so the
batching is identical to upstream's, including the ``nan_to_num`` that zeroes
the :math:`\\lambda = 0` modes -- **that** is what removes the centre of mass.
"""

from __future__ import annotations

import torch

from .graphs import PriorGraph


class GaussianPrior:
    """``mu + sigma * N(0, 1)``. Ignores the graph entirely."""

    name = "GaussianPrior"

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu = mu
        self.sigma = sigma

    def sample(
        self,
        shape,
        graph_prior: PriorGraph | None = None,  # noqa: ARG002
        *,
        device=None,
        dtype=torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return self.mu + self.sigma * noise


class HarmonicPrior:
    """Gaussian shaped by the bond-graph Laplacian's pseudo-inverse."""

    name = "HarmonicPrior"

    def sample(
        self,
        shape,
        graph_prior: PriorGraph,
        *,
        device=None,
        dtype=torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if graph_prior is None:
            msg = "HarmonicPrior.sample requires a PriorGraph"
            raise ValueError(msg)
        num_nodes = shape[0]
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        scaled = graph_prior.node_attr.to(dtype).unsqueeze(-1) * noise
        scaled = torch.nan_to_num(scaled)
        messages = graph_prior.edge_attr.to(dtype).unsqueeze(-1) * scaled.index_select(
            0, graph_prior.senders
        )
        out = messages.new_zeros((num_nodes, *shape[1:]))
        return out.index_add(0, graph_prior.receivers, messages)


PRIORS = {"harmonic": HarmonicPrior, "gaussian": GaussianPrior}


def build_prior(name: str, **kwargs):
    if name not in PRIORS:
        msg = f"prior must be one of {sorted(PRIORS)}, received {name!r}"
        raise ValueError(msg)
    return PRIORS[name](**kwargs)
