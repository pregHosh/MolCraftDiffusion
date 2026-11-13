"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.ditmc.priors``; moved here unchanged."""

from MolecularDiffusion.modules.models.ditmc.graphs import PriorGraph
import torch
from MolecularDiffusion.modules.models.ditmc.priors import (
    HarmonicPrior,
)


def _self_check() -> None:  # pragma: no cover
    """Sample covariance of the harmonic prior ~ pinv of the Laplacian."""
    import numpy as np

    from MolecularDiffusion.modules.models.ditmc.graph_features import laplacian_eigen

    adj = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int64
    )
    d, p = laplacian_eigen(adj, num_components=1)
    n = 4
    i, j = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    graph = PriorGraph(
        node_attr=torch.as_tensor(d),
        senders=j.reshape(-1),
        receivers=i.reshape(-1),
        edge_attr=torch.as_tensor(p).reshape(-1),
    )
    g = torch.Generator().manual_seed(0)
    samples = torch.stack(
        [HarmonicPrior().sample((n, 3), graph, generator=g) for _ in range(40000)]
    )
    cov = torch.einsum("sid,sjd->ij", samples, samples) / (samples.shape[0] * 3)
    lap = np.diag(adj.sum(1)) - adj
    expect = torch.as_tensor(np.linalg.pinv(lap.astype(np.float64)), dtype=torch.float32)
    assert torch.allclose(cov, expect, atol=0.05), (cov - expect).abs().max()
    # The lambda=0 mode is zeroed, which removes the centre of mass.
    assert samples.mean(dim=1).abs().max() < 1e-4, samples.mean(dim=1).abs().max()
    print("ditmc.priors self-check OK")


def test_models_ditmc_priors_self_check():
    _self_check()
