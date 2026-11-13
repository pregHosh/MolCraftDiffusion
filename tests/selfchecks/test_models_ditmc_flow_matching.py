"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.ditmc.flow_matching``; moved here unchanged."""

from MolecularDiffusion.modules.layers.e3x import random_rotation
from MolecularDiffusion.modules.models.ditmc.graphs import LatentGraph
import torch
from MolecularDiffusion.modules.models.ditmc.flow_matching import (
    FlowMatching,
    aggregate_node_error,
    kabsch_align,
)


def _self_check() -> None:  # pragma: no cover
    """Pins the loss reduction, the drift identity and the Kabsch alignment."""
    torch.manual_seed(0)
    segments = torch.tensor([0, 0, 0, 1, 1])
    num_graphs = 2

    # Loss reduction: a known constant error must come out as that constant.
    err = torch.full((5, 3), 4.0)
    assert abs(aggregate_node_error(err, segments, num_graphs).item() - 4.0) < 1e-6
    # A per-graph weight multiplies before the mean over graphs.
    w = torch.tensor([2.0, 0.0])
    assert (
        abs(aggregate_node_error(err, segments, num_graphs, graph_weight=w).item() - 4.0)
        < 1e-6
    )

    # Kabsch: a rotated + translated copy aligns back to ~0 RMSD.
    q = torch.randn(5, 3)
    rot = random_rotation(2)
    p = torch.einsum("bij,bj->bi", rot.index_select(0, segments), q) + torch.tensor(
        [1.0, -2.0, 3.0]
    )
    aligned = kabsch_align(p, q, segments, num_graphs)
    assert (aligned - q).abs().max() < 1e-4, (aligned - q).abs().max()
    assert (torch.linalg.det(rot) - 1).abs().max() < 1e-5

    # regress_x1: vt == (x1_hat - x_tau) / (1 - tau)
    fm = FlowMatching(prior=None)
    g = LatentGraph(
        atomic_numbers=torch.ones(5, dtype=torch.long),
        node_attr=torch.zeros(5, 4),
        positions=torch.randn(5, 3),
        senders=torch.zeros(0, dtype=torch.long),
        receivers=torch.zeros(0, dtype=torch.long),
        shortest_hops=torch.zeros(0, dtype=torch.long),
        batch_segments=segments,
        num_graphs=num_graphs,
        cond_scaling_nodes=torch.ones(5),
        cond_scaling_edges=torch.zeros(0),
    )
    tau = torch.full((5,), 0.25)
    out = torch.randn(5, 3)
    x1_pred, vt = fm.clean_prediction(out, g, tau)
    assert torch.allclose(vt, (x1_pred - g.positions) / 0.75, atol=1e-5)
    print("ditmc.flow_matching self-check OK")


def test_models_ditmc_flow_matching_self_check():
    _self_check()
