"""Checks that used to run via ``python -m MolecularDiffusion.modules.models.ditmc.graph_features``; moved here unchanged."""

from rdkit import Chem
from scipy.sparse.csgraph import floyd_warshall
import numpy as np
from MolecularDiffusion.modules.models.ditmc.graph_features import (
    UNREACHABLE_HOPS,
    all_pairs_edges,
    get_node_attr_from_mol,
    laplacian_eigen,
    node_attr_dim,
    shortest_hops_from_adjacency,
)


def _self_check() -> None:  # pragma: no cover - run via ``python -m``
    """Smallest thing that fails if the featurization drifts."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    attr = get_node_attr_from_mol(mol, "qm9")
    assert attr.shape[1] == node_attr_dim("qm9") == 64, attr.shape
    # 8 blocks always set exactly one bit; symbol, aromatic and ring bits vary.
    assert attr.sum(axis=1).min() >= 8, attr.sum(axis=1).min()

    # hops ordering must match all_pairs_edges
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    hops = shortest_hops_from_adjacency(adj)
    rec, sen = all_pairs_edges(3)
    dist = floyd_warshall(adj.astype(float), directed=False, unweighted=True)
    for k in range(len(hops)):
        assert hops[k] == dist[rec[k], sen[k]], (k, hops[k])

    # disconnected pair -> sentinel
    adj2 = np.zeros((2, 2), dtype=np.int64)
    assert (shortest_hops_from_adjacency(adj2) == UNREACHABLE_HOPS).all()

    # harmonic prior: one zero mode for a connected graph
    d, p = laplacian_eigen(adj, num_components=1)
    assert d[0] == 0.0 and (d[1:] > 0).all(), d
    assert np.allclose(p @ p.T, np.eye(3), atol=1e-5)
    print("ditmc.graph_features self-check OK")


def test_models_ditmc_graph_features_self_check():
    _self_check()
