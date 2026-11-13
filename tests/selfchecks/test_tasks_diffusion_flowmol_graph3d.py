"""Checks that used to run via ``python -m MolecularDiffusion.modules.tasks.diffusion_flowmol_graph3d``; moved here unchanged."""

from MolecularDiffusion.modules.models.flowmol_graph3d import get_upper_edge_mask
import torch
from MolecularDiffusion.modules.tasks.diffusion_flowmol_graph3d import (
    FlowMolGraph3DTaskFactory,
)


def _self_check() -> None:  # pragma: no cover - run via `python -m`
    """Smallest runnable check of the two things most likely to break silently.

    1. the three atom-type widths (10 data / 11 heads / 12 ``a`` embedding);
    2. the bond round-trip through the adapter: stored upper-triangular bonds ->
       dense adjacency -> mirrored one-hot -> back, with the upper/lower halves
       agreeing and ``build_edge_idxs``'s ordering respected.
    """
    from torch_geometric.data import Batch, Data

    vocab = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    task = FlowMolGraph3DTaskFactory(
        atom_vocab=vocab,
        vector_field_config={
            "n_vec_channels": 8,
            "n_hidden_scalars": 16,
            "n_hidden_edge_feats": 16,
            "n_molecule_updates": 1,
            "convs_per_update": 1,
            "a_token_dim": 8,
            "c_token_dim": 8,
            "e_token_dim": 8,
            "time_embedding_dim": 8,
            "self_conditioning": True,
            "rbf_dim": 8,
        },
    ).build()

    assert task.n_real_atom_types == 10, task.n_real_atom_types
    assert task.n_atom_types == 11, task.n_atom_types
    emb = task.vector_field.token_embeddings["a"].weight.shape
    assert emb == (12, 8), emb
    assert task.vector_field.to_edge_logits[-1].out_features == 4
    assert task.vector_field.node_output_head[-1].out_features == 17

    # propane-like: 3 atoms, bonds (0-1) single, (1-2) double
    item = Data(
        pos=torch.randn(3, 3),
        atom_idx=torch.tensor([0, 0, 0]),
        fc=torch.tensor([0, 0, 0]),
        bond_index=torch.tensor([[0, 1], [1, 2]]),
        bond_type=torch.tensor([1, 2]),
        n_nodes=3,
        smiles="CCC",
    )
    g = task.to_dgl(
        {"graph": Batch.from_data_list([item])}, use_fake_atoms=False
    )
    ue = get_upper_edge_mask(g)
    assert int(ue.sum()) * 2 == int(g.num_edges())
    labels = g.edata["e_1_true"].argmax(-1)
    # upper triangle of a 3-node graph, in triu_indices order: (0,1),(0,2),(1,2)
    assert labels[ue].tolist() == [1, 0, 2], labels[ue].tolist()
    assert labels[~ue].tolist() == [1, 0, 2], labels[~ue].tolist()

    # one training step must produce a finite loss and a gradient
    loss, stats = task({"graph": Batch.from_data_list([item, item])})
    assert torch.isfinite(loss), loss
    assert set(stats) >= {"x_loss", "a_loss", "c_loss", "e_loss"}
    loss.backward()

    # sampling: width-10 one-hot out, atom count <= requested, bonds stashed
    task.eval()
    one_hot, charges, coords, node_mask = task.sample(
        nodesxsample=torch.tensor([5, 6]), num_steps=3
    )
    assert one_hot.shape[-1] == 10, one_hot.shape
    assert coords.shape[:2] == node_mask.shape == charges.shape
    assert int(node_mask.sum(dim=1).max()) <= 6
    assert len(task.last_bond_types) == 2

    print("diffusion_flowmol_graph3d self-check OK")  # noqa: T201


def test_tasks_diffusion_flowmol_graph3d_self_check():
    _self_check()
