

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
import torch
from ase.data import covalent_radii


def create_pyg_graph(cartesian_coordinates_tensor,
                     atomic_numbers_tensor,
                     xyz_filename=None,
                     r=5.0):
    """
    Creates a PyTorch Geometric graph from given cartesian coordinates and atomic numbers.
    Args:
        cartesian_coordinates_tensor (torch.Tensor): A tensor containing the cartesian coordinates of the atoms.
        atomic_numbers_tensor (torch.Tensor): A tensor containing the atomic numbers of the atoms.
        xyz_filename (str): The filename of the XYZ file.
        r (float, optional): The radius within which to consider edges between nodes. Default is 5.0.
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object containing the graph representation of the molecule.
    """


    edge_index = radius_graph(cartesian_coordinates_tensor, r=r)

    data = Data(x=atomic_numbers_tensor.view(-1, 1).float(),
                pos=cartesian_coordinates_tensor,
                edge_index=edge_index,
                filename=xyz_filename
                )

    return data


def correct_edges(data, scale_factor=1.3):
    """
    Corrects the edges in a molecular grapSCALE_FACTORh based on covalent radii.
    This function iterates over the nodes and their adjacent nodes in the given
    molecular graph data. It calculates the bond length between each pair of nodes
    and checks if it is within the allowed bond length threshold (sum of covalent radii plus relaxation factor).
    If the bond length is valid, the edge is kept; otherwise, it is removed.

    Parameters:
    data (torch_geometric.data.Data): The input molecular graph data containing node features,
                                      edge indices, and positions.
    scale_factor (float): The scaling factor to apply to the covalent radii. Default is 1.3.

    Returns:
    torch_geometric.data.Data: The corrected molecular graph data with updated edge indices.
    """
    atomic_nums = data.x.view(-1).int().tolist()
    edge_index = data.edge_index
    valid_edges = []

    for node in range(len(atomic_nums)):
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        for adj_node in adjacent_nodes:
            bond_length = torch.norm(data.pos[node] - data.pos[adj_node]).item()

            # Get covalent radii from ASE
            r1 = covalent_radii[atomic_nums[node]]*scale_factor
            r2 = covalent_radii[atomic_nums[adj_node]]*scale_factor
            max_bond_length = r1 + r2

            if bond_length <= max_bond_length:
                valid_edges.append([node, adj_node])

    data.edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
    return data


def remove_mean_pyG(x, batch_idx):
    """Removes the mean of the node positions for each graph individually."""

    batch_size = batch_idx.max().item() + 1

    # Sum of positions for each graph
    sum_x = torch.zeros((batch_size, x.size(-1)), device=x.device)  # (batch_size, n_features)
    num_nodes = torch.zeros((batch_size, 1), device=x.device)       # (batch_size, 1)

    sum_x.index_add_(0, batch_idx, x)            # Sum positions per graph
    num_nodes.index_add_(0, batch_idx, torch.ones_like(x[:, [0]]))  # Count nodes per graph

    mean = sum_x / num_nodes.clamp(min=1.0)      # (batch_size, n_features)

    # Expand mean back to each node
    mean_per_node = mean[batch_idx]              # (total_num_nodes, n_features)

    # Subtract mean
    x_centered = x - mean_per_node

    # Optional: check that overall mean is (almost) zero
    # total_mean_abs = x_centered.index_add_(0, batch_idx, x_centered).abs().sum().item()
    # assert total_mean_abs < 1e-5, f"Error {total_mean_abs} too high"

    return x_centered



