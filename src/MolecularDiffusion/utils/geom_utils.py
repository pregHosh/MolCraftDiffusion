import torch
from typing import Tuple
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from ase.io import read
from ase.data import covalent_radii
import numpy as np
import os
import shutil


def translate_to_origine(coords, node_mask):
    centroid = coords.mean(dim=1, keepdim=True)  
    translation_vector = -centroid
    translated_coords = coords + translation_vector * node_mask
    return translated_coords

    
def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask, std=1.0):
    assert len(size) == 3
    x = torch.randn(size, device=device) * std

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size, device, node_mask, std=1.0):
    x = torch.randn(size, device=device) * std
    x_masked = x * node_mask
    return x_masked


def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = torch.pi * 2
    if n_dims == 2:
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)

    elif n_dims == 3:

        # Build Rx
        Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rx[:, 1:2, 1:2] = cos
        Rx[:, 1:2, 2:3] = sin
        Rx[:, 2:3, 1:2] = -sin
        Rx[:, 2:3, 2:3] = cos

        # Build Ry
        Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Ry[:, 0:1, 0:1] = cos
        Ry[:, 0:1, 2:3] = -sin
        Ry[:, 2:3, 0:1] = sin
        Ry[:, 2:3, 2:3] = cos

        # Build Rz
        Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        Rz[:, 0:1, 0:1] = cos
        Rz[:, 0:1, 1:2] = sin
        Rz[:, 1:2, 0:1] = -sin
        Rz[:, 1:2, 1:2] = cos

        x = x.transpose(1, 2)
        x = torch.matmul(Rx, x)
        # x = torch.matmul(Rx.transpose(1, 2), x)
        x = torch.matmul(Ry, x)
        # x = torch.matmul(Ry.transpose(1, 2), x)
        x = torch.matmul(Rz, x)
        # x = torch.matmul(Rz.transpose(1, 2), x)
        x = x.transpose(1, 2)
    else:
        raise Exception("Not implemented Error")

    return x.contiguous()



def coord2cosine(x, edge_index, epsilon=1e-8):
    row, col = edge_index
    tensor1, tensor2 = x[row], x[col]
    dot_product = torch.sum(tensor1 * tensor2, dim=-1)
    magnitude1 = torch.sqrt(torch.sum(tensor1**2, dim=-1)) + epsilon
    magnitude2 = torch.sqrt(torch.sum(tensor2**2, dim=-1)) + epsilon

    cosine_sim = dot_product / (magnitude1 * magnitude2)

    return cosine_sim



def coord2diff(x: torch.Tensor, edge_index: torch.Tensor, norm_constant: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the radial distance and normalized coordinate difference between nodes connected by edges.

    Args:
        x (torch.Tensor): Node coordinates of shape (num_nodes, 3).
        edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
        norm_constant (float, optional): Constant added to the normalization term for numerical stability. Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Radial distances of shape (num_edges, 1) and normalized coordinate differences of shape (num_edges, 3).
    """
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff


def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    return x - mean


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1, considering a node mask.

    Args:
        x (torch.Tensor): Input tensor.
        node_mask (torch.Tensor): Boolean mask indicating valid nodes.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(x, dim=1, keepdim=True) / N
    return x - mean * node_mask


def remove_mean_with_mask_v2(pos: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Removes the mean from a tensor along dimension 1, considering a node mask.

    Args:
        pos (torch.Tensor): Input tensor of shape (bs, n, 3).
        node_mask (torch.Tensor): Boolean mask of shape (bs, n) indicating valid nodes.

    Returns:
        torch.Tensor: Mean-centered tensor.
    """
    # assert node_mask.dtype == torch.bool, f"Wrong dtype for the mask: {node_mask.dtype}"
    N = node_mask.sum(1, keepdims=True)
    mean = torch.sum(pos, dim=1, keepdim=True) / N
    return pos - mean * node_mask



def assert_mean_zero(x: torch.Tensor) -> None:
    """
    Asserts that the mean of a tensor along dimension 1 is close to zero.

    Args:
        x (torch.Tensor): Input tensor.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    """
    Asserts that the masked values in the variable are close to zero.

    Args:
        variable (torch.Tensor): Input tensor.
        node_mask (torch.Tensor): Boolean mask indicating valid nodes.
    """
    assert (
        variable * (1 - node_mask)
    ).abs().max().item() < 1e-4, "Variables not masked properly."
    
def check_mask_correct(variables: list, node_mask: torch.Tensor) -> None:
    """
    Checks if variables are correctly masked using assert_correctly_masked.

    Args:
        variables (list): List of variables to check.
        node_mask (torch.Tensor): Node mask to apply.
    """
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)



def read_xyz_file(xyz_file):
    """
    Reads an XYZ file and extracts atomic positions and atomic numbers.
    Args:
        xyz_file (str): Path to the XYZ file.
    Returns:
        tuple: A tuple containing:
            - cartesian_coordinates_tensor (torch.Tensor): Tensor of shape (N, 3) with the Cartesian coordinates of the atoms.
            - atomic_numbers_tensor (torch.Tensor): Tensor of shape (N,) with the atomic numbers of the atoms.
    """
    atoms = read(xyz_file)
    cartesian_coordinates = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    
    
    cartesian_coordinates_tensor = torch.tensor(cartesian_coordinates, dtype=torch.float32)
    atomic_numbers_tensor = torch.tensor(atomic_numbers, dtype=torch.int16)
    
    return cartesian_coordinates_tensor, atomic_numbers_tensor


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

def save_xyz_file(
    path,
    one_hot,
    positions,
    atom_decoder,
    id_from=0,
    name="molecule",
    node_mask=None,
    idxs=None,
):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        try:
        
            if idxs is None:
                idx = batch_i + id_from
            else:
                idx = idxs[batch_i]

            f = open(name + "_" + "%03d.xyz" % (idx), "w")
            f.write("%d\n\n" % atomsxmol[batch_i])
            atoms = torch.argmax(one_hot[batch_i], dim=1)
            n_atoms = int(atomsxmol[batch_i])
            for atom_i in range(n_atoms):
                atom = atoms[atom_i]
                atom = atom_decoder[atom.item() if isinstance(atom, torch.Tensor) else atom]
                f.write(
                    "%s %.9f %.9f %.9f\n"
                    % (
                        atom,
                        positions[batch_i, atom_i, 0],
                        positions[batch_i, atom_i, 1],
                        positions[batch_i, atom_i, 2],
                    )
                )
            f.close()
            filename_xyz = name + "_" + "%03d.xyz" % (idx)
            if os.path.exists(filename_xyz):
                shutil.move(filename_xyz, path)
            else:
                pass
        except Exception as e:
            # print("Error in saving molecule: ", idx)
            pass

