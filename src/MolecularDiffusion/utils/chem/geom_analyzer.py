from typing import Dict, List, Optional, Tuple

import os

import ase
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy as sp
import torch
import wandb

from ase.data import covalent_radii
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from MolecularDiffusion.utils.chem.smilify import smilify_xyz2mol as smilify
# %% predefined data


SCALE_FACTOR = 1.3
EDGE_THRESHOLD = 2

# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
# TODO are these comprehensive enougn for FORMED

num2symbol = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    33: "As",
    34: "Se",
    35: "Br",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    80: "Hg",
    83: "Bi",
    13: "Al",
}

symbol2num = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Bi": 83,
    "Hg": 80,
    "Al": 13,
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "B": 132,  # **
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "As": 185,  # **
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "B": 145,  # **
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Si": 170,
        "P": 177,
        "S": 168,
        "Cl": 175,
        "As": 222,  # **
        "Br": 214,
        "I": 222,
    },
    "O": {
        "H": 96,
        "B": 137,  # **
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Si": 163,
        "P": 163,
        "S": 151,
        "Cl": 164,
        "As": 178,  # **
        "Br": 172,
        "I": 194,
    },
    "F": {
        "H": 92,
        "B": 130,  # **
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "Si": 160,
        "P": 156,
        "S": 158,
        "Cl": 166,
        "As": 171,  # **
        "Br": 178,
        "I": 187,
    },
    "B": {
        "H": 119,
        "B": 165,  # **
        "C": 132,  # **
        "O": 137,  # **
        "N": 145,  # **
        "F": 130,  # **
        "Si": 175,  # **
        "P": 185,  # **
        "S": 175,  # **
        "Cl": 175,
        "As": 185,  # **
        "Br": 194,
        "I": 214,
    },
    "Si": {
        "H": 148,
        "B": 175,  # **
        "C": 185,
        "N": 170,  # *
        "O": 163,
        "F": 160,
        "Si": 233,
        "P": 220,  # **
        "S": 200,
        "Cl": 202,
        "As": 243,  # **
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "H": 127,
        "B": 175,  # **
        "C": 177,
        "N": 175,
        "O": 164,
        "F": 166,
        "Si": 202,
        "P": 203,
        "S": 207,
        "Cl": 199,
        "Br": 214,
        "As": 216,  # **
        "I": 233,
    },
    "S": {
        "H": 134,
        "B": 175,  # **
        "C": 182,
        "N": 168,
        "O": 151,
        "F": 158,
        "S": 204,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "B": 194,  # **
        "C": 194,
        "O": 172,
        "N": 214,
        "F": 178,
        "Si": 215,
        "P": 222,
        "S": 225,
        "Cl": 214,
        "As": 233,
    },
    "P": {
        "P": 221,
        "H": 144,
        "B": 185,  # **
        "C": 184,
        "O": 163,
        "N": 177,
        "F": 156,
        "Si": 220,
        "S": 210,
        "Cl": 203,
        "As": 243,
        "Br": 222,
        "I": 254,
    },
    "I": {
        "H": 161,
        "C": 214,
        "N": 222,
        "O": 194,
        "F": 187,
        "Si": 243,
        "P": 254,
        "S": 234,
        "Cl": 233,
        "Br": 228,
        "As": 254,
        "I": 266,
    },
    "As": {
        "H": 152,
        "C": 185,  # **
        "N": 222,
        "O": 178,  # **
        "F": 171,
        "Si": 243,  # **
        "As": 243,
        "Cl": 216,
        "Br": 233,
        "I": 254,
    },
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}

stdv = {"H": 5, "C": 1, "N": 1, "O": 2, "F": 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": [3, 5],
    "Se": 4,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}


# %% 3D mol quality check

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


def is_fully_connected(edge_index, num_nodes):
    """
    Determines if the graph is fully connected.
    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int): The number of nodes in the graph.
    Returns:
        bool: True if the graph is fully connected, False otherwise.
        int: The number of connected components in the graph.
    """
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    try:
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
    except nx.NetworkXPointlessConcept:
        is_connected = False
        num_components = 100
    return is_connected, num_components
    
    
def check_quality(cartesian_coordinates_tensor, atomic_numbers_tensor):
    data = create_pyg_graph(cartesian_coordinates_tensor, 
                                    atomic_numbers_tensor,
                                    r=EDGE_THRESHOLD)
    data = correct_edges(data, scale_factor=SCALE_FACTOR) 
    is_connected, num_components = is_fully_connected(data.edge_index, data.num_nodes)
    
    n_degrees = []
    edge_index = data.edge_index
    for node in range(len(atomic_numbers_tensor)):
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        n_degree = len(adjacent_nodes)
        n_degrees.append(n_degree)
    return is_connected, num_components, n_degrees


    

# %% Utilis functions


def build_molecule(positions, atom_types, atom_decoder):
    X, A, E = build_xae_molecule(positions, atom_types, atom_decoder)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(
            bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
        )
    return mol


def build_xae_molecule(positions, atom_types, atom_decoder):
    """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
    args:
    positions: N x 3  (already masked to keep final number nodes)
    atom_types: N
    returns:
    X: N         (int)
    A: N x N     (bool)                  (binary adjacency matrix)
    E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """

    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)

    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(
                atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j]
            )
            # order = geom_predictor(
            #     (atom_decoder[pair[0]], atom_decoder[pair[1]]),
            #     dists[i, j],
            #     limit_bonds_to_one=True,
            # ) NOTE whats diff between this and get_bond_order
            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    not_exist = False
    if check_exists:
        if atom1 not in bonds1:
            not_exist = True
        if atom2 not in bonds1[atom1]:
            not_exist = True
    if not_exist and distance < 180:
        return 1
    elif not_exist and distance >= 180:
        return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """p: atom pair (couple of str)
    l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order


def save_xyz_tmp(path, atom_type, position):
    with open(path, "w") as f:
        f.write(f"{len(atom_type)}\n")
        f.write("Suichan wa kyou mo kawaii~~\n")
        for i in range(len(atom_type)):
            f.write(
                f"{atom_type[i]} {position[i][0]} {position[i][1]} {position[i][2]}\n"
            )

def get_cutoffs(z, radii=ase.data.covalent_radii, mult=1):
    return [radii[zi] * mult for zi in z]

def check_symmetric(am, tol=1e-8):
    return sp.linalg.norm(am - am.T, np.Inf) < tol

def check_connected(am, tol=1e-8):
    sums = am.sum(axis=1)
    lap = np.diag(sums) - am
    eigvals, eigvects = np.linalg.eig(lap)
    return len(np.where(abs(eigvals) < tol)[0]) < 2


def check_stability(positions, atom_type):
    """
    Check the stability of a molecule based on atom positions and types.

    Parameters:
    - positions (np.ndarray): An array of shape (N, 3) containing the 3D coordinates of the atoms.
    - atom_type (List[int]): A list of atom types corresponding to the positions.
    - atom_decoder (Dict[int, str]): A dictionary mapping atom type indices to atom symbols.

    Returns:
    Tuple[bool, int, int]: A tuple containing a boolean indicating if the molecule is stable,
                        the number of stable bonds, and the total number of atoms.
    """
    molecule_stable = 0
    ratio_stable_atoms = 0
    n_stable_atom = 0
    TMP_XYZ_PATH = "tmp_analyze.xyz"
    atom_symbols = [num2symbol[atom] for atom in atom_type]
    save_xyz_tmp(TMP_XYZ_PATH, atom_symbols, positions)
    smiles, mol, AC = smilify(TMP_XYZ_PATH)
    if smiles is not None and mol is not None:
        molecule = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(molecule, canonical=True)
        molecule_stable = 1
        ratio_stable_atoms = 1
    else:
        k = 0
        for atom, bond in zip(atom_symbols, AC):
            nbond = len(np.where(bond == 1)[0])
            if nbond <= allowed_bonds[atom]:
                n_stable_atom += 1
            else:
                continue
                # print(f"{atom}{k} has {nbond} bonds")
            k += 1
        ratio_stable_atoms = n_stable_atom / len(atom_type)

    if os.path.exists(TMP_XYZ_PATH):
        os.remove(TMP_XYZ_PATH)

    return molecule_stable, ratio_stable_atoms, smiles