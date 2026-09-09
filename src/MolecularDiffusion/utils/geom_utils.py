import torch
from typing import Tuple
from typing import Any, Dict, List
from tqdm import tqdm
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from ase.io import read
from ase.data import covalent_radii, chemical_symbols
from scipy.spatial.transform import Rotation
import numpy as np
import os
import shutil
import logging

_logger = logging.getLogger(__name__)


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


def sample_uniform_rotation_matrices(n, device, dtype=torch.float32):
    """Haar-uniform SO(3) rotation matrices, shape ``(n, 3, 3)``.

    Unlike ``random_rotation`` below (independent uniform Euler angles,
    not exactly Haar-uniform), this samples true Haar-uniform rotations
    via ``scipy.spatial.transform.Rotation.random()`` -- the same
    sampler ``modules/models/tabasco/data/transforms.py::
    sample_uniform_rotation`` already uses for TABASCO's own N-fold
    rotation augmentation. Placed here rather than imported from that
    module so this generic geometry utility has no TABASCO-family
    dependency.
    """
    mats = Rotation.random(n).as_matrix()
    return torch.tensor(mats, device=device, dtype=dtype)


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
    tol=1e-4,
    atomic_numbers=None,
    use_unknown_fallback=False,
):
    """Save XYZ files for a batch of molecules, skipping atoms near (0,0,0).
    
    Args:
        path: Output directory
        one_hot: [B, N, C] one-hot encoding
        positions: [B, N, 3] coordinates
        atom_decoder: List mapping indices to atom symbols
        id_from: Starting index for filenames
        name: Filename prefix
        node_mask: Optional [B, N] or [B, N, 1] mask
        idxs: Optional indices for filenames
        tol: Tolerance for filtering atoms near origin
        atomic_numbers: Optional [B, N] atomic numbers for fallback
        use_unknown_fallback: If True and argmax hits unknown column, use atomic_numbers
    """
    os.makedirs(path, exist_ok=True)

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        try:
            idx = batch_i + id_from if idxs is None else idxs[batch_i]
            filename = f"{name}_{idx:03d}.xyz"

            atoms = torch.argmax(one_hot[batch_i], dim=1)
            n_atoms = int(atomsxmol[batch_i])

            # Filter out atoms near (0,0,0)
            coords = positions[batch_i, :n_atoms]
            mask = torch.any(torch.abs(coords) > tol, dim=1)  # keep atoms not at origin
            filtered_atoms = atoms[:n_atoms][mask]
            filtered_coords = coords[mask]
            n_valid = filtered_atoms.size(0)
            
            # Get atomic numbers for fallback if available
            if atomic_numbers is not None:
                filtered_Z = atomic_numbers[batch_i, :n_atoms][mask]
            else:
                filtered_Z = None

            with open(filename, "w") as f:
                f.write(f"{n_valid}\n\n")
                for i, (atom, pos) in enumerate(zip(filtered_atoms, filtered_coords)):
                    atom_idx = atom.item()
                    
                    # Get symbol from decoder
                    if atom_idx < len(atom_decoder):
                        symbol = atom_decoder[atom_idx]
                    else:
                        symbol = None  # Unknown index
                    
                    # Use atomic number fallback when:
                    # 1. use_unknown_fallback is True AND
                    # 2. (symbol is invalid/placeholder OR atom_idx >= len(atom_decoder))
                    is_invalid_symbol = (symbol is None or 
                                        symbol not in chemical_symbols or 
                                        symbol in ("Suisei", "X", "UNK", "?"))
                    
                    if use_unknown_fallback and is_invalid_symbol:
                        if filtered_Z is not None:
                            symbol = chemical_symbols[int(filtered_Z[i].item())]
                        else:
                            symbol = "X"  # Last resort fallback
                    elif symbol is None:
                        symbol = "X"
                        
                    f.write(f"{symbol} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}\n")

            if os.path.exists(filename):
                dest = os.path.join(path, os.path.basename(filename))
                if os.path.exists(dest):
                    _logger.warning(f"save_xyz_file: overwriting existing file {dest}")
                shutil.move(filename, dest)
        except Exception as _e:
            _logger.warning(f"save_xyz_file: failed to save molecule {batch_i}: {_e}")


def save_xyz_file_atomic_numbers(
    path: str,
    positions: torch.Tensor,
    atomic_numbers: torch.Tensor,
    id_from: int = 0,
    name: str = "molecule",
    node_mask: torch.Tensor | None = None,
    idxs=None,
    tol: float = 1e-4,
):
    """
    Save XYZ files for a batch of molecules, writing ATOMIC SYMBOLS in the first column.

    Args:
        path: output directory
        positions: (B, N, 3) tensor
        atomic_numbers: (B, N) long tensor
        id_from: starting index for filenames
        name: filename prefix
        node_mask: optional (B, N) mask; if provided, considers first sum(mask) atoms per molecule
        idxs: optional iterable of indices (len B) to use in filenames
        tol: atoms with coords ~ (0,0,0) are skipped (|coord| <= tol in all dims)
    """
    os.makedirs(path, exist_ok=True)

    if positions.ndim != 3 or positions.size(-1) != 3:
        raise ValueError("`positions` must have shape (B, N, 3).")
    if atomic_numbers.ndim != 2 or atomic_numbers.shape[:2] != positions.shape[:2]:
        raise ValueError("`atomic_numbers` must have shape (B, N) matching positions' (B, N, 3).")

    B, N, _ = positions.shape

    # How many atoms per mol to consider
    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1).to(torch.long)  # (B,)
    else:
        atomsxmol = torch.full((B,), N, dtype=torch.long, device=positions.device)

    for batch_i in range(B):
        try:
            idx = (batch_i + id_from) if idxs is None else int(idxs[batch_i])
            filename = f"{name}_{idx:03d}.xyz"
            outpath = os.path.join(path, filename)

            n_atoms = int(atomsxmol[batch_i].item())
            coords = positions[batch_i, :n_atoms]                      # (n_atoms, 3)
            Z = atomic_numbers[batch_i, :n_atoms].to(torch.long)       # (n_atoms,)

            # Filter out atoms near origin
            keep = torch.any(torch.abs(coords) > tol, dim=1)
            coords = coords[keep]
            Z = Z[keep]
            n_valid = int(Z.numel())

            with open(outpath, "w", encoding="utf-8") as f:
                f.write(f"{n_valid}\n\n")
                for zi, pos in zip(Z.tolist(), coords.tolist()):
                    symbol = chemical_symbols[zi]
                    f.write(f"{symbol} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}\n")

        except Exception as e:
            # keep behavior similar to your original (skip failures quietly)
            pass

def save_shepherd_outputs(output_dir: str, structures: list, idx_offset: int = 0, save_modalities: bool = False):
    """
    Save ShEPhERD generated structures to disk.

    Each structure is a dict returned by _extract_generated_samples():
        x1: {atoms: ndarray(N,), positions: ndarray(N,3), bonds: ndarray(E,)}
        x2: {positions: ndarray(75,3)}
        x3: {charges: ndarray(75,), positions: ndarray(75,3)}
        x4: {types: ndarray(M,), positions: ndarray(M,3), directions: ndarray(M,3)}

    Outputs per sample (zero-padded index):
        mol_{idx:04d}.xyz          x1 structure (standard XYZ)
        mol_{idx:04d}_surface.npy  x2 surface point cloud (75,3)
        mol_{idx:04d}_esp.npz      x3 electrostatics: positions(75,3) + charges(75,)
        mol_{idx:04d}_pharm.npz    x4 pharmacophores: types(M,) + positions(M,3) + directions(M,3)
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, s in enumerate(structures):
        idx = idx_offset + i

        # --- x1: write .xyz ---
        x1 = s.get('x1', {})
        atoms = x1.get('atoms', np.array([]))
        positions = x1.get('positions', np.array([]).reshape(0, 3))
        if len(atoms) > 0:
            xyz_path = os.path.join(output_dir, f"mol_{idx:04d}.xyz")
            with open(xyz_path, 'w') as f:
                f.write(f"{len(atoms)}\n")
                f.write(f"mol_{idx:04d}\n")
                for z, pos in zip(atoms, positions):
                    sym = chemical_symbols[int(z)] if 0 <= int(z) < len(chemical_symbols) else 'X'
                    f.write(f"{sym}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}\n")

        # --- x2: surface point cloud ---
        x2 = s.get('x2', {})
        x2_pos = x2.get('positions', None)
        if x2_pos is not None and len(x2_pos) > 0:
            np.save(os.path.join(output_dir, f"mol_{idx:04d}_surface.npy"), x2_pos)

        # --- x3: electrostatics ---
        x3 = s.get('x3', {})
        x3_pos = x3.get('positions', None)
        x3_charges = x3.get('charges', None)
        if x3_pos is not None and x3_charges is not None:
            np.savez(
                os.path.join(output_dir, f"mol_{idx:04d}_esp.npz"),
                positions=x3_pos,
                charges=x3_charges,
            )

        # --- x4: pharmacophores ---
        x4 = s.get('x4', {})
        x4_types = x4.get('types', None)
        x4_pos = x4.get('positions', None)
        x4_dir = x4.get('directions', None)
        if x4_types is not None and x4_pos is not None:
            np.savez(
                os.path.join(output_dir, f"mol_{idx:04d}_pharm.npz"),
                types=x4_types,
                positions=x4_pos,
                directions=x4_dir if x4_dir is not None else np.zeros_like(x4_pos),
            )

        # --- Optional: unified .npz for metrics ---
        if save_modalities:
            np.savez(
                os.path.join(output_dir, f"mol_{idx:04d}.npz"),
                surf_pts=x2_pos if x2_pos is not None else np.array([]),
                esp_vals=x3_charges if x3_charges is not None else np.array([]),
                pharm_pts=x4_pos if x4_pos is not None else np.array([]),
                pharm_types=x4_types if x4_types is not None else np.array([]),
            )



def random_rotation_matrix(validate: bool = False, device=None, dtype=None) -> torch.Tensor:
    """Generate a random 3x3 rotation matrix from a quaternion.
    
    Args:
        validate: If True, verify the matrix is orthogonal
        device: Target device for the tensor
        dtype: Target dtype for the tensor
        
    Returns:
        A (3, 3) rotation matrix
    """
    # Generate a random quaternion
    q = torch.rand(4, device=device, dtype=dtype)
    q = q / torch.linalg.norm(q)
    
    # Compute the rotation matrix from the quaternion
    rot_mat = torch.tensor([
        [
            1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
            2 * q[1] * q[2] - 2 * q[0] * q[3],
            2 * q[1] * q[3] + 2 * q[0] * q[2],
        ],
        [
            2 * q[1] * q[2] + 2 * q[0] * q[3],
            1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
            2 * q[2] * q[3] - 2 * q[0] * q[1],
        ],
        [
            2 * q[1] * q[3] - 2 * q[0] * q[2],
            2 * q[2] * q[3] + 2 * q[0] * q[1],
            1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
        ],
    ], device=device, dtype=dtype)
    
    if validate:
        eye = torch.eye(3, device=device, dtype=dtype)
        assert torch.allclose(
            rot_mat @ rot_mat.T, eye, atol=1e-5, rtol=1e-5
        ), "Not a valid rotation matrix."
    
    return rot_mat


def apply_rotation_augmentation(
    batch,
    rot_mat: torch.Tensor,
    rotate_cell: bool = False,
) -> None:
    """Apply rotation augmentation to batch positions (in-place).
    
    Args:
        batch: PyG Data/Batch with pos attribute
        rot_mat: (3, 3) rotation matrix
        rotate_cell: If True, also rotate cell (for crystals). Disabled for molecules.
    """
    # Rotate positions: pos' = pos @ R^T
    batch.pos = batch.pos @ rot_mat.T
    
    # For crystals: rotate cell vectors (disabled for molecules)
    if rotate_cell and hasattr(batch, 'cell') and batch.cell is not None:
        batch.cell = batch.cell @ rot_mat.T
        # Note: fractional coordinates are rotation invariant, no update needed


def compute_rmsd(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Compute RMSD between two sets of positions.
    
    Handles different number of atoms by returning inf.
    """
    if pos1.shape != pos2.shape:
        return float("inf")
    return np.sqrt(np.mean(np.sum((pos1 - pos2) ** 2, axis=-1)))


def compute_atom_type_accuracy(types1: np.ndarray, types2: np.ndarray) -> float:
    """Compute accuracy of atom type predictions.
    
    Handles different number of atoms by returning 0.0.
    """
    if types1.shape != types2.shape:
        return 0.0
    return np.mean(types1 == types2)


class MoleculeReconstructionEvaluator:
    """Evaluator for molecule reconstruction tasks.
    
    Simple evaluator that computes:
    - RMSD between predicted and ground truth positions
    - Atom type accuracy
    - Match rate (molecules with RMSD below threshold and perfect atom types)
    
    Does NOT require pymatgen or openbabel.
    
    Args:
        rmsd_threshold: RMSD threshold (in Angstroms) for considering a match.
    """
    
    def __init__(self, rmsd_threshold: float = 0.5):
        self.rmsd_threshold = rmsd_threshold
        self.pred_arrays_list: List[Dict[str, np.ndarray]] = []
        self.gt_arrays_list: List[Dict[str, np.ndarray]] = []
        self.device = torch.device("cpu")
    
    def append_pred_array(self, pred: Dict[str, np.ndarray]):
        """Append a prediction to the evaluator.
        
        Args:
            pred: Dict with keys:
                - 'atom_types': (n_atoms,) atomic numbers
                - 'pos': (n_atoms, 3) positions
                - 'sample_idx': sample index
        """
        self.pred_arrays_list.append(pred)
    
    def append_gt_array(self, gt: Dict[str, np.ndarray]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)
    
    def clear(self):
        """Clear stored predictions and ground truths for next epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
    
    def get_metrics(
        self, 
        current_epoch: int = 0, 
        save: bool = False, 
        save_dir: str = ""
    ) -> Dict[str, Any]:
        """Compute reconstruction metrics.
        
        Returns:
            Dict with:
                - match_rate: fraction of molecules with RMSD < threshold AND perfect atom types
                - mean_rms_dist: mean RMSD over all samples
                - atom_type_accuracy: mean atom type accuracy over all samples
        """
        assert len(self.pred_arrays_list) == len(self.gt_arrays_list), \
            "Number of predictions and ground truths must match."
        
        if len(self.pred_arrays_list) == 0:
            return {
                "match_rate": torch.tensor(0.0, device=self.device),
                "mean_rms_dist": torch.tensor(10.0, device=self.device),
                "atom_type_accuracy": torch.tensor(0.0, device=self.device),
            }
        
        rmsd_list = []
        accuracy_list = []
        match_list = []
        
        for i in tqdm(range(len(self.pred_arrays_list)), 
                      desc=f"Epoch {current_epoch}, reconstruction eval", leave=False):
            pred = self.pred_arrays_list[i]
            gt = self.gt_arrays_list[i]
            
            # Compute RMSD
            rmsd = compute_rmsd(pred["pos"], gt["pos"])
            rmsd_list.append(rmsd)
            
            # Compute atom type accuracy
            acc = compute_atom_type_accuracy(pred["atom_types"], gt["atom_types"])
            accuracy_list.append(acc)
            
            # Match: RMSD below threshold AND perfect atom types
            is_match = (rmsd < self.rmsd_threshold) and (acc == 1.0)
            match_list.append(float(is_match))
        
        # Convert to tensors
        rmsd_tensor = torch.tensor(rmsd_list, device=self.device)
        accuracy_tensor = torch.tensor(accuracy_list, device=self.device)
        match_tensor = torch.tensor(match_list, device=self.device)
        
        # Filter inf for mean RMSD
        valid_rmsd = rmsd_tensor[~torch.isinf(rmsd_tensor)]
        mean_rmsd = valid_rmsd.mean() if len(valid_rmsd) > 0 else torch.tensor(10.0, device=self.device)
        
        return {
            "match_rate": match_tensor.mean(),
            "mean_rms_dist": mean_rmsd,
            "atom_type_accuracy": accuracy_tensor.mean(),
        }
    
    def save_molecules(self, save_dir: str):
        """Save predicted and ground truth molecules as XYZ files."""
        from ase import Atoms
        from ase.io import write
        
        os.makedirs(f"{save_dir}/pred", exist_ok=True)
        os.makedirs(f"{save_dir}/gt", exist_ok=True)
        
        for i, (pred, gt) in enumerate(zip(self.pred_arrays_list, self.gt_arrays_list)):
            sample_idx = pred.get("sample_idx", i)
            
            # Predicted
            atoms_pred = Atoms(numbers=pred["atom_types"], positions=pred["pos"])
            write(f"{save_dir}/pred/molecule_{sample_idx}.xyz", atoms_pred)
            
            # Ground truth
            atoms_gt = Atoms(numbers=gt["atom_types"], positions=gt["pos"])
            write(f"{save_dir}/gt/molecule_{sample_idx}.xyz", atoms_gt)
