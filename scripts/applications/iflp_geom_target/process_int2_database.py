import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Assuming molSimplify.Classes.mol3D is available
# Ensure 'molSimplify' is installed or its path is correctly configured.
from molSimplify.Classes.mol3D import mol3D 

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default level
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)-10s %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        p1 (np.ndarray): First 3D point (e.g., array of [x, y, z]).
        p2 (np.ndarray): Second 3D point.

    Returns:
        float: The distance between the two points.
    """
    return np.linalg.norm(p1 - p2)


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angle in degrees between two 3D vectors.

    Args:
        v1 (np.ndarray): The first 3D vector.
        v2 (np.ndarray): The second 3D vector.

    Returns:
        float: The angle between the vectors in degrees. Returns NaN if either vector has zero norm.
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return np.nan

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def mol_reorient(mol: mol3D, filename: str, save_path: Path) -> Tuple[float, int, int, float]:
    """
    Reorients a molecule to align the B-N bond along the x-axis and B-H bond in the xy-plane.
    It then saves the reoriented XYZ file and calculates relevant geometric parameters.

    Assumptions for atom indexing (based on original code's implicit use):
    - B is the first Boron atom found.
    - N is the Nitrogen atom closest to B (distance > 1.8 Å).
    - H1 (bonded to B) is the second-to-last atom in the mol3D atom list.
    - H2 (bonded to N) is the last atom in the mol3D atom list.

    Args:
        mol (mol3D): The mol3D object representing the molecule.
        filename (str): The base filename of the molecule (e.g., "molecule.xyz").
        save_path (Path): The directory where the reoriented XYZ file will be saved.

    Returns:
        Tuple[float, int, int, float]: A tuple containing:
            - bn_distance (float): Distance between B and N atoms.
            - num_b_bonds (int): Number of bonds to the B atom (excluding the B-N bond).
            - num_n_bonds (int): Number of bonds to the N atom (excluding the N-B bond).
            - angle_bh_nh_deg (float): Angle between B-H1 and N-H2 vectors in degrees.

    Raises:
        ValueError: If no N atom satisfies the B-N distance condition, or if expected
                    H atoms are not found or not bonded correctly (commented out checks
                    from original code are noted as potential future additions).
    """
    atom_list = mol.getAtoms()
    b_atom = mol.getAtomwithSyms('B')[0]
    n_atoms = mol.getAtomwithSyms('N')
    n_indices = mol.getAtomwithSyms('N', return_index=True)
    b_indices = mol.getAtomwithSyms('B', return_index=True)

    # Find N atom closest to B with distance > 1.8 Å
    n_atom_distances = {}
    for i, n_atom in zip(n_indices, n_atoms):
        dist = b_atom.distance(n_atom)
        if dist >= 1.8:
            n_atom_distances[n_atom] = dist

    num_b_bonds_total = len(mol.getBondedAtoms(b_indices[0]))

    if not n_atom_distances:
        raise ValueError(f"No N atom satisfies the condition B-N > 1.8 in {filename}")

    min_dist_n_atom = min(n_atom_distances, key=n_atom_distances.get)
    min_idx_n = n_indices[n_atoms.index(min_dist_n_atom)]

    bn_distance = n_atom_distances[min_dist_n_atom]
    bn_vector = np.array(min_dist_n_atom.coords()) - np.array(b_atom.coords())
    bn_norm_vector = bn_vector / np.linalg.norm(bn_vector)

    # Translate molecule so B is at origin
    mol_coords = mol.coordsvect()
    translated_coords = mol_coords - b_atom.coords()

    # Rotate so B-N lies along x-axis
    target_x_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(bn_norm_vector, target_x_axis)
    rotation_angle = np.arccos(np.clip(np.dot(bn_norm_vector, target_x_axis), -1.0, 1.0))

    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
        r1 = R.from_rotvec(rotation_angle * rotation_axis)
        rotated_coords = r1.apply(translated_coords)
    else:
        rotated_coords = translated_coords.copy()

    # Rotate so B-H1 lies in xy-plane
    # H1 is assumed to be the second-to-last atom in the original atom_list
    h1_atom = atom_list[-2]
    h1_coords_rotated = rotated_coords[atom_list.index(h1_atom)]
    
    # Calculate angle around x-axis to bring H1 into xy-plane
    z_angle = np.arctan2(h1_coords_rotated[2], h1_coords_rotated[1])
    r2 = R.from_euler('x', -z_angle)
    rotated_coords = r2.apply(rotated_coords)

    # Optional shift to center B-N on origin (original code's behavior)
    rotated_coords[:, 0] -= bn_distance / 2
    
    # Remove mean (center the entire molecule)
    coords_mean = rotated_coords.mean(axis=0)
    rotated_coords -= coords_mean

    num_n_bonds_total = len(mol.getBondedAtoms(min_idx_n))
    logger.info(f"File: {filename}, B bonds: {num_b_bonds_total}, N bonds: {num_n_bonds_total}")

    # Compute angle between B-H1 and N-H2
    # H2 is assumed to be the last atom in the original atom_list
    h2_atom = atom_list[-1]
    
    # Get coordinates of B, N, H1, H2 in the new rotated system
    b_coords_new = rotated_coords[atom_list.index(b_atom)] # B is now at origin conceptually, but its transformed coords are here
    n_coords_new = rotated_coords[atom_list.index(min_dist_n_atom)]
    h1_coords_new = rotated_coords[atom_list.index(h1_atom)]
    h2_coords_new = rotated_coords[atom_list.index(h2_atom)]

    # Vectors for angle calculation
    bh_vector_final = h1_coords_new - b_coords_new
    nh_vector_final = h2_coords_new - n_coords_new

    angle_bh_nh_deg = calculate_angle_between_vectors(bh_vector_final, nh_vector_final)

    # Reorder atoms for writing: B, N, H1, H2, then others
    new_atom_order = [b_atom, min_dist_n_atom, h1_atom, h2_atom] + [
        a for a in atom_list if a not in (b_atom, min_dist_n_atom, h1_atom, h2_atom)
    ]
    # Map original atom objects to their new coordinates
    coords_map = {id(atom): coord for atom, coord in zip(atom_list, rotated_coords)}

    output_filepath = save_path / filename
    with open(output_filepath, 'w') as f:
        f.write(f"{len(new_atom_order)}\n\n")
        for atom in new_atom_order:
            coord = coords_map[id(atom)]
            f.write(f"{atom.symbol()} {' '.join(map(str, coord))}\n")

    # Return bond counts excluding the central B-N bond (as per original script's output)
    return bn_distance, num_b_bonds_total - 1, num_n_bonds_total - 1, angle_bh_nh_deg


def main():
    """
    Main function to parse command-line arguments, process XYZ files
    for reorientation and analysis, and save results to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Reorient molecules to align B–N and B–H bonds.")
    parser.add_argument(
        "input",
        type=str,
        help="Path to input directory containing .xyz files."
    )
    parser.add_argument(
        "output", 
        type=str,
        help="Path to output directory for reoriented .xyz files and CSV."
    )

    args = parser.parse_args()

    input_directory = Path(args.input)
    output_directory = Path(args.output)

    if not input_directory.is_dir():
        logger.critical(f"Error: Input directory '{input_directory}' does not exist or is not a directory.")
        sys.exit(1)

    output_directory.mkdir(parents=True, exist_ok=True)

    xyz_files_to_process = list(input_directory.rglob("*.xyz")) # Use rglob to find files recursively

    if not xyz_files_to_process:
        logger.warning(f"No .xyz files found in '{input_directory}'. Exiting.")
        sys.exit(0)

    results = []
    for xyz_filepath in tqdm(xyz_files_to_process, desc="Processing XYZ files"):
        try:
            mol = mol3D()
            mol.readfromxyz(str(xyz_filepath)) # molSimplify readfromxyz expects string path
            filename = xyz_filepath.name # Get just the filename

            bn_dist, b_bond_count, n_bond_count, phi_angle = \
                mol_reorient(mol, filename, output_directory)
            
            results.append({
                "filename": filename.replace(".xyz", ""),
                "BN_dist": bn_dist,
                "phi": phi_angle,
                "B_bonds": b_bond_count,
                "N_bonds": n_bond_count
            })
        except Exception as e:
            logger.error(f"Failed to process {xyz_filepath.name}: {e}. Skipping.")
            continue

    if results:
        csv_file_path = output_directory / "bond_counts.csv"
        fieldnames = ["filename", "BN_dist", "B_bonds", "N_bonds", "phi"]
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Analysis complete. Results saved to {csv_file_path}")
    else:
        logger.warning("No successful molecule reorientations. No CSV file generated.")


if __name__ == "__main__":
    main()