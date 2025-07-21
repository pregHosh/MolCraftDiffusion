import argparse
import glob
import os
import subprocess as sp
import logging 
import shutil

import torch
import yaml
import numpy as np  

from MolecularDiffusion.utils import create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_utils import read_xyz_file
from MolecularDiffusion.utils.geom_metrics import is_fully_connected


def check_neutrality(filename: str, charge: int = -1, timeout: int = 60) -> bool:
    """
    Checks if a molecule described in an XYZ file is neutral using xTB.

    Executes xTB with the `--ptb` (print properties) flag and parses its log output
    to detect if xTB reports a mismatch between the number of electrons and spin multiplicity,
    which indicates a non-neutral molecule. Temporary xTB output files are cleaned up.

    Args:
        filename (str): Path to the XYZ file.
        charge (int): Molecular charge to use for the xTB calculation. Defaults to -1.
        timeout (int): Maximum time in seconds to wait for the xTB process to complete.

    Returns:
        bool: True if the molecule is inferred to be neutral, False otherwise.
    """
    neutral_mol = True
    execution_command = ["xtb", filename, "--ptb", "-c", str(charge)]
    try:
        with open("xtb.log", "w") as f:
            sp.call(execution_command, stdout=f, stderr=sp.STDOUT, timeout=timeout)
    except sp.TimeoutExpired:           
        logging.warning("xTB calculation timed out during neutrality check.")
        return False 
    
    if os.path.exists("xtb.log"):
        with open("xtb.log", "r") as f:
            lines = f.readlines()
        
        for line in lines:
            if "Number of electrons and spin multiplicity do not match" in line:
                neutral_mol = False
                break
    else:
        logging.warning(f"xTB log file not found for {filename}. Cannot verify neutrality.")
        neutral_mol = False 
    
    temp_files = ["wbo", "charges", "xtb.log"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return neutral_mol

def check_xyz(filename: str, connector_dicts: dict, scale_factor: float = 1.3) -> tuple[bool, int, bool]:
    """
    Performs structural checks on an XYZ file, including zero coordinates, graph connectivity,
    and the degree of specified 'connector' nodes.

    Args:
        filename (str): Path to the XYZ file.
        connector_dicts (dict): Dictionary where keys are node indices and values are lists
                                of expected degrees for those nodes.
        scale_factor (float): Scaling factor for covalent radii in edge correction.

    Returns:
        tuple[bool, int, bool]: A tuple containing:
            - is_connected (bool): True if the molecule's graph is fully connected.
            - num_components (int): Number of connected components in the graph.
            - match_n_degree (bool): True if all specified connector nodes have
                                     degrees matching their expected values, False otherwise.
    """
    cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(filename)
    if torch.all(cartesian_coordinates_tensor == 0):
        logging.error(f"Error: All coordinates in {filename} are zero.")
        return False, 100, False
    
    mol_data = create_pyg_graph(cartesian_coordinates_tensor, atomic_numbers_tensor, xyz_filename=filename, r=5)
    mol_data = correct_edges(mol_data, scale_factor=scale_factor)
    
    num_node = mol_data.num_nodes   
    edge_index = mol_data.edge_index
    
    is_connected, num_components = is_fully_connected(edge_index, num_node)
    match_n_degree = True
    
    for node in range(num_node):
        if node in connector_dicts: 
            adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
            n_degree = len(adjacent_nodes)
            if n_degree not in connector_dicts[node]:
                logging.error(f"Error: Node {node} has {n_degree} neighbors, expected {connector_dicts[node]} in {filename}.")
                match_n_degree = False
                break
            
    return is_connected, num_components, match_n_degree


if __name__ == "__main__":
    """
    Script to filter XYZ files based on structural properties (connectivity, coordination
    number of specific atoms) and molecular neutrality using xTB.
    Bad molecules are moved to a 'bad' subdirectory.
    """
    parser = argparse.ArgumentParser(description="Filter XYZ files based on structural and neutrality checks.")
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        help="Path to the directory containing the XYZ files.", 
        default="input_directory"
    )
    parser.add_argument(
        "-c", "--connectors", 
        type=str, 
        help="Path to the YAML file defining connector atom degrees.", 
        default="connectors.yaml"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=60, # Default timeout for subprocess calls in seconds
        help="Timeout for xTB subprocess calls in seconds. Default is 60."
    )
    parser.add_argument(
        "-s", "--scale_factor",
        type=float,
        default=1.3, # Scale factor for covalent radii in edge correction
        help="Scale factor for covalent radii in edge correction. Default is 1.3."
    )
    parser.add_argument(
        "-ch", "--charge",
        type=int,
        default=-1, # Default charge for neutrality check
        help="Molecular charge for xTB neutrality check. Default is -1."
    )
    args = parser.parse_args()   
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    xyz_path = args.input
    connectors_path = args.connectors
    timeout = args.timeout
    scale_factor = args.scale_factor
    charge = args.charge # Get charge from argparse

    if not os.path.isdir(xyz_path):
        logging.critical(f"Input directory '{xyz_path}' does not exist or is not a directory.")
        exit(1)
    if not os.path.exists(connectors_path):
        logging.critical(f"Connectors YAML file '{connectors_path}' not found.")
        exit(1)

    with open(connectors_path, "r") as yaml_file:
        connector_dicts = yaml.safe_load(yaml_file)
    
    xyz_files = glob.glob(os.path.join(xyz_path, "*.xyz"))   
    xyz_files.sort()
    
    bad_output_dir = os.path.join(xyz_path, "bad")
    os.makedirs(bad_output_dir, exist_ok=True)
    
    bad_mols_list = []
    
    for xyz_file in xyz_files:
        try:
            is_connected, num_components, match_n_degree = check_xyz(xyz_file, connector_dicts, scale_factor=scale_factor)
            neutral_mol = check_neutrality(xyz_file, charge=charge, timeout=timeout) # Pass charge to function
            
            good_xyz = is_connected and (num_components == 1) and match_n_degree and neutral_mol
        except Exception as e:
            logging.error(f"Error processing {xyz_file}: {e}")
            good_xyz = False
        
        if not good_xyz:
            shutil.move(xyz_file, os.path.join(bad_output_dir, os.path.basename(xyz_file)))
            logging.info(f"Moved bad molecule: {xyz_file}")
            logging.info(f"  - Connected: {is_connected}")
            logging.info(f"  - Num Components: {num_components}")
            logging.info(f"  - Match N Degree: {match_n_degree}")
            logging.info(f"  - Neutral Mol: {neutral_mol}")
            bad_mols_list.append(xyz_file)
        else:
            logging.info(f"Passed checks: {xyz_file}")
    
    np.savetxt(os.path.join(xyz_path, "bad_mols.txt"), bad_mols_list, fmt="%s")
    logging.info(f"Processed {len(xyz_files)} files. Found {len(bad_mols_list)} bad molecules.")
