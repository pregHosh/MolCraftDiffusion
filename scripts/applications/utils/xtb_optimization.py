import subprocess as sp
import os
import glob
import shutil
from tqdm import tqdm
import argparse
import torch

from MolecularDiffusion.utils import create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_utils import read_xyz_file
from MolecularDiffusion.utils.geom_metrics import is_fully_connected

def check_neutrality(filename: str, 
                     charge: int = -1, 
                     timeout: int=180) -> bool:
    """
    Checks if a molecule described in an XYZ file is neutral using xTB.

    This function executes the `xtb` command with the `--ptb` (print properties)
    flag and parses its log output to detect if xTB reports a mismatch
    between the number of electrons and spin multiplicity, which indicates
    a non-neutral molecule. Temporary xTB output files are cleaned up afterwards.

    Note: This function assumes `xtb` is installed and accessible in the system's PATH.
    This functionality could potentially be integrated with other molecular property
    calculation modules if available.

    Args:
        filename (str): The path to the XYZ file of the molecule to check.
        charge (int): The molecular charge to use for the xTB calculation. Defaults to -1.
        timeout (int): The maximum time in seconds to wait for the xTB process to complete.

    Returns:
        bool: True if the molecule is inferred to be neutral based on xTB's output,
              False otherwise.
    """
    neutral_mol = True
    execution_command = ["xtb", filename, "--ptb", "-c", str(charge)]

    try:
        with open("xtb.log", "w") as f:
            sp.call(execution_command, stdout=f, stderr=sp.STDOUT, timeout=timeout)
    except sp.TimeoutExpired:
        print("xTB calculation timed out during neutrality check.")
        return False

    if os.path.exists("xtb.log"):
        with open("xtb.log", "r") as f:
            lines = f.readlines()

        for line in lines:
            if "Number of electrons and spin multiplicity do not match" in line:
                neutral_mol = False
                break
    else:
        print(f"Warning: xTB log file not found for {filename}. Cannot verify neutrality.")
        neutral_mol = False

    temp_files = ["wbo", "charges", "xtb.log"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return neutral_mol


def check_xyz(filename: str, connector_dicts: dict = None, scale_factor: float = 1.3) -> tuple[bool, int, bool]:
    """
    Performs a series of checks on an XYZ file to validate its molecular structure.

    This includes checking for zero coordinates, graph connectivity, and
    optionally, the degree of specific 'connector' nodes.

    Args:
        filename (str): The path to the XYZ file to be checked.
        connector_dicts (dict, optional): A dictionary where keys are node indices
            and values are lists of expected degrees for those nodes. Used to
            validate connectivity at specific points in the molecule. Defaults to None.
        scale_factor (float, optional): The scaling factor for covalent radii in edge correction. Defaults to 1.3.

    Returns:
        tuple[bool, int, bool]: A tuple containing:
            - is_connected (bool): True if the molecule's graph is fully connected.
            - num_components (int): The number of connected components in the graph.
            - match_n_degree (bool): True if all specified connector nodes have
              degrees matching their expected values in `connector_dicts`, False otherwise.
    """
    cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(filename)

    if torch.all(cartesian_coordinates_tensor == 0):
        print(f"Error: All coordinates in {filename} are zero.")
        return False, 100, False

    mol_data = create_pyg_graph(cartesian_coordinates_tensor, atomic_numbers_tensor, xyz_filename=filename)

    mol_data = correct_edges(mol_data, scale_factor=scale_factor)

    num_node = mol_data.num_nodes
    edge_index = mol_data.edge_index

    is_connected, num_components = is_fully_connected(edge_index, num_node)

    match_n_degree = True
    if connector_dicts:
        for node_idx in range(num_node):
            if node_idx in connector_dicts:
                adjacent_nodes = edge_index[1][edge_index[0] == node_idx].tolist()

                node_degree = len(adjacent_nodes)

                if node_degree not in connector_dicts[node_idx]:
                    print(f"Error: Node {node_idx} has {node_degree} neighbors, expected {connector_dicts[node_idx]} in {filename}.")
                    match_n_degree = False
                    break

    return is_connected, num_components, match_n_degree


def xyz2mol_xtb(filename: str, charge: int, level: str, timeout: int) -> str | None:
    """
    Optimizes the geometry of a molecule from an XYZ file using xTB.

    This function attempts to optimize the molecule with a specified charge and
    xTB calculation level. If the optimization is successful, it moves the
    `xtbopt.xyz` output file to a new name based on the input filename.

    Args:
        filename (str): The path to the input XYZ file.
        charge (int): The molecular charge to use for the xTB calculation.
        level (str): The xTB calculation level (e.g., "gfn1", "gfn2", "gfn-ff").
        timeout (int): The maximum time in seconds to wait for the xTB process to complete.

    Returns:
        str | None: The path to the optimized XYZ file if successful,
                    otherwise None if xTB times out or fails to produce output.
    """
    execution_command = ["xtb", filename, "--opt", "crude", "-c", str(charge), f"-{level}"]

    try:
        sp.call(execution_command, stdout=sp.DEVNULL, stderr=sp.STDOUT, timeout=timeout)
    except sp.TimeoutExpired:
        print(f"xTB optimization timed out for {filename}.")
        return None

    if os.path.exists("xtbopt.xyz"):
        base_filename = os.path.basename(filename).split(".")[0]
        optimized_filename = f"{base_filename}_opt.xyz"
        shutil.move("xtbopt.xyz", optimized_filename)
        return optimized_filename
    else:
        print(f"xTB optimization failed to produce output for {filename}.")
        return None


def get_xtb_optimized_xyz(
    input_directory: str,
    output_directory: str = None,
    charge: int = -1,
    level: str = "gfn1",
    timeout: int = 240,
    scale_factor: float = 1.3,
    optimize_all: bool = True
) -> list[str]:
    """
    Optimizes all XYZ files in a given input directory using xTB and saves them
    to an output directory.

    This function iterates through all `.xyz` files, performs initial structural
    checks (connectivity, zero coordinates, and optional degree checks), and then
    attempts to optimize valid structures using `xyz2mol_xtb`. It skips files
    that already have an optimized counterpart in the output directory.

    Args:
        input_directory (str): The path to the directory containing the input XYZ files.
        output_directory (str, optional): The path to the directory where optimized
            XYZ files will be saved. If None, optimized files are saved in the
            `input_directory`. Defaults to None.
        charge (int, optional): The molecular charge to use for xTB optimizations. Defaults to -1.
        level (str, optional): The xTB calculation level (e.g., "gfn1", "gfn2", "gfn-ff"). Defaults to "gfn1".
        timeout (int, optional): The maximum time in seconds to wait for each xTB process. Defaults to 240.
        scale_factor (float, optional): The scaling factor for covalent radii in edge correction. Defaults to 1.3.
        optimize_all (bool, optional): If True, optimizes all files regardless of existing optimized versions.

    Returns:
        list[str]: A list of paths to the successfully optimized XYZ files.
    """
    if output_directory is None:
        output_directory = input_directory

    os.makedirs(output_directory, exist_ok=True)

    xyz_files = glob.glob(os.path.join(input_directory, "*.xyz"))
    optimized_files = []

    for xyz_file in tqdm(xyz_files, desc="Optimizing XYZ files", total=len(xyz_files)):
        output_file_path = os.path.join(output_directory, os.path.basename(xyz_file[:-4] + "_opt.xyz"))

        if os.path.exists(output_file_path):
            print(f"Skipping {xyz_file} as {output_file_path} already exists.")
            continue

        is_connected, num_components, match_n_degree = check_xyz(xyz_file, scale_factor=scale_factor)
        is_neutral = check_neutrality(xyz_file, charge=charge, timeout=timeout)    
        good_xyz =is_neutral and is_connected and (num_components == 1) and match_n_degree

        if good_xyz or optimize_all:
            optimized_file_basename = xyz2mol_xtb(xyz_file, charge, level, timeout)
            if optimized_file_basename is not None:
                shutil.move(optimized_file_basename, output_file_path)
                optimized_files.append(output_file_path)
            else:
                print(f"Optimization failed for {xyz_file}.")
        else:
            print(f"Error: {xyz_file} failed initial structural checks.")

    return optimized_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize XYZ files using xTB.")
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="Input directory containing XYZ files."
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for optimized XYZ files. Defaults to input directory if not provided."
    )
    parser.add_argument(
        "--charge",
        "-c",
        type=int,
        default=-1,
        help="Molecular charge for xTB optimization. Defaults to -1."
    )
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        default="gfn1",
        help="xTB calculation level (e.g., 'gfn1', 'gfn2', 'gfn-ff'). Defaults to 'gfn1'."
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=240,
        help="Maximum time in seconds for xTB processes to complete. Defaults to 240."
    )
    parser.add_argument(
        "--scale_factor",
        "-s",
        type=float,
        default=1.3,
        help="Scaling factor for covalent radii in edge correction. Defaults to 1.3."
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir is not None else input_dir

    optimized_files = get_xtb_optimized_xyz(
        input_dir,
        output_dir,
        charge=args.charge,
        level=args.level,
        timeout=args.timeout,
        scale_factor=args.scale_factor
    )

    print(f"Successfully optimized {len(optimized_files)} XYZ files and saved them in '{output_dir}'.")
