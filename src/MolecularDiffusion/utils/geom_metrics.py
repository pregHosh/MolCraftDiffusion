
import glob
import logging
import multiprocessing
import os
import subprocess as sp

import networkx as nx
import numpy as np
import pandas as pd
import torch
from MolecularDiffusion.device import get_device
from tqdm import tqdm

try:
    from ase.data import covalent_radii
except ImportError:
    covalent_radii = None
try:
    from openbabel import pybel
except ImportError:
    pybel = None
try:
    from posebusters import PoseBusters
except ImportError:
    PoseBusters = None
try:
    from rdkit import Chem
except ImportError:
    Chem = None
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_networkx
except ImportError:
    Data = None
    to_networkx = None

from .geom_constant import (
    allow_n_bonds,
    allowed_shape,
    degree_angles_ref,
    valid_valencies,
    vertices_labels,
)
from .geom_utils import correct_edges, create_pyg_graph, read_xyz_file
from .sascore import calculateScore

from .shape_measure import shape_measure


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("posebusters").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
SCORES_THRESHOLD = 3.0

EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2

# Canonical ShEPhERD constants
ALPHA_DEFAULT = 0.81
COULOMB_SCALING = 1e4 / (4 * 55.263 * np.pi)
LAM_SCALING = COULOMB_SCALING ** 2

# Pharmacophore types mapping and alphas
P_TYPES = [
    'Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 
    'Halogen', 'Cation', 'Anion', 'ZnBinder', 'Dummy'
]
P_ALPHAS = {
    0: 1.0, # Acceptor
    1: 1.0, # Donor
    2: 0.7, # Aromatic
    3: 0.7, # Hydrophobe
    4: 1.0, # Halogen
    5: 1.0, # Cation
    6: 1.0, # Anion
    7: 1.0, # ZnBinder
    8: 1.0  # Dummy
}


def _require_optional(name: str, value) -> None:
    if value is None:
        raise ImportError(
            f"{name} is required for this analysis. Install optional dependencies with: "
            "pip install 'molcraftdiffusion[analyze]'"
        )


#%%
def check_neutrality(filename):
    """
    Checks if a molecule (specified by an XYZ file) is neutral by running an xTB calculation.
    
    It runs 'xtb <filename> --ptb' and checks the output log for messages indicating
    a mismatch between electrons and spin multiplicity.
    
    Args:
        filename (str): Path to the XYZ file containing the molecule.
        
    Returns:
        bool: True if the molecule is neutral (no mismatch found), False
        otherwise, or None when xTB is unavailable or failed.
    """
    neutral_mol = True
    execution = ["xtb", filename, "--ptb"]
    lines = []
    try:
        with open("xtb.log", "w") as f:
            sp.call(execution, stdout=f, stderr=sp.STDOUT, timeout=60)
    except sp.TimeoutExpired:
        print("xTB calculation timed out.")
    except (FileNotFoundError, OSError) as e:
        # no xtb on PATH -- report unknown rather than killing the whole run
        print(f"xTB unavailable ({e}), skipping neutrality check.")
        return None

    if os.path.exists("xtb.log"):
        with open("xtb.log", "r") as f:
            lines = f.readlines()

    for line in lines:
        if "Number of electrons and spin multiplicity do not match" in line:
            neutral_mol = False
            break
    
    if os.path.exists("wbo"):
        os.remove("wbo")
    if os.path.exists("charges"):
        os.remove("charges")
    if os.path.exists("xtb.log"):
        os.remove("xtb.log")

    return neutral_mol


def compare_graph_topology(graph1, graph2):
    """
    Compare the topology of two PyG graphs by checking their edge indices.
    
    Args:
        graph1 (torch_geometric.data.Data): The first PyG graph.
        graph2 (torch_geometric.data.Data): The second PyG graph.
        
    Returns:
        bool: True if the graphs have the same topology, False otherwise.
    """
    edge_index1 = graph1.edge_index
    edge_index2 = graph2.edge_index
    
    # Sort the edge indices for comparison
    edge_index1_sorted = edge_index1[:, edge_index1[0].argsort()]
    edge_index2_sorted = edge_index2[:, edge_index2[0].argsort()]
    
    return torch.equal(edge_index1_sorted, edge_index2_sorted)



def is_fully_connected(edge_index, num_nodes):
    """
    Determines if the graph is fully connected.
    
    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        num_nodes (int): The number of nodes in the graph.
        
    Returns:
        tuple: (bool, int)
            - bool: True if the graph is fully connected, False otherwise.
            - int: The number of connected components in the graph.
    """
    _require_optional("torch_geometric", Data)
    _require_optional("torch_geometric", to_networkx)
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    try:
        is_connected = nx.is_connected(G)
        num_components = nx.number_connected_components(G)
    except nx.NetworkXPointlessConcept:
        is_connected = False
        num_components = 100
    return is_connected, num_components


def check_validity_v0(data, 
                   angle_relax=10,
                   scale_factor=1.3,
                   verbose=False):
    
    """
    Validate a molecular structure based on atomic distances and angles (Version 0).

    Args:
        data: A dictionary containing molecular information.
            - 'atomic_numbers': List of integers representing the atomic number of each atom.
            - 'positions': List of tuples (x, y, z) representing the position of each atom.
        angle_relax (float): Tolerance allowed for bond angles in degrees. Default is 10.0.
        scale_factor (float): The scaling factor to apply to the covalent radii. Default is 1.3.
        verbose (bool): Whether to print debug messages during validation. Default is False.

    Returns:
        tuple: Contains the following elements:
            - is_valid (bool): Boolean indicating if the structure is valid.
            - percent_atom_valid (float): Percentage of atoms that meet the criteria.
            - num_components (int): Number of connected components in the molecular graph.
            - bad_atoms (list): List of indices for atoms that do not meet the criteria.
            - needs_rechecking (bool): Whether further checks are needed due to borderline cases or special conditions.

    Notes:
        This function assumes that 'data' contains valid atomic numbers and positions. 
        The validation process involves checking bond distances and angles against predefined reference values.
        Special handling is applied for atoms with certain atomic numbers, such as carbon (atomic number 6), which may require different criteria due to their bonding behavior.

    """
    _require_optional("ase", covalent_radii)
    atomic_numbers = data.x.view(-1).int().tolist()
    edge_index = data.edge_index
    good_atoms = []
    bad_atoms = []
    
    to_be_recheck_flag = False # soem angles are too distorted
    all_node_pass_flag = True

    is_connected, num_components = is_fully_connected(data.edge_index, data.num_nodes)
    if verbose:
        print(f"Is fully connected: {is_connected}, Number of components: {num_components}")
    
    if num_components == 100:
        return False, 0, 100, torch.arange(0,data.num_nodes), False
        
    for node in range(len(atomic_numbers)):
        is_valid_node = True
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        n_degree = len(adjacent_nodes)
        if n_degree > 1:
            bond_angles = []
            for i in range(len(adjacent_nodes)):
                
                distance = torch.norm(data.pos[adjacent_nodes[i]] - data.pos[node]).item()
                r1 = covalent_radii[atomic_numbers[node]]*(1/scale_factor)
                r2 = covalent_radii[atomic_numbers[adjacent_nodes[i]]]*(1/scale_factor)
                min_dist_ref = r1 + r2
                if distance < min_dist_ref:
                    if verbose:
                        print(f"Bad distance Node {node} (Atomic Number: {atomic_numbers[node]}): Distance -> {distance:.2f}")
                    is_valid_node = False
                    all_node_pass_flag = False
                    
                    
                
                for j in range(i + 1, len(adjacent_nodes)):
                    node_i = adjacent_nodes[i]
                    node_j = adjacent_nodes[j]
                    
                    vec1 = data.pos[node_i] - data.pos[node]
                    vec2 = data.pos[node_j] - data.pos[node]
                    
                    cos_angle = torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))
                    angle = torch.acos(cos_angle).item() * (180.0 / 3.141592653589793)
                    bond_angles.append(angle)
                    # print(f"Bond angle between nodes {node}, {node_i}, and {node_j}: {angle:.2f} degrees")
            bond_angles = [angle for angle in bond_angles if not torch.isnan(torch.tensor(angle))]
            if len(bond_angles) > 0:
                average_bond_angle = sum(bond_angles) / len(bond_angles)
            else:
                average_bond_angle =0 
        else:
            bond_angles = [180.0]   
            average_bond_angle = 180.0
        ref_angle = degree_angles_ref.get(atomic_numbers[node], {}).get(n_degree, [])
        
        if ref_angle and is_valid_node:
            if any(abs(average_bond_angle - angle) <= angle_relax for angle in ref_angle):
                if any(abs(angle - ref) <= angle_relax*2 for angle in bond_angles for ref in ref_angle):                
                    is_valid_node = True
                else:
                    if verbose:
                        print(f"Bad angle Node {node} (Atomic Number: {atomic_numbers[node]}): Angle -> {average_bond_angle:.2f}")
                    is_valid_node = False
                    # Assume most false negatives are due to distorted angles (particularly n_degree = 2)
                    if n_degree > 1 and atomic_numbers[node] != 6 and all_node_pass_flag: # carbon is not influenced by electron pair
                        to_be_recheck_flag = True
            else:
                if verbose:
                    print(f"Bad angle Node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}, Angle -> {average_bond_angle:.2f}")
                is_valid_node = False
                if n_degree > 1 and atomic_numbers[node] != 6 and all_node_pass_flag:
                    to_be_recheck_flag = True
        else:
            all_node_pass_flag = False
            if verbose:
                print(f"Bad dist or not in stat node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}, Angle -> {average_bond_angle:.2f}, Distance -> {distance:.2f}")
            is_valid_node = False
        if is_valid_node:
            good_atoms.append(node)
        else:
            bad_atoms.append(node)
    percent_atom_valid = len(good_atoms) / len(atomic_numbers)
    if percent_atom_valid == 1:
        is_valid = True
    else:
        is_valid = False
    return is_valid, percent_atom_valid, num_components, bad_atoms, to_be_recheck_flag


def check_validity_v1(data, 
                   score_threshold=3,
                   scale_factor=1.3,
                   skip_indices=[],
                   verbose=False):
    
    """
    Validate a molecular structure based on atomic distances and angles (Version 1).

    Args:
        data: A dictionary containing molecular information.
            - 'atomic_numbers': List of integers representing the atomic number of each atom.
            - 'positions': List of tuples (x, y, z) representing the position of each atom.
        score_threshold (float): Tolerance allowed for shape values. Default is 3.0.
        scale_factor (float): The scaling factor to apply to the covalent radii. Default is 1.3.
        skip_indices (list): List of atom indices to skip during validation.
        verbose (bool): Whether to print debug messages during validation. Default is False.

    Returns:
        tuple: Contains the following elements:
            - is_valid (bool): Boolean indicating if the structure is valid.
            - percent_atom_valid (float): Percentage of atoms that meet the criteria.
            - num_components (int): Number of connected components in the molecular graph.
            - bad_atom_chem (list): List of indices for atoms that do not meet chemical valency criteria.
            - bad_atom_distort (list): List of indices for atoms that are geometrically distorted.

    Notes:
        This function assumes that 'data' contains valid atomic numbers and positions. 
        The validation process involves checking bond distances and angles against predefined reference values.
        Special handling is applied for atoms with certain atomic numbers, such as carbon (atomic number 6), which may require different criteria due to their bonding behavior.

    """
    
    _require_optional("ase", covalent_radii)
    atomic_numbers = data.x.view(-1).int().tolist()
    edge_index = data.edge_index
    good_atoms = []
    bad_atom_distort = []
    bad_atom_chem = []
    
    is_connected, num_components = is_fully_connected(data.edge_index, data.num_nodes)
    if verbose:
        print(f"Is fully connected: {is_connected}, Number of components: {num_components}")
    
    if num_components == 100:
        return False, 0, 100, torch.arange(0,data.num_nodes), torch.arange(0,data.num_nodes)
       
    for node in range(len(atomic_numbers)):
        
        if node in skip_indices:
            continue
        shp_scores = {}
        is_valid_node = True
        adjacent_nodes = edge_index[1][edge_index[0] == node].tolist()
        n_degree = len(adjacent_nodes)
        
        # 1 check number of bonds
        if n_degree not in allow_n_bonds.get(atomic_numbers[node], []):
            bad_atom_chem.append(node)
            if verbose:
                print(f"Bad atom Node {node} (Atomic Number: {atomic_numbers[node]}): Degree -> {n_degree}")
            is_valid_node = False
            
            
        # 2 check bond lengths
        if is_valid_node:
            for i in range(len(adjacent_nodes)):
                
                distance = torch.norm(data.pos[adjacent_nodes[i]] - data.pos[node]).item()
                r1 = covalent_radii[atomic_numbers[node]]*(1/scale_factor)
                r2 = covalent_radii[atomic_numbers[adjacent_nodes[i]]]*(1/scale_factor)
                min_dist_ref = r1 + r2
                if distance < min_dist_ref:
                    if verbose:
                        print(f"Bad distance Node {node} (Atomic Number: {atomic_numbers[node]}): Distance -> {distance:.2f}")
                    bad_atom_distort.append(node)
                    is_valid_node = False
                
        # 3 check the shape of the molecule
        if is_valid_node:
            nodes_all = [node] + adjacent_nodes

            positions = data.pos[nodes_all].tolist()
            if len(adjacent_nodes) > 1:
                shp_types = vertices_labels[len(adjacent_nodes)]
                for shp_type in shp_types:
                    shp_scores[shp_type] = shape_measure(
                        positions, shp_type, central_atom=1
                    )


                shp_type = min(shp_scores, key=shp_scores.get)
                shp_score = shp_scores[shp_type]
                ref_shp = allowed_shape[atomic_numbers[node]]
                if shp_type not in ref_shp or shp_score > score_threshold:
                    bad_atom_distort.append(node)
                    is_valid = False
                    if verbose:
                        print(f"Bad shape Node {node} (Atomic Number: {atomic_numbers[node]}): Shape -> {shp_type}")
                else:
                    good_atoms.append(node)
            else:
                good_atoms.append(node)
    # denominator counts only the atoms actually examined, otherwise every
    # skipped atom is scored as invalid (biases --skip-atoms runs low)
    n_examined = len(atomic_numbers) - len(
        {i for i in skip_indices if 0 <= i < len(atomic_numbers)}
    )
    percent_atom_valid = len(good_atoms) / n_examined if n_examined > 0 else 0.0
    if len(bad_atom_chem) == 0  and len(bad_atom_distort) == 0:
        is_valid = True   
    else:
        is_valid = False        

    return (is_valid, percent_atom_valid, num_components, bad_atom_chem, bad_atom_distort)
    
def check_chem_validity(mol_list, 
                        skip_idx=[],
                        verbose=0):
    _require_optional("rdkit", Chem)
    """
    Analyze a list of RDKit molecules for chemical validity based on atom valencies
    and identify broken or disconnected fragments.

    Parameters
    ----------
    mol_list : list of rdkit.Chem.Mol
        A list of RDKit Mol objects to be checked.
    skip_idx : list of int, optional
        Atom indices to skip when counting and checking (currently unused in logic;
        provided for future extension). Default is None.
    verbose : int, default=0
        If > 0, prints detailed information about each valency violation as:
        "<SMILES> has invalid valency: <AtomSymbol> <TotalValence> <FormalCharge>".

    Returns
    -------
    natom_stability_dicts : dict[str, int]
        Counts of atoms whose total electron count (valence minus formal charge)
        matches a known valid valency (i.e. “stable” atoms), keyed by atom symbol.
    natom_tot_dicts : dict[str, int]
        Total counts of atoms encountered, keyed by atom symbol.
    good_smiles : list of str
        Canonical SMILES strings for molecules deemed chemically valid and not
        containing disconnected fragments (".").
    bad_smiles_broken : list of str
        Canonical SMILES for molecules that are chemically valid but contain
        disconnected fragments (i.e. salts or mixtures with "." in the SMILES).
    bad_smiles_chem : list of str
        Canonical SMILES for molecules that failed the valency checks.

    Notes
    -----
    - This function relies on a pre-defined mapping `valid_valencies`:
        >>> valid_valencies = {
        ...     'H': {1}, 'C': {4}, 'N': {3, 5}, 'O': {2}, ...
        ... }
      which maps each element symbol to the set of permitted electron counts.
    - The `skip_idx` argument is currently not applied in the loop; if you wish
      to ignore certain atoms (for example, metals or explicit hydrogens), you
      could uncomment and adapt the skip logic.

    Example
    -------
    >>> from rdkit.Chem import MolFromSmiles
    >>> smiles_list = ['CCO', 'C[N+](C)(C)C', 'C.C']  # ethanol, tetramethylammonium, disconnected C and C
    >>> mols = [MolFromSmiles(s) for s in smiles_list]
    >>> valid, total, good, broken, bad = check_chem_validity(mols, verbose=1)
     # would print any valency errors if present
    """
    natom_stability_dicts = {}
    natom_tot_dicts = {}

    bad_smiles_broken = []
    bad_smiles_chem = []
    good_smiles = []

    for mol in mol_list:
        is_chem_valid = True

        for atom in mol.GetAtoms():
            if atom.GetIdx() in skip_idx:
                continue
            symbol = atom.GetSymbol()
            if symbol not in natom_tot_dicts:
                natom_tot_dicts[symbol] = 1
            else:
                natom_tot_dicts[symbol] += 1

            nelectron = atom.GetTotalValence() - atom.GetFormalCharge()
            if nelectron in valid_valencies[symbol]:
                if symbol not in natom_stability_dicts:
                    natom_stability_dicts[symbol] = 1
                else:
                    natom_stability_dicts[symbol] += 1

            else:
                is_chem_valid = False
                if verbose > 0:
                    smiles = Chem.MolToSmiles(mol,canonical=True)
                    print(
                        f"{smiles} has invalid valency: {symbol} {atom.GetTotalValence()} {atom.GetFormalCharge()}"
                    )

        smiles = Chem.MolToSmiles(mol, canonical=True)
        if is_chem_valid:
            # check for broken molecules
            if "." not in smiles:
                good_smiles.append(smiles)
            else:
                bad_smiles_broken.append(smiles)
        else:
            bad_smiles_chem.append(smiles)

    return (
        natom_stability_dicts,
        natom_tot_dicts,
        good_smiles,
        bad_smiles_broken,
        bad_smiles_chem,
    )
    
def smilify_wrapper(xyzs, xyz2mol):
    """
    Convert a list of XYZ files to SMILES strings using a provided xyz2mol function.
    
    Args:
        xyzs (list of str): List of paths to XYZ files.
        xyz2mol (callable): A function that takes an XYZ file path and returns (smiles, mol).
        
    Returns:
        tuple: (validity, smiles_list, mol_list, dicts)
            - validity (float): Fraction of successful conversions.
            - smiles_list (list of str): List of SMILES strings (None for failures).
            - mol_list (list of RDKit Mol): List of RDKit Mol objects (None for failures).
            - dicts (dict): Dictionary with 'smiles' and 'filename' lists.
    """
    smiles_list = []
    mol_list = []
    dicts = {"smiles": [], "filename": []}
    for xyz in xyzs:
        try:
            (
                smiles,
                mol,
            ) = xyz2mol(xyz)
        except Exception as e:
            print("Error in converting {xyz} to smiles due to", e)
            smiles = None
            mol = None

        if smiles is None:
            continue

        # Canonicalize the SMILES
        # smiles = Chem.MolToSmiles(mol, canonical=True)
        mol_list.append(mol)
        dicts["smiles"].append(smiles)
        dicts["filename"].append(xyz)
        smiles_list.append(smiles)

    validity = len(smiles_list) / len(xyzs)
    return validity, smiles_list, mol_list, dicts

#%% postbuster

def xyz_to_pdb(xyz_file_path, pdb_file_path):
    """
    Converts an XYZ file to a PDB file using OpenBabel (via pybel).
    
    Args:
        xyz_file_path (str): Path to input XYZ file.
        pdb_file_path (str): Path to output PDB file.
    """
    _require_optional("openbabel", pybel)
    try:
        mol = next(pybel.readfile("xyz", xyz_file_path))
        mol.write("pdb", pdb_file_path, overwrite=True)
    except Exception as e:
        logger.error(f"Error converting {xyz_file_path} to PDB: {e}")


def load_molecules_from_xyz(xyz_dir):
    """
    Converts all XYZ files in a directory to RDKit Mol objects.
    
    It first converts XYZ files to PDB using OpenBabel, then loads the PDBs into RDKit.
    
    Args:
        xyz_dir (str): Directory containing XYZ files.
        
    Returns:
        tuple: (valid_molecules, pass_xyz_files)
            - valid_molecules (list of RDKit Mol): Successfully loaded molecules.
            - pass_xyz_files (list of str): Filenames of the successfully loaded molecules.
    """
    _require_optional("rdkit", Chem)
    _require_optional("openbabel", pybel)
    xyz_files = glob.glob(os.path.join(xyz_dir, "*.xyz"))
    valid_molecules = []
    pass_xyz_files = []

    for xyz_file in xyz_files:
        try:
            pdb_file = os.path.splitext(xyz_file)[0] + ".pdb"
            xyz_to_pdb(xyz_file, pdb_file)
            mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
            if mol is not None:
                valid_molecules.append(mol)
                pass_xyz_files.append(xyz_file)

        except Exception as e:
            logger.warning(f"Skipping {xyz_file} due to error: {e}")

    total = len(xyz_files)
    success = len(valid_molecules)
    percent = (success / total) * 100 if total > 0 else 0.0

    logger.info(f"Successfully converted {success} out of {total} XYZ files ({percent:.2f}%).")

    return valid_molecules, pass_xyz_files



def _run_buster(mols, queue):
    """
    Helper function to run PoseBusters on a list of molecules in a separate process.
    
    Args:
        mols (list of RDKit Mol): List of molecules to process.
        queue (multiprocessing.Queue): Queue to put the result (DataFrame or Exception).
    """
    _require_optional("posebusters", PoseBusters)
    try:
        buster = PoseBusters(config="mol")
        results = buster.bust(mols)
        queue.put(results)
    except Exception as e:
        queue.put(e)


def run_postbuster(mols, timeout=60, batch_size=1, names=None):
    """
    Run PoseBusters on a list of RDKit molecules, optionally in batches, with a timeout per batch.
    
    This function processes molecules using the PoseBusters library to compute various geometric checks.
    Processing happens in separate processes to enforce timeouts.
    
    Args:
        mols (list of RDKit Mol): List of molecules to evaluate.
        timeout (int, optional): Maximum time (in seconds) allowed for each batch calculation. Default is 60.
        batch_size (int, optional): Number of molecules to process in a single batch. 
                                    If None, processes all molecules in one batch. Default is None.
        names (list of str, optional): One label per molecule. When given, every row gets a
                                    ``filename`` column, and a batch that fails or times out
                                    yields one row per molecule (checks NaN, ``posebuster_error``
                                    set) instead of being dropped. Default None keeps the old
                                    drop-the-batch behaviour that training eval relies on.
                                    
    Returns:
        pd.DataFrame or None: DataFrame containing PoseBusters results for all processed molecules.
                              Returns None if no results could be obtained.
    """
    _require_optional("posebusters", PoseBusters)
    if not mols:
        logger.warning("No valid molecules loaded. Exiting.")
        return None

    if batch_size is None:
        batch_size = len(mols)

    all_results = []
    n_succeeded = 0
    num_batches = (len(mols) + batch_size - 1) // batch_size

    def _keep_failed(i, error):
        # without a row the caller's per-molecule columns misalign and crash
        if names is not None:
            all_results.append(pd.DataFrame({
                "filename": names[i * batch_size : (i + 1) * batch_size],
                "posebuster_error": error,
            }))
    
    for i in tqdm(range(num_batches), desc="Processing PoseBusters batches"):
        batch_mols = mols[i * batch_size : (i + 1) * batch_size]
        
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=_run_buster, args=(batch_mols, queue))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            logger.warning(f"PoseBusters timed out after {timeout} seconds (Batch {i+1}/{num_batches}). Skipping batch.")
            _keep_failed(i, f"timed out after {timeout} s")
            continue
        
        try:
            result = queue.get(timeout=5)
        except Exception:
            logger.error(f"PoseBusters failed to return result (Batch {i+1}/{num_batches}). Skipping batch.")
            _keep_failed(i, "no result returned")
            continue

        if isinstance(result, Exception):
            logger.error(f"PoseBusters failed with an exception: {result}. Skipping batch.")
            _keep_failed(i, f"{type(result).__name__}: {result}")
            continue
        
        if names is not None:
            result["filename"] = names[i * batch_size : (i + 1) * batch_size]
        all_results.append(result)
        n_succeeded += 1

    # no batch ran: None as before, there are no check columns to report
    if n_succeeded == 0:
        return None

    try:
        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    except Exception as e:
        logger.error(f"Error concatenating batch results: {e}")
        return None


def xyz_to_rdkit_mol(xyz_path: str, charges: list = None):
    """
    Convert an XYZ file to an RDKit molecule with perceived bonds.
    Tries multiple charges to find a valid structure without radicals.

    Args:
        charges: candidate total charges to try, in order. Defaults to
            ``[0, 1, -1, 2, -2]`` (a blind sweep); pass e.g. ``[total_charge]``
            to try only a single, already-known charge instead of sweeping.
    """
    _require_optional("rdkit", Chem)
    from rdkit.Chem import rdDetermineBonds
    try:
        with open(xyz_path, "r") as f:
            xyz_block = f.read()
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None

        # Try different charges to perceive bonds correctly
        for charge in (charges if charges is not None else [0, 1, -1, 2, -2]):
            try:
                mol_copy = Chem.Mol(mol)
                rdDetermineBonds.DetermineBonds(mol_copy, charge=charge, embedChiral=True)
                Chem.SanitizeMol(mol_copy)
                if any(atom.GetNumRadicalElectrons() > 0 for atom in mol_copy.GetAtoms()):
                    continue
                smiles = Chem.MolToSmiles(mol_copy)
                if "." in smiles:
                    continue
                return mol_copy
            except:
                continue
        return None
    except Exception as e:
        logger.error(f"Error in xyz_to_rdkit_mol for {xyz_path}: {e}")
        return None

def compute_drug_likeness(mol) -> dict:
    """Compute RDKit-based drug-likeness metrics including SAScore and QED."""
    _require_optional("rdkit", Chem)
    from rdkit.Chem import QED, Descriptors, rdMolDescriptors
    if mol is None:
        return {}
    return {
        # SA is defined on the heavy-atom graph (explicit H atoms would
        # inflate it by ~5 points), same as the db `sascore` annotation.
        "SA_score": calculateScore(Chem.RemoveHs(mol)),
        "QED": QED.qed(mol),
        "LogP": Descriptors.MolLogP(mol),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "MW": Descriptors.MolWt(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol)
    }


def _pca_axes(pts: np.ndarray) -> np.ndarray:
    """Return (3,3) matrix whose rows are the principal axes of pts (sorted by decreasing variance)."""
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt  # rows are principal axes, descending singular value order


def pca_align(pts_fit: np.ndarray, pts_ref: np.ndarray) -> np.ndarray:
    """Rotate pts_fit to align its principal axes with those of pts_ref.

    Works for point clouds of **different sizes** (no correspondence required).
    Tries all 4 axis-flip combinations and returns the rotation that maximises
    the sum of dot products between corresponding principal axes (best sign match).

    Both inputs must already be centered at the origin.

    Parameters
    ----------
    pts_fit : np.ndarray (N, 3)  — points to rotate (pre-centered)
    pts_ref : np.ndarray (M, 3)  — reference points (pre-centered)

    Returns
    -------
    pts_fit_rotated : np.ndarray (N, 3)
    """
    axes_ref = _pca_axes(pts_ref)  # (3, 3) rows = principal axes of ref
    axes_fit = _pca_axes(pts_fit)  # (3, 3) rows = principal axes of fit

    # Try the 4 sign combinations (flip 1st and/or 2nd axis; 3rd is determined by right-hand rule)
    best_score = -np.inf
    best_R = np.eye(3)
    for s0 in (1, -1):
        for s1 in (1, -1):
            ax_ref = axes_ref.copy()
            ax_ref[0] *= s0
            ax_ref[1] *= s1
            ax_ref[2] = np.cross(ax_ref[0], ax_ref[1])  # ensure right-handed frame
            # R maps fit axes to ref axes: R = axes_ref.T @ axes_fit
            R = ax_ref.T @ axes_fit
            score = np.sum(np.diag(axes_ref @ R @ axes_fit.T))
            if score > best_score:
                best_score = score
                best_R = R
    return pts_fit @ best_R.T  # (N, 3)


def kabsch_align(pts_fit: np.ndarray, pts_ref: np.ndarray) -> np.ndarray:
    """
    Standard Kabsch alignment for 1:1 corresponding point sets.
    Returns pts_fit rotated and translated to align with pts_ref.
    """
    if len(pts_fit) != len(pts_ref):
        # Fallback to PCA alignment if sizes differ (Kabsch requires 1:1)
        return pca_align(pts_fit - pts_fit.mean(axis=0), pts_ref - pts_ref.mean(axis=0)) + pts_ref.mean(axis=0)

    c_fit = pts_fit.mean(axis=0)
    c_ref = pts_ref.mean(axis=0)
    pts_fit_c = pts_fit - c_fit
    pts_ref_c = pts_ref - c_ref

    H = pts_fit_c.T @ pts_ref_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    return (pts_fit_c @ R.T) + c_ref
