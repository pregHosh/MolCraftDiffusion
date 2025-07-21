import subprocess as sp
import numpy as np
import os
import pandas as pd
import glob
from morfeus import read_xyz, BuriedVolume, Sterimol
import tqdm
from itertools import compress
import warnings
import networkx as nx
from ase.data import covalent_radii, atomic_numbers
import ase
from collections import Counter

# --- Global Constants and Look-up Tables ---
ATOM2Z = atomic_numbers
COVALENT_RADII = covalent_radii

kier_radii = {
    (6, 4): 0.77, (6, 3): 0.67, (6, 2): 0.6,
    (7, 3): 0.74, (7, 2): 0.62, (7, 1): 0.55,
    (8, 2): 0.74, (8, 1): 0.62,
    (9, 1): 0.72,
    (15, 5): 1.1, (15, 4): 1.1, (15, 3): 1.1, (15, 2): 1.0, (15, 1): 0.95,
    (16, 4): 1.04, (16, 3): 1.04, (16, 2): 1.04, (16, 1): 0.94,
    (17, 1): 0.99,
    (35, 1): 1.14,
    (53, 1): 1.33,
}

# Default parameters for buried volume and electronic analyses
M_IDX = 1  # Metal atom index (1-based)
L_IDX = 0  # Ligand index (not directly used in provided code, but kept for context)
XZ_ATOMS = [9] # Atoms defining the XZ plane for buried volume (1-based)
Z_ATOMS = [1]  # Atoms defining the Z axis for buried volume (1-based)
TIMEOUT = 30 # Default timeout for subprocess calls (seconds)
RADIUS = 5   # Default radius for buried volume calculation (Angstroms)


# --- Rigidity and Complexity Descriptors ---

def remove_hydrogen_from_graph(graph: nx.Graph, copy_graph: bool = True) -> nx.Graph:
    """
    Removes all hydrogen atoms (atomic number = 1) from a molecular graph.

    Args:
        graph (networkx.Graph): The input molecular graph. Nodes are expected to have
                                an 'atomic_number' attribute.
        copy_graph (bool): If True, returns a new graph (a copy of the subgraph).
                           If False, returns a view of the subgraph. Defaults to True.

    Returns:
        networkx.Graph: A subgraph containing only non-hydrogen atoms.
    """
    non_hydrogens = [node for node, data in graph.nodes(data=True) if data.get("atomic_number") != 1]
    subgraph = graph.subgraph(non_hydrogens)
    
    if copy_graph:
        g_sub = subgraph.copy()
    else:
        g_sub = subgraph

    g_sub.graph["graph_type"] = "anydrogram"
    return g_sub


def compute_kier_alpha(graph: nx.Graph, radii_data: dict = ase.data.covalent_radii, mode: str = "a") -> float:
    """
    Computes the alpha correction factor for Kier indices based on atomic radii or bond lengths.

    Args:
        graph (networkx.Graph): The molecular graph. Nodes must have 'atomic_number' attribute.
                                Edges must have 'distance' attribute if mode 'b' is used.
        radii_data (dict): Dictionary of covalent radii (e.g., ase.data.covalent_radii).
                           Defaults to ASE covalent radii.
        mode (str): The mode of alpha computation:
                    - 'a': Based on atomic radii relative to carbon (sp3).
                    - 'b': Based on bond lengths relative to C-C sp3 bond.
                    - 'legacy': Uses specific Kier radii and requires 'plerogram' graph type.
                                This mode is for uncharged/non-radical molecules.

    Returns:
        float: The alpha correction value.

    Raises:
        NotImplementedError: If an unsupported mode is provided.
        AssertionError: In 'legacy' mode, if the graph type is not 'plerogram'.
    """
    k_alpha = 0.0

    if mode in ["a", "b"]:
        if graph.graph.get("graph_type") == "plerogram":
            warnings.warn("The graph type is 'plerogram'. This may lead to undesired results for Kier descriptors.")
    
    if mode == "a":
        for node_id in graph.nodes():
            atomic_num = graph.nodes[node_id]["atomic_number"]
            k_alpha += (radii_data.get(atomic_num, radii_data[6]) / radii_data[6]) - 1
    elif mode == "b":
        for u, v, data in graph.edges(data=True):
            bond_distance = data.get("distance")
            if bond_distance is None:
                warnings.warn(f"Edge ({u}, {v}) missing 'distance' attribute, skipping for Kier alpha mode 'b'.")
                continue
            k_alpha += (bond_distance / 1.535) - 1
    elif mode == "legacy":
        warnings.warn("Legacy mode. Use only for uncharged/non-radical molecules.")
        assert graph.graph.get("graph_type") == "plerogram", f"Plerogram required to compute hybridization, got '{graph.graph.get('graph_type')}'."
        for node_id in graph.nodes():
            atomic_num = graph.nodes[node_id]["atomic_number"]
            if atomic_num != 1:
                coord_num = len(list(graph.neighbors(node_id)))
                try:
                    k_alpha += (kier_radii[(atomic_num, coord_num)] / kier_radii[(6, 4)]) - 1
                except KeyError:
                    warnings.warn(f"Atomic number '{atomic_num}' with coordination '{coord_num}' not tabulated. Using sp3 carbon (6,4) for alpha correction.")
                    k_alpha += (kier_radii[(6, 4)] / kier_radii[(6, 4)]) - 1
    else:
        raise NotImplementedError(f"Unsupported mode '{mode}'. Choose from 'a', 'b', or 'legacy'.")

    return k_alpha


def molecular_shannon_entropy(graph: nx.Graph) -> float:
    """
    Computes the Shannon entropy of the atom types in a molecular graph.
    Atom types are defined by (atomic_number, coordination_number).

    Args:
        graph (networkx.Graph): The molecular graph. Nodes must have 'atomic_number' attribute.

    Returns:
        float: The Shannon entropy value.
    """
    atom_types = []
    num_nodes = graph.number_of_nodes()
    
    if num_nodes == 0:
        return 0.0

    for node_id in graph.nodes():
        atomic_num = graph.nodes[node_id]["atomic_number"]
        coordination_num = len(list(graph.neighbors(node_id)))
        atom_types.append((atomic_num, coordination_num))
    
    frequencies = Counter(atom_types)
    
    shannon_entropy = 0.0
    for count in frequencies.values():
        probability = count / num_nodes
        if probability > 0:
            shannon_entropy -= probability * np.log10(probability)
            
    return shannon_entropy


def _find_simple_paths(graph: nx.Graph, path_length: int) -> list[list[int]]:
    """
    Finds all unique simple paths of a given length in the graph.
    A simple path does not repeat any nodes.

    Args:
        graph (networkx.Graph): The molecular graph.
        path_length (int): The desired length of the paths (number of edges).

    Returns:
        list[list[int]]: A list of paths, where each path is a list of node indices.
                         Paths are returned such that the first node in the path is
                         always smaller than the second node in the path to ensure uniqueness
                         for undirected graphs (e.g., [0,1,2] is same as [2,1,0]).
    """
    all_paths = []

    if path_length < 0:
        raise ValueError("Path length must be non-negative.")
    elif path_length == 0:
        return [[node] for node in graph.nodes()]
    
    for start_node in graph.nodes():
        for path in nx.all_simple_paths(graph, source=start_node, cutoff=path_length):
            if len(path) == path_length + 1:
                if path[0] < path[-1]:
                    all_paths.append(path)
                elif path[0] == start_node:
                    pass
                
    unique_paths = list(map(list, set(tuple(sorted(path)) for path in all_paths)))
    
    return unique_paths


def compute_kier_kappa(graph: nx.Graph, m: int, alpha: bool = False, mode: str = "a") -> float:
    """
    Computes the m-th order Kier kappa shape index.

    Args:
        graph (networkx.Graph): The molecular graph.
        m (int): Order of kappa index (0, 1, 2, or 3).
                 - m=0: Molecular Shannon entropy.
                 - m=1: Kappa 1 index (related to number of edges).
                 - m=2: Kappa 2 index (related to paths of length 2).
                 - m=3: Kappa 3 index (related to paths of length 3).
        alpha (bool): Whether to include alpha correction. Defaults to False.
        mode (str): Mode for alpha correction ('a', 'b', 'legacy'). Defaults to 'a'.

    Returns:
        float: The m-th order kappa shape index.

    Raises:
        NotImplementedError: If an invalid 'm' order is provided.
        AssertionError: If 'm=3' is used with a graph having fewer than 3 atoms.
    """
    if alpha:
        alpha_val = compute_kier_alpha(graph, mode=mode)
    else:
        alpha_val = 0.0

    if mode == "legacy":
        graph_processed = remove_hydrogen_from_graph(graph)
    else:
        graph_processed = graph

    num_atoms_effective = graph_processed.number_of_nodes() + alpha_val

    if m == 0:
        return molecular_shannon_entropy(graph_processed) * graph_processed.number_of_nodes()
    elif m == 1:
        numerator = (num_atoms_effective) * (num_atoms_effective - 1) ** 2
        p_val = graph_processed.number_of_edges()
    elif m == 2:
        numerator = (num_atoms_effective - 1) * (num_atoms_effective - 2) ** 2
        paths_of_length_2 = _find_simple_paths(graph_processed, m)
        p_val = len(paths_of_length_2)
    elif m == 3:
        if num_atoms_effective < 3:
             raise AssertionError(f"Kappa 3 needs at least 3 effective atoms, got '{num_atoms_effective:.2f}'.")
        warnings.warn("Kappa 3 may not work reliably with cyclopropanes or very small rings.")
        if int(num_atoms_effective) % 2 == 0:
            numerator = (num_atoms_effective - 3) * ((num_atoms_effective - 2) ** 2)
        else:
            numerator = (num_atoms_effective - 1) * ((num_atoms_effective - 3) ** 2)
        paths_of_length_3 = _find_simple_paths(graph_processed, m)
        p_val = len(paths_of_length_3)
    else:
        raise NotImplementedError(f"Invalid 'm' order '{m}'. Supported orders are 0, 1, 2, 3.")
    
    if p_val + alpha_val == 0:
        return 0.0
    
    return numerator / ((p_val + alpha_val) ** 2)


def compute_kier_phi(graph: nx.Graph, alpha: bool = False, mode: str = "a") -> float:
    """
    Computes the Kier phi descriptor of a molecule.
    This descriptor quantifies molecular shape and branching in a size-normalized way.
    A higher `kier_phi` value suggests a more branched and/or complex molecule,
    while a lower value indicates a more linear or simple structure.

    Args:
        graph (networkx.Graph): The molecular graph.
        alpha (bool): Whether to include alpha correction in kappa calculations. Defaults to False.
        mode (str): Mode for alpha correction ('a', 'b', 'legacy'). Defaults to 'a'.

    Returns:
        float: The Kier phi descriptor value.
    """
    if mode == "legacy":
        num_atoms_for_phi = remove_hydrogen_from_graph(graph).number_of_nodes()
    else:
        num_atoms_for_phi = graph.number_of_nodes()
    
    if num_atoms_for_phi == 0:
        return 0.0

    kappa1 = compute_kier_kappa(graph, 1, alpha=alpha, mode=mode)
    kappa2 = compute_kier_kappa(graph, 2, alpha=alpha, mode=mode)
    
    return (kappa1 * kappa2) / num_atoms_for_phi


def xyz_to_networkx_graph(xyz_string: str, graph_type: str = "kenogram", bond_factor: float = 1.2) -> nx.Graph:
    """
    Converts an XYZ format string into a NetworkX graph representation of a molecule.
    Bonds are identified based on interatomic distances and covalent radii.

    Args:
        xyz_string (str): The content of an XYZ file as a string.
        graph_type (str): A label to assign to the graph's 'graph_type' attribute.
                          Defaults to 'kenogram'.
        bond_factor (float): A multiplier applied to the sum of covalent radii
                             to determine the maximum allowed distance for a bond.
                             A value > 1.0 allows for slightly longer bonds. Defaults to 1.2.

    Returns:
        networkx.Graph: A molecular graph where:
                        - Nodes are labeled by integer indices.
                        - Node attributes include 'atomic_number' (integer) and 'element' (string).
                        - Edges represent bonds, with 'bond_order' (default 1.0) and 'distance' attributes.
    Raises:
        ValueError: If the first line of the XYZ string is not a valid number of atoms.
    """
    lines = xyz_string.strip().splitlines()
    if not lines:
        return nx.Graph()

    try:
        num_atoms = int(lines[0])
    except ValueError:
        raise ValueError("First line of XYZ string must be the number of atoms.")
    
    if num_atoms == 0:
        return nx.Graph()

    atom_lines = lines[2:2+num_atoms]

    atoms_elements = []
    atom_coords = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            warnings.warn(f"Skipping malformed line in XYZ string: {line}")
            continue
        atom_symbol = parts[0]
        try:
            position = list(map(float, parts[1:4]))
        except ValueError:
            warnings.warn(f"Skipping line with non-numeric coordinates: {line}")
            continue
        atoms_elements.append(atom_symbol)
        atom_coords.append(position)
    
    atom_coords = np.array(atom_coords)

    graph = nx.Graph()
    for i, atom_symbol in enumerate(atoms_elements):
        graph.add_node(i, atomic_number=ATOM2Z.get(atom_symbol, 0), element=atom_symbol)

    for i in range(len(atoms_elements)):
        for j in range(i + 1, len(atoms_elements)):
            atomic_num_i = graph.nodes[i]["atomic_number"]
            atomic_num_j = graph.nodes[j]["atomic_number"]
            
            if atomic_num_i == 0 or atomic_num_j == 0:
                continue
            
            covalent_radius_sum = COVALENT_RADII[atomic_num_i] + COVALENT_RADII[atomic_num_j]
            distance = np.linalg.norm(atom_coords[i] - atom_coords[j])
            
            if distance <= bond_factor * covalent_radius_sum:
                graph.add_edge(i, j, bond_order=1.0, distance=distance)

    graph.graph["graph_type"] = graph_type
    return graph


def compute_crest_flexibility(graph: nx.Graph, bond_order_label: str = "bond_order") -> float:
    """
    Computes the flexibility score of a molecule from its graph, based on the CREST method.

    The flexibility is computed as:
        Flexibility = sqrt((1/m) * sum(val^2 for all bonds))
    where 'val' for each bond depends on branching, ring membership, bond order, and hybridization.

    Args:
        graph (networkx.Graph): The molecular graph. Nodes should have 'atomic_number'
                                and edges should have 'bond_order_label' attribute.
        bond_order_label (str): The name of the edge attribute representing bond order.
                                Defaults to 'bond_order'.

    Returns:
        float: The flexibility score. Returns 0.0 if the graph has no edges.
    """
    if graph.graph.get("graph_type") != "kenogram":
        warnings.warn("By definition, 'compute_crest_flexibility' works best on kenograms.")
    
    if not nx.get_edge_attributes(graph, bond_order_label) or bond_order_label == "":
        nx.set_edge_attributes(graph, 0.0, bond_order_label)
        warnings.warn(f"No bond orders found with label '{bond_order_label}'. Defaulting to 0.0 for all edges.")
    
    cycles = nx.cycle_basis(graph)
    edges = list(graph.edges())
    num_edges = len(edges)
    
    if num_edges == 0:
        return 0.0

    sum_val_squared = 0.0
    for edge in edges:
        node1, node2 = edge
        
        coord_num1 = len(list(graph.neighbors(node1)))
        coord_num2 = len(list(graph.neighbors(node2)))
        
        hybf = 1.0
        if graph.nodes[node1]['atomic_number'] == 6 and coord_num1 < 4:
            hybf *= 0.5
        if graph.nodes[node2]['atomic_number'] == 6 and coord_num2 < 4:
            hybf *= 0.5
        
        bond_order = graph.edges[edge].get(bond_order_label, 1.0)
        doublef = 1.0 - np.exp(-4.0 * (bond_order - 2.0)**6)
        
        branch = 2.0 / np.sqrt(coord_num1 * coord_num2) if coord_num1 > 0 and coord_num2 > 0 else 0.0
        
        ringf = 1.0
        is_in_ring = False
        for cycle in cycles:
            if node1 in cycle and node2 in cycle:
                is_in_ring = True
                break
        
        if is_in_ring:
            k_min_ring_size = 0
            ring_sizes_containing_edge = []
            for cycle in cycles:
                if node1 in cycle and node2 in cycle:
                    ring_sizes_containing_edge.append(len(cycle))
            if ring_sizes_containing_edge:
                k_min_ring_size = min(ring_sizes_containing_edge)
                ringf = 0.5 * (1.0 - np.exp(-0.06 * k_min_ring_size))
        
        val = branch * ringf * doublef * hybf
        sum_val_squared += val**2
    
    flexibility_score = np.sqrt(sum_val_squared / num_edges) if num_edges > 0 else 0.0
    return flexibility_score


# --- Steric and Electronic Descriptors ---

def get_electronic_features(filename: str, charge_index: int, timeout: int) -> tuple[float, float, float, float]:
    """
    Runs an xTB/PTB calculation on the given .xyz file and extracts FMO energies (HOMO, LUMO, HL-Gap)
    and the charge of a target atom. It attempts a calculation with default settings, and if that
    fails to yield FMOs, it retries with a charge of -1.

    Args:
        filename (str): Path to the input XYZ file.
        charge_index (int): The 0-based index of the atom whose charge is to be extracted.
        timeout (int): Maximum time in seconds to wait for each xTB process.

    Returns:
        tuple[float, float, float, float]: A tuple containing:
            - fmo_gap (float): HOMO-LUMO energy gap in eV. Returns np.nan if not found.
            - homo (float): HOMO energy in eV. Returns np.nan if not found.
            - lumo (float): LUMO energy in eV. Returns np.nan if not found.
            - q_target (float): Charge of the target atom. Returns np.nan if not found.
    """
    fmo_gap, homo, lumo, q_target = np.nan, np.nan, np.nan, np.nan
    
    def run_xtb_and_parse(xyz_file: str, extra_args: list, log_file: str, current_timeout: int):
        cmd = ["xtb", xyz_file, "--ptb"] + extra_args
        try:
            with open(log_file, "w") as f:
                sp.call(cmd, stdout=f, stderr=sp.STDOUT, timeout=current_timeout)
            
            parsed_homo, parsed_lumo, parsed_gap = np.nan, np.nan, np.nan
            if os.path.exists(log_file):
                with open(log_file, "r") as f_log:
                    for line in f_log:
                        if "HL-Gap" in line:
                            parsed_gap = float(line.split()[3])
                        elif "(HOMO)" in line:
                            parsed_homo = float(line.split()[3])
                        elif "(LUMO)" in line:
                            parsed_lumo = float(line.split()[2])
                return parsed_gap, parsed_homo, parsed_lumo
            return np.nan, np.nan, np.nan
        except sp.TimeoutExpired:
            print(f"xTB calculation timed out for {xyz_file} with args {extra_args}.")
            return np.nan, np.nan, np.nan
        except (IndexError, ValueError) as e:
            print(f"Error parsing xTB output for {xyz_file} with args {extra_args}: {e}")
            return np.nan, np.nan, np.nan

    fmo_gap, homo, lumo = run_xtb_and_parse(filename, [], "xtbcalc.log", timeout)

    if np.isnan(homo) or np.isnan(lumo) or np.isnan(fmo_gap):
        print(f"FMOs not found with default xTB settings for {filename}. Retrying with charge -1.")
        fmo_gap, homo, lumo = run_xtb_and_parse(filename, ["-c", "-1"], "xtbcalc.log", timeout)

    if os.path.exists("charges"):
        try:
            with open("charges", "r") as f:
                lines = f.readlines()
                if 0 <= charge_index < len(lines):
                    q_target = float(lines[charge_index].strip())
                else:
                    warnings.warn(f"Charge index {charge_index} out of bounds for 'charges' file in {filename}.")
        except (ValueError, IndexError) as e:
            warnings.warn(f"Error reading charge from 'charges' file for {filename}: {e}")
    
    temp_files = ["xtbtopo.mol", "wbo", "xtbrestart", "charges", "xtbcalc.log"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return fmo_gap, homo, lumo, q_target


def calculate_angle_between_vectors(v: np.ndarray, w: np.ndarray) -> float:
    """
    Calculates the angle in degrees between two 3D vectors.

    Args:
        v (np.ndarray): The first 3D vector.
        w (np.ndarray): The second 3D vector.

    Returns:
        float: The angle between the vectors in degrees.
    """
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    if norm_v == 0 or norm_w == 0:
        return np.nan

    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def perform_buried_volume_analysis(
    filename: str, 
    metal_atom_indices: list[int] = None, 
    radius: float = 5.0, 
    compute_dihedral_angle: bool = False
) -> dict:
    """
    Computes buried volume, quadrant/octant analysis, Sterimol parameters, and
    optionally a specific dihedral angle for a molecular structure.

    Args:
        filename (str): Path to the input XYZ file.
        metal_atom_indices (list[int], optional): 1-based index(es) of the metal atom(s).
                                                  Defaults to [1] if None.
        radius (float, optional): Radius in Angstroms for the buried volume calculation.
                                  Defaults to 5.0.
        compute_dihedral_angle (bool, optional): If True, computes a specific dihedral angle
                                                 between two pairs of ring centroids.
                                                 Requires specific atom indices to be hardcoded.
                                                 Defaults to False.

    Returns:
        dict: A dictionary containing the computed steric descriptors:
              - "buried_volume": Fractional buried volume.
              - "quadrants": Dictionary of percent buried volume per quadrant.
              - "octants": Dictionary of percent buried volume per octant.
              - "L_value", "B1_value", "B5_value": Sterimol parameters.
              - "dihedral_angle" (optional): Dihedral angle in degrees, if `compute_dihedral_angle` is True.
    """
    if metal_atom_indices is None:
        metal_atom_indices = [M_IDX]

    cp_atom_idxs_0based = [4, 5, 6, 7, 8]
    excluded_atoms_1based = [2, 3, 4, 5]

    results_compilation = {}
    
    elements_orig, coords_orig = read_xyz(filename)
    
    cp_coords = coords_orig[cp_atom_idxs_0based]
    cp_centroid = np.mean(cp_coords, axis=0)

    temp_xyz_file = "tmp_bv.xyz"
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
    
    lines.insert(2, f"H {cp_centroid[0]:.6f} {cp_centroid[1]:.6f} {cp_centroid[2]:.6f}\n")
    lines[0] = str(int(lines[0].strip()) + 1) + "\n"
    
    with open(temp_xyz_file, 'w') as f_out:
        f_out.writelines(lines)

    elements_modified, coords_modified = read_xyz(temp_xyz_file)
    
    metal_atom_indices_for_bv = [idx + 1 for idx in metal_atom_indices]
    excluded_atoms_for_bv = [1] + excluded_atoms_1based

    bv_calculator = BuriedVolume(elements_modified,
                                 coords_modified,
                                 metal_atom_indices_for_bv,
                                 excluded_atoms=excluded_atoms_for_bv,
                                 z_axis_atoms=Z_ATOMS, 
                                 xz_plane_atoms=XZ_ATOMS,
                                 radius=radius)
    bv_calculator.octant_analysis()
    
    results_compilation["buried_volume"] = bv_calculator.fraction_buried_volume
    results_compilation["quadrants"] = bv_calculator.quadrants
    results_compilation["octants"] = bv_calculator.octants
    
    elements_orig, coords_orig = read_xyz(filename)
    
    sterimol_excluded_atoms_1based = [2, 3, 4, 5]
    
    sterimol_calc = Sterimol(elements_orig,
                             coords_orig,
                             dummy_index=2,
                             attached_index=[idx + 1 for idx in cp_atom_idxs_0based],
                             excluded_atoms=sterimol_excluded_atoms_1based)
    
    results_compilation["L_value"] = sterimol_calc.L_value
    results_compilation["B1_value"] = sterimol_calc.B_1_value
    results_compilation["B5_value"] = sterimol_calc.B_5_value
    
    if os.path.exists(temp_xyz_file):
        os.remove(temp_xyz_file)
    
    if compute_dihedral_angle:
        ring_a1_idxs_0based = [21, 22, 23, 24, 25, 26]
        ring_a2_idxs_0based = [20, 21, 26, 27, 28, 29]
        ring_b1_idxs_0based = [13, 14, 15, 16, 17, 18]
        ring_b2_idxs_0based = [10, 11, 12, 13, 18, 19]

        ring_a1_centroid = coords_orig[ring_a1_idxs_0based].mean(axis=0)
        ring_a2_centroid = coords_orig[ring_a2_idxs_0based].mean(axis=0)
        ring_b1_centroid = coords_orig[ring_b1_idxs_0based].mean(axis=0)
        ring_b2_centroid = coords_orig[ring_b2_idxs_0based].mean(axis=0)

        vec1 = ring_a1_centroid - ring_a2_centroid
        vec2 = ring_b1_centroid - ring_b2_centroid
        
        dihedral_angle = calculate_angle_between_vectors(vec1, vec2)
        results_compilation["dihedral_angle"] = dihedral_angle
    
    return results_compilation


# --- Main Descriptor Calculation Function ---

def get_molecular_descriptors(
    xyz_directory: str, 
    output_csv_filename: str = "descriptor.csv",
    radius_for_bv: float = 5.0,
    compute_electronic_features: bool = False,
    compute_dihedral_angle: bool = False,
    xtb_timeout: int = 30
) -> None:
    """
    Computes a comprehensive set of molecular descriptors (steric, electronic,
    rigidity, and complexity) for all XYZ files in a given directory and
    saves them to a CSV file.

    Args:
        xyz_directory (str): Path to the directory containing the XYZ files.
        output_csv_filename (str, optional): Name of the output CSV file. Defaults to "descriptor.csv".
        radius_for_bv (float, optional): Radius for buried volume analysis. Defaults to 5.0.
        compute_electronic_features (bool, optional): If True, electronic descriptors (HOMO, LUMO, etc.)
                                                      are computed using xTB. Defaults to False.
        compute_dihedral_angle (bool, optional): If True, a specific dihedral angle is computed.
                                                 Defaults to False.
        xtb_timeout (int, optional): Timeout in seconds for xTB calls within electronic feature computation.
                                     Defaults to 30.
    """
    descriptor_data = {
        "filename": [], "buried_volume": [], "L_value": [], "B1_value": [], "B5_value": [],
        "q1": [], "q2": [], "q3": [], "q4": [],
        "o1": [], "o2": [], "o3": [], "o4": [], "o5": [], "o6": [], "o7": [], "o8": [],
        "flexibility": [], "kier_phi": []
    }

    if compute_electronic_features:
        descriptor_data["homo"] = []
        descriptor_data["lumo"] = []
        descriptor_data["hlg"] = []
        descriptor_data["q"] = []
        descriptor_data["chem_potential"] = []
        descriptor_data["hardness"] = []
    if compute_dihedral_angle:
        descriptor_data["dihedral_angle"] = []

    xyz_files = glob.glob(os.path.join(xyz_directory, "*.xyz"))
    
    if not xyz_files:
        print(f"No .xyz files found in '{xyz_directory}'. No descriptors to compute.")
        return

    for xyz_file_path in tqdm.tqdm(xyz_files, desc="Computing Descriptors"):
        try:
            xyz_file_basename = os.path.basename(xyz_file_path)
            
            steric_results = perform_buried_volume_analysis(
                xyz_file_path, 
                metal_atom_indices=[M_IDX],
                radius=radius_for_bv,
                compute_dihedral_angle=compute_dihedral_angle
            )
            
            with open(xyz_file_path, "r") as f:
                xyz_content = f.read()
            graph = xyz_to_networkx_graph(xyz_content)
            
            flexibility_score = compute_crest_flexibility(graph)
            kier_phi_value = compute_kier_phi(graph)

            if compute_electronic_features:
                fmo_gap, homo_energy, lumo_energy, target_charge = get_electronic_features(
                    xyz_file_path, M_IDX - 1, xtb_timeout
                )
                descriptor_data["homo"].append(homo_energy)
                descriptor_data["lumo"].append(lumo_energy)
                descriptor_data["hlg"].append(fmo_gap)
                descriptor_data["q"].append(target_charge)
                descriptor_data["chem_potential"].append((homo_energy + lumo_energy) / 2 if not (np.isnan(homo_energy) or np.isnan(lumo_energy)) else np.nan)
                descriptor_data["hardness"].append((homo_energy - lumo_energy) / 2 if not (np.isnan(homo_energy) or np.isnan(lumo_energy)) else np.nan)
            
            descriptor_data["filename"].append(xyz_file_basename)
            descriptor_data["buried_volume"].append(steric_results["buried_volume"])
            descriptor_data["L_value"].append(steric_results["L_value"])
            descriptor_data["B1_value"].append(steric_results["B1_value"])
            descriptor_data["B5_value"].append(steric_results["B5_value"])

            descriptor_data["q1"].append(steric_results["quadrants"]["percent_buried_volume"].get(1, np.nan))
            descriptor_data["q2"].append(steric_results["quadrants"]["percent_buried_volume"].get(2, np.nan))
            descriptor_data["q3"].append(steric_results["quadrants"]["percent_buried_volume"].get(3, np.nan))
            descriptor_data["q4"].append(steric_results["quadrants"]["percent_buried_volume"].get(4, np.nan))
            
            descriptor_data["o1"].append(steric_results["octants"]["percent_buried_volume"].get(0, np.nan))
            descriptor_data["o2"].append(steric_results["octants"]["percent_buried_volume"].get(1, np.nan))
            descriptor_data["o3"].append(steric_results["octants"]["percent_buried_volume"].get(2, np.nan))
            descriptor_data["o4"].append(steric_results["octants"]["percent_buried_volume"].get(3, np.nan))
            descriptor_data["o5"].append(steric_results["octants"]["percent_buried_volume"].get(4, np.nan))
            descriptor_data["o6"].append(steric_results["octants"]["percent_buried_volume"].get(5, np.nan))
            descriptor_data["o7"].append(steric_results["octants"]["percent_buried_volume"].get(6, np.nan))
            descriptor_data["o8"].append(steric_results["octants"]["percent_buried_volume"].get(7, np.nan))
            
            descriptor_data["flexibility"].append(flexibility_score)
            descriptor_data["kier_phi"].append(kier_phi_value)

            if compute_dihedral_angle:
                descriptor_data["dihedral_angle"].append(steric_results["dihedral_angle"])

        except Exception as e:
            print(f"Error processing {xyz_file_path}: {e}")
            for key in descriptor_data:
                if key != "filename":
                    descriptor_data[key].append(np.nan)
                else:
                    descriptor_data[key].append(xyz_file_basename)
            continue

    df = pd.DataFrame(descriptor_data)
    df = df.sort_values(by="filename").reset_index(drop=True)
    df.to_csv(output_csv_filename, index=False)
    print(f"Descriptors computed for {len(xyz_files)} files and saved to '{output_csv_filename}'.")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute steric, electronic, rigidity, and complexity descriptors for molecular structures.
        This script assumes a specific molecular setup for buried volume and Sterimol calculations:
        - The metal atom index (M_IDX) defaults to 1 (1-based).
        - The Cp atoms (for centroid calculation and Sterimol attached_index) are hardcoded as 5,6,7,8,9 (1-based).
        - Excluded atoms for steric calculations (e.g., CO and Cl) are hardcoded as 2,3,4,5 (1-based).
        - Dihedral angle calculation (if enabled) relies on hardcoded ring atom indices.
        Ensure your XYZ files conform to these indexing assumptions for accurate results.
        """
    )
    parser.add_argument(
        "-i", 
        "--input_dir",
        type=str, 
        required=True,
        help="Path to the directory containing the XYZ files."
    )
    parser.add_argument(
        "-el",
        "--electronic", 
        action="store_true",
        help="Flag to compute electronic descriptors (HOMO, LUMO, etc.) using xTB."
    )
    parser.add_argument(
        '-r', 
        '--radius', 
        type=float,
        default=RADIUS,
        help=f'Radius in Angstroms for buried volume analysis. Defaults to {RADIUS}.'
    )
    parser.add_argument(
        "-dih",
        "--dihedral", 
        action="store_true",
        help="""Flag to compute a specific dihedral angle between two BINOL-like rings.
        This calculation relies on hardcoded atom indices within the script:
        ring_a1_idxs = [22,23,24,25,26,27] (1-based)
        ring_a2_idxs = [21,22,27,28,29,30] (1-based)
        ring_b1_idxs = [14,15,16,17,18,19] (1-based)
        ring_b2_idxs = [11,12,13,14,19,20] (1-based)
        Ensure your molecular structure matches these indices for correct results.
        """
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        default="descriptor.csv",
        help="Name of the output CSV file. Defaults to 'descriptor.csv'."
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=TIMEOUT,
        help=f"Maximum time in seconds for xTB subprocess calls. Defaults to {TIMEOUT}."
    )

    args = parser.parse_args()
    
    input_directory_path = args.input_dir
    output_csv_filename = args.output_csv
    compute_electronic = args.electronic
    compute_dihedral = args.dihedral
    bv_radius = args.radius
    xtb_process_timeout = args.timeout
    
    if not os.path.isdir(input_directory_path):
        raise FileNotFoundError(f"Input directory '{input_directory_path}' does not exist or is not a directory.")

    get_molecular_descriptors(
        xyz_directory=input_directory_path,
        output_csv_filename=output_csv_filename,
        radius_for_bv=bv_radius,
        compute_electronic_features=compute_electronic,
        compute_dihedral_angle=compute_dihedral,
        xtb_timeout=xtb_process_timeout
    )

