"""
Molecular Stability Assessment Module for 3D Molecular Generation Evaluation.

This module provides aromatic-aware stability validation by separating aromatic
and non-aromatic bond contributions. Ported from geom-drugs-3dgen-evaluation.
"""

import json
import warnings
from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms


# ============================================================================
# Valency Table Loading
# ============================================================================

def load_valency_table(name: str = "tuple", validate_schema: bool = False) -> Dict[str, Any]:
    """
    Load valency table from package JSON data.
    
    Args:
        name: Table name - "tuple" (aromatic-aware) or "legacy" (backward compat)
        validate_schema: Whether to validate against JSON schema (requires jsonschema)
        
    Returns:
        dict: Valency table mapping element -> charge -> allowed valencies
    """
    table_files = {
        "tuple": "geom_drugs_h_tuple_valencies.json",
        "legacy": "legacy_valencies.json",
    }
    
    filename = table_files.get(name, f"{name}.json")
    
    pkg = files("MolecularDiffusion.data.valency_tables")
    table_path = pkg.joinpath(filename)
    
    with table_path.open("r") as f:
        data = json.load(f)
    
    return _convert_to_internal_format(data, name)


def _convert_to_internal_format(data: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    """Convert JSON format to internal format."""
    valency_table = data["valency_table"]
    converted = {}
    
    for element, charge_dict in valency_table.items():
        converted[element] = {}
        for charge_str, valencies in charge_dict.items():
            charge_int = int(charge_str)
            
            if table_name == "tuple":
                # Convert list of lists to list of tuples
                if isinstance(valencies, list) and valencies and isinstance(valencies[0], list):
                    converted[element][charge_int] = [tuple(v) for v in valencies]
                else:
                    converted[element][charge_int] = valencies
            else:
                converted[element][charge_int] = valencies
    
    return converted


# Default valency table (loaded lazily)
_CACHED_TUPLE_VALENCIES = None
_CACHED_SIMPLE_VALENCIES = None


def get_default_valencies() -> Dict[str, Any]:
    """Get default aromatic-aware valency table (lazy loading)."""
    global _CACHED_TUPLE_VALENCIES
    if _CACHED_TUPLE_VALENCIES is None:
        _CACHED_TUPLE_VALENCIES = load_valency_table("tuple")
    return _CACHED_TUPLE_VALENCIES


def get_simple_valencies() -> Dict[str, Any]:
    """Get simple valency table for 1.5 aromatic mode (lazy loading)."""
    global _CACHED_SIMPLE_VALENCIES
    if _CACHED_SIMPLE_VALENCIES is None:
        _CACHED_SIMPLE_VALENCIES = load_valency_table("legacy")
    return _CACHED_SIMPLE_VALENCIES


# ============================================================================
# Molecule Validity Check
# ============================================================================

def is_valid(mol, verbose: bool = False) -> bool:
    """
    Validate molecule for single fragment and successful sanitization.
    
    Args:
        mol: RDKit molecule object
        verbose: Print error messages if validation fails
        
    Returns:
        True if valid, otherwise False
    """
    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        if verbose:
            print(f"Kekulization failed: {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"Sanitization failed: {e}")
        return False

    if len(Chem.GetMolFrags(mol)) > 1:
        if verbose:
            print("Molecule has multiple fragments.")
        return False

    return True


# ============================================================================
# Aromatic-Aware Molecule Stability
# ============================================================================

def _is_valid_valence_tuple(
    combo: Tuple[int, int],
    allowed: Union[tuple, list, set, dict],
    charge: int,
    element_symbol: Optional[str] = None
) -> bool:
    """Validate valence tuple against allowed configurations."""
    if isinstance(allowed, tuple):
        return combo == allowed
    elif isinstance(allowed, (list, set)):
        return combo in allowed
    elif isinstance(allowed, dict):
        if charge not in allowed:
            return False
        return _is_valid_valence_tuple(combo, allowed[charge], charge, element_symbol)
    elif allowed == []:
        return False
    return False


def _is_valid_simple_valence(
    total_valence: int,
    allowed: Union[int, list, dict],
    charge: int,
    element_symbol: Optional[str] = None
) -> bool:
    """Validate simple valence (total bond order sum) against allowed values."""
    if isinstance(allowed, int):
        return total_valence == allowed
    elif isinstance(allowed, list):
        return total_valence in allowed
    elif isinstance(allowed, dict):
        if charge not in allowed:
            return False
        return _is_valid_simple_valence(total_valence, allowed[charge], charge, element_symbol)
    return False


def compute_molecules_stability_from_graph(
    adjacency_matrices: torch.Tensor,
    numbers: torch.Tensor,
    charges: torch.Tensor,
    allowed_bonds: Optional[Dict] = None,
    aromatic: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute molecular stability from graph representations.
    
    Args:
        adjacency_matrices: Bond matrices [batch, n_atoms, n_atoms]
                           Values: 1=single, 2=double, 3=triple, 1.5=aromatic
        numbers: Atomic numbers [batch, n_atoms]
        charges: Formal charges [batch, n_atoms]
        allowed_bonds: Valency lookup table
        aromatic: If True, use tuple valencies (n_aromatic, v_other).
                  If False, sum all bonds (MS 1.5 Arom mode)
        
    Returns:
        (stable_mask, n_stable_atoms, n_atoms)
    """
    if adjacency_matrices.ndim == 2:
        adjacency_matrices = adjacency_matrices.unsqueeze(0)
        numbers = numbers.unsqueeze(0)
        charges = charges.unsqueeze(0)

    # Select appropriate valency table based on mode
    if allowed_bonds is None:
        if aromatic:
            allowed_bonds = get_default_valencies()  # Tuple valencies
        else:
            allowed_bonds = get_simple_valencies()   # Simple valencies

    batch_size = adjacency_matrices.shape[0]
    stable_mask = torch.zeros(batch_size)
    n_stable_atoms = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size)

    for i in range(batch_size):
        adj = adjacency_matrices[i]
        atom_nums = numbers[i]
        atom_charges = charges[i]

        mol_stable = True
        n_atoms_i, n_stable_i = 0, 0

        for j, (a_num, charge) in enumerate(zip(atom_nums, atom_charges)):
            if a_num.item() == 0:
                continue
                
            row = adj[j]
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(a_num))
            allowed = allowed_bonds.get(symbol, {})
            
            if aromatic:
                # Tuple mode: (n_aromatic_bonds, valence_from_non_aromatic_bonds)
                aromatic_count = int((row == 1.5).sum().item())
                normal_valence = float((row * (row != 1.5)).sum().item())
                combo = (aromatic_count, int(normal_valence))
                
                if _is_valid_valence_tuple(combo, allowed, int(charge), symbol):
                    n_stable_i += 1
                else:
                    mol_stable = False
            else:
                # Simple mode: sum all bond orders (aromatic=1.5 contributes 1.5)
                total_valence = round(float(row.sum().item()))
                
                if _is_valid_simple_valence(total_valence, allowed, int(charge), symbol):
                    n_stable_i += 1
                else:
                    mol_stable = False

            n_atoms_i += 1

        stable_mask[i] = float(mol_stable)
        n_stable_atoms[i] = n_stable_i
        n_atoms[i] = n_atoms_i

    return stable_mask, n_stable_atoms, n_atoms


def compute_molecules_stability(
    rdkit_molecules: List,
    aromatic: bool = True,
    allowed_bonds: Optional[Dict] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute stability metrics for RDKit molecules.
    
    Args:
        rdkit_molecules: List of RDKit Mol objects
        aromatic: Whether aromatic bonds are expected
        allowed_bonds: Custom valency table (defaults to aromatic-aware tuple table)
        
    Returns:
        (validity, stability, n_stable_atoms, n_atoms)
    """
    stable_list, stable_atoms_list, atom_counts_list, validity_list = [], [], [], []

    for mol in rdkit_molecules:
        if mol is None:
            continue
            
        n_atoms = mol.GetNumAtoms()
        
        adj = torch.zeros((1, n_atoms, n_atoms))
        numbers = torch.zeros((1, n_atoms), dtype=torch.long)
        charges = torch.zeros((1, n_atoms), dtype=torch.long)

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            numbers[0, idx] = atom.GetAtomicNum()
            charges[0, idx] = atom.GetFormalCharge()

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adj[0, i, j] = adj[0, j, i] = bond_type

        stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            adj, numbers, charges, allowed_bonds, aromatic
        )
        
        stable_list.append(stable.item())
        stable_atoms_list.append(stable_atoms.item())
        atom_counts_list.append(atom_count.item())
        validity_list.append(float(is_valid(mol)))

    return (
        torch.tensor(validity_list),
        torch.tensor(stable_list),
        torch.tensor(stable_atoms_list),
        torch.tensor(atom_counts_list)
    )


# ============================================================================
# Helper Utilities
# ============================================================================

def generate_canonical_key(*components) -> tuple:
    """Generate canonical key for molecular components (order-independent)."""
    key1 = tuple(components)
    key2 = tuple(reversed(components))
    return min(key1, key2)


def compute_rmsd(init_mol, opt_mol, hydrogens: bool = True) -> float:
    """Compute RMSD between initial and optimized molecule coordinates."""
    init_mol = Chem.Mol(init_mol)
    init_mol.AddConformer(opt_mol.GetConformer(), assignId=True)
    if not hydrogens:
        init_mol = Chem.RemoveAllHs(init_mol)
    return AllChem.AlignMol(init_mol, init_mol, prbCid=0, refCid=1)


def compute_mmff_energy_drop(mol, max_iters: int = 1000) -> Optional[float]:
    """
    Compute MMFF energy drop after optimization.
    
    Returns:
        Energy difference (E_before - E_after), or None if failed.
    """
    try:
        mol_copy = Chem.Mol(mol)
        props = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant='MMFF94')
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        e_before = ff.CalcEnergy()

        success = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=max_iters)
        if success != 0:
            return None

        ff_opt = AllChem.MMFFGetMoleculeForceField(mol_copy, props)
        e_after = ff_opt.CalcEnergy()

        return e_before - e_after
    except Exception:
        return None


def bond_type_to_symbol(bond_type_numeric: int) -> str:
    """Convert bond type numeric to chemical symbol."""
    if bond_type_numeric == 1:
        return "-"
    elif bond_type_numeric == 2:
        return "="
    elif bond_type_numeric == 3:
        return "#"
    elif bond_type_numeric == 12:
        return ":"
    else:
        return "?"


def compute_statistics(diff_sums: Dict) -> Dict:
    """
    Compute statistics: average difference, standard deviation, and weight.
    
    Args:
        diff_sums: Dict mapping keys to (diff_list, count)
        
    Returns:
        Dict mapping keys to (avg_diff, std_dev, weight)
    """
    total = sum(count for (diff_list, count) in diff_sums.values())
    
    avg_diffs = {}
    for key, (diff_list, count) in diff_sums.items():
        avg_diff = np.mean(diff_list) if count > 0 else 0
        std_dev = np.std(diff_list) if count > 0 else 0
        weight = count / total if total > 0 else 0
        avg_diffs[key] = (avg_diff, std_dev, weight)
    return avg_diffs


def compute_differences(pairs: List[Tuple], compute_function, show_progress: bool = False) -> Dict:
    """
    Compute geometry differences using a specific compute_function.
    
    Args:
        pairs: List of (init_mol, opt_mol) pairs
        compute_function: Function that computes differences for a molecule pair
        show_progress: Whether to show a tqdm progress bar
        
    Returns:
        Dict with geometry difference stats
    """
    from collections import defaultdict
    from tqdm import tqdm
    
    diff_sums = defaultdict(lambda: [[], 0])
    iterator = tqdm(pairs, total=len(pairs), desc="Processing Molecules") if show_progress else pairs

    for pair in iterator:
        result = compute_function(pair)
        for key, (diff_list, count) in result.items():
            diff_sums[key][0].extend(diff_list)
            diff_sums[key][1] += count

    return compute_statistics(diff_sums)


# ============================================================================
# Pair Geometry Comparisons
# ============================================================================

def compute_bond_lengths_diff(pair: Tuple) -> Dict:
    """
    Compute bond length differences between initial and optimized structures.
    
    Args:
        pair: (initial_mol, optimized_mol) RDKit molecule objects
        
    Returns:
        Dict mapping bond type keys to (diff_list, count)
    """
    init_mol, opt_mol = pair
    bond_lengths = {}
    
    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()
    
    for bond in init_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        atom1_type = init_mol.GetAtomWithIdx(idx1).GetAtomicNum()
        atom2_type = init_mol.GetAtomWithIdx(idx2).GetAtomicNum()
        bond_type = int(bond.GetBondType())
        
        init_length = rdMolTransforms.GetBondLength(init_conf, idx1, idx2)
        opt_length = rdMolTransforms.GetBondLength(opt_conf, idx1, idx2)
        diff = np.abs(init_length - opt_length)
        
        key = generate_canonical_key(atom1_type, bond_type, atom2_type)
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(diff)
        bond_lengths[key][1] += 1
        
    return bond_lengths


def compute_bond_angles_diff(pair: Tuple) -> Dict:
    """
    Compute bond angle differences between initial and optimized structures.
    """
    init_mol, opt_mol = pair
    bond_angles = {}
    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    for atom in init_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                
                atom1_type = init_mol.GetAtomWithIdx(idx1).GetAtomicNum()
                atom2_type = init_mol.GetAtomWithIdx(idx2).GetAtomicNum()
                atom3_type = init_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                bond_type_1 = int(init_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(init_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                angle_init = rdMolTransforms.GetAngleDeg(init_conf, idx1, idx2, idx3)
                angle_opt = rdMolTransforms.GetAngleDeg(opt_conf, idx1, idx2, idx3)
                diff = min(np.abs(angle_init - angle_opt), 360 - np.abs(angle_init - angle_opt))

                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2, atom3_type)
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(diff)
                bond_angles[key][1] += 1

    return bond_angles


def compute_torsion_angles_diff(pair: Tuple) -> Dict:
    """
    Compute torsion angle differences using SMARTS-based rotatable bond identification.
    """
    init_mol, opt_mol = pair
    
    torsion_smarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsion_smarts)

    torsion_angles = {}
    init_conf = init_mol.GetConformer()
    opt_conf = opt_mol.GetConformer()

    torsion_matches = init_mol.GetSubstructMatches(torsion_query)

    for match in torsion_matches:
        idx2, idx3 = match[0], match[1]
        bond = init_mol.GetBondBetweenAtoms(idx2, idx3)

        for b1 in init_mol.GetAtomWithIdx(idx2).GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            
            for b2 in init_mol.GetAtomWithIdx(idx3).GetBonds():
                if b2.GetIdx() == bond.GetIdx() or b2.GetIdx() == b1.GetIdx():
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                if idx4 == idx1:
                    continue

                atom1_type = init_mol.GetAtomWithIdx(idx1).GetAtomicNum()
                atom2_type = init_mol.GetAtomWithIdx(idx2).GetAtomicNum()
                atom3_type = init_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                atom4_type = init_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                bond_type_1 = int(b1.GetBondType())
                bond_type_2 = int(bond.GetBondType())
                bond_type_3 = int(b2.GetBondType())

                init_angle = rdMolTransforms.GetDihedralDeg(init_conf, idx1, idx2, idx3, idx4)
                opt_angle = rdMolTransforms.GetDihedralDeg(opt_conf, idx1, idx2, idx3, idx4)
                diff = min(np.abs(init_angle - opt_angle), 360 - np.abs(init_angle - opt_angle))
                
                key = generate_canonical_key(
                    atom1_type, bond_type_1, atom2_type, bond_type_2,
                    atom3_type, bond_type_3, atom4_type
                )

                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(diff)
                torsion_angles[key][1] += 1

    return torsion_angles
