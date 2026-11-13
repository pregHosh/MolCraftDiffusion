"""Helpers for comparing an initial geometry with an optimized counterpart.

This is an internal helper library, not a command. The directory-walking
front end that used to live here is now the ``xyz+optimized`` layout of
``--metrics conformer`` (see ``conformer_metrics.py``).

Two consumers depend on it:

* ``conformer_metrics.py`` -- ``compute_all_metrics`` for the legacy layout;
* ``runmodes/generate/tasks_conformer.py`` -- ``get_xtb_energy`` for the
  optional energy column. Removing that function breaks conformer
  *generation*, not just analysis.
"""

from __future__ import annotations

import argparse
import subprocess as sp
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

try:
    from openbabel import pybel
except ImportError:
    pass

from MolecularDiffusion.utils.geometry.geom_utils import read_xyz_file
# Import from local package assuming this script is in src/MolecularDiffusion/runmodes/analyze/
from MolecularDiffusion.runmodes.analyze.xtb_optimization import check_xyz
from MolecularDiffusion.utils.metrics.geom_stability import (
    compute_bond_lengths_diff,
    compute_bond_angles_diff,
    compute_torsion_angles_diff,
    compute_differences,
)


def get_xtb_energy(xyz_path: str, charge: int = 0, level: str = "gfn2", timeout: int = 120) -> Optional[float]:
    """Compute xTB single-point energy for an XYZ file (returns Hartree)."""
    cmd = ["xtb", xyz_path, f"-{level}", "-c", str(charge), "--sp"]
    try:
        result = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
        for line in result.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Eh" and i > 0:
                        return float(parts[i - 1])
        return None
    except (sp.TimeoutExpired, Exception):
        return None


def xyz2mol_openbabel(xyz_file: str) -> Optional[Chem.Mol]:
    """Convert XYZ to RDKit Mol using OpenBabel (via pybel)."""
    try:
        mol_pb = next(pybel.readfile("xyz", str(xyz_file)))
        mol_sdf = mol_pb.write("sdf")
        mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False, sanitize=False)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def xyz2mol_converter(xyz_file: str, timeout: int = 10) -> Optional[Chem.Mol]:
    """Convert XYZ to RDKit Mol using xyz2mol logic."""
    from MolecularDiffusion.utils.chem.smilify import smilify_xyz2mol
    try:
        _, mol = smilify_xyz2mol(str(xyz_file), timeout=timeout)
        return mol
    except Exception:
        return None


def compute_coord_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def compute_all_metrics(
    init_file: Path,
    opt_file: Path,
    args: argparse.Namespace
) -> dict:
    """Compute all available metrics for a single pair."""
    results = {}
    
    # 1. Connectivity Check
    if check_xyz is not None:
        is_conn_init, n_comp_init, _ = check_xyz(str(init_file), scale_factor=1.3)
        if not (is_conn_init and n_comp_init == 1):
            return {"error": "Initial molecule not fully connected"}
        
        is_conn_opt, n_comp_opt, _ = check_xyz(str(opt_file), scale_factor=1.3)
        if not (is_conn_opt and n_comp_opt == 1):
            return {"error": "Optimized molecule not fully connected"}
    
    # 2. RMSD & Energy
    try:
        init_coords, _ = read_xyz_file(init_file)
        opt_coords, _ = read_xyz_file(opt_file)
        
        init_np = init_coords.numpy() if hasattr(init_coords, 'numpy') else np.array(init_coords)
        opt_np = opt_coords.numpy() if hasattr(opt_coords, 'numpy') else np.array(opt_coords)
        
        results["rmsd"] = compute_coord_rmsd(init_np, opt_np)
        
        e_init = get_xtb_energy(str(init_file), args.charge, args.level, args.timeout)
        e_opt = get_xtb_energy(str(opt_file), args.charge, args.level, args.timeout)
        
        results["e_init_Ha"] = e_init
        results["e_opt_Ha"] = e_opt
        
        if e_init is not None and e_opt is not None:
            results["energy_diff_kcal"] = (e_init - e_opt) * 627.5
        else:
            results["energy_diff_kcal"] = None
            
    except Exception as e:
        return {"error": f"RMSD/Energy failed: {e}"}

    # 3. Bond Geometry
    try:
        if args.mol_converter == "openbabel":
            init_mol = xyz2mol_openbabel(str(init_file))
            opt_mol = xyz2mol_openbabel(str(opt_file))
        elif args.mol_converter == "xyz2mol":
            init_mol = xyz2mol_converter(str(init_file), timeout=args.timeout)
            opt_mol = xyz2mol_converter(str(opt_file), timeout=args.timeout)
        else:
            return {"error": f"Unknown converter {args.mol_converter}"}
            
        if init_mol is None or opt_mol is None:
             return {"error": "Failed to load molecules for bond analysis"}
             
        pair = (init_mol, opt_mol)

        # Helper to extract mean diff from result tuple (avg_diff, std, weight)
        def get_diff(res_dict):
            # Typically returns a dict key -> (list of diffs, list of weights) in run_bond_analysis 
            # But compute_differences returns {k: (avg, std, weight)} ??
            # Wait, check `compute_pair_geometry.py` logic:
            # compute_differences returns dict: key -> (avg_diff, std_dev, weight)
            # Actually compute_differences returns {bond_type: (avg_diff, std, weight)}
            # We want the weighted mean across all bond types.
            total_w_diff = 0
            total_weight = 0
            for k, (avg, s, w) in res_dict.items():
                total_w_diff += avg * w
                total_weight += w
            return total_w_diff / total_weight if total_weight > 0 else None

        # Check `geom_stability.py` - checking `compute_pair_geometry.py` usage:
        # It aggregates results. Here we are doing per-pair.
        # `compute_differences` takes a List of pairs. We pass list of 1 pair.
        
        b_len_res = compute_differences([pair], compute_bond_lengths_diff)
        b_ang_res = compute_differences([pair], compute_bond_angles_diff)
        tor_res = compute_differences([pair], compute_torsion_angles_diff)
        
        results["bond_length_mean"] = get_diff(b_len_res)
        results["bond_angle_mean"] = get_diff(b_ang_res)
        results["torsion_angle_mean"] = get_diff(tor_res)

    except Exception as e:
         return {"error": f"Bond analysis failed: {e}"}

    return results
