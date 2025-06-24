import warnings
import numpy as np
import scipy as sp
import subprocess
import tempfile
import os
import re
import signal
import csv
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Sequence

import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

import ase
from ase.io import read as ase_read
from ase import neighborlist, data
from ase.io.extxyz import read_xyz
from ase.data import atomic_numbers, covalent_radii

from cell2mol.xyz2mol import xyz2mol

from tqdm import tqdm

from rdkit.Chem import MolToSmiles as mol2smi

# -----------------------------------------------------------------------------
# Quiet RDKit & constants
# -----------------------------------------------------------------------------
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
TIMEOUT = 300

# silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

#%% cell2mol

def simple_idx_match_check(rdkit_mol, ase_symbols):
    """Check that RDKit atom order matches the ASE symbol list exactly."""
    for rd_atom, sym in zip(rdkit_mol.GetAtoms(), ase_symbols):
        if rd_atom.GetSymbol() != sym:
            return False
    return True



def simple_idx_match_check(mol, ase_atoms):
    match = True
    for rd_atom, ase_atom in zip(mol.GetAtoms(), ase_atoms):
        if rd_atom.GetSymbol() != ase_atom:
            match = False
            break
    return match

def get_cutoffs(z, radii=ase.data.covalent_radii, mult=1):
    return [radii[zi] * mult for zi in z]

def check_symmetric(am, tol=1e-8):
    return sp.linalg.norm(am - am.T, np.inf) < tol

def check_connected(am, tol=1e-8):
    sums = am.sum(axis=1)
    lap = np.diag(sums) - am
    eigvals, eigvects = np.linalg.eig(lap)
    return len(np.where(abs(eigvals) < tol)[0]) < 2

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException()

def smilify_cell2mol(filename, z=None, coordinates=None, timeout=30):
    if timeout is not None:
        signal.signal(signal.SIGALRM, _timeout_handler)
    covalent_factors = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    ok = False
    
    try:
        for covalent_factor in covalent_factors:

            if (z is None) and (coordinates is None):
                assert filename.endswith(".xyz"), "Input file must be an .xyz"
                mol = next(read_xyz(open(filename)))
                # initial charge from file, but default to 0 for neutral guess
                charge = sum(mol.get_initial_charges())
                charge = 0
                atoms = mol.get_chemical_symbols()
                z = [int(zi) for zi in mol.get_atomic_numbers()]
                coordinates = mol.get_positions()

            cutoff = get_cutoffs(z, radii=ase.data.covalent_radii, mult=covalent_factor)
            nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
            nl.update(mol)
            AC = nl.get_connectivity_matrix(sparse=False)

            try:
                assert check_connected(AC) and check_symmetric(AC)
                # attempt mol generation, possibly multiple times
                mol = xyz2mol(
                    z,
                    coordinates,
                    AC,
                    covalent_factor,
                    charge=charge,
                    use_graph=True,
                    allow_charged_fragments=True,
                    embed_chiral=True,
                    use_huckel=True,
                )
                if isinstance(mol, list):
                    mol = mol[0]

                # sanitize and check formal charges
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
                # check for pathological case: every atom has nonzero formal charge
                fcharges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
                heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
                n_fcharge_nonzero = sum(1 for fc in fcharges if fc != 0)
            
                if n_fcharge_nonzero > len(heavy_atoms)/2: # If more than half of heavy atoms are charged, it is probably charge
                    best = _pick_best_charge(z, coordinates, AC, covalent_factor)
                    mol, charge = best

                smiles = mol2smi(mol)
                if isinstance(smiles, list):
                    smiles = smiles[0]

                # final checks
                if mol is None:
                    warnings.warn(f"{filename}: RDKit failed to convert to mol. Skipping.")
                    ok = False
                else:
                    match_idx = simple_idx_match_check(mol, atoms)
                    if not match_idx:
                        warnings.warn(
                            f"{filename}: Index mismatch between RDKit and ASE atoms. Skipping."
                        )
                        return None, None
                ok = True
                # print("Yay passed with covalent factor", covalent_factor)
                break

            except Exception as e:
                print("Attempt failed for factor", covalent_factor, "error:", e) # verb
                continue
    except TimeoutException:
        print(f"{filename}: timed out after {timeout} seconds")
        return None, None
    finally:
        # make sure no alarm is left pending
        if timeout is not None:
            signal.alarm(0)

    return (smiles, mol) if ok else (None, None)
    
def _singlepoint_energy(z, coords, charge, tmp_prefix="xtb"):
    """Helper: write XYZ, run xtb --sp, parse and return energy (Hartree)."""
    # 1) dump XYZ
    with tempfile.NamedTemporaryFile(prefix=tmp_prefix, suffix=".xyz", delete=False) as tmp:
        fname = tmp.name
        lines = [str(len(z)), ""]
        for Zi, (x, y, zc) in zip(z, coords):
            lines.append(f"{Zi} {x:.6f} {y:.6f} {zc:.6f}")
        tmp.write("\n".join(lines).encode("utf-8"))

    try:
        # 2) call xtb
        res = subprocess.run(
            ["xtb", fname, "--sp", "--chrg", str(charge)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # 3) parse energy
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)", res.stdout)
        if not m:
            raise RuntimeError("Unable to parse xtb energy")
        return float(m.group(1))
    finally:
        os.remove(fname)


def _pick_best_charge(z, coords, adj, covalent_factor):
    """
    Compute E(0), E(-1), E(+1) with GFN2-xTB single-point.
    Return the (mol, charge) pair with lowest ΔE, where
      ΔE(-1) = E(-1) - E(0)
      ΔE(+1) = E(0) - E(+1)
    """
    mols = {}
    energies = {}

    # --- neutral baseline ---
    mol0 = xyz2mol(
        z, coords, adj, covalent_factor,
        charge=0, use_graph=True, allow_charged_fragments=True,
        embed_chiral=True, use_huckel=True
    )
    if isinstance(mol0, list): mol0 = mol0[0]
    Chem.SanitizeMol(mol0, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
    E0 = _singlepoint_energy(z, coords, 0, tmp_prefix="xtb_neutral")
    mols[0] = mol0
    energies[0] = E0

    # --- charged states ---
    for q in (-1, +1):
        try:
            molq = xyz2mol(
                z, coords, adj, covalent_factor,
                charge=q, use_graph=True, allow_charged_fragments=True,
                embed_chiral=True, use_huckel=True
            )
            if isinstance(molq, list): molq = molq[0]
            Chem.SanitizeMol(molq, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)

            Eq = _singlepoint_energy(z, coords, q, tmp_prefix=f"xtb_charge{q}")
            mols[q] = molq
            energies[q] = Eq

        except Exception as e:
            print(f"xTB/SP failed for charge {q}: {e}")

    # --- compute ΔE with the updated definitions ---
    deltas = {}
    if -1 in energies:
        deltas[-1] = energies[-1] - E0
    if +1 in energies:
        deltas[+1] = E0 - energies[+1]

    if not deltas:
        raise RuntimeError("No charged-state energies available")

    # pick charge with smallest ΔE
    best_q = min(deltas, key=deltas.get) # verb
    print(f"E(0) = {E0:.6f}  E(-1) = {energies.get(-1,'n/a')}  E(+1) = {energies.get(+1,'n/a')}")
    print(f"ΔE(-1) = {deltas.get(-1,'n/a'):.6f}  ΔE(+1) = {deltas.get(+1,'n/a'):.6f}")
    print(f"Best charge: {best_q}  (ΔE = {deltas[best_q]:.6f})")

    return mols[best_q], best_q

#%% Openbabel

bond_dict = [None,
             Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]

def cov_radius(symbol: str) -> float:
    """Return covalent radius (Å) for a chemical symbol (via ASE)."""
    Z = atomic_numbers.get(symbol.capitalize(), None)
    if Z is None:
        raise ValueError(f"Unknown element symbol '{symbol}'.")
    return covalent_radii[Z]


def _try_import_openbabel():
    try:
        from openbabel import openbabel as ob  # type: ignore
        ob.obErrorLog.SetOutputLevel(0)
        return ob  # noqa: E501
    except ImportError:
        return None


_ob = _try_import_openbabel()


def _warn_openbabel_once():
    if not hasattr(_warn_openbabel_once, "done"):
        print("[INFO] Open Babel not found – falling back to single‑bond guessing.")
        _warn_openbabel_once.done = True  # type: ignore[attr-defined]


def guess_bond_matrix(
    symbols: List[str],
    positions: torch.Tensor,
    *,
    scale: float = 1.2,
    total_charge: int | None = None,
) -> torch.LongTensor:
    """Return an (n, n) bond‐order matrix, using Open Babel if available.
    If total_charge is given, we tell OBMol about it so it can distribute
    formal charges when perceiving bond orders."""
    if _ob is not None:
        obmol = _ob.OBMol()
        # build OBMol...
        for sym, coord in zip(symbols, positions):
            a = obmol.NewAtom()
            a.SetAtomicNum(int(atomic_numbers[sym]))
            a.SetVector(*[float(c) for c in coord.tolist()])
        if total_charge is not None:
            obmol.SetTotalCharge(int(total_charge))
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.SetTotalSpinMultiplicity(1)
        obmol.FindRingAtomsAndBonds()
        obmol.SetAromaticPerceived()

        n = len(symbols)
        mat = torch.zeros((n, n), dtype=torch.long)
        for bond in _ob.OBMolBondIter(obmol):
            i = bond.GetBeginAtomIdx() - 1
            j = bond.GetEndAtomIdx() - 1
            order = bond.GetBondOrder()
            if bond.IsAromatic():
                order = 4
            mat[i, j] = mat[j, i] = order
        return mat

    # Fallback to distance cutoff (single‐bonds only):
    _warn_openbabel_once()
    n = len(symbols)
    adj = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(i+1, n):
            cutoff = scale * (cov_radius(symbols[i]) + cov_radius(symbols[j]))
            if torch.dist(positions[i], positions[j]) <= cutoff:
                adj[i, j] = adj[j, i] 
    return adj

def read_xyz(path: os.PathLike | str) -> Tuple[List[str], torch.Tensor]:
    atoms = ase_read(path)
    return atoms.get_chemical_symbols(), torch.tensor(atoms.get_positions(), dtype=torch.float)

class Molecule:
    def __init__(
        self,
        symbols: List[str],
        positions: torch.Tensor,
        *,
        scale: float = 1.2,
        total_charge: int = 0,
    ):
        self.symbols = [s.capitalize() for s in symbols]
        self.positions = positions.float()
        self.num_atoms = len(self.symbols)
        self.total_charge = total_charge
        self.bond_matrix = guess_bond_matrix(
            self.symbols, self.positions,
            scale=scale, total_charge=total_charge
        )
        self.rdkit_mol = self._build_rdkit_mol()

    def _build_rdkit_mol(self) -> Chem.Mol | None:
        """Build an RDKit molecule, setting the total charge via atom formal charges."""
        rwm = Chem.RWMol()
        # add atoms
        for sym in self.symbols:
            a = Chem.Atom(sym)
            # we don't know per‐atom formal charges here; they should
            # already have been set in the OBMol step if needed
            rwm.AddAtom(a)
        # add bonds
        for i, j in torch.nonzero(torch.triu(self.bond_matrix,1), as_tuple=False):
            bo = int(self.bond_matrix[i,j].item())
            rwm.AddBond(int(i), int(j), bond_dict[min(bo,4)])
        mol = rwm.GetMol()
        # set up 3D coords so sanitize can check valence
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx in range(mol.GetNumAtoms()):
            x,y,z = self.positions[idx].tolist()
            conf.SetAtomPosition(idx, Point3D(x,y,z))
        mol.AddConformer(conf)
        # store total charge as a prop so failures in sanitize show it
        mol.SetProp("_TotalCharge", str(self.total_charge))
        return mol

def smilify_openbabel(filename):
    
    SCALES = [1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    
    for scale in SCALES:
        symbols, pos = read_xyz(filename)
        mol = Molecule(symbols, pos, scale=scale)
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            smiles_list: list[str] = []
            for frag in frags:
                Chem.SanitizeMol(frag)
                smiles_list.append(Chem.MolToSmiles(frag))
        
        except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException,
            Chem.rdchem.AtomKekulizeException, ValueError):
            continue
        return smiles_list, mol.rdkit_mol
        # if len(smiles_list) > 1:
        #     # If multiple fragments, return the largest one
        #     largest_mol = None
        #     largest_smiles = None
        #     for frag in frags:
        #         if largest_mol is None or frag.GetNumAtoms() > largest_mol.GetNumAtoms():
        #             largest_mol = frag
        #             largest_smiles = Chem.MolToSmiles(frag)
        #     return largest_smiles, largest_mol
        # else:
        #     return smiles_list[0], mol.rdkit_mol
    return None, None

#%% xTB

def smilfy_xtb(filename):

    """
    Assume neutral molecule
    If fails, change charge to +1 (there could be a better method to detect charge as either +1 or-1)
    
    Then if fails again, loosen the critaria to not sanitize the molecule.

        
    """
    mol = None
    execution = ["xtb", filename]

    try:
        sp.call(execution, stdout=sp.DEVNULL, stderr=sp.STDOUT, timeout=TIMEOUT)
    except sp.TimeoutExpired:
        print("xTB calculation timed out.")
        return None, None
    
    mol = Chem.rdmolfiles.MolFromMolFile("xtbtopo.mol", removeHs=False, sanitize=True)

    if mol is None:
        execution.extend(["-c", "1"])
        sp.call(execution, stdout=sp.DEVNULL, stderr=sp.STDOUT, timeout=TIMEOUT)
        mol = Chem.rdmolfiles.MolFromMolFile("xtbtopo.mol", removeHs=False, sanitize=True
        )
    
    if mol is None:
        mol = Chem.rdmolfiles.MolFromMolFile("xtbtopo.mol", removeHs=False, sanitize=False
        )
        
    if mol is None:
        smiles = None
    else:
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

    os.remove("xtbtopo.mol")
    os.remove("wbo")
    os.remove("xtbrestart")
    os.remove("charges")

    return smiles, mol
