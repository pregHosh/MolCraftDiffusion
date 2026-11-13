import math
import warnings
from dataclasses import dataclass

import torch

__ATOM_LIST__ = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V ",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
]


class PointCloud_Mol:
    """
    Point Cloud representation of a molecule.

    Parameters:
    _atoms: List of atoms in the molecule. Each atom is a list with the following elements:
        - Atomic symbol (str)
        - x coordinate (float)
        - y coordinate (float)
        - z coordinate (float)
    """

    def __init__(self, _atoms):
        if isinstance(_atoms[0], Atom):
            self.atoms = _atoms
        else:
            self.atoms = []
            i = 0
            for atom in _atoms:
                self.atoms.append(Atom(i, atom[0], atom[1], atom[2], atom[3]))
                i += 1

    def __str__(self):
        length = math.floor(math.log10(len(self.atoms))) + 1
        return "\n".join(
            f"{atom.index:{length}}  {atom.element:2} {atom.x:8.5f} {atom.y:8.5f} {atom.z:8.5f}"
            for atom in self.atoms
        )

    def __getitem__(self, index):
        return self.atoms[index]

    def get_coord(self):
        return torch.stack(
            [torch.tensor([atom.x, atom.y, atom.z]) for atom in self.atoms]
        )

    @classmethod
    def from_arrays(
        cls,
        zs: torch.Tensor,
        coords: torch.Tensor,
        with_hydrogen: bool = False,
        forbidden_atoms=[],
    ):
        """
        load arrays to mol representation

        parameters:
        zs: Tensor with the atomic numbers.
        coords: Tensor with the atomic coordinates.
        with_hydrogen: Boolean to include Hydrogen atoms in the representation.
        forbidden_atoms: List of forbidden atoms to be excluded from the representation.

        out:
        molrepr: Mol Object containing the molecular information (coordinates and elements).

        """
        natom = zs.size(0)
        molrepr = []
        for i in range(natom):
            atomic_symbol = int(zs[i])
            atomic_symbol = cls.str_atom(atomic_symbol)
            if not (with_hydrogen) and atomic_symbol == "H":
                continue
            if atomic_symbol in forbidden_atoms:
                return None
            molrepr.append([atomic_symbol, coords[i, 0], coords[i, 1], coords[i, 2]])
        return cls(molrepr)

    @classmethod
    def from_xyz(cls, _path: str, with_hydrogen: bool = False, forbidden_atoms=[]):
        """
        load_xyz(_path: str) -> molrepr: Mol

        Load molecule from an XYZ input file and initialize a Mol Object for it.

        parameters:
        _path: String with the path of the input file.
        with_hydrogen: Boolean to include Hydrogen atoms in the representation.
        forbidden_atoms: List of forbidden atoms to be excluded from the representation.

        out:
        molrepr: Mol Object containing the molecular information (coordinates and elements).

        """

        molrepr = []
        with open(_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number > 1:
                    if len(line.split()) < 4:
                        continue
                    if len(line.split()) == 4:
                        atomic_symbol, x, y, z = line.split()
                    else:
                        atomic_symbol, x, y, z = line.split()[:4]

                    if atomic_symbol in forbidden_atoms:
                        return None

                    if not (with_hydrogen) and atomic_symbol == "H":
                        continue
                    if not atomic_symbol.isalpha():
                        atomic_symbol = int(atomic_symbol)
                        atomic_symbol = cls.str_atom(atomic_symbol)
                    molrepr.append(
                        [atomic_symbol.capitalize(), float(x), float(y), float(z)]
                    )

        return cls(molrepr)

    # TODO get atom_type
    @classmethod
    def str_atom(cls, _atom: int) -> str:
        """
        str_atom(_atom: int) -> atom: str

        Convert integer atom to string atom.

        in:
        _atom: Integer with the atomic number.

        out:
        atom: String with the atomic element.

        """

        atom = __ATOM_LIST__[_atom - 1]
        return atom


@dataclass
class Atom:
    index: int
    element: str
    x: float
    y: float
    z: float

    def __repr__(self):
        return (
            f"{self.index:4} {self.element:2} {self.x:8.5f} {self.y:8.5f} {self.z:8.5f}"
        )

    def __hash__(self):
        return hash(f"{self.index}{self.element}{self.x}{self.y}{self.z}")

    def __eq__(self, other):
        return (
            self.index == other.index
            and self.element == other.element
            and math.isclose(self.x, other.x, 1e-9, 1e-9)
            and math.isclose(self.y, other.y, 1e-9, 1e-9)
            and math.isclose(self.z, other.z, 1e-9, 1e-9)
        )

    def get_coord(self):
        return [self.x, self.y, self.z]


try:
    import ase
    import numpy as np
    import scipy as sp
    from ase import neighborlist
    from ase.io.extxyz import read_xyz
    from MolecularDiffusion.utils.xyz2mol import xyz2mol
    from rdkit import RDLogger
    from rdkit.Chem import MolToSmiles as mol2smi
    from rdkit import Chem
    RDLogger.DisableLog("rdApp.*")

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

    def smilify(filename, z=None, coordinates=None):
        covalent_factors = [1.0, 1.05, 1.10, 1.15, 1.20]
        ok = False
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

                    # retry with explicit net charge adjustments
                    for trial_charge in (+1, -1):
                        try:
                            trial_mol = xyz2mol(
                                z,
                                coordinates,
                                AC,
                                covalent_factor,
                                charge=trial_charge,
                                use_graph=True,
                                allow_charged_fragments=True,
                                embed_chiral=True,
                                use_huckel=True,
                            )
                            if isinstance(trial_mol, list):
                                trial_mol = trial_mol[0]
                            Chem.SanitizeMol(trial_mol, Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True)
                            # accept first successful retry
                            mol = trial_mol
                            break
                        except Exception:
                            continue

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
                print("Yay passed with covalent factor", covalent_factor)
                break

            except Exception as e:
                print("Attempt failed for factor", covalent_factor, "error:", e)
                continue

        if ok:
            return smiles, mol
        else:
            return None, None

except ImportError:
    smilify = None
    warnings.warn("xyz2mol not installed; xyz-to-SMILES conversion unavailable.", ImportWarning, stacklevel=2)

