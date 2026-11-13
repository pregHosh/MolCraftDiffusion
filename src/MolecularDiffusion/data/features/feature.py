import warnings
from itertools import combinations

import ase
import networkx as nx
import numpy as np
import scipy.spatial
import torch
from ase import Atoms, neighborlist
from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from networkx.algorithms import community as nx_comm
from rdkit import Chem
from rdkit.Chem import AllChem

from MolecularDiffusion.utils.xyz2mol import ElementData

from MolecularDiffusion.utils.geometry.shape_measure import shape_measure


# less than 4 bonds
# 0 for S, 180 for SP, 120 for SP2, 109.5 for SP3
hybridization_dicts = {
    0: 0,
    180: 1,
    120: 2,
    109.5: 3,
}

# more than 4 bonds
# 4 for SP3D, 5 for SP3D2, 6 for SP3D3
hybridization_dicts_2 = {
    120: 4,
    90: 5,
    70: 6,    
}

vertices_labels = {
    2: ["L-2", "vT-2",],
    3: ["TP-3", "vT-3", "mvOC-3"],
    4: ["SP-4", "T-4", ],
    5: ["PP-5", "TBPY-5", "SPY-5"],
    6: ["HP-6", "PPY-6","TPR-6"],
    7: ["HPY-7", "PBPY-7",],
    8: ["HPY-8", "HBPY-8", "SAPR-8"],
}


vertices_labels_all = [
    "L-0", # for atoms with one neighbors
    "L-2",
    "vT-2",
    "TP-3",
    "vT-3",
    "mvOC-3",
    "SP-4",
    "T-4",
    "PP-5",
    "TBPY-5",
    "SPY-5",
    "HP-6",
    "PPY-6",
    "TPR-6",
    "HPY-7",
    "PBPY-7",
    "HPY-8",
    "HBPY-8",
    "SAPR-8",
    "XX" # for atoms with more than 8 neighbors
]
vertices_labels_vocab = {b: i for i, b in enumerate(vertices_labels_all)}
N_vertices_labels = len(vertices_labels_all)


# orderd by perodic table
atom_vocab = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Cu",
    "Zn",
    "Ge",
    "As",
    "Se",
    "Br",
    "Sn",
    "I",
]

atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(8)
num_hs_vocab = range(7)
formal_charge_vocab = range(-6, 9)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

bond_type_vocab = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(
                "Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab)
            )
        feature[index] = 1

    return feature


class NodeFeaturizer:
    """
    Unified node featurization for both GraphDataset and PointCloudDataset.
    
    Supports:
    - OHE (one-hot encoding) of atom types
    - Geom-based features from atomic numbers and coordinates
    
    Usage:
        featurizer = NodeFeaturizer(atom_vocab, use_ohe=True, geom_feature="geometric_fast")
        features = featurizer.featurize_all(atom_symbols, charges, coords)
    """
    
    # Feature name -> function mapping
    # Includes both intuitive names and legacy aliases
    GEOM_FEATURES = None  # Populated after function definitions
    
    @classmethod
    def _init_features(cls):
        """Initialize feature mapping after functions are defined."""
        if cls.GEOM_FEATURES is None:
            cls.GEOM_FEATURES = {
                # Intuitive names
                "topological": atom_topological,
                "geometric": atom_geom,
                "geometric_full": atom_geom_v2,
                "geometric_fast": atom_geom_compact,
                "geometric_hybrid": atom_geom_opt,
                # Legacy aliases (backward compat)
                "atom_topological": atom_topological,
                "atom_geom": atom_geom,
                "atom_geom_v2": atom_geom_v2,
                "atom_geom_v2_trun": atom_geom_v2_trun,
                "atom_geom_opt": atom_geom_opt,
                "atom_geom_compact": atom_geom_compact,
            }
    
    @classmethod
    def available_features(cls) -> list:
        """Return list of available feature names."""
        cls._init_features()
        return list(cls.GEOM_FEATURES.keys())
    
    def __init__(
        self,
        atom_vocab: list,
        use_ohe: bool = True,
        geom_feature: str = None,
        allow_unknown: bool = False,
    ):
        """
        Args:
            atom_vocab: List of atom symbols for OHE encoding
            use_ohe: Whether to include one-hot encoding of atom type
            geom_feature: Name of geom feature type (None for OHE only)
            allow_unknown: Whether to allow unknown atoms in OHE
        """
        self._init_features()
        self.atom_vocab = atom_vocab
        self.use_ohe = use_ohe
        self.geom_feature = geom_feature
        self.allow_unknown = allow_unknown
        
        if geom_feature is not None and geom_feature not in self.GEOM_FEATURES:
            raise ValueError(
                f"Unknown geom feature type '{geom_feature}'. "
                f"Available: {list(self.GEOM_FEATURES.keys())}"
            )
        self._geom_fn = self.GEOM_FEATURES.get(geom_feature) if geom_feature else None
    
    def compute_ohe(self, atom_symbols: list) -> torch.Tensor:
        """
        Compute one-hot encoding for atom symbols.
        
        Args:
            atom_symbols: List of atom symbols (e.g., ['C', 'H', 'O'])
            
        Returns:
            OHE tensor (N, len(vocab)) or (N, len(vocab)+1) if allow_unknown
        """
        features = []
        for symbol in atom_symbols:
            features.append(onehot(symbol, self.atom_vocab, allow_unknown=self.allow_unknown))
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_geom(
        self,
        charges: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geom features from atomic numbers and coordinates.
        
        Args:
            charges: Atomic numbers tensor (N,)
            coords: Coordinates tensor (N, 3)
            
        Returns:
            Geom features tensor (N, F)
        """
        if self._geom_fn is None:
            raise ValueError("No geom feature specified")
        return self._geom_fn(charges, coords)
    
    def featurize_all(
        self,
        atom_symbols: list,
        charges: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute all features: OHE + geom (if specified).
        
        Args:
            atom_symbols: List of atom symbols
            charges: Atomic numbers tensor (N,)
            coords: Coordinates tensor (N, 3)
            
        Returns:
            Combined features tensor (N, total_dim) or None if no features
        """
        features = None
        
        if self.use_ohe:
            features = self.compute_ohe(atom_symbols)
        
        if self._geom_fn is not None:
            geom_feats = self.compute_geom(charges, coords)
            if features is not None:
                features = torch.cat([features, geom_feats], dim=1)
            else:
                features = geom_feats
        
        return features
    
    # Legacy method for backward compatibility
    def featurize(
        self,
        charges: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geom features only (legacy interface)."""
        return self.compute_geom(charges, coords)


#%% RDKIT
def atom_default(atom):
    """Default atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetChiralTag(), chiral_tag_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetNumRadicalElectrons(), num_radical_vocab)
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


def atom_default_extra(atom):
    """Default atom feature.

    Features:

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    num_hs_vocab = range(5)
    valence_vocab = range(8)
    return (
        # onehot(atom.GetChiralTag(), chiral_tag_vocab)
        onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), valence_vocab)
        # + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + onehot(atom.GetTotalNumHs(True), num_hs_vocab)
        # + onehot(atom.GetNumRadicalElectrons(), num_radical_vocab)
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


def atom_default_condense(atom):
    """Default atom feature.

    Features:

        GetTotalDegree(): the degree of the atom in the molecule including Hs

        GetTotalDegree(): the degree of the atom in the molecule including Hs

        GetTotalValence(): the total valence (explicit + implicit) of the atom

        GetTotalNumHs(): the total number of Hs (explicit and implicit) on the atom

        GetHybridization(): one-hot embedding for the atom's hybridization

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        # [atom.GetChiralTag()] # not specified
        [atom.GetTotalDegree()]
        + [atom.GetTotalValence()]
        + [atom.GetFormalCharge()]
        + [atom.GetTotalNumHs(True)]
        # + [atom.GetTotalNumHs()] # somehow return all 0
        # + [atom.GetNumRadicalElectrons()]
        + onehot(atom.GetHybridization(), hybridization_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


def atom_center_identification(atom):
    """Reaction center identification atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetIsAromatic(): whether the atom is aromatic

        IsInRing(): whether the atom is in a ring
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab)
        + [atom.GetIsAromatic(), atom.IsInRing()]
    )


def atom_synthon_completion(atom):
    """Synthon completion atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs

        IsInRing(): whether the atom is in a ring

        IsInRingSize(3, 4, 5, 6): whether the atom is in a ring of a particular size

        IsInRing() and not IsInRingSize(3, 4, 5, 6): whether the atom is in a ring and not in a ring of 3, 4, 5, 6
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab)
        + onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True)
        + [
            atom.IsInRing(),
            atom.IsInRingSize(3),
            atom.IsInRingSize(4),
            atom.IsInRingSize(5),
            atom.IsInRingSize(6),
            atom.IsInRing()
            and (not atom.IsInRingSize(3))
            and (not atom.IsInRingSize(4))
            and (not atom.IsInRingSize(5))
            and (not atom.IsInRingSize(6)),
        ]
    )


def atom_symbol(atom):
    """Symbol atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)


def atom_explicit_property_prediction(atom):
    """Explicit property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab)
        + [atom.GetIsAromatic()]
    )


def atom_property_prediction(atom):
    """Property prediction atom feature.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetDegree(): one-hot embedding for the degree of the atom in the molecule

        GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom

        GetTotalValence(): one-hot embedding for the total valence (explicit + implicit) of the atom

        GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule

        GetIsAromatic(): whether the atom is aromatic
    """
    return (
        onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True)
        + onehot(atom.GetDegree(), degree_vocab, allow_unknown=True)
        + onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True)
        + onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True)
        + onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True)
        + [atom.GetIsAromatic()]
    )


def atom_pretrain(atom):
    """Atom feature for pretraining.

    Features:
        GetSymbol(): one-hot embedding for the atomic symbol

        GetChiralTag(): one-hot embedding for atomic chiral tag
    """
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + onehot(
        atom.GetChiralTag(), chiral_tag_vocab
    )


def bond_default(bond):
    """Default bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond

        GetStereo(): one-hot embedding for the stereo configuration of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated
    """
    return (
        onehot(bond.GetBondType(), bond_type_vocab)
        + onehot(bond.GetBondDir(), bond_dir_vocab)
        + onehot(bond.GetStereo(), bond_stereo_vocab)
        + [int(bond.GetIsConjugated())]
    )


def bond_length(bond):
    """
    Bond length in the molecular conformation.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]


def bond_property_prediction(bond):
    """Property prediction bond feature.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetIsConjugated(): whether the bond is considered to be conjugated

        IsInRing(): whether the bond is in a ring
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + [
        int(bond.GetIsConjugated()),
        bond.IsInRing(),
    ]


def bond_pretrain(bond):
    """Bond feature for pretraining.

    Features:
        GetBondType(): one-hot embedding for the type of the bond

        GetBondDir(): one-hot embedding for the direction of the bond
    """
    return onehot(bond.GetBondType(), bond_type_vocab) + onehot(
        bond.GetBondDir(), bond_dir_vocab
    )


def ExtendedConnectivityFingerprint(mol, radius=2, length=1024):
    """Extended Connectivity Fingerprint molecule feature.

    Features:
        GetMorganFingerprintAsBitVect(): a Morgan fingerprint for a molecule as a bit vector
    """
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    return list(ecfp)


def molecule_default(mol):
    """Default molecule feature."""
    return ExtendedConnectivityFingerprint(mol)


ECFP = ExtendedConnectivityFingerprint

#%% Geom


def get_cutoffs(z, radii=ase.data.covalent_radii, mult=1):
    return [radii[zi] * mult for zi in z]

def get_radii(z, radii="covalent"):
    if radii == "covalent":
        return np.array([covalent_radii[i] for i in z], dtype=float)
    if radii == "vdw":
        return np.array([vdw_radii[i] for i in z], dtype=float)

def get_other_features(z):
    ed = ElementData()
    sym = [ed.elementsym[zi.item()] for zi in z]
    ve = [ed.valenceelectrons[symi] for symi in sym]
    en = [ed.ElectroNegativityPauling[symi] for symi in sym]
    return ve, en

def get_dm(z, coordinates):
    n = len(z)
    dm = scipy.spatial.distance.pdist(coordinates)
    return dm

def get_am(z, coordinates, radii, dm, scale_factor=1.15):
    n = len(z)
    am = np.zeros((n, n), dtype=float)
    row, col = np.triu_indices(n, 1)
    rm = scale_factor * scipy.spatial.distance.pdist(
        radii.reshape(-1, 1), metric=lambda x, y: x + y
    )
    am[row, col] = am[col, row] = dm - rm
    return am < 0

def calculate_bond_angles(coordinates, connectivity_matrix, central_atom, degree=True):
    """
    coordinates: (N, 3) torch tensor
    connectivity_matrix: (N, N) torch tensor (uint8 or bool)
    central_atom: int index
    Returns:
        angles: (M, 3) torch.FloatTensor of (atom1, atom2, angle)
    """
    # Find bonded neighbors
    bonded_atoms = torch.nonzero(connectivity_matrix[central_atom], as_tuple=False).squeeze(-1)

    if bonded_atoms.numel() < 2:
        return torch.tensor([[0.0, 0.0, 0.0]], device=coordinates.device)

    # Central atom's coordinate
    central_coord = coordinates[central_atom]  # (3,)
    vectors = coordinates[bonded_atoms] - central_coord  # (n_bonds, 3)

    norms = torch.norm(vectors, dim=1)  # (n_bonds,)
    if torch.any(norms == 0):
        raise ValueError("Zero-length bond detected.")

    unit_vectors = vectors / norms[:, None]  # (n_bonds, 3)

    angles = []
    for (i, atom1), (j, atom2) in combinations(enumerate(bonded_atoms.tolist()), 2):
        vec1 = unit_vectors[i]
        vec2 = unit_vectors[j]
        dot = torch.clamp(torch.dot(vec1, vec2), -1.0, 1.0)
        angle_rad = torch.acos(dot)
        angle = torch.rad2deg(angle_rad) if degree else angle_rad
        # angles.append((atom1, atom2, angle.item()))
        angles.append(angle.item())

    return torch.tensor(angles, dtype=torch.float32, device=coordinates.device)


def bin(arr, n_bins=5):
    intervals = np.linspace(min(arr), max(arr), n_bins)
    return np.digitize(arr, intervals)

def atom_topological(z, coords):

    TOL = 0.2
    cov_radii = get_radii(z, radii="covalent")
    cov_dm = get_dm(z, coords)
    cov_am = get_am(z, coords, cov_radii, cov_dm, scale_factor=1.15)
    G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
    degree = [val for (i, val) in G.degree()]
    if not (cov_dm > TOL).all():
        print("Some atoms are incredibly close to each other!")
    ve, en = get_other_features(z)

    # TODO implement 3D info later
    # TODO these in float atm, how to bin them?
    closeness = nx.closeness_centrality(G, distance="weight")
    closeness_arr = np.array([closeness[i] for i in range(len(z))])

    centrality = nx.betweenness_centrality(G, weight="weight")
    centrality_arr = np.array([centrality[i] for i in range(len(z))])

    communities = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
    community_dict = {
        node: idx for idx, comm in enumerate(communities) for node in comm
    }
    community_arr = np.array([community_dict[i] for i in range(len(z))])

    geom_node_feat = torch.tensor(
        [
            ve,
            en,
            cov_radii,
            degree,
            closeness_arr,
            centrality_arr,
            community_arr,
        ],
        dtype=torch.float32,
    ).T

    return geom_node_feat

def atom_geom(z, coords):

    TOL = 0.2
    cov_radii = get_radii(z, radii="covalent")
    vdw_radii = get_radii(z, radii="vdw")
    cov_dm = get_dm(z, coords)
    cov_am = get_am(z, coords, cov_radii, cov_dm, scale_factor=1.15)
    G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
    degree = [val for (i, val) in G.degree()]
    if not (cov_dm > TOL).all():
        print("Some atoms are incredibly close to each other!")

    # a_volume, a_surface = atomic_vs(coords, vdw_radii)
    from morfeus import SASA  # lazy: only SASA features need morfeus-ml

    sasa = SASA(z, coords, vdw_radii, probe_radius=0.0, density=0.01)
    sa_volume = np.fromiter(sasa.atom_volumes.values(), dtype=float)
    sa_surface = np.fromiter(sasa.atom_areas.values(), dtype=float)
    ve, en = get_other_features(z)

    geom_node_feat = torch.tensor(
        [degree, ve, en, sa_volume, sa_surface, cov_radii], dtype=torch.float32
    ).T
    return geom_node_feat

def atom_geom_v2_trun(z, coords):

    TOL = 0.2

    mol = Atoms(symbols=z, positions=coords)
    cutoff = get_cutoffs(z, radii=ase.data.covalent_radii, mult=1)
    nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(mol)
    AC = nl.get_connectivity_matrix(sparse=False)

    cov_radii = get_radii(z, radii="covalent")
    cov_dm = get_dm(z, coords)
    cov_am = get_am(z, coords, cov_radii, cov_dm, scale_factor=1.15)
    G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
    degree = [val for (i, val) in G.degree()]
    if not (cov_dm > TOL).all():
        print("Some atoms are incredibly close to each other!")
    ve, en = get_other_features(z)

    # TODO implement 3D info later
    closeness = nx.closeness_centrality(G, distance="weight")
    closeness_arr = np.array([closeness[i] for i in range(len(z))])

    centrality = nx.betweenness_centrality(G, weight="weight")
    centrality_arr = np.array([centrality[i] for i in range(len(z))])

    communities = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
    community_dict = {
        node: idx for idx, comm in enumerate(communities) for node in comm
    }
    community_arr = np.array([community_dict[i] for i in range(len(z))])

    angle_all = np.zeros_like(z, dtype=float)
    for central_atom in range(len(z)):
        angle_ = calculate_bond_angles(coords, AC, central_atom, degree=True)
        print(angle_)
        if (angle_ == 0).all():
            avergage_angle = 0
        else:
            avergage_angle = np.mean(angle_[:, -1])
        angle_all[central_atom] = avergage_angle

    geom_node_feat = torch.tensor(
        [
            ve,
            en,
            cov_radii,
            degree,
            closeness_arr,
            centrality_arr,
            community_arr,
            angle_all,
        ],
        dtype=torch.float32,
    ).T
    return geom_node_feat


def atom_geom_compact(z, coords, scale_factor=1.3):

    device = coords.device
    N = coords.size(0)

    # Get covalent radii
    r = torch.tensor(covalent_radii[z], dtype=torch.float32, device=device)  # (N,)

    # Compute distance matrix
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
    dists = diff.norm(dim=-1)  # (N, N)

    # Build adjacency mask
    r_sum = r.unsqueeze(1) + r.unsqueeze(0)  # (N, N)
    cutoff = r_sum * scale_factor 
    adj = (dists < cutoff) & (dists > 1e-6)  # Avoid self-loops (d > 0)
    # Degree
    degree_tensor = adj.sum(dim=-1)

    # degree = torch.tensor(degree, dtype=torch.float32)
    ve, _ = get_other_features(z)

    ve, _ = get_other_features(z)
    ve = torch.tensor(ve, dtype=torch.float32)  

    # hybridizations = torch.zeros_like(ve)

    geom_node_feat = torch.stack([ve, degree_tensor], dim=-1)
    return geom_node_feat

def atom_geom_opt(z, coords, scale_factor = 1.3):

    device = coords.device
    N = coords.size(0)

    # Get covalent radii
    r = torch.tensor(covalent_radii[z], dtype=torch.float32, device=device)  # (N,)
    degree_tensor = torch.zeros(z.shape[0], dtype=torch.float32)

    # Compute distance matrix
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
    dists = diff.norm(dim=-1)  # (N, N)

    # Build adjacency mask
    r_sum = r.unsqueeze(1) + r.unsqueeze(0)  # (N, N)
    cutoff = r_sum * scale_factor 
    adj = (dists < cutoff) & (dists > 1e-6)  # Avoid self-loops (d > 0)
    connectivity_matrix = adj.to(torch.uint8)
    # Degree
    degree_tensor = adj.sum(dim=-1)

    hybridizations = torch.zeros(z.shape[0], dtype=torch.float32)
    use_hybridization_set2 = False

    for central_atom in range(len(z)):
        angles = calculate_bond_angles(coords, connectivity_matrix, central_atom, degree=True)
        if (angles == 0).all():
            probe_angle = 0 if z[central_atom] == 1 else 180
        else:
            if degree_tensor[central_atom] > 4:
                probe_angle = angles.max()
                use_hybridization_set2 = True
            else:
                probe_angle = angles.mean()

        if use_hybridization_set2:
            closest_key = min(hybridization_dicts_2.keys(), key=lambda k: abs(k - probe_angle))
            hybr = hybridization_dicts_2[closest_key]
        else:
            closest_key = min(hybridization_dicts.keys(), key=lambda k: abs(k - probe_angle))
            hybr = hybridization_dicts[closest_key]

        hybridizations[central_atom] = hybr

    hybridizations = torch.tensor(hybridizations, dtype=torch.float32)
    ve, _ = get_other_features(z)
    ve = torch.tensor(ve, dtype=torch.float32)  

    geom_node_feat = torch.stack([ve, degree_tensor, hybridizations], dim=-1)
    return geom_node_feat

def atom_geom_shape(z, coords, scale_factor = 1.3):

    device = coords.device
    N = coords.size(0)

    # Get covalent radii
    r = torch.tensor(covalent_radii[z], dtype=torch.float32, device=device)  # (N,)

    # Compute distance matrix
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
    dists = diff.norm(dim=-1)  # (N, N)

    # Build adjacency mask
    r_sum = r.unsqueeze(1) + r.unsqueeze(0)  # (N, N)
    cutoff = r_sum * scale_factor 
    adj = (dists < cutoff) & (dists > 1e-6)  # Avoid self-loops (d > 0)
    connectivity_matrix = adj.to(torch.uint8)
    # Degree
    degree = adj.sum(dim=-1)
    
    shps = []
    for node in range(z.shape[0]):
        shp_scores = {}
        adjacent_nodes = np.where(connectivity_matrix[node] == 1)[0]
        n_degree = len(adjacent_nodes)
        nodes_all = np.array([node] + adjacent_nodes.tolist())
        positions = coords[nodes_all].detach().cpu().numpy()
        if n_degree > 1 and n_degree < 9:
            shp_types = vertices_labels[n_degree]

            for shp_type in shp_types:
                shp_scores[shp_type] = shape_measure(
                    positions, shp_type, central_atom=1
                )
            shp_type = min(shp_scores, key=shp_scores.get)
        elif n_degree == 1:
            shp_type = "L-0"
        else:
            shp_type = "XX"
        shp_label = vertices_labels_vocab[shp_type]
        # shp_oh =  onehot(shp_type, vertices_labels_vocab)   
        shps.append(shp_label)  
    
    shp_ohs = torch.tensor(shps, dtype=torch.float32)
    degree = torch.tensor(degree, dtype=torch.float32)

    geom_node_feat = torch.cat(
        [
            degree.unsqueeze(-1),
            shp_ohs.unsqueeze(-1),
        ],
        dim=-1,
    )

    return geom_node_feat



def atom_geom_v2(z, coords):

    mol = Atoms(symbols=z, positions=coords)
    cutoff = get_cutoffs(z, radii=ase.data.covalent_radii, mult=1)
    nl = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(mol)
    AC = nl.get_connectivity_matrix(sparse=False)

    TOL = 0.2
    cov_radii = get_radii(z, radii="covalent")
    vdw_radii = get_radii(z, radii="vdw")
    cov_dm = get_dm(z, coords)
    cov_am = get_am(z, coords, cov_radii, cov_dm, scale_factor=1.15)
    G = nx.from_numpy_array(cov_am, create_using=nx.Graph)
    degree = [val for (i, val) in G.degree()]
    if not (cov_dm > TOL).all():
        print("Some atoms are incredibly close to each other!")

    # a_volume, a_surface = atomic_vs(coords, vdw_radii)
    from morfeus import SASA  # lazy: only SASA features need morfeus-ml

    sasa = SASA(z, coords, vdw_radii, probe_radius=0.0, density=0.01)
    sa_volume = np.fromiter(sasa.atom_volumes.values(), dtype=float)
    sa_surface = np.fromiter(sasa.atom_areas.values(), dtype=float)
    ve, en = get_other_features(z)

    # TODO implement 3D info later
    # TODO these in float atm, how to bin them?
    closeness = nx.closeness_centrality(G, distance="weight")
    closeness_arr = np.array([closeness[i] for i in range(len(z))])

    centrality = nx.betweenness_centrality(G, weight="weight")
    centrality_arr = np.array([centrality[i] for i in range(len(z))])

    communities = np.array(
        nx_comm.greedy_modularity_communities(G, weight="weight")
    )
    community_dict = {
        node: idx for idx, comm in enumerate(communities) for node in comm
    }
    community_arr = np.array([community_dict[i] for i in range(len(z))])

    angle_all = np.zeros_like(z, dtype=float)
    for central_atom in range(len(z)):
        angle_ = calculate_bond_angles(coords, AC, central_atom, degree=True)
        if (angle_ == 0).all():
            avergage_angle = 0
        else:
            avergage_angle = np.mean(angle_[:, -1])
        angle_all[central_atom] = avergage_angle

    geom_node_feat = torch.tensor(
        [
            ve,
            en,
            cov_radii,
            degree,
            closeness_arr,
            centrality_arr,
            community_arr,
            sa_volume,
            sa_surface,
            angle_all,
        ],
        dtype=torch.float32,
    ).T
    return geom_node_feat


__all__ = [
    "NodeFeaturizer",
    "onehot",
    "atom_default",
    "atom_center_identification",
    "atom_synthon_completion",
    "atom_symbol",
    "atom_explicit_property_prediction",
    "atom_property_prediction",
    "atom_position",
    "atom_pretrain",
    "atom_residue_symbol",
    "atom_geom",
    "atom_geom_compact",    
    "atom_topological",
    "atom_geom_v2",
    "atom_geom_opt",
    "bond_default",
    "bond_length",
    "bond_property_prediction",
    "bond_pretrain",
    "residue_symbol",
    "residue_default",
    "ExtendedConnectivityFingerprint",
    "molecule_default",
    "ECFP",
]

