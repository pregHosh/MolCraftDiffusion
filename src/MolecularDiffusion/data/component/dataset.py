
import csv
import logging
import math
from typing import Callable, Dict, List, Optional, Union, Any
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils import data as torch_data
from tqdm import tqdm

from ase.data import atomic_numbers
from ase.db import connect
try:
    from rdkit import Chem
except ImportError:
    Chem = None
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph

from MolecularDiffusion.utils import sascore
from MolecularDiffusion import core, utils
from MolecularDiffusion.utils.smilify import smilify_cell2mol as smilify
from .feature import (
    onehot,
    atom_default_condense,
    atom_default_extra,
    atom_geom,
    atom_geom_v2,
    atom_geom_v2_trun,
    atom_topological,
    atom_geom_compact,
    atom_geom_opt
)
from .pointcloud import PointCloud_Mol

logger = logging.getLogger(__name__)

hybiridization_map = {
    "S": 0, 'SP': 1, 'SP2': 2, 'SP3': 3, 'SP3D': 4, 'SP3D2': 5, 'UNSPECIFIED': -1
}
        
BASE_ATOM_VOCAB = [
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

class GraphDataset(torch_data.Dataset):
    
    def load_smiles(self):
        pass
    
    def load_csv(
        self,
        csv_file: str,
        xyz_dir: str,
        xyz_field: str = "xyz",
        smiles_field: str = "smiles",
        target_fields: Optional[List[str]] = None,
        atom_vocab: List[str] = [],
        node_feature: Optional[str] = None,
        forbidden_atoms: List[str] = [],
        verbose: int = 0,
        **kwargs,
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            xyz_dir (str): directory to store XYZ files
            xyz_field (str): name of the XYZ column in the table
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            atom_vocab (list of str, optional): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        if xyz_field is None:
            raise ValueError("xyz_field must be provided")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            print("atom vocabulary not provided, using defaul in constant.py")
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            xyzs = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == xyz_field:
                        xyz_path = os.path.join(xyz_dir, f"{value}.xyz")
                        xyzs.append(xyz_path)
                    elif field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
        assert len(xyzs) > 0, "No XYZ files found"
        # TODO to deal with when xyz but absence smiles and vice versa, skip it for now
        self.load_xyz(
            xyzs,
            smiles,
            targets,
            atom_vocab,
            forbidden_atoms=forbidden_atoms,
            node_feature=node_feature,
            verbose=verbose,
            **kwargs,
        )
    
    def load_xyz(
        self,
        xyz_list: List[str],
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        edge_type: str = "distance",
        radius: float = 4.0,
        n_neigh: int = 5,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """
        Load the dataset from XYZ and targets.

        Parameters:
            xyz_list (list of str): XYZ file names
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            edge_type (str, optional): type of edge to construct the graph (default: distance, neighbor)
            radius (float, optional): radius to construct the graph (default: 4.0)
            n_neigh (int, optional): number of neighbors to consider (default: 5)
            verbose (int, optional): output verbose level
            **kwargs
        """

        num_sample = len(xyz_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            xyz_list = tqdm(xyz_list, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            print("Hydrogen atoms are considered")
        else:
            print("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.targets = defaultdict(list)
        self.atom_vocab = atom_vocab
        self.graph_data_list = []
        self.n_atoms = []
        
        for i, xyz in enumerate(xyz_list):
            try:
                if os.path.exists(xyz):

                    mol_xyz = PointCloud_Mol.from_xyz(
                        xyz, with_hydrogen, forbidden_atoms=forbidden_atoms
                    )
                    if mol_xyz is None:
                        if verbose > 0:
                            print(f"Skipping {xyz} due to containing forbidden atoms")
                        continue
                    if len(mol_xyz.atoms) > max_atom:
                        if verbose > 0:
                            print(
                                f"Skipping {xyz} due to too many atoms {len(mol_xyz.atoms)}"
                            )
                        continue
                    coords = mol_xyz.get_coord()

                    if i < len(smiles_list):
                        smiles = smiles_list[i]
                        if node_feature is not None and smilify is not None:
                            if "rdkit" in node_feature:
                                _, mol = smilify(xyz)
                            else:
                                mol = None
                        else:
                            mol = None
                    else:
                        if verbose > 0:
                            print("Cannot find smiles for ", xyz)
                        smiles = None

                else:
                    print(f"File {xyz} does not exist")
                    continue

                node_features = []
                for atom in mol_xyz.atoms:
                    node_features.append(onehot(
                        atom.element, atom_vocab, allow_unknown=False
                    ))
                charges = [atomic_numbers[atom.element]
                           for atom in mol_xyz.atoms
                           if atom.element in atomic_numbers]
                if node_feature:
                    if mol is None and "rdkit" in node_feature:
                        continue
                    if node_feature == "rdkit_oh":
                        node_features_extra = torch.tensor(
                            [atom_default_extra(atom) for atom in mol.GetAtoms()]
                        )
                    elif node_feature == "rdkit":
                        node_features_extra = torch.tensor(
                            [atom_default_condense(atom) for atom in mol.GetAtoms()]
                        )
                    elif node_feature in [
                        "atom_topological",
                        "atom_geom",
                        "atom_geom_v2",
                        "atom_geom_v2_trun",
                        "atom_geom_opt",
                        "atom_geom_compact"
                    ]:
                        feature_mapping = {
                            "atom_topological": atom_topological,
                            "atom_geom": atom_geom,
                            "atom_geom_v2": atom_geom_v2,
                            "atom_geom_v2_trun": atom_geom_v2_trun,
                            "atom_geom_opt": atom_geom_opt,
                            "atom_geom_compact": atom_geom_compact,
                        }
                        feature_function = feature_mapping.get(node_feature)
                        if feature_function is not None:
                            node_features_extra = feature_function(
                                charges, coords
                            )                                                           
                    else:
                        raise ValueError(
                            "Unknown node feature type, not yet installed dependency (cell2mol or libarvo)"
                        )
                    node_features = torch.cat(
                        [torch.tensor(node_features), node_features_extra], dim=1
                    )
                node_features = torch.tensor(node_features, dtype=torch.float32)

                charges = torch.as_tensor(charges, dtype=torch.long)

                n_nodes = len(mol_xyz.atoms)
                self.n_atoms.append(n_nodes)
                
                if edge_type == "distance":
                    edge_index = radius_graph(coords, r=radius)
                elif edge_type == "neighbor":
                    edge_index = knn_graph(coords, k=n_neigh)
                elif edge_type == "fully_connected":
                    num_nodes = coords.size(0)
                    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
                    col = torch.arange(num_nodes).repeat(num_nodes)
                    edge_index = torch.stack([row, col], dim=0)
                    edge_index = edge_index[:, row != col]  # Remove self-loops if needed
                else:
                    raise ValueError("Unknown edge type %s" % edge_type)
                
                tags = torch.zeros(n_nodes, dtype=torch.long) + i
                graph_data = Data(
                            x=node_features,
                            pos=coords,
                            atomic_numbers=charges,
                            natoms=n_nodes,
                            smiles=smiles,
                            xyz=xyz,
                            edge_index=edge_index,
                            tags=tags,
                        )
                self.graph_data_list.append(graph_data)
                for field in targets:
                    self.targets[field].append(targets[field][i])

            except Exception as e:
                logging.error(f"Error in loading {xyz}: {e}")
                continue

    def load_db(
        self,
        db_path: str,
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[List[str]] = None,
        consider_global_attributes: bool = True,
        target_fields: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        edge_type: str = "distance",
        radius: float = 4.0,
        n_neigh: int = 5,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """
        Load the dataset from an ASE db file.

        Parameters:
            db_path (str): path to ASE db file
            atom_vocab (list of str, optional): atom types
            node_feature_choice (list of str, optional): RDKit atom features to extract
            target_fields (list of str, optional): name of target columns in the table.
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            edge_type (str, optional): type of edge to construct the graph (default: distance, neighbor)
            radius (float, optional): radius to construct the graph (default: 4.0)
            n_neigh (int, optional): number of neighbors to consider (default: 5)
            verbose (int, optional): output verbose level
            **kwargs
        """
        if Chem is None and node_feature_choice is not None:
            raise ImportError("RDKit is required for node_feature_choice. Please install it.")

        if not atom_vocab:
            atom_vocab = BASE_ATOM_VOCAB
            print("atom vocabulary not provided, using default")

        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.targets = defaultdict(list)
        self.atom_vocab = atom_vocab
        self.graph_data_list = []
        self.n_atoms = []
        
        db = connect(db_path)
        iterator = db.select()
        if verbose:
            iterator = tqdm(iterator, "Processing ASE db files", total=len(db))

        for i, row in enumerate(iterator):
            try:
                mol_ase = row.toatoms()

                if len(mol_ase) > max_atom:
                    if verbose > 0:
                        logger.warning(f"Skipping entry {i} with {len(mol_ase)} atoms (> {max_atom})")
                    continue

                if any(atom.symbol in forbidden_atoms for atom in mol_ase):
                    if verbose > 0:
                        logger.warning(f"Skipping entry {i} due to forbidden atoms")
                    continue

                coords = torch.from_numpy(mol_ase.get_positions()).to(torch.float32)
                charges = torch.from_numpy(mol_ase.get_atomic_numbers()).to(torch.long)
                n_nodes = len(mol_ase)

                atomic_symbols = mol_ase.get_chemical_symbols()
                node_features = [onehot(atom, atom_vocab, allow_unknown=False) for atom in atomic_symbols]
                node_features = torch.tensor(node_features, dtype=torch.float32)

                mol_rdkit = None
                if node_feature_choice:
                    if "mol_block" not in row.data:
                        if verbose > 0:
                            logger.warning(f"Skipping entry {i} as it lacks 'mol_block' for rdkit features")
                        continue
                    
                    mol_block = row.data.get('mol_block')
                    if isinstance(mol_block, bytes):
                        mol_block = mol_block.decode('utf-8')

                    mol_rdkit = Chem.MolFromMolBlock(mol_block, removeHs=False)
                    if not mol_rdkit:
                        logger.warning(f"RDKit failed to parse mol_block for entry {i}")
                        continue

                    ase_atomic_num = mol_ase.get_atomic_numbers()
                    rdkit_atomic_num = np.array([atom.GetAtomicNum() for atom in mol_rdkit.GetAtoms()])
                    if not np.array_equal(ase_atomic_num, rdkit_atomic_num):
                         if verbose > 0:
                            logger.warning(f"Atom order mismatch for entry {i}. Skipping.")
                         continue

                    atom_feats = defaultdict(list)
                    for atom in mol_rdkit.GetAtoms():
                        atom_feats['degree'].append(atom.GetDegree())
                        atom_feats['formal_charge'].append(atom.GetFormalCharge())
                        atom_feats['hybridization'].append(hybiridization_map.get(str(atom.GetHybridization()), -1))
                        atom_feats['is_aromatic'].append(atom.GetIsAromatic())
                        atom_feats['valence'].append(atom.GetTotalValence())
                    
                    node_features_extra = torch.tensor([
                        atom_feats[key] for key in node_feature_choice
                    ], dtype=torch.float32).T
                    
                    node_features = torch.cat((node_features, node_features_extra), dim=1)

                if torch.isnan(coords).any() or torch.isnan(node_features).any():
                    if verbose > 0:
                        print(f"Skipping entry {i} due to NaN values in coordinates or node features")
                    continue
                
                smiles = Chem.MolToSmiles(mol_rdkit) if mol_rdkit else None

                if edge_type == "distance":
                    edge_index = radius_graph(coords, r=radius)
                elif edge_type == "neighbor":
                    edge_index = knn_graph(coords, k=n_neigh)
                elif edge_type == "fully_connected":
                    num_nodes = coords.size(0)
                    row = torch.arange(num_nodes).repeat_interleave(num_nodes)
                    col = torch.arange(num_nodes).repeat(num_nodes)
                    edge_index = torch.stack([row, col], dim=0)
                    edge_index = edge_index[:, row != col]
                else:
                    raise ValueError(f"Unknown edge type {edge_type}")

                tags = torch.zeros(n_nodes, dtype=torch.long) + i
                graph_data = Data(
                    x=node_features,
                    pos=coords,
                    atomic_numbers=charges,
                    natoms=n_nodes,
                    smiles=smiles,
                    xyz=f"db_entry_{i}",
                    edge_index=edge_index,
                    tags=tags,
                )
                
                self.graph_data_list.append(graph_data)
                self.n_atoms.append(n_nodes)

                if target_fields:
                    for field in target_fields:
                        value = row.data.get(field, math.nan)
                        try:
                            value = utils.literal_eval(str(value))
                        except (ValueError, SyntaxError):
                            if isinstance(value, (np.ndarray, torch.Tensor)):
                                value = value.tolist()
                        if value == "":
                            value = math.nan
                        self.targets[field].append(value)
                
                if consider_global_attributes:
                    self.targets['total_charge'].append(row.data.get('total_charge', 0))
                    self.targets['num_graph'].append(row.data.get('num_graph', 1))
                    self.targets['distortion_d'].append(row.data.get('distortion_d', 0))

                    sascore_value = 0
                    if mol_rdkit is not None:
                        if 'sascore' in row.data:
                            sascore_value = row.data.get('sascore', 0)
                        else:
                            try:
                                mol_withoutH = Chem.RemoveHs(mol_rdkit)
                                mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol_withoutH))
                                sascore_value = sascore.calculateScore(mol_clean)
                            except Exception:
                                sascore_value = 0
                    self.targets['sascore'].append(sascore_value)

            except Exception as e:
                logging.error(f"Error in loading db entry {i}: {e}")
                continue
            
         
    def load_pickle(self, pkl_file, verbose=0):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        self.transform = None

        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample, tasks = pickle.load(fin)

            self.graph_data_list = []
            self.targets = defaultdict(list)
            self.n_atoms = []

            for task in tasks:
                self.targets[task] = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            for i in indexes:
                graph_data, values = pickle.load(fin)
                self.graph_data_list.append(graph_data)
                self.n_atoms.append(graph_data.natoms)
                for task, value in zip(tasks, values):
                    self.targets[task].append(value)
            self.atom_vocab, self.with_hydrogen = pickle.load(fin)

    def save_pickle(self, pkl_file, verbose=0):
        """
        Save the dataset to a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.graph_data_list)
            tasks = list(self.targets.keys())
            pickle.dump((num_sample, tasks), fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                values = [v[i] for v in self.targets.values()]
                pickle.dump(
                    (
                        self.graph_data_list[i],
                        values,
                    ),
                    fout,
                )
            pickle.dump(
                (
                    self.atom_vocab,
                    self.with_hydrogen,
                ),
                fout,
            )
    
    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
        # item = {"Point Cloud": self.data[index]}

        item = {k: v[index] for k, v in self.targets.items()}
        item.update({"graph": self.graph_data_list[index]})
        if self.transform:
            item = self.transform(item)
        return item
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    def atom_types(self):
        """All atom types."""
        atom_types = set()
        for symbol in self.atom_vocab:
            atom_types.add(atomic_numbers[symbol])
        if 0 in atom_types:
            atom_types.discard(0)
        atom_types = sorted(atom_types)
        return atom_types
 
    @property
    def num_atom_type(self):
        """Number of different atom types."""
        return len(self.atom_types)
       

    def __len__(self):
        return len(self.graph_data_list)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: %d" % len(self.tasks),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))



class PointCloudDataset(torch_data.Dataset):

    def load_xyz(
        self,
        xyz_list: List[str],
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """
        Load the dataset from XYZ and targets.

        Parameters:
            xyz_list (list of str): XYZ file names
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            pad_data (bool, optional): whether to pad data to max_atom
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            **kwargs
        """

        num_sample = len(xyz_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            xyz_list = tqdm(xyz_list, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            print("Hydrogen atoms are considered")
        else:
            print("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []

        self.atom_vocab = atom_vocab

        for i, xyz in enumerate(xyz_list):
            try:
                if os.path.exists(xyz):

                    mol_xyz = PointCloud_Mol.from_xyz(
                        xyz, with_hydrogen, forbidden_atoms=forbidden_atoms
                    )
                    if mol_xyz is None:
                        if verbose > 0:
                            print(f"Skipping {xyz} due to containing forbidden atoms")
                        continue
                    if len(mol_xyz.atoms) > max_atom:
                        if verbose > 0:
                            print(
                                f"Skipping {xyz} due to too many atoms {len(mol_xyz.atoms)}"
                            )
                        continue
                    coords = mol_xyz.get_coord()

                    if i < len(smiles_list):
                        smiles = smiles_list[i]
                        if node_feature is not None and smilify is not None:
                            if "rdkit" in node_feature:
                                _, mol = smilify(xyz)
                            else:
                                mol = None
                        else:
                            mol = None
                    else:
                        if verbose > 0:
                            print("Cannot find smiles for ", xyz)
                        smiles = None

                else:
                    print(f"File {xyz} does not exist")
                    continue

                node_features = []
                for atom in mol_xyz.atoms:
                    node_features.append(
                        onehot(
                            atom.element, atom_vocab, allow_unknown=False
                        ))
                charges = [atomic_numbers[atom.element]
                           for atom in mol_xyz.atoms
                           if atom.element in atomic_numbers]
                charges = torch.as_tensor(charges, dtype=torch.long)
                if node_feature:
                    if node_feature == "rdkit_oh":
                        node_features_extra = torch.tensor(
                            [atom_default_extra(atom) for atom in mol.GetAtoms()]
                        )
                    elif node_feature == "rdkit":
                        node_features_extra = torch.tensor(
                            [atom_default_condense(atom) for atom in mol.GetAtoms()]
                        )
                    elif node_feature in [
                        "atom_topological",
                        "atom_geom",
                        "atom_geom_v2",
                        "atom_geom_v2_trun",
                        "atom_geom_opt",
                        "atom_geom_compact"
                    ]:
                        feature_mapping = {
                            "atom_topological": atom_topological,
                            "atom_geom": atom_geom,
                            "atom_geom_v2": atom_geom_v2,
                            "atom_geom_v2_trun": atom_geom_v2_trun,
                            "atom_geom_opt": atom_geom_opt,
                            "atom_geom_compact": atom_geom_compact,
                        }
                        feature_function = feature_mapping.get(node_feature)
                        if feature_function is not None:
                            node_features_extra = feature_function(
                                charges, coords
                            )

                    else:
                        raise ValueError(
                            "Unknown node feature type, not yet installed dependency (cell2mol or libarvo)"
                        )
                    node_features = torch.cat(
                        [torch.tensor(node_features, dtype=torch.float32), node_features_extra], dim=1
                    )
                node_features = torch.tensor(node_features, dtype=torch.float32)


                # adjust shape to max_atom
                n_nodes = len(mol_xyz.atoms)
                node_mask = torch.ones(n_nodes, dtype=torch.int8)

                if pad_data:
                    coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                    charges_mask = torch.zeros(max_atom, dtype=torch.long)

                    node_mask = torch.zeros(max_atom, dtype=torch.int8)
                    coords_full[:n_nodes] = coords
                    node_mask[:n_nodes] = 1
                    node_feat_full = torch.zeros(max_atom, node_features.size(1))
                    node_feat_full[:n_nodes] = node_features
                    charges_mask[:n_nodes] = charges
                    coords = coords_full
                    node_features = node_feat_full
                    charges = charges_mask
                    # NOTE basically fully-conneted graph
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
                else:
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
                edge_mask *= diag_mask

                if torch.isnan(coords).any() or torch.isnan(node_features).any():
                    if verbose > 0:
                        print(f"Skipping {xyz} due to NaN values in coordinates or node features")
                    continue
                
                self.coords_list.append(coords)
                self.n_atoms.append(n_nodes)
                self.node_mask_list.append(node_mask)
                self.edge_mask_list.append(edge_mask)
                self.node_feature_list.append(node_features)
                self.charges_list.append(charges)
                self.smiles_list.append(smiles)
                self.xyzs.append(xyz)
                for field in targets:
                    self.targets[field].append(targets[field][i])

            except Exception as e:
                logging.error(f"Error in loading {xyz}: {e}")
                continue

    def load_npy(
        self,
        coords: torch.Tensor,
        natoms: torch.Tensor,
        smiles_list: List[str],
        targets: Dict[str, List[Union[float, int]]],
        atom_vocab: List[str] = [],
        node_feature: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """
        Load the dataset from npy and targets.

        Parameters:
            coords (tensor): tensor of coordinates
            natoms (tensor): tensor of number of atoms
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            atom_vocab (list of str): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule (default: 120)
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            pad_data (bool, optional): whether to pad data to max_atom
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = natoms.size(0)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError(
                    "Number of target `%s` doesn't match with number of molecules. "
                    "Expect %d but found %d" % (field, num_sample, len(target_list))
                )
        if verbose:
            natoms = tqdm(natoms, "Constructing point cloud molecules from XYZs")

        if with_hydrogen:
            print("Hydrogen atoms are considered")
        else:
            print("Hydrogen atoms are not considered")
        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []

        self.atom_vocab = atom_vocab

        start_index = 0
        mol = None
        for i, natom in enumerate(natoms):
            # try:
            end_index = start_index + natom.item()
            molecule_data = coords[start_index:end_index, :]
            start_index = end_index

            if natom > max_atom:
                if verbose > 0:
                    print(f"Skipping {i} due to too many atoms {natom} > {max_atom}")
                    continue
            zs = torch.zeros(natom, dtype=torch.long)
            coord = torch.zeros((natom, 3), dtype=torch.float32)
            for i, row in enumerate(molecule_data):
                atomic_number = int(row[1])
                zs[i] = atomic_number
                coord[i] = row[2:]
            mol_xyz = PointCloud_Mol.from_arrays(
                zs, coord, with_hydrogen, forbidden_atoms=forbidden_atoms
            )

            if mol_xyz is None:
                if verbose > 0:
                    print(f"Skipping {i} due to containing forbidden atoms")
                continue

            coords_mol = mol_xyz.get_coord()

            if i < len(smiles_list):
                smiles = smiles_list[i]
                if node_feature is not None and smilify is not None:
                    if "rdkit" in node_feature:
                        _, mol = smilify(None, zs, coords_mol)
                    else:
                        mol = None
                else:
                    mol = None
            else:
                if verbose > 0:
                    print("Cannot find smiles for ", i)
                smiles = None
            self.smiles_list.append(smiles)

            node_features = []
            for atom in mol_xyz.atoms:
                node_features.append(
                    onehot(atom.element, atom_vocab, allow_unknown=False)
                )
            charges = [atomic_numbers[atom.element]
                       for atom in mol_xyz.atoms
                       if atom.element in atomic_numbers]
            charges = torch.as_tensor(charges, dtype=torch.long)
            if node_feature:
                if node_feature == "rdkit_oh":
                    node_features_extra = torch.tensor(
                        [atom_default_extra(atom) for atom in mol.GetAtoms()]
                    )
                elif node_feature == "rdkit":
                    node_features_extra = torch.tensor(
                        [atom_default_condense(atom) for atom in mol.GetAtoms()]
                    )
                elif node_feature == "topological" and atom_topological is not None:
                    node_features_extra = atom_topological(
                        np.array(charges), coords_mol
                    )
                elif node_feature in [
                    "atom_topological",
                    "atom_geom",
                    "atom_geom_v2",
                    "atom_geom_v2_trun",
                    "atom_geom_opt",
                    "atom_geom_compact"
                ]:
                    feature_mapping = {
                        "atom_topological": atom_topological,
                        "atom_geom": atom_geom,
                        "atom_geom_v2": atom_geom_v2,
                        "atom_geom_v2_trun": atom_geom_v2_trun,
                        "atom_geom_opt": atom_geom_opt,
                        "atom_geom_compact": atom_geom_compact,
                    }
                    feature_function = feature_mapping.get(node_feature)
                    if feature_function is not None:
                        node_features_extra = feature_function(
                            charges, coords_mol
                        )
                else:
                    raise ValueError(
                        "Unknown node feature type, not yet installed dependency (cell2mol or libarvo)"
                    )
                print(feature_function, node_feature)
                node_features = torch.cat(
                    [torch.tensor(node_features), node_features_extra],
                    dim=1,
                )

            node_features = torch.tensor(node_features, dtype=torch.float32)


            # adjust shape to max_atom
            n_nodes = len(mol_xyz.atoms)
            node_mask = torch.ones(n_nodes, dtype=torch.int8)

            if pad_data:
                coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                charges_mask = torch.zeros(max_atom, dtype=torch.long)

                node_mask = torch.zeros(max_atom, dtype=torch.int8)
                coords_full[:n_nodes] = coords_mol
                node_mask[:n_nodes] = 1
                node_feat_full = torch.zeros(max_atom, node_features.size(1))
                node_feat_full[:n_nodes] = node_features
                charges_mask[:n_nodes] = charges
                coords_mol = coords_full
                node_features = node_feat_full
                charges = charges_mask
                # NOTE basically fully-conneted graph
                edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
            else:
                edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
            edge_mask *= diag_mask

            if torch.isnan(coords).any() or torch.isnan(node_features).any():
                if verbose > 0:
                    print(f"Skipping {i} due to NaN values in coordinates or node features")
                continue

            self.coords_list.append(coords_mol)
            self.n_atoms.append(n_nodes)
            self.node_mask_list.append(node_mask)
            self.edge_mask_list.append(edge_mask)
            self.node_feature_list.append(node_features)
            self.charges_list.append(charges)
            self.xyzs.append(i)
            for field in targets:
                self.targets[field].append(targets[field][i])

    def load_csv(
        self,
        csv_file,
        xyz_dir,
        xyz_field="xyz",
        smiles_field="smiles",
        target_fields=None,
        atom_vocab=[],
        node_feature=None,
        forbidden_atoms=[],
        verbose=0,
        **kwargs,
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            xyz_dir (str): directory to store XYZ files
            xyz_field (str): name of the XYZ column in the table
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            atom_vocab (list of str, optional): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)

        if xyz_field is None:
            raise ValueError("xyz_field must be provided")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            print("atom vocabulary not provided, using defaul in constant.py")
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            xyzs = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == xyz_field:
                        xyz_path = os.path.join(xyz_dir, f"{value}.xyz")
                        xyzs.append(xyz_path)
                    elif field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
        assert len(xyzs) > 0, "No XYZ files found"
        # TODO to deal with when xyz but absence smiles and vice versa, skip it for now
        self.load_xyz(
            xyzs,
            smiles,
            targets,
            atom_vocab,
            forbidden_atoms=forbidden_atoms,
            node_feature=node_feature,
            verbose=verbose,
            **kwargs,
        )

    def load_csv_npy(
        self,
        csv_file,
        coords_file,
        natoms_file,
        smiles_field="smiles",
        target_fields=None,
        atom_vocab=[],
        node_feature=None,
        forbidden_atoms=[],
        verbose=0,
        **kwargs,
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            coords_file (str): npy file containing coordinates
            natoms_file (str): npy file containing number of atoms
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            atom_vocab (list of str, optional): atom types
            node_feature (bool, optional): atom features to extract [rdkit, rdkit_oh, geom]
            forbidden_atoms (list of str, optional): forbidden atoms
            verbose (int, optional): output verbose level
            **kwargs
        """
        coords = np.load(coords_file)
        coords = torch.tensor(coords, dtype=torch.float32)
        natoms = np.load(natoms_file)
        natoms = torch.tensor(natoms, dtype=torch.long)

        if target_fields is not None:
            target_fields = set(target_fields)

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            print("atom vocabulary not provided, using defaul in constant.py")
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
        # TODO to deal with when xyz but absence smiles and vice versa, skip it for now
        self.load_npy(
            coords,
            natoms,
            smiles,
            targets,
            atom_vocab,
            forbidden_atoms=forbidden_atoms,
            node_feature=node_feature,
            verbose=verbose,
            **kwargs,
        )

    def load_db(
        self,
        db_path: str,
        atom_vocab: List[str] = [],
        node_feature_choice: Optional[List[str]] = None,
        consider_global_attributes: bool = True,
        target_fields: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        max_atom: int = 200,
        with_hydrogen: bool = True,
        forbidden_atoms: List[str] = [],
        pad_data: bool = False,
        verbose: int = 0,
        **kwargs: Any,
    ):
        """
        Load the dataset from an ASE db file.

        Parameters:
            db_path (str): path to ASE db file
            atom_vocab (list of str, optional): atom types
            node_feature_choice (list of str, optional): RDKit atom features to extract
            target_fields (list of str, optional): name of target columns in the table.
            transform (Callable, optional): data transformation function
            max_atom (int, optional): maximum number of atoms in a molecule
            with_hydrogen (bool, optional): whether to add hydrogen atoms
            forbidden_atoms (list of str, optional): forbidden atoms
            pad_data (bool, optional): whether to pad data to max_atom)
            verbose (int, optional): output verbose level
            **kwargs
        """
        if Chem is None and node_feature_choice is not None:
            raise ImportError("RDKit is required for node_feature_choice. Please install it.")

        if atom_vocab == []:
            atom_vocab = BASE_ATOM_VOCAB
            print("atom vocabulary not provided, using default")

        self.with_hydrogen = with_hydrogen
        self.transform = transform
        self.kwargs = kwargs
        self.xyzs = []
        self.smiles_list = []
        self.coords_list = []
        self.node_mask_list = []
        self.edge_mask_list = []
        self.node_feature_list = []
        self.charges_list = []
        self.targets = defaultdict(list)
        self.n_atoms = []
        self.atom_vocab = atom_vocab

        db = connect(db_path)


        iterator = db.select()
        if verbose:
            iterator = tqdm(iterator, "Processing ASE db files", total=len(db))

        for i, row in enumerate(iterator):
            try:
                mol_ase = row.toatoms()

                if len(mol_ase) > max_atom:
                    if verbose > 0:
                        logger.warning(f"Skipping entry {i} with {len(mol_ase)} atoms (> {max_atom})")
                    continue

                if any(atom.symbol in forbidden_atoms for atom in mol_ase):
                    if verbose > 0:
                        logger.warning(f"Skipping entry {i} due to forbidden atoms")
                    continue

                coords = torch.from_numpy(mol_ase.get_positions()).to(torch.float32)
                charges = torch.from_numpy(mol_ase.get_atomic_numbers()).to(torch.long)
                n_nodes = len(mol_ase)

                atomic_symbols = mol_ase.get_chemical_symbols()
                node_features = [onehot(atom, atom_vocab, allow_unknown=False) for atom in atomic_symbols]
                node_features = torch.tensor(node_features, dtype=torch.float32)

                mol_rdkit = None
                    
                mol_block = row.data.get('mol_block')
                if isinstance(mol_block, bytes):
                    mol_block = mol_block.decode('utf-8')

                mol_rdkit = Chem.MolFromMolBlock(mol_block, removeHs=False)
                if node_feature_choice:
                    if "mol_block" not in row.data:
                        if verbose > 0:
                            logger.warning(f"Skipping entry {i} as it lacks 'mol_block' for rdkit features")
                        continue

                    if not mol_rdkit:
                        logger.warning(f"RDKit failed to parse mol_block for entry {i}")
                        continue

                    ase_atomic_num = mol_ase.get_atomic_numbers()
                    rdkit_atomic_num = np.array([atom.GetAtomicNum() for atom in mol_rdkit.GetAtoms()])
                    if not np.array_equal(ase_atomic_num, rdkit_atomic_num):
                         if verbose > 0:
                            logger.warning(f"Atom order mismatch for entry {i}. Skipping.")
                         continue

                    atom_feats = defaultdict(list)
                    for atom in mol_rdkit.GetAtoms():
                        atom_feats['degree'].append(atom.GetDegree())
                        atom_feats['formal_charge'].append(atom.GetFormalCharge())
                        atom_feats['hybridization'].append(hybiridization_map.get(str(atom.GetHybridization()), -1))
                        atom_feats['is_aromatic'].append(atom.GetIsAromatic())
                        atom_feats['valence'].append(atom.GetTotalValence())
                    
                    node_features_extra = torch.tensor([
                        atom_feats[key] for key in node_feature_choice
                    ], dtype=torch.float32).T
                    
                    node_features = torch.cat((node_features, node_features_extra), dim=1)

                node_mask = torch.ones(n_nodes, dtype=torch.int8)

                if pad_data:
                    coords_full = torch.zeros(max_atom, 3, dtype=torch.float32)
                    charges_mask = torch.zeros(max_atom, dtype=torch.long)
                    node_mask = torch.zeros(max_atom, dtype=torch.int8)
                    coords_full[:n_nodes] = coords
                    node_mask[:n_nodes] = 1
                    node_feat_full = torch.zeros(max_atom, node_features.size(1), dtype=torch.float32)
                    node_feat_full[:n_nodes] = node_features
                    charges_mask[:n_nodes] = charges
                    coords = coords_full
                    node_features = node_feat_full
                    charges = charges_mask
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(max_atom, dtype=torch.bool)
                    edge_mask *= diag_mask
                else:
                    edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
                    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool)
                    edge_mask *= diag_mask

                if torch.isnan(coords).any() or torch.isnan(node_features).any():
                    if verbose > 0:
                        print(f"Skipping entry {i} due to NaN values in coordinates or node features")
                    continue

                self.coords_list.append(coords)
                self.node_mask_list.append(node_mask)
                self.edge_mask_list.append(edge_mask)
                self.node_feature_list.append(node_features)
                self.charges_list.append(charges)
                self.xyzs.append(f"db_entry_{i}")
                self.n_atoms.append(n_nodes)
                
        
                if target_fields:
                    for field in target_fields:
                        value = row.data.get(field, math.nan)
                        try:
                            value = utils.literal_eval(str(value))
                        except (ValueError, SyntaxError):
                            if isinstance(value, (np.ndarray, torch.Tensor)):
                                value = value.tolist()
                        if value == "":
                            value = math.nan
                        self.targets[field].append(value)
                
                # consider global attributes of molecules
                # total_charge, num_graph, sascore, distortion_d
                if consider_global_attributes:
                    self.targets['total_charge'].append(row.data.get('total_charge', 0))
                    self.targets['num_graph'].append(row.data.get('num_graph', 1))
                    self.targets['distortion_d'].append(row.data.get('distortion_d', 0))

                    if mol_rdkit is not None:
                        if 'sascore' in row.data:
                            self.targets['sascore'].append(row.data.get('sascore', 0))
                        else:
                            try:
                                mol_withoutH = Chem.RemoveHs(mol_rdkit)
                                mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol_withoutH))
                                sascore_value = sascore.calculateScore(mol_clean)
                            except ImportError:
                                sascore_value = 0
                            self.targets['sascore'].append(sascore_value)
                    
                if mol_rdkit is not None:
                    smiles = Chem.MolToSmiles(mol_rdkit) if mol_rdkit else None
                    self.smiles_list.append(smiles)


            except Exception as e:
                logging.error(f"Error in loading db entry {i}: {e}")
                continue

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
        # item = {"Point Cloud": self.data[index]}

        item = {k: v[index] for k, v in self.targets.items()}
        item.update({"coords": self.coords_list[index]})
        item.update({"node_mask": self.node_mask_list[index]})
        item.update({"edge_mask": self.edge_mask_list[index]})
        item.update({"node_feature": self.node_feature_list[index]})
        item.update({"charges": self.charges_list[index]})
        item.update({"natoms": self.n_atoms[index]})
        item.update({"xyz": self.xyzs[index]})
        if self.transform:
            item = self.transform(item)
        return item

    def load_pickle(self, pkl_file, verbose=0, cheap_data=False):
        """
        Load the dataset from a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """
        self.transform = None

        if cheap_data:
            float_dtype = torch.float16
            long_dtype = torch.int16
            mask_dtype = torch.int8
        else:
            float_dtype = torch.float32
            long_dtype = torch.long
            mask_dtype = torch.int8

        with utils.smart_open(pkl_file, "rb") as fin:
            num_sample, tasks = pickle.load(fin)

            self.xyzs = []
            self.smiles_list = []
            self.coords_list = []
            self.node_mask_list = []
            self.edge_mask_list = []
            self.node_feature_list = []
            self.charges_list = []
            self.targets = defaultdict(list)
            self.n_atoms = []

            for task in tasks:
                self.targets[task] = []
            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Loading %s" % pkl_file)
            # To discard nmax
            for i in indexes:
                (
                    natom,
                    coord,
                    node_mask,
                    edge_mask,
                    node_feature,
                    charge,
                    xyz,
                    values,
                ) = pickle.load(fin)

                if natom > self.max_atom:
                    print(f"Skipping {xyz} due to too many atoms")
                    continue
                else:
                    if cheap_data:
                        coord = torch.tensor(coord, dtype=float_dtype)
                        node_mask = torch.tensor(node_mask, dtype=mask_dtype)
                        edge_mask = torch.tensor(edge_mask, dtype=mask_dtype)
                        node_feature = torch.tensor(node_feature, dtype=float_dtype)
                        charge = torch.tensor(charge, dtype=long_dtype)
                        xyz = None
                    self.n_atoms.append(natom)
                    self.coords_list.append(coord)
                    self.node_mask_list.append(node_mask)
                    self.edge_mask_list.append(edge_mask)
                    self.node_feature_list.append(node_feature)
                    self.charges_list.append(charge)
                    self.xyzs.append(xyz)
                    for task, value in zip(tasks, values):
                        self.targets[task].append(value)
            self.smiles_list, self.atom_vocab, self.with_hydrogen = pickle.load(fin)


    def save_pickle(self, pkl_file, verbose=0, cheap_data=False):
        """
        Save the dataset to a pickle file.

        Parameters:
            pkl_file (str): file name
            verbose (int, optional): output verbose level
        """

        if cheap_data:
            float_dtype = torch.float16
            long_dtype = torch.int16
            mask_dtype = torch.int8
            self.xyzs = [None] * len(self.xyzs)
        else:
            float_dtype = torch.float32
            long_dtype = torch.long
            mask_dtype = torch.int8

        with utils.smart_open(pkl_file, "wb") as fout:
            num_sample = len(self.xyzs)
            tasks = list(self.targets.keys())
            pickle.dump((num_sample, tasks), fout)

            indexes = range(num_sample)
            if verbose:
                indexes = tqdm(indexes, "Dumping to %s" % pkl_file)
            for i in indexes:
                values = [v[i] for v in self.targets.values()]
                pickle.dump(
                    (
                        self.n_atoms[i],
                        self.coords_list[i].to(float_dtype),
                        self.node_mask_list[i].to(mask_dtype),
                        self.edge_mask_list[i].to(mask_dtype),
                        self.node_feature_list[i].to(float_dtype),
                        self.charges_list[i].to(long_dtype),
                        self.xyzs[i],
                        values,
                    ),
                    fout,
                )
            pickle.dump(
                (
                    self.smiles_list,
                    self.atom_vocab,
                    self.with_hydrogen,
                ),
                fout,
            )
            
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

    @property
    def tasks(self):
        """List of tasks."""
        return list(self.targets.keys())

    def atom_types(self):
        """All atom types."""

        # if len(self.smiles_list) == 0:
        #     raise ValueError(
        #         "No SMILES available in the dataset or not yet converted from XYZ."
        #     )
        atom_types = set()
        for symbol in self.atom_vocab:
            atom_types.add(atomic_numbers[symbol])
        if 0 in atom_types:
            atom_types.discard(0)
        atom_types = sorted(atom_types)
        return atom_types

    @property
    def num_atom_type(self):
        """Number of different atom types."""
        return len(self.atom_types)

    @property
    def num_atoms(self):
        """Number of atoms in each molecule."""
        num_atoms = torch.tensor(self.n_atoms, dtype=torch.long)

        return num_atoms

    # property
    def get_property(self, task):
        if len(list(self.targets.keys())) == 0:
            return None
        else:
            prop = torch.tensor(self.targets[task], dtype=torch.float32)
            return prop

    def __len__(self):
        return len(self.xyzs)

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: %d" % len(self.tasks),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
