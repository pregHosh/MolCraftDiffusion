
import csv
import logging
import math
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils import data as torch_data
from tqdm import tqdm

from ase.data import atomic_numbers
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph

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
    
    def load_xyz(
        self,
        xyz_list,
        smiles_list,
        targets,
        atom_vocab=[],
        node_feature=None,
        transform=None,
        max_atom=200,
        with_hydrogen=True,
        forbidden_atoms=[],
        edge_type="distance",
        radius=4.0,
        n_neigh=5,
        verbose=0,
        **kwargs,
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
                    elif node_feature == "topological" and atom_topological is not None:
                        node_features_extra = atom_topological(
                            np.array(charges), coords
                        )
                    elif node_feature == "geomv1" and atom_geom is not None:
                        node_features_extra = atom_geom(np.array(charges), coords)
                    elif node_feature == "geomv2" and atom_geom_v2 is not None:
                        node_features_extra = atom_geom_v2(np.array(charges), coords)
                    elif (
                        node_feature == "geomv2_trun" and atom_geom_v2_trun is not None
                    ):
                        node_features_extra = atom_geom_v2_trun(
                            np.array(charges), coords
                        )
                    elif node_feature == "geomv2_compact" and atom_geom_compact is not None:
                        node_features_extra = atom_geom_compact(
                            np.array(charges), coords
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
            
         
    def load_pickle(self, pkl_file, verbose=0):
        pass
    
    def save_pickle(self, pkl_file, verbose=0):
        pass
    
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
        xyz_list,
        smiles_list,
        targets,
        atom_vocab=[],
        node_feature=None,
        transform=None,
        max_atom=200,
        with_hydrogen=True,
        forbidden_atoms=[],
        pad_data=False,
        verbose=0,
        **kwargs,
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
        coords,
        natoms,
        smiles_list,
        targets,
        atom_vocab=[],
        node_feature=None,
        transform=None,
        max_atom=200,
        with_hydrogen=True,
        forbidden_atoms=[],
        pad_data=False,
        verbose=0,
        **kwargs,
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

        fname, _ = os.path.splitext(pkl_file)
        fname = fname + ".smi"
        self.smiles_list = []

        if os.path.exists(fname):
            with open(fname, "r") as f:
                for line in f:
                    self.smiles_list.append(line.strip())

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
        if len(self.smiles_list) > 0:
            fname = pkl_file.split(".")[0]
            fname = fname + ".smi"
            with open(fname, "w") as f:
                for smi in self.smiles_list:
                    if smi is None:
                        smi = ""
                    f.write(smi + "\n")

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

        if len(self.smiles_list) == 0:
            raise ValueError(
                "No SMILES available in the dataset or not yet converted from XYZ."
            )
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
