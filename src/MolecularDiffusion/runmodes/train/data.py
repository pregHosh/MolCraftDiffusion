import os
import torch
from MolecularDiffusion.data.dataset import pointcloud_dataset, pointcloud_dataset_pyG
from MolecularDiffusion.data.component.dataset import PointCloudDataset
from MolecularDiffusion.data.dataloader import pointcloud_collate, graph_collate, pointcloud_collate_v0
from MolecularDiffusion.utils import get_vram_size

class DataModule:
    """
    DataModule to load, optionally save/load pickle, and split datasets for diffusion or predictive tasks.

    Usage:
        module = DataModule(
            filename="data.pkl",
            task_type="diffusion",
            atom_vocab=atom_vocab_list,
            with_hydrogen=True,
            node_feature="geom",
            max_atom=50,
            xyz_dir="xyz/",
            coord_file="coords.csv",
            natoms_file="natoms.csv",
            forbidden_atom=None,
            data_efficient_collator=False,
            train_ratio=0.8,
            load_pkl=None,            # path to load dataset pickle
            save_pkl="cached_dataset.pkl"  # path to save dataset pickle
        )
        module.load()
        train_ds, valid_ds, test_ds = module.train_set, module.valid_set, module.test_set
    """
    def __init__(
        self,
        root: str,
        filename: str,
        task_type: str,
        atom_vocab: list,
        with_hydrogen: bool,
        node_feature: str,
        max_atom: int,
        xyz_dir: str = None,
        coord_file: str = None,
        natoms_file: str = None,
        forbidden_atom: list = None,
        data_efficient_collator: bool = False,
        train_ratio: float = 0.8,
        load_pkl: str = None,
        save_pkl: str = None,
        data_type: str = "pointcloud",
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_name: str = "suisei",
    ):
        self.root = root
        self.filename = filename
        self.task_type = task_type
        self.atom_vocab = atom_vocab
        self.with_hydrogen = with_hydrogen
        self.node_feature = node_feature
        self.max_atom = max_atom
        self.xyz_dir = xyz_dir
        self.coord_file = coord_file
        self.natoms_file = natoms_file
        self.forbidden_atom = forbidden_atom
        self.data_efficient_collator = data_efficient_collator
        self.train_ratio = train_ratio
        self.load_pkl = load_pkl
        self.save_pkl = save_pkl
        self.data_type = data_type.lower()
        self.dataset_name = dataset_name    
        
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if self.data_type == "pyg":
            self.collate_fn = graph_collate
        else:
            if self.data_efficient_collator:
                self.VRAM_SIZE = int(get_vram_size())
                self.collate_fn = pointcloud_collate(self.VRAM_SIZE)
            else:
                self.collate_fn = pointcloud_collate_v0  # or None

    def load(self):
        """
        Load dataset (from pickle if available), optionally save pickle, then split into train/valid/test.
        """
        # load from pickle if specified and exists
        if self.load_pkl and os.path.exists(self.load_pkl):     
            dataset = PointCloudDataset()
            dataset.max_atom = self.max_atom
            dataset.load_pickle(self.filename)
            dataset.atom_vocab = self.atom_vocab

            class Config:
                def __init__(self, with_hydrogen, node_feature, max_atom):
                    self.config_dict = {
                        "with_hydrogen": with_hydrogen,
                        "atom_feature": node_feature,
                        "max_atom": max_atom,
                    }

                def __call__(self):
                    return self.config_dict

            dataset.config_dict = Config(
                self.with_hydrogen,
                self.node_feature,
                self.max_atom
            )
        else:
            # instantiate dataset for task
            if self.task_type == "diffusion":
        
                if self.data_type == "pyg":
                    dataset = pointcloud_dataset_pyG(
                        root=self.root,
                        df_path=self.filename,
                        xyz_dir=self.xyz_dir,
                        coord_file=self.coord_file,
                        natoms_file=self.natoms_file,
                        max_atom=self.max_atom,
                        node_feature=self.node_feature,
                        atom_vocab=self.atom_vocab,
                        with_hydrogen=self.with_hydrogen,
                        forbidden_atoms=self.forbidden_atom,
                        pad_data=not self.data_efficient_collator,
                        dataset_name=self.dataset_name,
                        verbose=1,
                    )
                else:
                    dataset = pointcloud_dataset(
                        root=self.root,
                        df_path=self.filename,
                        xyz_dir=self.xyz_dir,
                        coord_file=self.coord_file,
                        natoms_file=self.natoms_file,
                        max_atom=self.max_atom,
                        node_feature=self.node_feature,
                        atom_vocab=self.atom_vocab,
                        with_hydrogen=self.with_hydrogen,
                        forbidden_atoms=self.forbidden_atom,
                        pad_data=not self.data_efficient_collator,
                        dataset_name=self.dataset_name,
                        verbose=1,
                    )
            elif self.task_type in ("regression", "guidance"):
                dataset = pointcloud_dataset_pyG(
                    root=self.root,
                    df_path=self.filename,
                    xyz_dir=self.xyz_dir,
                    coord_file=self.coord_file,
                    natoms_file=self.natoms_file,
                    max_atom=self.max_atom,
                    node_feature=self.node_feature,
                    atom_vocab=self.atom_vocab,
                    with_hydrogen=self.with_hydrogen,
                    forbidden_atoms=self.forbidden_atom,
                    pad_data=not self.data_efficient_collator,
                    dataset_name=self.dataset_name,
                    verbose=1,
                )
            else:
                raise ValueError(f"Unknown task_type '{self.task_type}'. Choose 'diffusion', 'regression', or 'guidance'.")

            # save to pickle if requested
            if self.save_pkl:
                dataset.save_pickle(self.save_pkl)

        # split
        total = len(dataset)
        test_ratio = (1 - self.train_ratio) / 2
        lengths = [int(self.train_ratio * total), int(test_ratio * total)]
        lengths.append(total - sum(lengths))
        self.train_set, self.valid_set, self.test_set = torch.utils.data.random_split(
            dataset, lengths
        )

        # attach metadata
        for subset in (self.train_set, self.valid_set, self.test_set):
            subset.atom_types = dataset.atom_types
        if self.task_type == "diffusion":
            for subset in (self.train_set, self.valid_set, self.test_set):
                subset.smiles_list = dataset.smiles_list
                subset.num_atoms = dataset.num_atoms
                subset.get_property = dataset.get_property

        print(
            f"Total: {total}, Train: {len(self.train_set)}, Valid: {len(self.valid_set)}, Test: {len(self.test_set)}"
        )
        return self.train_set, self.valid_set, self.test_set
