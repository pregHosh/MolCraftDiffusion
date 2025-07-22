# %%
from glob import glob
import os

import pandas as pd
from MolecularDiffusion.data.component import dataset as data
import logging

logger = logging.getLogger(__name__)


class pointcloud_dataset(data.PointCloudDataset):
    """
    Point cloud dataset for EDM archietcture

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(
        self,
        root,
        df_path,
        xyz_dir=None,
        coord_file=None,
        natoms_file=None,
        max_atom=0,
        with_hydrogen=True,
        forbidden_atom=[],
        verbose=1,
        node_feature=None,
        pad_data=False,
        dataset_name="suisei",
        **kwargs,
    ):
        self._processed_file = os.path.join(root, f"processed_data_{dataset_name}.pt")
        df = pd.read_csv(df_path)
        columns_to_discard = ["smiles", "filename", "name", "xyz"]
        all_columns = df.columns.tolist()
        target_fields = [col for col in all_columns if col not in columns_to_discard]

        for col in target_fields[:]:
            if df[col].apply(lambda x: isinstance(x, str)).any():
                target_fields.remove(col)

        logging.info("Target fields:", target_fields)
        xyz_field = "filename"
        smiles_field = "smiles"
        self.max_atom = max_atom

        if max_atom == 0 or max_atom is None:
            logging.info("Maximum number of atoms not specifed, determining it....")
            self.max_atom = 0
            xyzs = glob(f"{xyz_dir}/*.xyz")
            for xyz in xyzs:
                with open(xyz, "r") as f:
                    n_atom = int(f.readlines()[0])
                    if n_atom > self.max_atom:
                        self.max_atom = n_atom
            logging.info("The max atom is ", self.max_atom)

        if os.path.exists(self._processed_file):
            logging.info(f"Found processed file at {self._processed_file}, loading it.")
            self.load_pickle(
                self._processed_file
            )
        else:
            logging.info("Processing data and saving to pickle file")
            if (coord_file is not None) and (natoms_file is not None):

                logging.info("Reading from coordinates from npy files")
                self.load_csv_npy(
                    df_path,
                    coord_file,
                    natoms_file,
                    xyz_field=xyz_field,
                    smiles_field=smiles_field,
                    verbose=verbose,
                    target_fields=target_fields,
                    with_hydrogen=with_hydrogen,
                    max_atom=self.max_atom,
                    forbidden_atom=forbidden_atom,
                    node_feature=node_feature,
                    pad_data=pad_data,
                    **kwargs,
                )
            else:
                logging.info("Reading coodinates from xyz files")
                self.load_csv(
                    df_path,
                    xyz_dir,
                    xyz_field=xyz_field,
                    smiles_field=smiles_field,
                    verbose=verbose,
                    target_fields=target_fields,
                    with_hydrogen=with_hydrogen,
                    max_atom=self.max_atom,
                    forbidden_atom=forbidden_atom,
                    node_feature=node_feature,
                    pad_data=pad_data,
                    **kwargs,
                )
            self.save_pickle(
                self._processed_file
            )




class pointcloud_dataset_pyG(data.GraphDataset):

    def __init__(
        self,
        root,
        df_path,
        xyz_dir=None,
        coord_file=None,
        natoms_file=None,
        max_atom=0,
        with_hydrogen=True,
        forbidden_atom=[],
        verbose=1,
        node_feature=None,
        pad_data=False,
        dataset_name="suisei",
        **kwargs,
    ):
        self._processed_file = os.path.join(root, f"processed_data_{dataset_name}.pt")
        df = pd.read_csv(df_path)
        columns_to_discard = ["smiles", "filename", "name", "xyz"]
        all_columns = df.columns.tolist()
        target_fields = [col for col in all_columns if col not in columns_to_discard]

        for col in target_fields[:]:
            if df[col].apply(lambda x: isinstance(x, str)).any():
                target_fields.remove(col)

        logging.info("Target fields:", target_fields)
        xyz_field = "filename"
        smiles_field = "smiles"
        self.max_atom = max_atom

        if max_atom == 0 or max_atom is None:
            logging.info("Maximum number of atoms not specifed, determining it....")
            self.max_atom = 0
            xyzs = glob(f"{xyz_dir}/*.xyz")
            for xyz in xyzs:
                with open(xyz, "r") as f:
                    n_atom = int(f.readlines()[0])
                    if n_atom > self.max_atom:
                        self.max_atom = n_atom
            logging.info("The max atom is ", self.max_atom)

        if os.path.exists(self._processed_file):
            logging.info(f"Found processed file at {self._processed_file}, loading it.")
            self.load_pickle(
                self._processed_file
            )
        else:
            logging.info("Processing data and saving to pickle file")
            self.load_csv(
                df_path,
                xyz_dir,
                xyz_field=xyz_field,
                smiles_field=smiles_field,
                verbose=verbose,
                target_fields=target_fields,
                with_hydrogen=with_hydrogen,
                max_atom=self.max_atom,
                forbidden_atom=forbidden_atom,
                node_feature=node_feature,
                pad_data=pad_data,
                edge_type="fully_connected",
                **kwargs,
            )
            self.save_pickle(
                self._processed_file
            )

# %%
if __name__ == "__main__":
    import torch

    dataset = pointcloud_dataset(
        "data/formed/Data_FORMED_sampled.csv",
        "data/formed/XYZ_FORMED/",
        max_atom=57,
        verbose=1,
    )
    train_ratio = 0.8
    test_ratio = (1 - train_ratio) / 2
    lengths = [
        int(train_ratio * len(dataset)),
        int(test_ratio * len(dataset)),
    ]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)
    # %%
    dataset[0]

# %%

