import glob
import os
import torch
import pandas as pd
from tqdm import tqdm
from MolecularDiffusion.utils.geom_utils import read_xyz_file, create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_metrics import check_validity_v1, check_chem_validity, run_postbuster, smilify_wrapper, load_molecules_from_xyz
from MolecularDiffusion.utils import smilify_cell2mol, smilify_openbabel

import logging
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Constants
EDGE_THRESHOLD = 4
SCALE_FACTOR = 1.2
SCORES_THRESHOLD = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def runner(args):
    
    xyz_dir = args.input
    recheck_topo = args.recheck_topo
    check_strain = args.check_strain
    check_postbuster = args.check_postbuster
    skip_idx = args.skip_atoms
    
    if skip_idx is None:
        skip_idx = []

    xyzs = [
    path for path in glob.glob(f"{xyz_dir}/*.xyz")
    if 'opt' not in os.path.basename(path)
]

    df_res_dict = {
        "file": [],
        "percent_atom_valid": [],
        "valid": [],
        "valid_connected": [],
        "num_graphs": [],
        "bad_atom_distort": [],
        "bad_atom_chem": [],
        "smiles": [],
    }
    

        
    for xyz in tqdm(xyzs, desc="Processing XYZ files", total=len(xyzs)):

        try:
            cartesian_coordinates_tensor, atomic_numbers_tensor = read_xyz_file(xyz)
            data = create_pyg_graph(cartesian_coordinates_tensor, 
                                        atomic_numbers_tensor,
                                        xyz_filename=xyz,
                                        r=EDGE_THRESHOLD)
            data = correct_edges(data, scale_factor=SCALE_FACTOR)           
            (is_valid, percent_atom_valid, num_components, bad_atom_chem, bad_atom_distort) = \
                check_validity_v1(data, score_threshold=SCORES_THRESHOLD, 
                                  skip_indices=skip_idx,
                                  verbose=False)

        except Exception as e:
            logging.error(f"Error processing {xyz}: {e}")
            is_valid = False
            percent_atom_valid = 0
            num_components = 100
            bad_atom_chem = torch.arange(0,data.num_nodes)
            bad_atom_distort = torch.arange(0,data.num_nodes)
 
        try:
            smiles_list, mol_list = smilify_openbabel(xyz)
        except:
            # logging.warning(f"fail to convert xyz to mol with openbabel, retry with cell2mol")
            mol_list = None
        
        to_recheck = recheck_topo and (len(bad_atom_distort) > 0) and (len(bad_atom_chem) == 0)
        
        if mol_list is None and num_components < 3:
            xyz2mol_fn = smilify_cell2mol
            try:
                _, smiles_list, mol_list, _ = smilify_wrapper([xyz], xyz2mol_fn)
                mol_list = mol_list[0]
            except Exception as e:
                # logging.warning(f"fail to convert xyz to mol with v0, skip and assign invalid")
                to_recheck = False
                is_valid = False   

        if to_recheck:
            try:
                (natom_stability_dicts,
                    _,
                    _,
                    _,
                    bad_smiles_chem) = check_chem_validity([mol_list], skip_idx=skip_idx)
                natom_stable = sum(natom_stability_dicts.values())
                percent_atom_valid = natom_stable/cartesian_coordinates_tensor.size(0)
    
                if len(bad_smiles_chem) == 0:
                
                    is_valid = True
                else:
                    logging.warning("Detect bad smiles in ", xyz, bad_smiles_chem)
            except Exception as e:
                logging.error(f"Fail to check on {xyz} due to {e}, asssign invalid")
                is_valid = False
                percent_atom_valid = 0
                
        if is_valid and num_components == 1:
            is_valid_connected = True   
        else:
            is_valid_connected = False         
            
        df_res_dict["smiles"].append(smiles_list[0] if len(smiles_list) == 1 else smiles_list)
        df_res_dict["file"].append(xyz)
        df_res_dict["percent_atom_valid"].append(percent_atom_valid)
        df_res_dict["valid"].append(is_valid)
        df_res_dict["valid_connected"].append(is_valid_connected)
        df_res_dict["num_graphs"].append(num_components)
        df_res_dict["bad_atom_distort"].append(bad_atom_distort)
        df_res_dict["bad_atom_chem"].append(bad_atom_chem)

    df = pd.DataFrame(df_res_dict)
    df = df.sort_values(by="file")
    fully_connected = [1 if num == 1 else 0 for num in df_res_dict["num_graphs"]]

    logging.info(f"{df['percent_atom_valid'].mean() * 100:.2f}% of atoms are stable")
    logging.info(f"{df['valid'].mean() * 100:.2f}% of 3D molecules are valid")
    logging.info(f"{df['valid_connected'].mean() * 100:.2f}% of 3D molecules are valid and fully-connected")
    logging.info(f"{sum(fully_connected) / len(fully_connected) * 100:.2f}% of 3D molecules are fully connected")
    
    if check_strain:
        rmsd_mean = df["rmsd"].dropna().mean()
        delta_energy_mean = df["delta_energy"].dropna().mean()
        intact_topology = [1 if top else 0 for top in df_res_dict["same_topology"] if not pd.isna(top)]
        logging.info(f"RMSD mean: {rmsd_mean:.2f}")
        logging.info(f"Delta Energy mean: {delta_energy_mean:.2f}")
        logging.info(f"{sum(intact_topology) / len(intact_topology) * 100:.2f}% of 3D molecules have intact topology after the optimization")
    
    if check_postbuster:
        mols = load_molecules_from_xyz(xyz_dir)
        postbuster_results = run_postbuster(mols)
        if postbuster_results is not None:
            if args.output is None:
                postbuster_output_path = f"{xyz_dir}/postbuster_metrics.csv"
            else:
                base, ext = os.path.splitext(args.output)
                postbuster_output_path = f"{base}_postbuster{ext}"
            postbuster_results.to_csv(postbuster_output_path, index=False)

            logging.info(f"Sanitization: {postbuster_results['sanitization'].mean() * 100:.2f}%")
            logging.info(f"InChI Convertible: {postbuster_results['inchi_convertible'].mean() * 100:.2f}%")
            logging.info(f"All Atoms Connected: {postbuster_results['all_atoms_connected'].mean() * 100:.2f}%")
            logging.info(f"Bond Lengths: {postbuster_results['bond_lengths'].mean():.2f}")
            logging.info(f"Bond Angles: {postbuster_results['bond_angles'].mean():.2f}")
            logging.info(f"Internal Steric Clash: {postbuster_results['internal_steric_clash'].mean():.2f}")
            logging.info(f"Aromatic Ring Flatness: {postbuster_results['aromatic_ring_flatness'].mean():.2f}")
            logging.info(f"Non-Aromatic Ring Non-Flatness: {postbuster_results['non-aromatic_ring_non-flatness'].mean():.2f}")
            logging.info(f"Double Bond Flatness: {postbuster_results['double_bond_flatness'].mean():.2f}")
            logging.info(f"Internal Energy: {postbuster_results['internal_energy'].mean():.2f}")

    if args.output is None:
        output_path = f"{xyz_dir}/output_metrics.csv"
    else:
        output_path = args.output
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="input directory with xyz files")
    parser.add_argument("-o", "--output", type=str, default=None, help="output csv file")
    parser.add_argument("--recheck_topo", action="store_true", help="recheck topology")
    parser.add_argument("--check_strain", action="store_true", help="check strain")
    parser.add_argument("--check_postbuster", action="store_true", help="check postbuster metrics")
    parser.add_argument("--skip_atoms", type=int, nargs="+", default=None, help="skip atoms")
    args = parser.parse_args()
    runner(args)
