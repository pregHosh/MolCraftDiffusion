import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Optional
import argparse
import multiprocessing as mp
import json
import logging
from pathlib import Path # Import Path for easier path manipulation

from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import BRICS
from ase.data import chemical_symbols
from rdkit import Chem

from MolecularDiffusion.utils.smilify import smilify_cell2mol, smilify_openbabel
from MolecularDiffusion.utils.geom_utils import read_xyz_file   

def extract_scaffold_and_fingerprints(smiles_iter, fp_bits=2048):
    """
    Sanitizes SMILES strings and computes molecular descriptors:
    Morgan fingerprints, Bemis-Murcko scaffolds, and BRICS substructure counts.

    Args:
        smiles_iter: An iterable of SMILES strings.
        fp_bits (int): Number of bits for the Morgan fingerprint.

    Returns:
        tuple: A tuple containing:
            - fps (np.ndarray): Array of Morgan fingerprints.
            - scaffolds (list): List of Bemis-Murcko scaffold SMILES.
            - clean_smiles (list): List of canonicalized and sanitized SMILES.
            - n_fail (int): Number of SMILES strings that failed processing.
            - substruct_counts (dict): Dictionary of BRICS substructure counts.
    """
    fps = []
    scaffolds = []
    clean_smiles = []
    substruct_counts = {}
    n_fail = 0

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(fpSize=fp_bits)

    for smi in smiles_iter:
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                n_fail += 1
                continue
            Chem.SanitizeMol(mol)
            Chem.AddHs(mol)
        except Exception:
            n_fail += 1
            continue

        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((fp_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.append(scaffold)

        clean_smi = Chem.MolToSmiles(mol, canonical=True)
        clean_smiles.append(clean_smi)

        try:
            fragments = BRICS.BRICSDecompose(mol)
            for frag in fragments:
                substruct_counts[frag] = substruct_counts.get(frag, 0) + 1
        except Exception:
            continue

    return np.array(fps, dtype=np.uint8), scaffolds, clean_smiles, n_fail, substruct_counts


def sanitize_smiles(smiles: str) -> Optional[str]:
    """
    Sanitizes a single SMILES string and returns its canonical form.

    Args:
        smiles (str): The SMILES string to sanitize.

    Returns:
        Optional[str]: Canonical SMILES string if valid and sanitization succeeds,
                       otherwise None.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_file_list_from_dir(xyz_dir: str) -> pd.DataFrame:
    """
    Lists all .xyz files in the given directory and returns a DataFrame
    with a single column 'xyz_file'.

    Args:
        xyz_dir (str): Path to the directory containing .xyz files.

    Returns:
        pd.DataFrame: DataFrame with 'xyz_file' column.
    """
    files = [f for f in os.listdir(xyz_dir) if f.lower().endswith('.xyz')]
    df = pd.DataFrame({'xyz_file': files})
    return df


def _process_row(full_path: str, scale_factors: list[float], smilify_func, queue: mp.Queue, verbose: bool):
    """
    Worker function for one XYZ to SMILES conversion.
    Attempts conversion with multiple scale factors for OpenBabel,
    falls back to a different method if needed, and sanitizes the result.

    Args:
        full_path (str): Full path to the XYZ file.
        scale_factors (list[float]): List of scale factors to try for OpenBabel.
        smilify_func: Fallback SMILES generation function (e.g., smilify_cell2mol).
        queue (mp.Queue): Queue to put the result (sanitized SMILES or None).
        verbose (bool): If True, print detailed messages.
    """
    try:
        coords, atomic_nums = read_xyz_file(full_path)
        symbols = [chemical_symbols[int(n.item())] for n in atomic_nums]
        
        mol_list = None
        smiles_list = None

        for sf in scale_factors:
            ml, _, sl, _ = smilify_openbabel(symbols, coords, scale=sf)
            if ml is not None:
                mol_list, smiles_list = ml, sl
                if verbose:
                    logging.info(f"[{os.path.basename(full_path)}] Create {smiles_list} with scale factor {sf}")
                break

        if mol_list is None:
            smiles, _ = smilify_func(full_path)
            if verbose:
                logging.info(f"[{os.path.basename(full_path)}] Fallback SMILES {smiles}")
        else:
            smiles = smiles_list[0]
        
        queue.put(sanitize_smiles(smiles))

    except Exception as e:
        if verbose:
            logging.error(f"[{os.path.basename(full_path)}] Error: {e}")
        queue.put(None)


def run_processing(
    df: pd.DataFrame, 
    xyz_dir: str, 
    label: Optional[str], 
    output_csv_filepath: Path, # New argument for explicit output path
    timeout: int = 30, 
    verbose: bool = True
) -> None:
    """
    Processes each row of a DataFrame to generate SMILES strings from XYZ files.
    Uses multiprocessing to handle files, with a timeout for each conversion.
    Saves the results to a CSV file at the specified output_csv_filepath.

    Args:
        df (pd.DataFrame): DataFrame containing a column with XYZ file names.
        xyz_dir (str): Path to the directory containing the XYZ files.
        label (Optional[str]): Label to assign to the processed files.
        output_csv_filepath (Path): The full path where the processed SMILES CSV will be saved.
        timeout (int): Maximum time in seconds for each XYZ to SMILES conversion process.
                       Defaults to 30.
        verbose (bool): If True, print per-row messages and a summary. Defaults to True.
    """
    scale_factors = [1.0, 1.05, 1.10, 1.15, 1.20]

    fname_col = next(col for col in ('xyz_file', 'file', 'filename') if col in df.columns)

    smiles_results = []
    dir_results = []
    labels_list = []
    success_count = 0
    fail_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating SMILES"):
        raw_path = row[fname_col]
        base_name = os.path.basename(raw_path)
        full_path = os.path.join(xyz_dir, base_name) if xyz_dir else raw_path

        dir_results.append(os.path.dirname(full_path))
        labels_list.append(label)

        queue = mp.Queue()
        p = mp.Process(
            target=_process_row,
            args=(full_path, scale_factors, smilify_cell2mol, queue, verbose)
        )
        p.start()
        p.join(timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            if verbose:
                logging.warning(f"[{base_name}] Timeout after {timeout}s, skipping.")
            smiles_results.append(None)
            fail_count += 1
            continue

        smiles = queue.get() if not queue.empty() else None
        if smiles:
            smiles_results.append(smiles)
            success_count += 1
        else:
            if verbose:
                logging.warning(f"[{base_name}] Failed to convert.")
            smiles_results.append(None)
            fail_count += 1

    df['smiles'] = smiles_results
    df['xyz_dir'] = dir_results
    df['label'] = labels_list
    df = df[df['smiles'].notnull()].reset_index(drop=True)

    df.to_csv(output_csv_filepath, index=False)
    if verbose:
        logging.info(f"Processed {len(df)} molecules; results saved to {output_csv_filepath}")
        logging.info(f"Total files: {success_count + fail_count}, Successes: {success_count}, Failures: {fail_count}")

    return df

def main():
    """
    Main function to parse command-line arguments and initiate the processing
    of XYZ files to generate SMILES strings, followed by fingerprint and scaffold extraction.
    All 2D representation outputs are saved in a '2d_reprs' subdirectory within xyz_dir.
    """
    parser = argparse.ArgumentParser(
        description="Process XYZ files (from CSV or directory) to generate SMILES, fingerprints, and scaffolds."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input CSV containing xyz file column. If provided, --xyz_dir is used as base path."
    )
    parser.add_argument(
        "-x", "--xyz_dir",
        type=str,
        help="Path to the directory containing the XYZ files. Required if --input is not used.",
        default="input_directory"
    )
    parser.add_argument(
        "-l", "--label",
        default=None,
        help="Label for these files (e.g., 'CFGGG')."
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=30,
        help="Timeout in seconds for each XYZ to SMILES conversion process. Default is 30."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for per-row processing messages."
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=2048,
        help="Number of bits in the Morgan fingerprint (default: 2048)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define the base directory for all 2D representation outputs
    two_d_reprs_dir = Path(args.xyz_dir) / "2d_reprs"
    two_d_reprs_dir.mkdir(parents=True, exist_ok=True) # Create the directory and any necessary parent directories

    # Define the output path for the processed SMILES CSV
    smiles_csv_output_path = two_d_reprs_dir / "smiles_processed.csv"

    # 1. SMILES Generation
    df = pd.DataFrame()
    if args.input:
        if not os.path.exists(args.input):
            logging.critical(f"Input CSV file '{args.input}' not found.")
            exit(1)
        df = pd.read_csv(args.input)
        run_processing(df, args.xyz_dir, args.label, output_csv_filepath=smiles_csv_output_path, timeout=args.timeout, verbose=args.verbose)
    else:
        if not os.path.isdir(args.xyz_dir):
            logging.critical(f"XYZ directory '{args.xyz_dir}' does not exist or is not a directory.")
            exit(1)
        df = load_file_list_from_dir(args.xyz_dir)
        df_smiles = run_processing(df, args.xyz_dir, args.label, output_csv_filepath=smiles_csv_output_path, timeout=args.timeout, verbose=args.verbose)

    if 'smiles' not in df.columns or df['smiles'].isnull().all():
        logging.error("No valid SMILES generated. Cannot proceed with fingerprint/scaffold extraction.")
        exit(1)

    # 2. Extract scaffolds and fingerprints
    fps, scaffolds, clean_smiles, n_fail_extract, substruct_counts = \
        extract_scaffold_and_fingerprints(df_smiles["smiles"].dropna().values, fp_bits=args.bits)

    np.save(two_d_reprs_dir / "fingerprints.npy", fps)
    with open(two_d_reprs_dir / "scaffolds.txt", "w") as f:
        f.write("\n".join(scaffolds))
    with open(two_d_reprs_dir / "smiles_cleaned.txt", "w") as f:
        f.write("\n".join(clean_smiles))
    with open(two_d_reprs_dir / "substructures.json", "w") as f:
        json.dump(substruct_counts, f, indent=2)

    # Report results for the second stage
    total_smiles_input = len(df_smiles["smiles"].dropna())
    logging.info(f"--- Fingerprint and Scaffold Extraction Summary ---")
    logging.info(f"Input SMILES for extraction: {total_smiles_input}")
    logging.info(f"Failed to extract FP/Scaffold: {n_fail_extract} ({n_fail_extract/total_smiles_input:.1%})")
    logging.info(f"Unique substructures found: {len(substruct_counts)}")
    logging.info(f"All 2D representation outputs written to: {two_d_reprs_dir}")
    
if __name__ == "__main__":
    main()
