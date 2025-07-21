import argparse
import glob
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
# This setup ensures logs go to stdout and can be controlled by verbosity
logger = logging.getLogger(__name__) # Use __name__ for logger name
logger.setLevel(logging.INFO) # Default level
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)-10s %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Constants
HARTREE_TO_EV = 27.211386245988 # Conversion factor from Hartree to eV


def process_relaxed_log_file(log_filepath: Path) -> Tuple[float, float, float, float, Optional[float], float]:
    """
    Processes a Gaussian log file for a relaxed (optimized) calculation to extract
    vertical excitation energies (T1, S1, T2), S1 oscillator strength,
    S1 electron-hole distance (if Theodore analysis is enabled), and SCF energy.

    Args:
        log_filepath (Path): Path to the Gaussian log file.

    Returns:
        Tuple[float, float, float, float, Optional[float], float]: A tuple containing:
            - T1 (float): Vertical Triplet 1 energy in Hartree.
            - S1 (float): Vertical Singlet 1 energy in Hartree.
            - T2 (float): Vertical Triplet 2 energy in Hartree.
            - osc_strength_S1 (float): Oscillator strength of Singlet 1.
            - S1ehdist (Optional[float]): S1 electron-hole distance. Currently returns np.nan.
            - scf_energy (float): Final SCF energy in Hartree.

    Raises:
        Exception: If the log file does not indicate normal termination or
                   if required energy states/oscillator strengths are not found.
    """
    t1_vert, s1_vert, t2_vert, osc_strength_s1, s1_eh_dist = (np.nan,) * 5

    normal_termination = False
    singlet_energies = []
    triplet_energies = []
    osc_strengths_singlets = []
    scf_energies = []

    with open(log_filepath, 'r') as f:
        for line in f:
            if "Singlet-" in line:
                try:
                    singlet_energies.append(float(line.split()[4]))
                    osc_strengths_singlets.append(line.split()[8])
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse singlet energy/oscillator strength from line: {line.strip()} in {log_filepath} - {e}")

            if "Triplet-" in line:
                try:
                    triplet_energies.append(float(line.split()[4]))
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse triplet energy from line: {line.strip()} in {log_filepath} - {e}")

            if "Normal termination" in line:
                logger.info(f"Normal termination detected in {log_filepath.name}")
                normal_termination = True

            if "SCF Done:" in line:
                try:
                    scf_energies.append(float(line.split()[4]))
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse SCF energy from line: {line.strip()} in {log_filepath} - {e}")

    if not scf_energies:
        logger.error(f"No SCF Done energy found in {log_filepath.name}")
        raise Exception(f"Failed to find SCF energy in {log_filepath.name}")
    final_scf_energy = scf_energies[-1]

    if normal_termination:
        try:
            s1_vert = singlet_energies[0]
            t1_vert = triplet_energies[0]
            t2_vert = triplet_energies[1]
            osc_strength_s1 = float(osc_strengths_singlets[0].split("=")[1])
        except IndexError:
            logger.error(f"Missing expected excited state energies or oscillator strength in {log_filepath.name}")
            raise Exception(f"Incomplete excited state data in {log_filepath.name}")
        except ValueError as e:
            logger.error(f"Error parsing oscillator strength in {log_filepath.name}: {e}")
            raise Exception(f"Parsing error in {log_filepath.name}")
    else:
        logger.error(f"Normal termination not found in {log_filepath.name}")
        raise Exception(f"Calculation not terminated normally in {log_filepath.name}")

    # S1ehdist from theodore_analysis is commented out in original code.
    # If you enable this, ensure 'theodore' is installed and `self.config.path_dens_ana_in` is defined.
    # try:
    #     S1ehdist = theodore_analysis.theodore_workflow_S1_excdist(
    #         self.config.path_dens_ana_in, log_file_relaxed
    #     )
    # except theodore.error_handler.MsgError:
    #     logger.warning("Theodore error during S1 electron-hole distance calculation.")
    #     S1ehdist = np.nan # Use np.nan for numerical consistency

    return t1_vert, s1_vert, t2_vert, osc_strength_s1, s1_eh_dist, final_scf_energy


def process_log_file_d(log_filepath: Path, nstate: int = 3) -> Tuple[float, float]:
    """
    Processes a Gaussian log file for a "D" (presumably optimized excited state)
    calculation to extract the energy of the N-th excited state (S1 or T1)
    and the final SCF energy.

    Args:
        log_filepath (Path): Path to the Gaussian log file.
        nstate (int): The N-th excited state to consider (e.g., 3 for S3 or T3).

    Returns:
        Tuple[float, float]: A tuple containing:
            - excited_state_energy (float): Energy of the N-th excited state in Hartree.
            - scf_energy (float): Final SCF energy in Hartree.

    Raises:
        Exception: If the log file does not indicate normal termination or
                   if the specified N-th state cannot be found.
    """
    logger.info(f"Processing {log_filepath.name} for {nstate}-th state")

    normal_termination = False
    scf_energies = []
    singlet_energies = []
    triplet_energies = []

    with open(log_filepath, 'r') as f:
        for line in f:
            if "SCF Done:" in line:
                try:
                    scf_energies.append(float(line.split()[4]))
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse SCF energy from line: {line.strip()} in {log_filepath} - {e}")
            if "Singlet" in line:
                try:
                    singlet_energies.append(float(line.split()[4]))
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse singlet energy from line: {line.strip()} in {log_filepath} - {e}")
            if "Triplet" in line:
                try:
                    triplet_energies.append(float(line.split()[4]))
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not parse triplet energy from line: {line.strip()} in {log_filepath} - {e}")
            if "Normal termination" in line:
                normal_termination = True

    if not normal_termination:
        raise Exception(f"Calculation not terminated normally in {log_filepath.name}")

    if not scf_energies:
        raise Exception(f"No SCF Done energy found in {log_filepath.name}")
    final_scf_energy = scf_energies[-1]

    excited_state_energy = np.nan
    if len(singlet_energies) >= nstate:
        excited_state_energy = singlet_energies[nstate - 1] # nstate is 1-based index
    elif len(triplet_energies) >= nstate:
        excited_state_energy = triplet_energies[nstate - 1] # nstate is 1-based index
    else:
        raise Exception(f"Cannot find {nstate}-th Singlet or Triplet state in {log_filepath.name}")

    return excited_state_energy, final_scf_energy


def extract_base_filename(filepath: Path) -> str:
    """
    Extracts the filename without extension from a given file path.

    Args:
        filepath (Path): The Path object of the file.

    Returns:
        str: The filename without its extension.
    """
    return filepath.stem


def group_log_files_by_basename(folder_path: Path) -> dict[str, list[Path]]:
    """
    Groups Gaussian log files by their base name (without _s1.log, _t1.log, or .log suffixes).
    Files are sorted such that the main log file (without _s1 or _t1) comes first.

    Args:
        folder_path (Path): Path to the directory containing log files.

    Returns:
        dict[str, list[Path]]: A dictionary where keys are base filenames and values
                               are lists of corresponding log file paths, sorted.
    """
    log_dict = defaultdict(list)

    for file_path in folder_path.glob("*.log"):
        file_name = file_path.name
        base_name = file_name.replace("_s1.log", "").replace("_t1.log", "").replace(".log", "")
        log_dict[base_name].append(file_path)

    for base_name in log_dict:
        # Sort files to ensure the main log (e.g., 'mol.log') comes before
        # excited state logs (e.g., 'mol_s1.log', 'mol_t1.log')
        log_dict[base_name].sort(key=lambda p: (not p.name.endswith(f'{base_name}.log'), p.name))
        
    return dict(log_dict)


def main():
    """
    Main function to parse command-line arguments, process Gaussian log files,
    and save extracted energies and properties to a CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Extract energies and properties from Gaussian log files of DFT/TDA calculations."
    )
    parser.add_argument(
        "--dir",
        dest="dir",
        type=str,
        default="",
        help="Path to the .log files of the DFT/TDA calculations. Defaults to current directory.",
    )
    parser.add_argument(
        "--nstate",
        dest="nstate",
        type=int,
        default=3,
        help="Number of states to be considered for 'D' log files (e.g., S3 or T3). Defaults to 3.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="energies.csv",
        help="Name of the output CSV file. Defaults to 'energies.csv'."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (sets logging level to DEBUG)."
    )

    args = parser.parse_args()
    nstate = args.nstate

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    working_directory = Path(args.dir) if args.dir else Path.cwd()

    if not working_directory.is_dir():
        logger.critical(f"Error: Directory '{working_directory}' does not exist or is not a directory.")
        sys.exit(1)

    # 1 Extract data from log files
    df_data = {
        "filename": [],
        "T1_vert": [],
        "S1_vert": [],
        "T1_d": [],
        "S1_d": [],
        "T2": [],
        "osc_strength_S1": [],
        "S1ehdist": [], # Currently np.nan as Theodore analysis is commented out
    }

    log_file_groups = group_log_files_by_basename(working_directory)

    if not log_file_groups:
        logger.warning(f"No Gaussian log files found in '{working_directory}'. Exiting.")
        sys.exit(0)

    for base_filename, log_files in log_file_groups.items():
        # Initialize values for current molecule
        current_filename = base_filename
        t1_vert_val, s1_vert_val, t2_vert_val, osc_strength_s1_val, s1_eh_dist_val, scf_e_vert_val = (np.nan,) * 6
        s1_d_val, t1_d_val = np.nan, np.nan

        for log_file_path in log_files:
            try:
                if "_s1" not in log_file_path.name and "_t1" not in log_file_path.name:
                    # This is the main relaxed log file
                    t1_vert_val, s1_vert_val, t2_vert_val, osc_strength_s1_val, s1_eh_dist_val, scf_e_vert_val = \
                        process_relaxed_log_file(log_file_path)
                    
                elif "_s1" in log_file_path.name:
                    # This is the S1 optimized log file
                    s1_d_energy, s1_d_scf_energy = process_log_file_d(log_file_path, nstate)
                    # Calculate S1_d (adiabatic S1)
                    if not np.isnan(s1_d_energy) and not np.isnan(s1_d_scf_energy) and not np.isnan(scf_e_vert_val):
                        s1_d_val = s1_d_energy + (s1_d_scf_energy - scf_e_vert_val) * HARTREE_TO_EV
                    else:
                        logger.warning(f"Could not calculate S1_d for {log_file_path.name} due to missing data.")
                    
                elif "_t1" in log_file_path.name:
                    # This is the T1 optimized log file
                    t1_d_energy, t1_d_scf_energy = process_log_file_d(log_file_path, nstate)
                    # Calculate T1_d (adiabatic T1)
                    if not np.isnan(t1_d_energy) and not np.isnan(t1_d_scf_energy) and not np.isnan(scf_e_vert_val):
                        t1_d_val = t1_d_energy + (t1_d_scf_energy - scf_e_vert_val) * HARTREE_TO_EV
                    else:
                        logger.warning(f"Could not calculate T1_d for {log_file_path.name} due to missing data.")
            except Exception as e:
                logger.error(f"Skipping processing for {log_file_path.name} due to error: {e}")
                # Reset values to NaN for this log group if an error occurs
                t1_vert_val, s1_vert_val, t2_vert_val, osc_strength_s1_val, s1_eh_dist_val, scf_e_vert_val = (np.nan,) * 6
                s1_d_val, t1_d_val = np.nan, np.nan
                break # Move to next base_filename if any log in group fails

        df_data["filename"].append(current_filename)
        df_data["T1_vert"].append(t1_vert_val)
        df_data["S1_vert"].append(s1_vert_val)
        df_data["T1_d"].append(t1_d_val)
        df_data["S1_d"].append(s1_d_val)
        df_data["T2"].append(t2_vert_val)
        df_data["osc_strength_S1"].append(osc_strength_s1_val)
        df_data["S1ehdist"].append(s1_eh_dist_val)

    output_df = pd.DataFrame(df_data)
    output_df = output_df.sort_values(by="filename", ascending=True).reset_index(drop=True)
    output_df.to_csv(args.output_csv, index=False)
    logger.info(f"Extracted data for {len(output_df)} molecules. Results saved to '{args.output_csv}'.")

    # 2 Filter and compute hit rates
    df_clean = output_df[output_df['filename'].notna()]

    # 3a. Original filters
    hit_vert      = df_clean[df_clean['S1_vert'] - 2 * df_clean['T1_vert'] > -1]
    hit_ad_orig   = df_clean[df_clean['S1_d']    - 2 * df_clean['T1_d']    > -0]

    # 3b. New, stricter "ad hit" criteria:
    #     S1_d - 2*T1_d > 0,  T1_d > 1,  S1_d < 4
    hit_ad_strict = df_clean[
        (df_clean['S1_d'] - 2 * df_clean['T1_d'] >  0  ) &
        (df_clean['T1_d']                          >  1  ) &
        (df_clean['S1_d']                          <  4  )
    ]

    # 4. Compute hit‐rates
    total       = len(df_clean)
    vert_rate   = len(hit_vert     ) / total if total else 0
    ad_rate     = len(hit_ad_orig  ) / total if total else 0
    ad_strict_rate = len(hit_ad_strict) / total if total else 0

    # 5. Print summary
    logging.info(f"Total non-empty rows:       {total}")
    logging.info(f"Original Hit-vert (S1-2T1 > -1) rows:     {len(hit_vert):4d}   ({vert_rate:.1%})")
    logging.info(f"Original Hit-ad (S1-2T1 > 0)   rows:     {len(hit_ad_orig):4d}   ({ad_rate:.1%})")
    logging.info(f"Strict Hit-ad (the three criterias)   rows:     {len(hit_ad_strict):4d}   ({ad_strict_rate:.1%})\n")

    # 6. Sample output
    logging.info("-- Sample strict 'ad hits' --")
    logging.info(hit_ad_strict)
    

if __name__ == "__main__":
    main()

