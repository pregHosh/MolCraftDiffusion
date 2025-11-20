#!/usr/bin/env python3
"""
Analyse a directory of .xyz files and their *_opt.xyz counterparts.

For every original structure ``molecule_XXXX.xyz`` the script tries to locate
``molecule_XXXX_opt.xyz``.

For each pair it

1. computes the RMSD via the external program **calculate_rmsd**
2. builds PyG graphs for both geometries and checks if the topology is identical

Progress is displayed with a **tqdm** progress‑bar so you can see how many
structures have already been processed.

After scanning the whole directory, results are written to a CSV file **and** a
summary of key metrics is printed:

* **mean RMSD** (ignoring NaN)
* **% same_topology** (fraction of pairs with identical graphs, ignoring NaN)
* **% success** (rows whose status is ``success``)

Every original file produces one CSV row with these columns:

======================= ================= ===================================
Column                   Type              Meaning
----------------------- ----------------- -----------------------------------
original                str               Original XYZ filename
optimised               str | None        Optimised XYZ filename (or "–")
rmsd                    float | None      RMSD in Å (NaN if missing)
same_topology           bool | None       Graphs are identical? (NaN if n/a)
status                  str               "success" or "fail"
======================= ================= ===================================

Required helper functions (import them from **your** codebase):
    • ``read_xyz_file``
    • ``create_pyg_graph``
    • ``correct_edges``
    • ``compare_graph_topology``

``calculate_rmsd`` **must** be on your ``$PATH`` and print the RMSD value as
the **first** token to **stderr** (exactly as in your current workflow).

Usage
-----
.. code-block:: bash

    # analyse current working directory and write xyz_analysis.csv
    python check_xyz_pairs.py

    # analyse a specific directory and write custom CSV file
    python check_xyz_pairs.py /data/xyzs --csv my_results.csv
"""

from __future__ import annotations

import argparse
import subprocess as sp
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm  # progress bar

# ── import your chemistry helpers here ──────────────────────────────────────
#     Replace `metrics` with the real module name if different.
from MolecularDiffusion.utils.geom_utils import read_xyz_file, create_pyg_graph, correct_edges
from MolecularDiffusion.utils.geom_metrics import (is_fully_connected, 
                                                   compare_graph_topology)

# ── tunable constants ───────────────────────────────────────────────────────
EDGE_THRESHOLD: float = 2.5   # Å – radius used in create_pyg_graph
SCALE_FACTOR: float = 1.3     #     – rescaling factor in correct_edges

# ── helper functions ────────────────────────────────────────────────────────

def compute_rmsd(original: Path, optimised: Path) -> float:
    """Run ``calculate_rmsd`` and return the RMSD (first token on *stderr*)."""
    cmd = ["calculate_rmsd", str(original), str(optimised)]
    result = sp.run(cmd, capture_output=True, text=True)
    return float(result.stderr.split()[0])


def graphs_have_same_topology(original: Path, optimised: Path) -> bool:
    """Return *True* if the two XYZ files yield identical PyG graph topology."""
    xyz, Z = read_xyz_file(original)
    g_orig = create_pyg_graph(xyz, Z, xyz_filename=str(original), r=EDGE_THRESHOLD)
    g_orig = correct_edges(g_orig, scale_factor=SCALE_FACTOR)

    xyz_opt, Z_opt = read_xyz_file(optimised)
    g_opt = create_pyg_graph(xyz_opt, Z_opt, xyz_filename=str(optimised), r=EDGE_THRESHOLD)
    g_opt = correct_edges(g_opt, scale_factor=SCALE_FACTOR)

    return compare_graph_topology(g_orig, g_opt)


def process_directory(directory: Path, skip_disconnected=True) -> pd.DataFrame:
    """Scan *directory* and return a *DataFrame* with the analysis results."""

    records: List[dict] = []

    xyz_files = sorted(directory.glob("*.xyz"))

    for xyz_file in tqdm(xyz_files, desc="Processing XYZ files", unit="file"):
        # Ignore all *_opt.xyz – we handle the pair via the original filename
        if xyz_file.stem.endswith("_opt"):
            continue
        try:
            xyz, Z = read_xyz_file(xyz_file)
            g_orig = create_pyg_graph(xyz, Z, xyz_filename=str(xyz_files), r=EDGE_THRESHOLD)
            g_orig = correct_edges(g_orig, scale_factor=1.3)
            is_connected, num_components = is_fully_connected(g_orig.edge_index, g_orig.num_nodes)
        
            if is_connected or not(skip_disconnected):

                opt_file: Path = xyz_file.with_name(f"{xyz_file.stem}_opt.xyz")
                row: dict[str, Optional[str | float | bool]] = {
                    "original": xyz_file.name,
                    "optimised": opt_file.name if opt_file.exists() else "–",
                    "rmsd": None,
                    "same_topology": None,
                    "status": "success" if opt_file.exists() else "fail",
                }

                if opt_file.exists():
                    try:
                        row["rmsd"] = compute_rmsd(xyz_file, opt_file)
                        row["same_topology"] = graphs_have_same_topology(xyz_file, opt_file)
                    except Exception as exc:  # noqa: BLE001
                        # Any error in the pipeline downgrades the row to *fail*
                        tqdm.write(f"[ERROR] {xyz_file.name}: {exc}")
                        row["status"] = "fail"
                records.append(row)
            else:
                print("skip, disconnected")
                continue
        except Exception as e:
            print(f"fail to process {xyz_file} due to {e}")
            continue
    df = pd.DataFrame.from_records(
        records,
        columns=["original", "optimised", "rmsd", "same_topology", "status"],
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Compute and print mean RMSD, % same_topology and % success."""

    mean_rmsd = df["rmsd"].mean(skipna=True)

    same_topology_non_nan = df["same_topology"].dropna()
    pct_same_topology = 100 * same_topology_non_nan.mean() if not same_topology_non_nan.empty else float("nan")

    pct_success = 100 * (df["status"] == "success").mean()

    print("\nSummary statistics:")
    print(f"  Mean RMSD          : {mean_rmsd:.4f} Å")
    print(f"  % same topology    : {pct_same_topology:.2f} %")
    print(f"  % success          : {pct_success:.2f} %")


# ── script entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="check_xyz_pairs",
        description="Compute RMSD and graph equivalence for xyz / *_opt.xyz pairs, "
                    "then write a CSV summary.",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        type=Path,
        help="Directory containing XYZ files (default: current directory).",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        default="xyz_analysis.csv",
        help="Output CSV filename (default: xyz_analysis.csv).",
    )
    args = parser.parse_args()

    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"{directory} is not a directory")

    df = process_directory(directory)
    df.to_csv(args.csv_path, index=False)

    print(f"\nAnalysis finished → {args.csv_path}\n{len(df)} structures processed.")

    # Print summary metrics requested by the user
    print_summary(df)

