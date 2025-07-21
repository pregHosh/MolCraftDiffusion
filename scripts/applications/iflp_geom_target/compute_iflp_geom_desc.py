#!/usr/bin/env python3
"""
Analyze IFLP XYZ files using PyTorch Geometric for graph construction.
Outputs a CSV with:
- filename
- distance_BN
- angle_BH_NH
- shortest_path_length (number of atoms separating B and N)

Assumes:
- B is atom 0
- N is atom 1
- H(B) is atom 2
- H(N) is atom 3

Usage:
    python analyze_iflp_pyg.py <xyz_dir> [--scaling 1.2]
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ase.data import atomic_numbers, covalent_radii
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)-10s %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants for atom indices (0-based)
B_ATOM_IDX = 0
N_ATOM_IDX = 1
HB_ATOM_IDX = 2
HN_ATOM_IDX = 3


def read_xyz(filepath: Path) -> Tuple[list[str], np.ndarray]:
    """
    Reads atomic symbols and coordinates from an XYZ file.

    Args:
        filepath (Path): Path to the XYZ file.

    Returns:
        Tuple[list[str], np.ndarray]: A tuple containing:
            - atoms (list[str]): List of atomic symbols (e.g., "C", "N").
            - positions (np.ndarray): Nx3 NumPy array of atomic coordinates.
    """
    atoms = []
    positions = []
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            positions.append([float(x) for x in parts[1:4]])
    return atoms, np.array(positions)


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two 3D points.

    Args:
        p1 (np.ndarray): First 3D point (e.g., array of [x, y, z]).
        p2 (np.ndarray): Second 3D point.

    Returns:
        float: The distance between the two points.
    """
    return np.linalg.norm(p1 - p2)


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculates the angle in degrees between two 3D vectors.

    Args:
        v1 (np.ndarray): The first 3D vector.
        v2 (np.ndarray): The second 3D vector.

    Returns:
        float: The angle between the vectors in degrees. Returns NaN if either vector has zero norm.
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return np.nan

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def build_pyg_graph(atoms: list[str], positions: np.ndarray, scaling: float) -> Data:
    """
    Builds a PyTorch Geometric (PyG) graph from atomic symbols and positions.
    Edges are determined by interatomic distances relative to covalent radii.

    Args:
        atoms (list[str]): List of atomic symbols.
        positions (np.ndarray): Nx3 NumPy array of atomic coordinates.
        scaling (float): Scaling factor for covalent radii to determine bond cutoffs.

    Returns:
        Data: A PyTorch Geometric Data object representing the molecular graph.
    """
    edge_index = []
    radii = [covalent_radii[atomic_numbers.get(a, 0)] for a in atoms]
    num_atoms = len(atoms)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            if radii[i] == 0 or radii[j] == 0:
                continue

            cutoff = scaling * (radii[i] + radii[j])
            if calculate_distance(positions[i], positions[j]) <= cutoff:
                edge_index.append([i, j])
                edge_index.append([j, i])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return Data(edge_index=edge_index, num_nodes=num_atoms)


def plot_kde(data: np.ndarray, task_name: str, output_filepath: Path) -> None:
    """
    Generates and saves a Kernel Density Estimate (KDE) plot.

    Args:
        data (np.ndarray): Numerical data to plot.
        task_name (str): Label for the x-axis and plot title.
        output_filepath (Path): Full path to save the plot (e.g., 'kde_plot.png').
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.kdeplot(
        data,
        label=task_name,
        color="#008fd5",
        fill=True,
        linewidth=3,
        alpha=0.5,
        zorder=2,
        ax=ax,
    )

    y_max = ax.get_ylim()[1] * 1.05
    ax.set_ylim(0, y_max)
    ax.set_xlabel(task_name, fontsize=36)
    ax.set_ylabel("Frequency", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def plot_hist(data: np.ndarray, task_name: str, output_filepath: Path, bins: int = 30) -> None:
    """
    Generates and saves a histogram plot.

    Args:
        data (np.ndarray): Numerical data to plot.
        task_name (str): Label for the x-axis and plot title.
        output_filepath (Path): Full path to save the plot (e.g., 'hist_plot.png').
        bins (int): Number of bins for the histogram.
    """
    is_int = np.allclose(data, data.astype(int))
    if is_int:
        min_val = int(data.min())
        max_val = int(data.max())
        bin_edges = np.arange(min_val - 0.5, max_val + 1.5, 1.0)
        xticks = np.arange(min_val, max_val + 1)
    else:
        bin_edges = bins
        xticks = None

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(
        data,
        bins=bin_edges,
        color="#008fd5",
        alpha=0.5,
        edgecolor="black",
        linewidth=2,
        zorder=2,
    )

    if is_int:
        ax.set_xticks(xticks)

    y_max = ax.get_ylim()[1] * 1.05
    ax.set_ylim(0, y_max)

    ax.set_xlabel(task_name, fontsize=36)
    ax.set_ylabel("Frequency", fontsize=36)
    ax.tick_params(axis="both", labelsize=32)

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)


def main():
    """
    Main function to analyze IFLP XYZ files.
    It reads XYZ files, calculates geometric properties (distances, angles),
    builds molecular graphs, determines shortest path lengths, and
    generates KDE and histogram plots of the calculated properties.
    Results are saved to a CSV file in the input directory.
    """
    parser = argparse.ArgumentParser(description="Analyze IFLP XYZ files with PyTorch Geometric.")
    parser.add_argument(
        "xyz_dir",
        type=str,
        help="Directory containing .xyz files."
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=1.2, # Default value moved here
        help="Scaling factor for covalent radii in graph construction (default: 1.2)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (sets logging level to DEBUG)."
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    xyz_directory = Path(args.xyz_dir)
    if not xyz_directory.is_dir():
        logger.critical(f"Error: Directory '{xyz_directory}' does not exist or is not a directory.")
        sys.exit(1)

    records = []
    for file_path in xyz_directory.glob("*.xyz"):
        try:
            atoms, positions = read_xyz(file_path)
            
            if len(positions) < max(B_ATOM_IDX, N_ATOM_IDX, HB_ATOM_IDX, HN_ATOM_IDX) + 1:
                logger.warning(f"[{file_path.name}]: Fewer than required atoms ({max(B_ATOM_IDX, N_ATOM_IDX, HB_ATOM_IDX, HN_ATOM_IDX) + 1}) – skipped.")
                continue

            dist_BN = calculate_distance(positions[B_ATOM_IDX], positions[N_ATOM_IDX])
            vec_BH = positions[HB_ATOM_IDX] - positions[B_ATOM_IDX]
            vec_NH = positions[HN_ATOM_IDX] - positions[N_ATOM_IDX]
            ang_BH_NH = calculate_angle_between_vectors(vec_BH, vec_NH)

            data = build_pyg_graph(atoms, positions, scaling=args.scaling)
            G = to_networkx(data, to_undirected=True)

            shortest_path_length = None
            try:
                shortest_path_length = nx.shortest_path_length(G, source=B_ATOM_IDX, target=N_ATOM_IDX)
            except nx.NetworkXNoPath:
                logger.warning(f"[{file_path.name}]: No path found between atoms {B_ATOM_IDX} (B) and {N_ATOM_IDX} (N).")
            except Exception as e:
                logger.error(f"[{file_path.name}]: Error calculating shortest path: {e}")

            records.append({
                'filename': file_path.name,
                'distance_BN': dist_BN,
                'angle_BH_NH': ang_BH_NH,
                'shortest_path_length': shortest_path_length
            })
            logger.info(f"Processed {file_path.name}")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e} – skipped.")
            continue

    if not records:
        logger.info("No suitable XYZ files found or all files were skipped. No output CSV generated.")
        sys.exit(0)

    df = pd.DataFrame(records)
    df.sort_values('filename', inplace=True)
    
    output_csv_path = xyz_directory / 'iflp_pyg_analysis.csv'
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved analysis to {output_csv_path}")

    plot_kde(df['distance_BN'].values, 'd$_{BN}$ (Å)', output_filepath=xyz_directory / 'kde_distance_BN.png')
    plot_kde(df['angle_BH_NH'].values, '$\Phi\degree$', output_filepath=xyz_directory / 'kde_angle_BH_NH.png')
    
    sep_vals = df['shortest_path_length'].dropna().values
    if sep_vals.size > 0:
        plot_kde(sep_vals, 'Number of atoms separating B and N', output_filepath=xyz_directory / 'kde_shortest_path_length.png')
        plot_hist(sep_vals, 'Number of atoms separating B and N', output_filepath=xyz_directory / 'hist_shortest_path_length.png')
    else:
        logger.warning("No valid shortest path lengths to plot.")

    plot_hist(df['distance_BN'].values, 'd$_{BN}$ (Å)', output_filepath=xyz_directory / 'hist_distance_BN.png')
    plot_hist(df['angle_BH_NH'].values, '$\Phi\degree$', output_filepath=xyz_directory / 'hist_angle_BH_NH.png')
    
    logger.info("All plots generated successfully.")


if __name__ == "__main__":
    main()
