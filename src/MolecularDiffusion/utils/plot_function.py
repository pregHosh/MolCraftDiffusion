import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from typing import Optional, Union, List
 
sns.set_palette("colorblind", 8)
sns.set_style("ticks", {"xtick.major.size": 18, "ytick.major.size": 18})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

def plot_correlation_with_histograms(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    property_name: str = "E",
    unit: str = "eV",
    output_path: Optional[str] = None
) -> None:
    """
    Plot a scatter plot comparing predicted vs. true values, with marginal histograms.

    Args:
        true_values (np.ndarray): True target values.
        predicted_values (np.ndarray): Predicted target values.
        property_name (str): Name of the property (for axis labels). Default is "E".
        unit (str): Unit of the property. Default is "eV".
        output_path (Optional[str]): If provided, path to save the figure.
    """
    # Compute metrics
    r2 = r2_score(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)

    # Set plot style
    prussian_blue = "#003153"
    plt.rc("font", size=24)

    # Setup figure and subplots using GridSpec
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_xhist = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Main scatter plot
    ax_main.scatter(
        predicted_values,
        true_values,
        color=prussian_blue,
        alpha=0.6,
        s=50
    )

    # Diagonal reference line
    min_val = min(true_values.min(), predicted_values.min())
    max_val = max(true_values.max(), predicted_values.max())
    ax_main.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="black",
        linewidth=2
    )

    ax_main.set_xlabel(f"{property_name},$_{{pred}}$ ({unit})", fontsize=30)
    ax_main.set_ylabel(f"{property_name},$_{{DFT}}$ ({unit})", fontsize=30)
    ax_main.tick_params(axis="both", labelsize=24)

    # R² and MAE annotation
    ax_main.text(
        0.5, 0.95,
        f"R$^2$: {r2:.2f}\nMAE: {mae:.2f} {unit}",
        transform=ax_main.transAxes,
        fontsize=30,
        ha="center",
        va="top",
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

    # Marginal histograms
    ax_xhist.hist(predicted_values, bins=30, color=prussian_blue, alpha=0.6)
    ax_yhist.hist(true_values, bins=30, orientation='horizontal', color=prussian_blue, alpha=0.6)

    # Remove ticks and labels from histograms
    ax_xhist.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    ax_yhist.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    # Hide spines on histograms
    for hist_ax in [ax_xhist, ax_yhist]:
        for spine in hist_ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def plot_kde_distribution(
    predictions: np.ndarray,
    task_name: str,
    output_path: Optional[str] = None
) -> None:
    """
    Plot the Kernel Density Estimate (KDE) of a single target distribution.

    Args:
        predictions (np.ndarray): Array of prediction values (1D).
        task_name (str): Name of the task/property for labeling.
        output_path (Optional[str]): Path to save the plot. Defaults to 'kde_targets.png'.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.kdeplot(predictions, label=task_name, color="blue", fill=True, linewidth=3, ax=ax)

    ax.set_title(f"{task_name} Distribution", fontsize=22)
    ax.set_xlabel(task_name, fontsize=32)
    ax.set_ylabel("Frequency", fontsize=32)
    ax.tick_params(labelsize=28)
    plt.tight_layout()

    if output_path is None:
        output_path = "kde_targets.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_kde_distribution_multiple(
    predictions: np.ndarray,
    task_names: List[str],
    output_path: Optional[str] = None
) -> None:
    """
    Plot KDEs of multiple target distributions on the same axes.

    Args:
        predictions (np.ndarray): 2D array of shape (n_samples, n_tasks).
        task_names (List[str]): List of task/property names.
        output_path (Optional[str]): Path to save the plot. Defaults to 'kde_targets.png'.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("hsv", len(task_names))

    for i, task_name in enumerate(task_names):
        sns.kdeplot(
            predictions[:, i],
            label=task_name,
            color=colors[i],
            fill=True,
            linewidth=3,
            alpha=0.5,
            ax=ax,
            zorder=2
        )

    # Highlight specified global regions
    x = np.linspace(-10, 10, 1000)
    y_max = ax.get_ylim()[1] * 1.05
    ax.fill_between(x, 0, y_max, where=((x >= 1.12) & (x <= 1.9)), color='#cdd14d58', alpha=0.15, zorder=0, hatch='////')
    ax.fill_between(x, 0, y_max, where=((x > 2.24) & (x <= 3.8)), color='#cdd14d58', alpha=0.15, zorder=0, hatch='////')

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Values", fontsize=36)
    ax.set_ylabel("Frequency", fontsize=36)
    ax.legend(fontsize=30)
    ax.tick_params(labelsize=32)
    plt.tight_layout()

    if output_path is None:
        output_path = "kde_targets.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_histogram_distribution(
    predictions: np.ndarray,
    task_name: str,
    output_path: Optional[str] = None,
    num_bins: int = 50
) -> None:
    """
    Plot a histogram of the distribution for a single target.

    Args:
        predictions (np.ndarray): Array of prediction values (1D).
        task_name (str): Name of the task/property for labeling.
        output_path (Optional[str]): Path to save the plot. Defaults to 'hist_targets.png'.
        num_bins (int): Number of histogram bins. Defaults to 50.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    bin_min = np.floor(np.min(predictions))
    bin_max = np.ceil(np.max(predictions))
    bin_width = np.ceil((bin_max - bin_min) / num_bins)
    bin_edges = np.arange(bin_min, bin_max + bin_width, bin_width)

    ax.hist(
        predictions,
        bins=bin_edges,
        color="#1266A4",
        edgecolor="#1266A4",
        alpha=0.8,
        align="mid",
        rwidth=0.8,
    )

    ax.set_title(f"{task_name} Distribution", fontsize=22)
    ax.set_xlabel(task_name, fontsize=36)
    ax.set_ylabel("Number of Occurrences", fontsize=36)
    ax.tick_params(labelsize=32)
    plt.tight_layout()

    if output_path is None:
        output_path = "hist_targets.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
