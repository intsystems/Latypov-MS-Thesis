from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

from experiment_runner import load_results


DPI = 400
FIGSIZE = (16, 6)
STD_MULTIPLIER = 1
COLORS = ["blue", "red", "black", "r", "black", "blue", "green", "y", "m", "y", "k"]
MARKERS = ["o", "s", "^", "v", "D", "p", "*", "h"]
LINESTYLES = [":", "--", "-.", "-"]


def save_plots(fig, filename: str | Path = "experiment_results.pdf"):
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_fig_set_style(lines_count, shape=(1, 1), figsize=None, params=None):
    if params is None:
        params = {
            "legend.fontsize": 17,
            "lines.markersize": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 10,
        }
    sns.set_context("paper", rc=params)
    if figsize is None:
        fig, ax = plt.subplots(*shape, dpi=DPI)
    else:
        fig, ax = plt.subplots(*shape, dpi=DPI, figsize=figsize, constrained_layout=True)
    plt.grid(which="both")
    return fig, ax


def plot_bar(ax, field_name, unique_groups, temp_map, k: int):
    pos = 0
    width = 0.8 / len(temp_map)
    for g_idx, g_name in enumerate(unique_groups):
        group_keys = sorted([k2 for k2 in temp_map.keys() if k2[0] == g_name])
        for alg_idx, key in enumerate(group_keys):
            values_t = temp_map[key]
            if g_name == "SmoothCORRAL":
                arr = [getattr(exp.algorithm, "selection_for_decisions") for exp in values_t]
            else:
                arr = [getattr(exp.algorithm, field_name) for exp in values_t]
            arr = np.array(arr)
            data = np.mean(arr, axis=0)
            x = np.arange(k) + pos * width
            pos += 1
            color = COLORS[g_idx % len(COLORS)]
            coeff = coeffs[alg_idx % len(coeffs)]
            ax.bar(x, data, width=width, color=color, alpha=coeff, label=f"{key[0]}, M={key[1]}")
    return ax


def plot_regret(ax, unique_groups, temp_map):
    for g_idx, g_name in enumerate(unique_groups):
        group_keys = sorted([k2 for k2 in temp_map.keys() if k2[0] == g_name])
        for alg_idx, key in enumerate(group_keys):
            values_t = temp_map[key]
            a = [exp.get_expected_regret() for exp in values_t]
            values_t = np.stack(a)
            data = np.array(values_t)
            mean_vals = np.mean(data, axis=0) if data.ndim > 1 else data
            std_vals = np.std(data, axis=0) if data.ndim > 1 else np.zeros_like(data)
            std_vals *= STD_MULTIPLIER
            x = np.arange(len(mean_vals))
            color = COLORS[g_idx % len(COLORS)]
            coeff = coeffs[alg_idx % len(coeffs)]
            marker = MARKERS[g_idx % len(COLORS)]
            linestyle = LINESTYLES[alg_idx % len(coeffs)]
            ax.plot(
                x,
                mean_vals,
                color=color,
                alpha=coeff,
                lw=2,
                label=f"{key[0]}, M={key[1]}",
                linestyle=linestyle,
                marker=marker,
                markevery=max(1, len(x) // 20),
                markersize=5,
            )
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2 * coeff)


def plot_data(temp_map, output_path: str | Path, k: int):
    global coeffs
    max_algs = max(k2[1] for k2 in temp_map.keys()) + 1
    coeffs = np.logspace(0, -0.7, max_algs)

    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    fig, (ax1, ax2, ax3) = get_fig_set_style(1, (1, 3), FIGSIZE)

    unique_groups = ["M-LCB", "LimitedAdvice", "SmoothCORRAL"]
    plot_regret(ax1, unique_groups, temp_map)
    plot_bar(ax2, "selection_for_decisions", unique_groups, temp_map, k)
    plot_bar(ax3, "counts", unique_groups, temp_map, k)

    ax1.set_title("(a) Cumulative Loss vs Steps")
    ax1.set_xlabel(r"$\#$ steps")
    ax1.set_ylabel("Cumulative Loss")
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(formatter)

    ax2.set_title("(b) Arm Selection Distribution")
    ax2.set_xlabel("Arm Index")
    ax2.set_ylabel("Selection Count")
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(formatter)

    ax3.set_title("(c) Arm Optimization Distribution")
    ax3.set_xlabel("Arm Index")
    ax3.set_ylabel("Optimization Count")
    ax3.grid(True)
    ax3.yaxis.set_major_formatter(formatter)

    h, legend_ = ax1.get_legend_handles_labels()
    rows = 3
    cols = 3
    total_slots = rows * cols

    empty_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    h_padded = list(h)
    l_padded = list(legend_)

    while len(h_padded) < total_slots:
        h_padded.append(empty_handle)
        l_padded.append("")

    fig.legend(
        h_padded,
        l_padded,
        loc="lower center",
        bbox_to_anchor=(0.025, -0.18, 0.95, 0.08),
        ncol=cols,
        mode="expand",
        borderaxespad=0.5,
        columnspacing=2.0,
        handletextpad=0.5,
        frameon=True,
        fontsize=15,
    )
    save_plots(fig, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render plots from experiment results.")
    parser.add_argument("--input", type=Path, default=Path("exp_result.pkl"), help="Path to experiment pickle file.")
    parser.add_argument("--output", type=Path, default=Path("bandit_experiment.pdf"), help="Path to output PDF.")
    parser.add_argument("--k", type=int, default=10, help="Number of arms on the x-axis.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    temp_map = load_results(args.input)
    plot_data(temp_map, args.output, args.k)
    print(f"Saved plots to {args.output}")


if __name__ == "__main__":
    main()
