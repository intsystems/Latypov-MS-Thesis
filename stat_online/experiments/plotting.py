"""Plotting helpers that consume experiment outputs or primitive artifacts."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stat_online.experiments.matplotlib import import_pyplot
from stat_online.experiments.storage import load_experiment_artifacts, sanitize_key

DEFAULT_PLOT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "plotting"


def _series_key(metric: str, algorithm: str, repeat: str | int) -> str:
    repeat_id = str(repeat)
    if not repeat_id.startswith("repeat_"):
        repeat_id = f"repeat_{repeat_id}"
    return sanitize_key(f"{metric}/{algorithm}/{repeat_id}")


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_plot_config(name: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load plotting config from ``configs/plotting`` with optional override."""
    default_path = DEFAULT_PLOT_CONFIG_DIR / f"{name}.json"
    config: dict[str, Any] = {}
    if default_path.exists():
        with default_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as f:
            config = _deep_update(config, json.load(f))
    return config


def _mean_std(series_list: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(series) for series in series_list)
    stacked = np.vstack([np.asarray(series[:min_len], dtype=float) for series in series_list])
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def _ordered_groups(keys: Sequence[tuple[str, int]], group_order: Sequence[str]) -> list[str]:
    present = {group for group, _m in keys}
    ordered = [group for group in group_order if group in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _coefficients(keys: Sequence[tuple[str, int]], config: Mapping[str, Any]) -> np.ndarray:
    max_m = max((m for _group, m in keys), default=0) + 1
    coeff = config.get("alpha_coefficients", {})
    start = float(coeff.get("start", 0.0))
    stop = float(coeff.get("stop", -0.7))
    return np.logspace(start, stop, max(1, max_m))


def _style_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("style", {}))


def _plot_bar_map(ax, data_map, keys, groups, k: int, config: Mapping[str, Any], ylabel: str):
    colors = config["colors"]
    coeffs = _coefficients(keys, config)
    width = float(config.get("bar_width_total", 0.8)) / max(1, len(keys))

    pos = 0
    for g_idx, group in enumerate(groups):
        group_keys = sorted([key for key in keys if key[0] == group], key=lambda item: item[1])
        for alg_idx, key in enumerate(group_keys):
            values = data_map.get(key)
            if not values:
                continue
            min_len = min(len(np.asarray(value)) for value in values)
            stacked = np.vstack([np.asarray(value[:min_len], dtype=float) for value in values])
            mean_vals = np.mean(stacked, axis=0)[:k]
            x = np.arange(len(mean_vals)) + pos * width
            pos += 1
            ax.bar(
                x,
                mean_vals,
                width=width,
                color=colors[g_idx % len(colors)],
                alpha=coeffs[alg_idx % len(coeffs)],
                label=f"{key[0]}, M={key[1]}",
            )
    ax.set_xlabel(config["labels"]["arm_index"])
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_classical_bandit_series(
    regrets: Mapping[tuple[str, int], Sequence[np.ndarray]],
    selection_counts: Mapping[tuple[str, int], Sequence[np.ndarray]],
    optimization_counts: Mapping[tuple[str, int], Sequence[np.ndarray]],
    *,
    output_dir: str | Path,
    k: int,
    plot_config: Mapping[str, Any] | None = None,
):
    """Render the current classical-bandit paper-style three-panel figure."""
    plt = import_pyplot(output_dir)
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    import seaborn as sns

    config = dict(plot_config or load_plot_config("classical_bandits"))
    sns.set_context("paper", rc=_style_config(config))

    keys = sorted(regrets.keys(), key=lambda item: (config["group_order"].index(item[0]) if item[0] in config["group_order"] else 99, item[1]))
    groups = _ordered_groups(keys, config["group_order"])
    colors = config["colors"]
    markers = config["markers"]
    linestyles = config["linestyles"]
    coeffs = _coefficients(keys, config)

    formatter = mtick.ScalarFormatter(useMathText=bool(config.get("use_math_text", True)))
    formatter.set_scientific(True)
    formatter.set_powerlimits(tuple(config.get("powerlimits", [-1, 1])))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        dpi=int(config.get("dpi", 400)),
        figsize=tuple(config.get("figsize", [16, 6])),
        constrained_layout=bool(config.get("constrained_layout", True)),
    )
    plt.grid(which="both")

    std_multiplier = float(config.get("std_multiplier", 1.0))
    for g_idx, group in enumerate(groups):
        group_keys = sorted([key for key in keys if key[0] == group], key=lambda item: item[1])
        for alg_idx, key in enumerate(group_keys):
            series = regrets[key]
            mean_vals, std_vals = _mean_std(series)
            std_vals *= std_multiplier
            x = np.arange(len(mean_vals))
            color = colors[g_idx % len(colors)]
            coeff = coeffs[alg_idx % len(coeffs)]
            ax1.plot(
                x,
                mean_vals,
                color=color,
                alpha=coeff,
                lw=float(config.get("line_width", 2)),
                label=f"{key[0]}, M={key[1]}",
                linestyle=linestyles[alg_idx % len(linestyles)],
                marker=markers[g_idx % len(markers)],
                markevery=max(1, len(x) // int(config.get("markevery_divisor", 20))),
                markersize=float(config.get("marker_size", 5)),
            )
            ax1.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2 * coeff)

    labels = config["labels"]
    ax1.set_title(labels["regret_title"])
    ax1.set_xlabel(labels["steps"])
    ax1.set_ylabel(labels["regret"])
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(formatter)

    _plot_bar_map(ax2, selection_counts, keys, groups, k, config, labels["selection_count"])
    ax2.set_title(labels["selection_title"])
    ax2.yaxis.set_major_formatter(formatter)

    _plot_bar_map(ax3, optimization_counts, keys, groups, k, config, labels["optimization_count"])
    ax3.set_title(labels["optimization_title"])
    ax3.yaxis.set_major_formatter(formatter)

    handles, legend_labels = ax1.get_legend_handles_labels()
    legend = config.get("legend", {})
    rows = int(legend.get("rows", 3))
    cols = int(legend.get("cols", 3))
    total_slots = rows * cols
    empty_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    handles = list(handles)
    legend_labels = list(legend_labels)
    while len(handles) < total_slots:
        handles.append(empty_handle)
        legend_labels.append("")

    fig.legend(
        handles,
        legend_labels,
        loc=legend.get("loc", "lower center"),
        bbox_to_anchor=tuple(legend.get("bbox_to_anchor", [0.025, -0.18, 0.95, 0.08])),
        ncol=cols,
        mode=legend.get("mode", "expand"),
        borderaxespad=float(legend.get("borderaxespad", 0.5)),
        columnspacing=float(legend.get("columnspacing", 2.0)),
        handletextpad=float(legend.get("handletextpad", 0.5)),
        frameon=bool(legend.get("frameon", True)),
        fontsize=int(legend.get("fontsize", 15)),
    )
    return fig


def plot_classical_bandit_results(
    temp_map,
    *,
    output_dir: str | Path,
    k: int,
    plot_config_path: str | Path | None = None,
):
    """Plot live classical-bandit experiment objects using the shared style."""
    config = load_plot_config("classical_bandits", plot_config_path)
    regrets: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    selection_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    optimization_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for key, runs in temp_map.items():
        group, _m = key
        for exp in runs:
            regrets[key].append(np.asarray(exp.get_expected_regret(), dtype=float))
            selection_counts[key].append(np.asarray(getattr(exp.algorithm, "selection_for_decisions"), dtype=float))
            if group == "SmoothCORRAL":
                optimization_counts[key].append(np.asarray(getattr(exp.algorithm, "selection_for_decisions"), dtype=float))
            else:
                optimization_counts[key].append(np.asarray(getattr(exp.algorithm, "counts"), dtype=float))

    return plot_classical_bandit_series(
        regrets,
        selection_counts,
        optimization_counts,
        output_dir=output_dir,
        k=k,
        plot_config=config,
    )



def _glm_algorithm_key_from_name(algorithm: str) -> tuple[str, int]:
    parts = algorithm.split("_")
    if len(parts) >= 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return algorithm, 0


def plot_glm_series(
    cumulative_losses: Mapping[tuple[str, int], Sequence[np.ndarray]],
    selection_counts: Mapping[tuple[str, int], Sequence[np.ndarray]],
    optimization_counts: Mapping[tuple[str, int], Sequence[np.ndarray]],
    *,
    output_dir: str | Path,
    k: int,
    plot_config: Mapping[str, Any] | None = None,
):
    """Render GLM/linear-bandit results as loss, selection, and optimization panels."""
    plt = import_pyplot(output_dir)
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mtick
    import seaborn as sns

    config = dict(plot_config or load_plot_config("glm_bandits"))
    sns.set_context("paper", rc=_style_config(config))

    keys = sorted(
        cumulative_losses.keys(),
        key=lambda item: (config["group_order"].index(item[0]) if item[0] in config["group_order"] else 99, item[1]),
    )
    groups = _ordered_groups(keys, config["group_order"])
    colors = config["colors"]
    markers = config["markers"]
    linestyles = config["linestyles"]
    coeffs = _coefficients(keys, config)

    formatter = mtick.ScalarFormatter(useMathText=bool(config.get("use_math_text", True)))
    formatter.set_scientific(True)
    formatter.set_powerlimits(tuple(config.get("powerlimits", [-1, 1])))

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        dpi=int(config.get("dpi", 400)),
        figsize=tuple(config.get("figsize", [16, 6])),
        constrained_layout=bool(config.get("constrained_layout", True)),
    )
    plt.grid(which="both")

    std_multiplier = float(config.get("std_multiplier", 1.0))
    for g_idx, group in enumerate(groups):
        group_keys = sorted([key for key in keys if key[0] == group], key=lambda item: item[1])
        for alg_idx, key in enumerate(group_keys):
            series = cumulative_losses[key]
            mean_vals, std_vals = _mean_std(series)
            std_vals *= std_multiplier
            x = np.arange(len(mean_vals))
            color = colors[g_idx % len(colors)]
            coeff = coeffs[alg_idx % len(coeffs)]
            ax1.plot(
                x,
                mean_vals,
                color=color,
                alpha=coeff,
                lw=float(config.get("line_width", 2)),
                label=f"{key[0]}, M={key[1]}",
                linestyle=linestyles[alg_idx % len(linestyles)],
                marker=markers[g_idx % len(markers)],
                markevery=max(1, len(x) // int(config.get("markevery_divisor", 20))),
                markersize=float(config.get("marker_size", 5)),
            )
            ax1.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.2 * coeff)

    labels = config["labels"]
    ax1.set_title(labels["loss_title"])
    ax1.set_xlabel(labels["steps"])
    ax1.set_ylabel(labels["loss"])
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(formatter)

    _plot_bar_map(ax2, selection_counts, keys, groups, k, config, labels["selection_count"])
    ax2.set_title(labels["selection_title"])
    ax2.yaxis.set_major_formatter(formatter)

    _plot_bar_map(ax3, optimization_counts, keys, groups, k, config, labels["optimization_count"])
    ax3.set_title(labels["optimization_title"])
    ax3.yaxis.set_major_formatter(formatter)

    handles, legend_labels = ax1.get_legend_handles_labels()
    legend = config.get("legend", {})
    rows = int(legend.get("rows", 3))
    cols = int(legend.get("cols", 3))
    total_slots = rows * cols
    empty_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    handles = list(handles)
    legend_labels = list(legend_labels)
    while len(handles) < total_slots:
        handles.append(empty_handle)
        legend_labels.append("")

    fig.legend(
        handles,
        legend_labels,
        loc=legend.get("loc", "lower center"),
        bbox_to_anchor=tuple(legend.get("bbox_to_anchor", [0.025, -0.18, 0.95, 0.08])),
        ncol=cols,
        mode=legend.get("mode", "expand"),
        borderaxespad=float(legend.get("borderaxespad", 0.5)),
        columnspacing=float(legend.get("columnspacing", 2.0)),
        handletextpad=float(legend.get("handletextpad", 0.5)),
        frameon=bool(legend.get("frameon", True)),
        fontsize=int(legend.get("fontsize", 15)),
    )
    return fig


def plot_glm_results(
    res_dict,
    *,
    output_dir: str | Path,
    k: int,
    plot_config_path: str | Path | None = None,
):
    """Plot live GLM experiment results using the shared three-panel style."""
    config = load_plot_config("glm_bandits", plot_config_path)
    cumulative_losses: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    selection_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    optimization_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for alg_name, runs in res_dict.items():
        key = _glm_algorithm_key_from_name(alg_name)
        for run in runs:
            losses = np.asarray(run.alg_loss_hist, dtype=float)
            selections = np.asarray(run.arm_selection_hist, dtype=int)
            n_arms = max(int(k), int(run.n_arms))
            cumulative_losses[key].append(np.cumsum(losses))
            selection_counts[key].append(np.bincount(selections, minlength=n_arms)[:n_arms])
            optimization_counts[key].append(
                np.asarray([arm.optimized_count for arm in run.learned_algorithm.arms], dtype=float)[:n_arms]
            )

    return plot_glm_series(
        cumulative_losses,
        selection_counts,
        optimization_counts,
        output_dir=output_dir,
        k=k,
        plot_config=config,
    )

def save_figure(fig, output_path: str | Path, output_dir: str | Path, *, dpi: int = 300, fmt: str = "pdf") -> Path:
    plt = import_pyplot(output_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lstrip(".") or fmt
    fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_glm_artifacts(output_dir: str | Path, plot_config_path: str | Path | None = None):
    """Plot GLM loss, selection counts, and optimization counts from artifacts."""
    artifacts = load_experiment_artifacts(output_dir)
    rows: Sequence[Mapping[str, Any]] = artifacts["runs"]
    arrays = artifacts["arrays"]
    config = load_plot_config("glm_bandits", plot_config_path)

    cumulative_losses: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    selection_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    optimization_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for row in rows:
        algorithm = row["algorithm"]
        repeat = row["repeat"]
        key = _glm_algorithm_key_from_name(algorithm)
        cumulative_loss = _series_from_arrays(arrays, "cum_loss", algorithm, repeat)
        selection = _series_from_arrays(arrays, "selection_count", algorithm, repeat)
        optimization = _series_from_arrays(arrays, "optimization_count", algorithm, repeat)

        if cumulative_loss is not None:
            cumulative_losses[key].append(np.asarray(cumulative_loss, dtype=float))
        if selection is not None:
            selection_counts[key].append(np.asarray(selection, dtype=float))
        if optimization is not None:
            optimization_counts[key].append(np.asarray(optimization, dtype=float))

    if not cumulative_losses:
        raise ValueError(f"No cumulative loss arrays found in {output_dir}")
    if not selection_counts:
        raise ValueError(f"No selection_count arrays found in {output_dir}")
    if not optimization_counts:
        raise ValueError(f"No optimization_count arrays found in {output_dir}")

    artifact_config = artifacts.get("config", {})
    k = int(artifact_config.get("K") or 0)
    if k <= 0:
        k = max((len(series[0]) for series in selection_counts.values() if series), default=1)

    return plot_glm_series(
        cumulative_losses,
        selection_counts,
        optimization_counts,
        output_dir=output_dir,
        k=k,
        plot_config=config,
    )

def save_glm_artifact_plot(
    output_dir: str | Path,
    filename: str = "regenerated_glm_summary.pdf",
    plot_config_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    config = load_plot_config("glm_bandits", plot_config_path)
    fig = plot_glm_artifacts(output_dir, plot_config_path=plot_config_path)
    return save_figure(fig, output_dir / filename, output_dir, dpi=int(config.get("save_dpi", 300)))


def _algorithm_key(row: Mapping[str, Any]) -> tuple[str, int]:
    group = row.get("group") or str(row["algorithm"]).rsplit("_M", 1)[0]
    try:
        m_value = int(row.get("M") or str(row["algorithm"]).rsplit("_M", 1)[1])
    except (IndexError, ValueError):
        m_value = 0
    return str(group), m_value


def _series_from_arrays(arrays, metric: str, algorithm: str, repeat: str | int) -> np.ndarray | None:
    key = _series_key(metric, algorithm, repeat)
    if key in arrays:
        return np.asarray(arrays[key])
    return None


def plot_classical_bandit_artifacts(output_dir: str | Path, plot_config_path: str | Path | None = None):
    """Plot classical-bandit summaries from saved artifacts in the paper style."""
    artifacts = load_experiment_artifacts(output_dir)
    rows: Sequence[Mapping[str, Any]] = artifacts["runs"]
    arrays = artifacts["arrays"]
    config = load_plot_config("classical_bandits", plot_config_path)

    regrets: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    selection_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    optimization_counts: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    for row in rows:
        algorithm = row["algorithm"]
        repeat = row["repeat"]
        key = _algorithm_key(row)
        regret = _series_from_arrays(arrays, "regret", algorithm, repeat)
        selection = _series_from_arrays(arrays, "selection_count", algorithm, repeat)
        optimization = _series_from_arrays(arrays, "optimization_count", algorithm, repeat)

        if regret is not None:
            regrets[key].append(np.asarray(regret, dtype=float))
        if selection is not None:
            selection_counts[key].append(np.asarray(selection, dtype=float))
        if optimization is not None:
            optimization_counts[key].append(np.asarray(optimization, dtype=float))

    if not regrets:
        raise ValueError(f"No regret arrays found in {output_dir}")
    if not selection_counts:
        raise ValueError(f"No selection_count arrays found in {output_dir}")
    if not optimization_counts:
        raise ValueError(f"No optimization_count arrays found in {output_dir}")

    artifact_config = artifacts.get("config", {})
    k = int(artifact_config.get("K_env") or artifact_config.get("K") or 0)
    if k <= 0:
        k = max((len(series[0]) for series in selection_counts.values() if series), default=1)

    return plot_classical_bandit_series(
        regrets,
        selection_counts,
        optimization_counts,
        output_dir=output_dir,
        k=k,
        plot_config=config,
    )


def save_classical_bandit_artifact_plot(
    output_dir: str | Path,
    filename: str = "regenerated_bandit_summary.pdf",
    plot_config_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    config = load_plot_config("classical_bandits", plot_config_path)
    fig = plot_classical_bandit_artifacts(output_dir, plot_config_path=plot_config_path)
    return save_figure(fig, output_dir / filename, output_dir, dpi=int(config.get("save_dpi", 300)))
