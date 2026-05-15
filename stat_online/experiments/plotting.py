"""Plotting helpers that consume saved primitive artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from stat_online.experiments.storage import load_experiment_artifacts, sanitize_key


def _series_key(metric: str, algorithm: str, repeat: str | int) -> str:
    repeat_id = str(repeat)
    if not repeat_id.startswith("repeat_"):
        repeat_id = f"repeat_{repeat_id}"
    return sanitize_key(f"{metric}/{algorithm}/{repeat_id}")


def plot_glm_artifacts(output_dir: str | Path):
    """Plot cumulative GLM loss using only saved CSV/NPZ artifacts."""
    artifacts = load_experiment_artifacts(output_dir)
    rows: Sequence[Mapping[str, Any]] = artifacts["runs"]
    arrays = artifacts["arrays"]

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = _series_key("cum_loss", row["algorithm"], row["repeat"])
        if key in arrays:
            grouped[row["algorithm"]].append(np.asarray(arrays[key], dtype=float))

    if not grouped:
        raise ValueError(f"No cumulative loss arrays found in {output_dir}")

    fig, ax = plt.subplots(figsize=(12, 6))
    for algorithm, series_list in grouped.items():
        min_len = min(len(series) for series in series_list)
        stacked = np.vstack([series[:min_len] for series in series_list])
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        x = np.arange(min_len)
        ax.plot(x, mean, label=algorithm)
        ax.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, alpha=0.2)

    ax.set_title("Cumulative Loss vs Steps")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def save_glm_artifact_plot(output_dir: str | Path, filename: str = "regenerated_cum_loss.pdf") -> Path:
    output_dir = Path(output_dir)
    fig = plot_glm_artifacts(output_dir)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_classical_bandit_artifacts(output_dir: str | Path):
    """Plot classical-bandit regret/reward summaries from saved artifacts."""
    artifacts = load_experiment_artifacts(output_dir)
    rows: Sequence[Mapping[str, Any]] = artifacts["runs"]
    arrays = artifacts["arrays"]

    regrets: dict[str, list[np.ndarray]] = defaultdict(list)
    rewards: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_experts: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_arms: dict[str, list[np.ndarray]] = defaultdict(list)

    for row in rows:
        algorithm = row["algorithm"]
        repeat = row["repeat"]
        for target, metric in [
            (regrets, "regret"),
            (rewards, "reward"),
            (selected_experts, "selected_expert"),
            (selected_arms, "selected_arm"),
        ]:
            key = _series_key(metric, algorithm, repeat)
            if key in arrays:
                target[algorithm].append(np.asarray(arrays[key]))

    if not regrets:
        raise ValueError(f"No regret arrays found in {output_dir}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    ax_regret, ax_reward, ax_selection = axes

    for algorithm, series_list in regrets.items():
        min_len = min(len(series) for series in series_list)
        stacked = np.vstack([series[:min_len] for series in series_list])
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        x = np.arange(min_len)
        ax_regret.plot(x, mean, label=algorithm)
        ax_regret.fill_between(x, mean - 0.5 * std, mean + 0.5 * std, alpha=0.2)

    for algorithm, series_list in rewards.items():
        min_len = min(len(series) for series in series_list)
        stacked = np.vstack([np.cumsum(series[:min_len]) for series in series_list])
        mean = np.mean(stacked, axis=0)
        ax_reward.plot(np.arange(min_len), mean, label=algorithm)

    selection_counts: dict[str, np.ndarray] = {}
    for algorithm, series_list in selected_experts.items():
        valid_values = [series[series >= 0].astype(int) for series in series_list]
        if not valid_values or not any(len(values) for values in valid_values):
            valid_values = [series.astype(int) for series in selected_arms.get(algorithm, [])]
        max_idx = max((int(values.max()) for values in valid_values if len(values)), default=-1)
        if max_idx >= 0:
            counts = np.vstack([np.bincount(values, minlength=max_idx + 1) for values in valid_values if len(values)])
            selection_counts[algorithm] = np.mean(counts, axis=0)

    if selection_counts:
        width = 0.8 / max(1, len(selection_counts))
        for idx, (algorithm, counts) in enumerate(selection_counts.items()):
            x = np.arange(len(counts)) + idx * width
            ax_selection.bar(x, counts, width=width, label=algorithm)

    ax_regret.set_title("Expected Regret vs Steps")
    ax_regret.set_xlabel("Step")
    ax_regret.set_ylabel("Expected Regret")
    ax_regret.grid(True)
    ax_regret.legend()

    ax_reward.set_title("Cumulative Reward vs Steps")
    ax_reward.set_xlabel("Step")
    ax_reward.set_ylabel("Cumulative Reward")
    ax_reward.grid(True)
    ax_reward.legend()

    ax_selection.set_title("Selection Distribution")
    ax_selection.set_xlabel("Expert/Arm Index")
    ax_selection.set_ylabel("Selection Count")
    ax_selection.grid(True)
    ax_selection.legend()

    fig.tight_layout()
    return fig


def save_classical_bandit_artifact_plot(
    output_dir: str | Path,
    filename: str = "regenerated_bandit_summary.pdf",
) -> Path:
    output_dir = Path(output_dir)
    fig = plot_classical_bandit_artifacts(output_dir)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
