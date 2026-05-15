"""Classical bandit experiment runner.

This module is intentionally import-safe: experiments run only through ``main``.
The implementation is still close to the original notebook-style script; deeper
storage/runner refactoring is planned in later phases.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Type

try:
    import fire
except ImportError:  # pragma: no cover - argparse fallback is for minimal environments.
    fire = None
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from tqdm import tqdm

from stat_online.classical_bandits.algorithm import (
    BaseModelSelection,
    EpsilonGreedy,
    LimitedAdvice,
    MLCB,
    SmoothCORRAL,
    UCB,
)
from stat_online.classical_bandits.environment import BernoulliBanditEnvironment
from stat_online.classical_bandits.experiment import ListExperiment
from stat_online.core.records import RunRecord
from stat_online.experiments.plotting import save_classical_bandit_artifact_plot
from stat_online.experiments.runner import run_tasks
from stat_online.experiments.storage import write_experiment_artifacts


COLORMAP_NAME = "tab20"
DPI = 400
FIGSIZE = (17, 8)
FONTSIZE = 20


def make_orcestra_algorithm(
    epsilon_list: list[float],
    M: int,
    K: int,
    c_scaler: float,
    alg_class: Type[BaseModelSelection],
    **kwargs,
) -> BaseModelSelection:
    n_exerts = len(epsilon_list)
    n_greedies = n_exerts // 2
    base_algorithms = [
        EpsilonGreedy(epsion=epsilon, K=K, c=c_scaler)
        for epsilon in epsilon_list[:n_greedies]
    ] + [UCB(K=K, c=c_scaler) for _ in epsilon_list[n_greedies:]]

    return alg_class(bandit_algorithms=base_algorithms, M=M, **kwargs)


def build_environments(K: int, K_env: int, delta: float) -> list[BernoulliBanditEnvironment]:
    base_rew = 2 * delta
    deltas = [
        np.linspace(
            base_rew - 2 * delta * k / K,
            base_rew + delta - delta * k / K,
            K_env,
        )
        for k in range(K)
    ]
    return [BernoulliBanditEnvironment(K=K_env, probs=p) for p in deltas]


def get_algorithms_list(
    epsilon_list: np.ndarray,
    K_env: int,
    T: int,
    c_scaler: float,
    m_values: tuple[int, ...],
    include_smooth_corral: bool = True,
) -> list[list[BaseModelSelection]]:
    algorithms = [
        [
            make_orcestra_algorithm(
                epsilon_list,
                m_i,
                K_env,
                c_scaler=c_scaler,
                alg_class=MLCB,
                c=0.5,
                delta=0.1,
            )
            for m_i in m_values
        ],
        [
            make_orcestra_algorithm(
                epsilon_list,
                m_i,
                K_env,
                c_scaler=c_scaler,
                alg_class=LimitedAdvice,
                eta_scaler=1,
            )
            for m_i in m_values
        ],
    ]
    if include_smooth_corral:
        algorithms.append([
            make_orcestra_algorithm(
                epsilon_list,
                1,
                K_env,
                c_scaler=c_scaler,
                alg_class=SmoothCORRAL,
                eta=(len(epsilon_list) / T) ** 0.5,
                T=T,
            )
        ])
    return algorithms


def getname(it) -> str:
    if hasattr(it, "name"):
        return it.name
    return type(it).__name__


def run_single_exp(alg: BaseModelSelection, env, T: int, indices):
    exp = ListExperiment(env, algorithm=alg)
    exp.run(n_steps=T)
    return indices, exp


def run_batch(
    T: int,
    K: int,
    K_env: int,
    num_repeats: int,
    n_jobs: int,
    c_scaler: float,
    m_values: tuple[int, ...],
    include_smooth_corral: bool = True,
):
    env_list = build_environments(K=K, K_env=K_env, delta=1 / 10)
    epsilon_list = np.full((K,), 1)
    algos_list = get_algorithms_list(
        epsilon_list=epsilon_list,
        K_env=K_env,
        T=T,
        c_scaler=c_scaler,
        m_values=m_values,
        include_smooth_corral=include_smooth_corral,
    )
    tasks = [
        (deepcopy(alg), i_group, j_alg)
        for i_group, group in enumerate(algos_list)
        for j_alg, alg in enumerate(group)
        for _ in range(num_repeats)
    ]

    def worker(task):
        alg, _ig, ja = task
        return run_single_exp(alg, env_list, T, (getname(alg), ja + 1))

    raw_results = run_tasks(tqdm(tasks, desc="Parallel Run"), worker, n_jobs=n_jobs)

    results_dict = defaultdict(list)
    for indices, exp in raw_results:
        results_dict[indices].append(exp)
    return dict(results_dict)


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
        fig, ax = plt.subplots(*shape, dpi=DPI, figsize=figsize)
    plt.grid(which="both")
    return fig, ax


def plot_bar(ax, temp_map, field_name, unique_groups, K):
    colors = ["blue", "red", "black", "r", "black", "blue", "green", "y", "m", "y", "k"]
    max_algs = max(k[1] for k in temp_map.keys()) + 1
    coeffs = np.logspace(0, -0.7, max_algs)

    pos = 0
    width = 0.8 / len(temp_map)
    for g_idx, g_name in enumerate(unique_groups):
        group_keys = sorted([k for k in temp_map.keys() if k[0] == g_name])
        for alg_idx, key in enumerate(group_keys):
            values_t = temp_map[key]
            if g_name == "SmoothCORRAL":
                a = [getattr(exp.algorithm, "selection_for_decisions") for exp in values_t]
            else:
                a = [getattr(exp.algorithm, field_name) for exp in values_t]
            data = np.mean(np.array(a), axis=0)
            x = np.arange(K) + pos * width
            pos += 1
            ax.bar(
                x,
                data,
                width=width,
                color=colors[g_idx % len(colors)],
                alpha=coeffs[alg_idx % len(coeffs)],
                label=f"{key[0]}, M={key[1]}",
            )
    return ax


def plot_regret(ax, temp_map, unique_groups, std_multiplier=1):
    colors = ["blue", "red", "black", "r", "black", "blue", "green", "y", "m", "y", "k"]
    markers = ["o", "s", "^", "v", "D", "p", "*", "h"]
    linestyles = [":", "--", "-.", "-"]
    max_algs = max(k[1] for k in temp_map.keys()) + 1
    coeffs = np.logspace(0, -0.7, max_algs)

    for g_idx, g_name in enumerate(unique_groups):
        group_keys = sorted([k for k in temp_map.keys() if k[0] == g_name])
        for alg_idx, key in enumerate(group_keys):
            regrets = np.stack([exp.get_expected_regret() for exp in temp_map[key]])
            mean_vals = np.mean(regrets, axis=0)
            std_vals = np.std(regrets, axis=0) * std_multiplier
            x = np.arange(len(mean_vals))
            coeff = coeffs[alg_idx % len(coeffs)]
            ax.plot(
                x,
                mean_vals,
                color=colors[g_idx % len(colors)],
                alpha=coeff,
                lw=2,
                label=f"{key[0]}, M={key[1]}",
                linestyle=linestyles[alg_idx % len(linestyles)],
                marker=markers[g_idx % len(markers)],
                markevery=max(1, len(mean_vals) // 20),
                markersize=5,
            )
            ax.fill_between(
                x,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=colors[g_idx % len(colors)],
                alpha=0.2 * coeff,
            )


def plot_results(temp_map, K):
    import matplotlib.patches as mpatches

    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    fig, (ax1, ax2, ax3) = get_fig_set_style(1, (3, 1), (10, 19))
    unique_groups = [name for name in ["M-LCB", "LimitedAdvice", "SmoothCORRAL"] if any(k[0] == name for k in temp_map)]
    plot_regret(ax1, temp_map, unique_groups)
    plot_bar(ax2, temp_map, "selection_for_decisions", unique_groups, K)
    plot_bar(ax3, temp_map, "counts", unique_groups, K)

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

    handles, labels = ax1.get_legend_handles_labels()
    rows = 4
    cols = 3
    total_slots = rows * cols
    empty_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    handles = list(handles)
    labels = list(labels)
    while len(handles) < total_slots:
        handles.append(empty_handle)
        labels.append("")

    fig.legend(
        handles,
        labels,
        ncol=cols,
        bbox_to_anchor=(0.0, -0.02, 1, 0.1),
        loc="outside upper left",
        mode="expand",
        borderaxespad=0.0,
    )
    return fig


def save_plots(fig, filename="experiment_results.pdf"):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_bandit_artifacts(output_dir: str | Path, temp_map, config: dict[str, Any]) -> Path:
    run_id = "classical_bandits"
    run_records: list[RunRecord] = []
    arrays: dict[str, Any] = {}
    T = int(config["T"])
    K = int(config["K"])
    for (group_name, m_value), runs in temp_map.items():
        algorithm = f"{group_name}_M{m_value}"
        for repeat_idx, exp in enumerate(runs):
            rewards = np.asarray(exp.reward_history, dtype=float)
            regret = np.asarray(exp.get_expected_regret(), dtype=float)
            selected_expert = []
            selected_arm = []
            for item in exp.arm_selection_history:
                if isinstance(item, tuple):
                    selected_expert.append(item[0])
                    selected_arm.append(item[1])
                else:
                    selected_expert.append(-1)
                    selected_arm.append(item)
            key_prefix = f"{algorithm}/repeat_{repeat_idx}"
            arrays[f"reward/{key_prefix}"] = rewards
            arrays[f"regret/{key_prefix}"] = regret
            arrays[f"selected_expert/{key_prefix}"] = np.asarray(selected_expert, dtype=int)
            arrays[f"selected_arm/{key_prefix}"] = np.asarray(selected_arm, dtype=int)
            run_records.append(RunRecord(
                run_id=run_id,
                experiment_name="classical_bandits",
                repeat=repeat_idx,
                algorithm=algorithm,
                group=group_name,
                M=int(m_value),
                K=K,
                T=T,
                final_regret=float(regret[-1]) if regret.size else 0.0,
                total_reward=float(np.sum(rewards)) if rewards.size else 0.0,
            ))
    return write_experiment_artifacts(
        output_dir,
        "classical_bandits",
        config,
        run_records,
        arrays,
        run_id=run_id,
    )


def main(
    T: int = 25_000,
    K: int = 10,
    K_env: int = 2,
    num_repeats: int = 101,
    n_jobs: int = -1,
    c_scaler: float = 0.5,
    output_path: str = "bandit_experiment.pdf",
    output_dir: str = "./exp_results/classical_bandits",
    smoke: bool = False,
    preset: str = "",
    regenerate_plot_from: str = "",
):
    if regenerate_plot_from:
        path = save_classical_bandit_artifact_plot(regenerate_plot_from)
        print(f"Regenerated plot saved to {path}")
        return str(path)

    if preset and preset != "smoke":
        raise ValueError(f"Unknown preset: {preset}")
    smoke_mode = smoke or preset == "smoke"
    if smoke_mode:
        T = 50
        num_repeats = 1
        n_jobs = 1
        K = min(K, 4)

    temp_map = run_batch(
        T=T,
        K=K,
        K_env=K_env,
        num_repeats=num_repeats,
        n_jobs=n_jobs,
        c_scaler=c_scaler,
        m_values=tuple(range(1, min(4, K) + 1)),
        include_smooth_corral=not smoke_mode,
    )
    config = {
        "T": T,
        "K": K,
        "K_env": K_env,
        "num_repeats": num_repeats,
        "n_jobs": n_jobs,
        "c_scaler": c_scaler,
        "output_path": output_path,
        "output_dir": output_dir,
        "smoke": smoke,
        "preset": preset,
        "include_smooth_corral": not smoke_mode,
    }
    write_bandit_artifacts(output_dir, temp_map, config)
    fig = plot_results(temp_map, K=K)
    save_plots(fig, output_path)
    return output_path


if __name__ == "__main__":
    if fire is None:
        import argparse

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--T", type=int, default=25_000)
        parser.add_argument("--K", type=int, default=10)
        parser.add_argument("--K_env", type=int, default=2)
        parser.add_argument("--num_repeats", type=int, default=101)
        parser.add_argument("--n_jobs", type=int, default=-1)
        parser.add_argument("--c_scaler", type=float, default=0.5)
        parser.add_argument("--output_path", default="bandit_experiment.pdf")
        parser.add_argument("--output_dir", default="./exp_results/classical_bandits")
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--preset", default="")
        parser.add_argument("--regenerate_plot_from", default="")
        main(**vars(parser.parse_args()))
    else:
        fire.Fire(main)
