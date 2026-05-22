"""Classical bandit experiment runner.

This module is intentionally import-safe: experiments run only through ``main``.
The implementation is still close to the original notebook-style script; deeper
storage/runner refactoring is planned in later phases.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Type

try:
    import fire
except ImportError:  # pragma: no cover - argparse fallback is for minimal environments.
    fire = None
import numpy as np
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
from stat_online.classical_bandits.runners import classical_results_to_artifacts
from stat_online.experiments.lifecycle import save_experiment_bundle
from stat_online.experiments.plotting import (
    plot_classical_bandit_results,
    save_classical_bandit_artifact_plot,
    save_figure,
)
from stat_online.experiments.runner import repeat_seeds, run_tasks, seed_numpy


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


def run_single_exp(alg: BaseModelSelection, env, T: int, indices, seed: int | None = None):
    if seed is not None:
        seed_numpy(seed)
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
    seed: int = 42,
):
    seed_numpy(seed)
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
    repeat_seed_values = repeat_seeds(seed, num_repeats)
    tasks = [
        (deepcopy(alg), i_group, j_alg, repeat_idx, repeat_seed_values[repeat_idx])
        for i_group, group in enumerate(algos_list)
        for j_alg, alg in enumerate(group)
        for repeat_idx in range(num_repeats)
    ]

    def worker(task):
        alg, _ig, ja, _repeat_idx, repeat_seed = task
        return run_single_exp(alg, env_list, T, (getname(alg), ja + 1), seed=repeat_seed)

    raw_results = run_tasks(tqdm(tasks, desc="Parallel Run"), worker, n_jobs=n_jobs)

    results_dict = defaultdict(list)
    for indices, exp in raw_results:
        results_dict[indices].append(exp)
    return dict(results_dict), repeat_seed_values


def get_fig_set_style(lines_count, shape=(1, 1), figsize=None, params=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

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


def main(
    T: int = 25_000,
    K: int = 10,
    K_env: int = 2,
    num_repeats: int = 101,
    n_jobs: int = -1,
    c_scaler: float = 0.5,
    output_path: str = "",
    output_dir: str = "./exp_results/classical_bandits",
    smoke: bool = False,
    preset: str = "",
    regenerate_plot_from: str = "",
    plot_config: str = "",
    include_smooth_corral: bool = False,
    skip_smooth_corral: bool = False,
    seed: int = 42,
):
    if regenerate_plot_from:
        path = save_classical_bandit_artifact_plot(regenerate_plot_from, plot_config_path=plot_config or None)
        print(f"Regenerated plot saved to {path}")
        return str(path)

    if preset and preset != "smoke":
        raise ValueError(f"Unknown preset: {preset}")
    smoke_mode = smoke or preset == "smoke"
    requested_include_smooth_corral = include_smooth_corral
    include_smooth_corral = (not smoke_mode) and requested_include_smooth_corral and not skip_smooth_corral
    if smoke_mode:
        T = 50
        num_repeats = 1
        n_jobs = 1
        K = min(K, 4)
    if not output_path:
        output_path = str(Path(output_dir) / "bandit_experiment.pdf")

    temp_map, repeat_seed_values = run_batch(
        T=T,
        K=K,
        K_env=K_env,
        num_repeats=num_repeats,
        n_jobs=n_jobs,
        c_scaler=c_scaler,
        m_values=tuple(range(1, min(4, K) + 1)),
        include_smooth_corral=include_smooth_corral,
        seed=seed,
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
        "include_smooth_corral": include_smooth_corral,
        "requested_include_smooth_corral": requested_include_smooth_corral,
        "skip_smooth_corral": skip_smooth_corral,
        "plot_config": plot_config,
        "seed": seed,
        "repeat_seeds": repeat_seed_values,
    }
    run_records, arrays = classical_results_to_artifacts(
        temp_map,
        run_id="classical_bandits",
        config=config,
    )
    save_experiment_bundle(
        output_dir,
        "classical_bandits",
        config,
        run_records,
        arrays,
        run_id="classical_bandits",
    )
    fig = plot_classical_bandit_results(
        temp_map,
        output_dir=output_dir,
        k=K_env,
        plot_config_path=plot_config or None,
    )
    save_figure(fig, output_path, output_dir)
    save_classical_bandit_artifact_plot(output_dir, plot_config_path=plot_config or None)
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
        parser.add_argument("--output_path", default="")
        parser.add_argument("--output_dir", default="./exp_results/classical_bandits")
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--preset", default="")
        parser.add_argument("--regenerate_plot_from", default="")
        parser.add_argument("--plot_config", default="")
        parser.add_argument("--include_smooth_corral", action="store_true")
        parser.add_argument("--skip_smooth_corral", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        main(**vars(parser.parse_args()))
    else:
        fire.Fire(main)
