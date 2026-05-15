"""
Experiment 1: Bandit Algorithms Comparison for Generalized Linear Functions

This experiment compares various bandit algorithms in a realizability assumption setting
with generalized linear functions. The experiment evaluates algorithms on synthetic data
generated using different nonlinearities.

Key Features:
- Generates synthetic data with various nonlinear transformations
- Compares multiple bandit strategies (UCB, LimitedAdvice, etc.)
- Supports different context types and optimization parameters
- Parallel execution for efficient experimentation
- Results visualization and saving capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from time import time
from itertools import chain, product
try:
    import fire
except ImportError:  # pragma: no cover - argparse fallback is for minimal environments.
    fire = None
from typing import Callable, List, Dict, Any, Optional, Union

from stat_online.lin_bandits.convex_functions_jax import get_functions
from stat_online.lin_bandits.contextual_mab import (
    OnlineGradArmOptimizer,
    GeneralizedLinearFunction
    )
from stat_online.lin_bandits.loss_functions import square_loss
from stat_online.lin_bandits.ucb_algorithm import (
    Strategy,
    UCBAlgorithm,
    EpsilonGreedyAlgorithm,
    Exp3Algorithm,
    GroupedUCB,
    Arm,
    ContextualArm,
    dummyContextualArm
)
from stat_online.core.records import RunRecord
from stat_online.experiments.plotting import save_glm_artifact_plot
from stat_online.experiments.runner import run_repeats
from stat_online.experiments.storage import write_experiment_artifacts
from stat_online.utils.visualization import AlgRes, plot_experiment_results, save_plots


def generate_data(T: int,
                  d: int,
                  context_d: int,
                  nonlinearity: Callable,
                  random_state: Optional[int] = None) -> tuple:
    """
    Generate synthetic data for bandit experiments.

    Args:
        T: Number of samples
        d: Dimension of feature vectors
        context_d: ground truth dimention of context vector.
        nonlinearity: Nonlinear transformation function
        random_state: Random seed for reproducibility
    Returns:
        tuple: (X, y, linears) where:
            - X: Feature matrix (T x d)
            - y: Target values (T,)
            - linears: Linear transformations before nonlinearity
    """
    rng = np.random.default_rng(random_state)

    # Generate X uniformly from a d-dimensional ball of radius 1
    X = rng.normal(size=(T, d))
    X /= np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    X = X * 2  # Scale to radius 2

    # Linear transformation with random weights
    W = rng.standard_cauchy(size=(context_d,))
    W /= np.linalg.norm(W)  # Normalize W to unit norm
    W = W * rng.uniform(0.7, 0.95)  # Scale W to have random magnitude
    linear = X[:, :context_d] @ W

    # Apply nonlinearity with noise
    y = nonlinearity(linear) + rng.normal(loc=0, scale=0.02, size=linear.shape)

    return X, y, linear


def get_arms_instance(K: int, d: int, nonlinearities: List[callable],
                     lr_scaler: float = 1.0, D: float = 0.1, G: float = 0.1,
                     is_context: str = "no") -> List[Arm]:
    """
    Create arms with generalized linear functions.

    Args:
        K: Number of arms
        d: Dimension of feature vectors
        nonlinearities: List of nonlinear functions for each arm
        lr_scaler: Learning rate scaling factor
        D: Diameter parameter for optimization
        G: Gradient bound parameter
        is_context: Context type ("no", "context", "dummy_context")

    Returns:
        List[Arm]: List of initialized arms
    """
    functions = [GeneralizedLinearFunction(n_params=d, nonlinearity=nonlinearities[i]) for i in range(K)]

    optimizers = [OnlineGradArmOptimizer(func, lr_scaler, square_loss, D, G, reg_term=1e-4) for func in functions]

    arms = [Arm(optimizer, delta=0.01) for optimizer in optimizers]
    return arms

def get_arms_instance_featured(
        K: int,
        d: int,
        d_base: int,
        nonlinearities: List[Callable],
        lr_scaler: float = 1.0,
        D: float = 0.1,
        G: float = 0.1,
        is_context: str = "no"
        ) -> List[Arm]:
    """
    Create arms with generalized linear functions.
    But for each nonlinearity create instances as a power of d_base: d_base, d_base^2 ... d^base^k < d

    Args:
        K: Number of arms
        d: Dimension of feature vectors
        d_base: base for power of dimentions in feature selection experiment
        nonlinearities: List of nonlinear functions for each arm
        lr_scaler: Learning rate scaling factor
        D: Diameter parameter for optimization
        G: Gradient bound parameter
        is_context: Context type ("no", "context", "dummy_context")

    Returns:
        List[Arm]: List of initialized arms
    """
    dimension = d_base
    arms = []
    while dimension < d:
        functions = [GeneralizedLinearFunction(n_params=dimension, nonlinearity=nonlinearities[i]) for i in range(K)]

        optimizers = [OnlineGradArmOptimizer(func, lr_scaler, square_loss, D, G, reg_term=1e-4) for func in functions]

        arms.append([Arm(optimizer, delta=0.01) for optimizer in optimizers])
        dimension *= d_base

    arms = list(chain.from_iterable(arms))
    return arms

def run_experiment(
        X: np.ndarray,
        y: np.ndarray,
        T: int,
        K: int,
        num_optimize: int,
        strategy_class: type, nonlinearities: List[Callable],
        strategy_params: dict,
        optimizer_params:dict = {},
        is_context: str = "no",
        featured: bool = False,
        d_base: int = 2,
        ) -> tuple:
    """
    Run a single experiment instance.

    Args:
        X: Feature matrix
        y: Target values
        T: Number of steps to run
        K: Number of arms
        num_optimize: Number of arms to optimize at each step
        strategy_class: Bandit strategy class
        nonlinearities: List of nonlinear functions
        is_context: Context type

    Returns:
        tuple: (algorithm_instance, runtime)
    """
    if featured:
        arms = get_arms_instance_featured(K,
                                          X.shape[1],
                                          d_base=d_base,
                                          nonlinearities=nonlinearities,
                                          is_context=is_context,
                                          **optimizer_params
                                          )
    else:
        arms = get_arms_instance(K, X.shape[1],
                                 nonlinearities=nonlinearities,
                                 is_context=is_context,
                                 **optimizer_params
                                 )

    # Initialize algorithm with appropriate parameters
    algo_params = {
        'arms': arms,
        'num_optimize': num_optimize,
        'T': T,
        'delta': 0.01
    }

    # Add strategy-specific parameters

    algo: Strategy = strategy_class(**algo_params)

    # Run algorithm
    start_time = time()
    algo.run(X[:T], y[:T])
    runtime = time() - start_time

    return algo, runtime


def run_experiment_batch(
        T: int,
        K: int,
        dim: int,
        nonlinearities: List[Callable],
        generation_nonlinearity: Callable,
        num_optimize: int,
        num_repeats: int,
        is_featured: bool = False,  # Whether to run feature selection experiment
        d_base: int = 2,      # Base for feature selection dimensions
        context_d: Union[int, None] = None,  # Ground truth dimension of context vector
        n_jobs: int = 12,
        output_dir: str = "./exp_results",
        best_arm_number: int = 0,
        save_debug_pickle: bool = False,
        config: Optional[dict[str, Any]] = None,
        ) -> Dict[str, List[AlgRes]]:
    """
    Run a batch of experiments with different strategies and parameters.

    Args:
        T: Maximum number of steps
        K: Number of arms
        dim: Dimension of feature vectors
        nonlinearities: List of nonlinear functions for arms
        generation_nonlinearity: Callable which will be used to generate data
        num_optimize: Maximum number of arms to optimize per step,
        num_repeats: Number of experiment repetitions,
        is_featured: Whether to run feature selection experiment
        d_base: Base for feature selection dimensions
        context_d:  Ground truth dimension of context vector
        n_jobs: Number of parallel jobs
        output_dir: Directory to save results

    Returns:
        Dict[str, List[AlgRes]]: Dictionary of algorithm results
    """
    from joblib import Parallel, delayed

    if is_featured:
        assert (context_d is not None) and (context_d < dim)
        assert d_base < dim
    else:
        context_d = dim

    # Generate data once for consistency
    X, y, linears = generate_data(T=T,
                                  d=dim,
                                  context_d=context_d,
                                  nonlinearity=generation_nonlinearity,
                                  random_state=42)

    # Define strategies to test
    strategies = [UCBAlgorithm, GroupedUCB, EpsilonGreedyAlgorithm, Exp3Algorithm]
    strategy_parameters = [{"delta": 0.1}, {"delta": 0.1}, {"delta": 0.1}, {"delta": 0.1},]  # additional parameters of strategies

    def run_single_experiment():
        """Run a single experiment with all strategy combinations."""
        n_optims = range(1, num_optimize + 1)
        context_types = ["no"]  # Can add "context", "dummy_context" for more experiments

        # Generate all parameter combinations
        strategy_n_opt_pairs = list(product(zip(strategies, strategy_parameters), n_optims, context_types))

        # Run experiments in parallel

        run_experiment_del = delayed(run_experiment)
        alg_results_list = Parallel(n_jobs=n_jobs)(
            run_experiment_del(X, y, T, K, n_optim,
                               strategy,
                               nonlinearities,
                               strategy_params,
                               is_context=is_context,
                               featured=is_featured, d_base=d_base)
            for (strategy, strategy_params), n_optim, is_context in strategy_n_opt_pairs
        )

        # Organize results
        alg_results = {}
        for (alg_res, runtime), ((strategy, _strategy_params), n_optim, is_context) in zip(alg_results_list, strategy_n_opt_pairs):
            alg_name = f"{strategy.__name__}_{n_optim}_{is_context}"

            res = AlgRes(
                algname=alg_name,
                strategy_class=strategy,
                num_optimized_arms=n_optim,
                arm_selection_hist=alg_res.selection_history,
                alg_loss_hist=alg_res.loss_history,
                max_steps=T,
                n_arms=len(alg_res.selection_history) and (max(alg_res.selection_history) + 1) or K,
                runtime=runtime,
                learned_algorithm=alg_res
            )
            alg_results[alg_name] = res

        return alg_results

    # Run multiple repetitions
    print(f"Running {num_repeats} experiment repetitions...")
    import os
    os.makedirs(output_dir, exist_ok=True)
    res_list = run_repeats(num_repeats, lambda _repeat: run_single_experiment(), n_jobs=1)

    # Combine results across repetitions
    res_dict = {}
    for key in res_list[0].keys():
        res_dict[key] = [r[key] for r in res_list]

    run_id = f"glm_best_{best_arm_number}"
    run_records: list[RunRecord] = []
    arrays: dict[str, Any] = {}
    for alg_name, runs in res_dict.items():
        for repeat_idx, run in enumerate(runs):
            losses = np.asarray(run.alg_loss_hist, dtype=float)
            selections = np.asarray(run.arm_selection_hist, dtype=int)
            key_prefix = f"{alg_name}/repeat_{repeat_idx}"
            arrays[f"loss/{key_prefix}"] = losses
            arrays[f"cum_loss/{key_prefix}"] = np.cumsum(losses)
            arrays[f"selected_arm/{key_prefix}"] = selections
            run_records.append(RunRecord(
                run_id=run_id,
                experiment_name="glm_bandits",
                repeat=repeat_idx,
                algorithm=alg_name,
                M=run.num_optimized_arms,
                K=run.n_arms,
                T=run.max_steps,
                runtime_sec=run.runtime,
                final_loss=float(np.sum(losses)) if losses.size else 0.0,
                extra={"strategy_class": run.strategy_class.__name__},
            ))

    artifact_config = config or {
        "T": T,
        "K": K,
        "dim": dim,
        "num_optimize": num_optimize,
        "num_repeats": num_repeats,
        "is_featured": is_featured,
        "d_base": d_base,
        "context_d": context_d,
        "best_arm_number": best_arm_number,
    }
    write_experiment_artifacts(
        output_dir,
        "glm_bandits",
        artifact_config,
        run_records,
        arrays,
        run_id=run_id,
    )

    # Generate and save plots
    print("Generating plots...")
    fig = plot_experiment_results(res_dict, strategies)
    save_plots(fig, f"{output_dir}/experiment_results_{best_arm_number}.pdf")

    if save_debug_pickle:
        import pickle
        with open(f"{output_dir}/debug.pkl", 'wb') as f:
            pickle.dump(res_dict, f)

    print(f"Experiment completed. Results saved to {output_dir}/")
    return res_dict


def main(T: int = 2500,
        dim: int = 5,
        K: int = 10,
        num_optimize: int = 3,
        best_arm_number: int = 9,
        is_featured: bool = False,
        d_base=2,
        context_d=None,
        num_repeats: int = 5,
        n_jobs: int = 12,
        output_dir: str = "./exp_results",
        smoke: bool = False,
        preset: str = "",
        save_debug_pickle: bool = False,
        regenerate_plot_from: str = ""):
    """
    Main function to run the bandit algorithms comparison experiment.

    Args:
        T: Maximum number of steps (default: 2500)
        d: Dimension of feature vectors (default: 5)
        K: Number of arms (default: 10)
        num_optimize: Maximum number of arms to optimize (default: 3)
        best_arm_number: Index of the best nonlinearity function (default: 9)
        num_repeats: Number of experiment repetitions (default: 5)
        n_jobs: Number of parallel jobs (default: 12)
        output_dir: Directory to save results (default: "./exp_results")
    """
    if regenerate_plot_from:
        path = save_glm_artifact_plot(regenerate_plot_from)
        print(f"Regenerated plot saved to {path}")
        return str(path)

    if preset and preset != "smoke":
        raise ValueError(f"Unknown preset: {preset}")
    if smoke or preset == "smoke":
        T = 20
        dim = 3
        K = 4
        num_optimize = 1
        num_repeats = 1
        n_jobs = 1

    print("Starting Experiment 1: Bandit Algorithms Comparison")
    print(f"Parameters: T={T}, d={dim}, K={K}, num_optimize={num_optimize}")
    print(f"best_arm_number={best_arm_number}, num_repeats={num_repeats}")
    if is_featured and context_d is None:
        context_d = max(1, dim // 2)
        print(f"context_d was not provided; using context_d={context_d} for featured run")

    try:
        nonlinearities = get_functions(x_scale=2)
        indices = [1, 4, 7, 9]
        nonlinearities = [nonlinearities[i] for i in indices]
        generation_nonlinearity = nonlinearities[0]

        results = run_experiment_batch(
                T=T,
                dim=dim,
                K=K,
                nonlinearities=nonlinearities,
                generation_nonlinearity=generation_nonlinearity,
                num_optimize=num_optimize,
                num_repeats=num_repeats,
                is_featured=is_featured,
                d_base=d_base,
                context_d=context_d,
                n_jobs=n_jobs,
                output_dir=output_dir,
                best_arm_number=best_arm_number,
                save_debug_pickle=save_debug_pickle,
                config={
                    "T": T,
                    "dim": dim,
                    "K": K,
                    "num_optimize": num_optimize,
                    "best_arm_number": best_arm_number,
                    "is_featured": is_featured,
                    "d_base": d_base,
                    "context_d": context_d,
                    "num_repeats": num_repeats,
                    "n_jobs": n_jobs,
                    "preset": preset,
                    "smoke": smoke,
                },
        )
        print("Experiment completed successfully!")
        return results
    except Exception as e:
        print(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    if fire is None:
        import argparse

        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--T", type=int, default=2500)
        parser.add_argument("--dim", type=int, default=5)
        parser.add_argument("--K", type=int, default=10)
        parser.add_argument("--num_optimize", type=int, default=3)
        parser.add_argument("--best_arm_number", type=int, default=9)
        parser.add_argument("--is_featured", action="store_true")
        parser.add_argument("--d_base", type=int, default=2)
        parser.add_argument("--context_d", type=int, default=None)
        parser.add_argument("--num_repeats", type=int, default=5)
        parser.add_argument("--n_jobs", type=int, default=12)
        parser.add_argument("--output_dir", default="./exp_results")
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--preset", default="")
        parser.add_argument("--save_debug_pickle", action="store_true")
        parser.add_argument("--regenerate_plot_from", default="")
        main(**vars(parser.parse_args()))
    else:
        fire.Fire(main)
