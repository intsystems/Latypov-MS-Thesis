"""Module for visualizing bandit algorithm experiment results."""
from typing import TYPE_CHECKING, Dict, List, Type


import numpy as np
from dataclasses import dataclass
import scipy.stats as st

from stat_online.lin_bandits.ucb_algorithm import Strategy

if TYPE_CHECKING:
    from matplotlib.figure import Figure



@dataclass
class AlgRes:
    algname: str
    strategy_class: Type[Strategy]
    num_optimized_arms: int
    n_arms: int
    max_steps: int
    runtime: float
    arm_selection_hist: list[int]
    alg_loss_hist: list[float]
    learned_algorithm: Strategy

def calculate_run_statistics(
    runs: List[AlgRes]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Calculate mean and confidence intervals for multiple runs.
    
    Args:
        runs: List of AlgRes from multiple runs of same algorithm
        
    Returns:
        Dictionary with numpy arrays for each statistic:
        - 'mean_loss': mean loss per step
        - 'loss_ci_lower': lower bound of confidence interval
        - 'loss_ci_upper': upper bound of confidence interval
        - 'mean_cum_loss': mean cumulative loss
        - 'mean_arm_selection': mean arm selection counts
        - 'mean_runtime': mean runtime (float)
    """




    # Stack all loss histories (assuming same length)
    loss_histories = np.array([run.alg_loss_hist for run in runs])
    mean_loss = np.mean(loss_histories, axis=0)
    # loss_ci = st.t.interval(0.95, len(runs)-1, loc=mean_loss, scale=st.sem(loss_histories, axis=0))
    
    # Calculate cumulative loss stats
    cum_loss = np.cumsum(loss_histories, axis=1)
    mean_cum_loss = np.mean(cum_loss, axis=0)
    # calculate confidence interval as disperce

    loss_ci_std = 0.5 * np.std(cum_loss - mean_cum_loss, axis=0) #/ np.sqrt(len(runs))
    loss_ci_cum = [loss_ci_std, loss_ci_std]
    # loss_ci_cum = [0, 0]
    # loss_ci_cum = st.t.interval(0.1, len(runs)-1, loc=mean_cum_loss, scale=st.sem(cum_loss, axis=0))
    
    # Calculate arm selection stats
    max_selected_arm = max((max(run.arm_selection_hist) for run in runs if run.arm_selection_hist), default=-1)
    n_arms = max(runs[0].n_arms, max_selected_arm + 1)
    if hasattr(runs[0].learned_algorithm, 'arms'):
        n_arms = max(n_arms, len(runs[0].learned_algorithm.arms))
    arm_selections = np.array([np.bincount(run.arm_selection_hist, minlength=n_arms)[:n_arms] for run in runs])
    mean_arm_selection = np.mean(arm_selections, axis=0)
    
    # Calculate mean runtime
    mean_runtime = np.mean([run.runtime for run in runs])
    
    return {
        'mean_loss': mean_loss,
        'loss_ci_lower': loss_ci_cum[0],
        'loss_ci_upper': loss_ci_cum[1],
        'mean_cum_loss': mean_cum_loss,
        'mean_arm_selection': mean_arm_selection,
        'mean_runtime': mean_runtime
    }



def get_color(strategy_colors, alg_results, algname,):
    import matplotlib.colors as mcolors

    color = strategy_colors.get(alg_results[algname][0].strategy_class.__name__)
    context_type = "_".join(algname.split("_")[2:])
    context_types = ["no", "context", "dummy_context"]
    is_context = False
    dark_factor = 1
    if context_type in context_types:
        if context_type == "context":
            dark_factor = 0.3  # Чем меньше, тем темнее
            is_context = True
        elif context_type == "dummy_context":
            dark_factor = 0.03  # Чем меньше, тем темнее
            is_context = True
    if is_context:
        # make color more dark
        rgb = mcolors.to_rgb(color)
        dark_rgb = np.array(rgb) * dark_factor

        # Преобразуем обратно в HEX или оставляем как кортеж
        color = mcolors.to_hex(dark_rgb)  # HEX-формат
    return color 

def plot_experiment_results(
        alg_results: Dict[str, List[AlgRes]],
        alg_classes: list[type[Strategy]]
    ) -> "Figure":
    import matplotlib.pyplot as plt


    """Plot experiment results in three different views for multiple runs.
    
    Args:
        alg_results: Dictionary of algorithm results (algname: list[AlgRes])
    """
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 24))

    # Color mapping for strategy classes
    strategy_colors = {
        0: 'blue',
        4: 'orange',
        3: 'purple',
        2: 'black',
        1: 'green',
        5: 'red',
        6: "darkblue",
        7: 'hotpink'
    }
    strategy_color2class = {cl.__name__: i for i, cl in enumerate(alg_classes)}
    strategy_colors = {cl_name: strategy_colors[val] for cl_name, val in strategy_color2class.items()}
    
    # Calculate statistics for each algorithm
    alg_stats = {algname: calculate_run_statistics(runs) for algname, runs in alg_results.items()}
    # print(alg_stats)



    # Plot 1: Cumulative loss vs steps
    for algname, stats in alg_stats.items():
        color = get_color(strategy_colors, alg_results, algname,)

        alpha = min(1.0, 0.2 + 0.8 * ((alg_results[algname][0].num_optimized_arms + 1) / alg_results[algname][0].n_arms))
        
        ax1.plot(
            range(len(stats['mean_cum_loss'])),
            stats['mean_cum_loss'],
            label=f"{algname}",
            color=color,
            alpha=alpha
        )
        
        # Add confidence interval for loss
        lower = stats['mean_cum_loss'] - stats['loss_ci_lower']
        upper = stats['mean_cum_loss'] + stats['loss_ci_upper']


        ax1.fill_between(
            range(len(stats['mean_cum_loss'])),
            lower,
            upper,
            color=color,
            alpha=0.2
        )
    
    ax1.set_title("Cumulative Loss vs Steps")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Cumulative loss vs normalized time
    for algname, stats in alg_stats.items():
        color = get_color(strategy_colors, alg_results, algname,)


        alpha = min(
            1.0,
            0.2 + 0.8 * ((alg_results[algname][0].num_optimized_arms + 1) / alg_results[algname][0].n_arms),
        )


        
        runtime = float(stats['mean_runtime'])
        time_steps = np.linspace(0, runtime, len(stats['mean_cum_loss']))


        ax2.plot(
            time_steps,
            stats['mean_cum_loss'],
            label=f"{algname}",
            color=color,
            alpha=alpha
        )
        
        # Add confidence interval
        lower = stats['mean_cum_loss'] - stats['loss_ci_lower']
        upper = stats['mean_cum_loss'] + stats['loss_ci_upper']


        ax2.fill_between(
            time_steps,
            lower,
            upper,
            color=color,
            alpha=0.2
        )
    
    ax2.set_title("Cumulative Loss vs Runtime")
    ax2.set_xlabel("Runtime (seconds)")
    ax2.set_ylabel("Cumulative Loss")
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Arm selection histogram
    n_arms = max(run.n_arms for runs in alg_results.values() for run in runs)
    width = 0.8 / len(alg_results)
    
    for i, (algname, stats) in enumerate(alg_stats.items()):
        color = get_color(strategy_colors, alg_results, algname,)

        alpha = min(1.0, 0.2 + 0.8 * ((alg_results[algname][0].num_optimized_arms + 1) / alg_results[algname][0].n_arms))
        
        x = np.arange(n_arms) + i * width
        
        ax3.bar(
            x,
            stats['mean_arm_selection'],
            width=width,
            label=f"{algname}",
            color=color,
            alpha=alpha
        )
    
    ax3.set_title("Arm Selection Distribution")
    ax3.set_xlabel("Arm Index")
    ax3.set_ylabel("Selection Count")
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Arm optimization histogram
    for i, (algname, runs) in enumerate(alg_results.items()):
        color = get_color(strategy_colors, alg_results, algname,)

        alpha = min(
            1.0,
            0.2 + 0.9 * ((runs[0].num_optimized_arms + 1) / runs[0].n_arms),
        )



        # Calculate mean optimization counts across runs
        opt_counts = []
        for run in runs:
            counts = [arm.optimized_count for arm in run.learned_algorithm.arms]
            if len(counts) < n_arms:
                counts = counts + [0] * (n_arms - len(counts))
            else:
                counts = counts[:n_arms]
            opt_counts.append(counts)
        
        # print(opt_counts)
        if len(opt_counts) > 1:
            mean_counts = np.mean(opt_counts, axis=0)
        else:
            mean_counts = opt_counts[0]
        x = np.arange(n_arms) + i * width
        # print(x, mean_counts)
        ax4.bar(
            x,
            mean_counts,
            width=width,
            label=f"{algname}",
            color=color,
            alpha=alpha
        )

    ax4.set_title("Arm Optimization Distribution")
    ax4.set_xlabel("Arm Index")
    ax4.set_ylabel("Optimization Count")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig


def save_plots(fig, filename="experiment_results.png"):
    """Save plots to file.
    
    Args:
        fig: matplotlib figure object
        filename: output filename
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
