import numpy as np


from stat_online.classical_bandits.experiment import (
    Experiment, ListExperiment
)

from stat_online.classical_bandits.environment import BernoulliBanditEnvironment

from stat_online.classical_bandits.algorithm import (
        BaseModelSelection,
        EpsilonGreedy,
        UCB,
        MLCB,
        LimitedAdvice,
        DDRB,
        SmoothCORRAL
    )


from typing import Type


def make_orcestra_algorithm(epsilon_list: list[float],
                        M: int,
                        K: int,
                        c_scaler: float,
                        alg_class: Type[BaseModelSelection],
                        **kwargs
                        ) -> BaseModelSelection:
    n_exerts = len(epsilon_list)
    n_greedies = n_exerts//2

    base_algorithms = [EpsilonGreedy(epsion=epsilon, K = K, c=c_scaler) for epsilon in epsilon_list[:n_greedies]] + [UCB(K =2, c = c_scaler) for _ in epsilon_list[n_greedies:]]

    return alg_class(
                bandit_algorithms= base_algorithms,
                M = M,
                **kwargs
            )

from tqdm import tqdm
from joblib import Parallel, delayed
n_repeats = 101

T = 25_000
K = 10
M = 4


st = 1
K_env = 2

deltas = [np.linspace(1/k**2, 1/k, K_env) for k in range(st, K + st)]

delta = 1/10
base_rew = 2 * delta
deltas = [ np.linspace(base_rew - 2 * delta * k/K, base_rew + delta - delta * k/K, K_env) for k in range(0, K)]


env_list = [BernoulliBanditEnvironment(K = K_env, probs=p) for p in deltas]
# env = BernoulliBanditEnvironment(K = K_env, probs=[0.45, 0.5] )
epsilon_list = np.full((K,), 1) # np.logspace(-3, 0, K)

c_scaler = 0.5
def get_algorithms_list():
    algorithms = [
        [make_orcestra_algorithm(epsilon_list, m_i, K_env,
                                 c_scaler = c_scaler,
                                 alg_class=MLCB,
                                 c = 0.5, delta = 0.1) for m_i in [1,2,3, 4]
         ],
        [make_orcestra_algorithm(epsilon_list, m_i, K_env,
                                 c_scaler = c_scaler,
                                 alg_class=LimitedAdvice,
                                 eta_scaler=1) for m_i in [1,2,3, 4]
         ],
        # [make_orcestra_algorithm(epsilon_list, m_i, K,
        #                          c_scaler = 0.5,
        #                          alg_class=DDRB,
        #                          d_min = 1.0, delta = 0.05,) for m_i in [1,2,3]
        #  ],
        [make_orcestra_algorithm(epsilon_list, 1, K_env,
                                 c_scaler = c_scaler,
                                 alg_class=SmoothCORRAL,
                                 eta = (len(epsilon_list)/ T) ** 0.5, T=T,) for m_i in [1]
         ]
        ]
    return algorithms


from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy
def run_single_exp(alg, env, T, indices):
    exp = ListExperiment(env, algorithm=alg)
    exp.run(n_steps=T)
    return indices, exp  # Возвращаем индексы вместе с результатом

# 2. Формируем плоский список задач с индексами (i_group, j_alg)
algos_list = get_algorithms_list()
tasks = [
    (deepcopy(alg), i_group, j_alg)
    for i_group, group in enumerate(algos_list)
    for j_alg, alg in enumerate(group)
    for _ in range(n_repeats)
]

def getname(it):
    if hasattr(it, "name"):
        return it.name
    return type(it).__name__
# 3. Выполняем параллельно
raw_results = Parallel(n_jobs=-1)(
    delayed(run_single_exp)(alg, env_list, T, (getname(alg), ja + 1))
    for alg, ig, ja in tqdm(tasks, desc="Parallel Run")
)

# 4. Собираем в словарь (ig, ja) -> [exp, exp, ...]
results_dict = defaultdict(list)
for (indices, exp) in raw_results:
    results_dict[indices].append(exp)

temp_map = dict(results_dict)


import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


COLORMAP_NAME = "tab20"
DPI = 400
FIGSIZE = (17, 8)
FONTSIZE = 20

def get_fig_set_style(lines_count, shape=(1, 1), figsize=None, params = None):
    # colors_list = [ "indigo", "blue", "grey", "red", "#0b5509", "pink", "coral", "black", "y", "c", "g"]
    # colors_list = [ "#a0a0a0","#303000","#406080", "#500010","#606030", "#800080", "goldenrod", "goldenrod", "goldenrod"]

    if params is None:
        params = {
            "legend.fontsize": 17,
            "lines.markersize": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 10,
            #  "text.usetex": True
        }
    sns.set_context("paper", rc=params)
    # sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})
    if figsize is None:
        fig, ax = plt.subplots(*shape, dpi=DPI)
    else:
        fig, ax = plt.subplots(*shape, dpi=DPI, figsize=figsize,)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.unicode'] = True
    plt.grid(which="both")
    return fig, ax

import matplotlib.pyplot as plt
import numpy as np


# Настройка цветов и прозрачности
std_multiplier = 1
colors = ['blue', 'red', 'black', 'r', 'black', 'blue', 'green','y', 'm', 'y', 'k']
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
linestyles = [ ':', '--', '-.','-']
# Определяем макс. количество алгоритмов в самой большой группе для шкалы прозрачности
max_algs = max(k[1] for k in temp_map.keys()) + 1
coeffs = np.logspace(0, -0.7, max_algs)

# Внутри цикла:


def plot_bar(ax, field_name, unique_groups):

    pos = 0
    width = 0.8 / len(temp_map)
    for g_idx, g_name in enumerate(unique_groups):
        # Находим все алгоритмы j, принадлежащие данной группе i
        group_keys = sorted([k for k in temp_map.keys() if k[0] == g_name])
        group_size = len(group_keys)
        print(group_size)
        for alg_idx, key in enumerate(group_keys):

            # Получаем список значений по t для конкретного алгоритма
            values_t = temp_map[key] # list(experiment)

            # a: list[dict[arm: n_selections] ]
            if g_name == "SmoothCORRAL":
                a = [getattr(exp.algorithm, "selection_for_decisions") for exp in values_t]
            else:
                a = [getattr(exp.algorithm, field_name) for exp in values_t]
            a = np.array(a)
            data = np.mean(a, axis = 0)

            print(f'{key[0]}, M={key[1]}', sum(data))
            x = np.arange(K) + pos * width
            pos += 1
            # Переводим в numpy для расчетов

            color = colors[g_idx % len(colors)]
            coeff = coeffs[alg_idx % len(coeffs)]

            # Основная линия (среднее)
            ax.bar(
                x,
                data,
                width=width,
                color=color,
                alpha=coeff,
                label=f'{key[0]}, M={key[1]}'
            )

    return ax



def plot_regret(ax, unique_groups):
    for g_idx, g_name in enumerate(unique_groups):
        # Находим все алгоритмы j, принадлежащие данной группе i
        group_keys = sorted([k for k in temp_map.keys() if k[0] == g_name])
        group_size = len(group_keys)
        print(group_size)
        for alg_idx, key in enumerate(group_keys):
            # Получаем список значений по t для конкретного алгоритма
            values_t = temp_map[key] # list(experiment)
            a = [exp.get_expected_regret() for exp in values_t]
            print(len(a), len(values_t), key)
            values_t = np.stack(a)

            # Переводим в numpy для расчетов
            data = np.array(values_t)

            # Если данных несколько, считаем среднее и стандартное отклонение
            # (Confidence Interval ~ mean ± std)
            mean_vals = np.mean(data, axis=0) if data.ndim > 1 else data
            std_vals = np.std(data, axis=0) if data.ndim > 1 else np.zeros_like(data)
            std_vals *=std_multiplier

            x = np.arange(len(mean_vals))
            color = colors[g_idx % len(colors)]
            coeff = coeffs[alg_idx % len(coeffs)]
            marker = markers[g_idx % len(colors)]
            linestyle = linestyles[alg_idx % len(coeffs)]
            # Основная линия (среднее)
            ax.plot(x, mean_vals, color=color, alpha=coeff, lw=2,
                    label=f'{key[0]}, M={key[1]}',
                    linestyle=linestyle,
                    marker=marker,        # Сами значки (круг, квадрат и т.д.)
                    markevery=1200,        # Ставим маркер только на каждую 10-ю точку
                    markersize=5,        # Размер значка
                    )



            # Доверительный интервал (заливка области std)
            # alpha=0.2 * coeff делает тень светлее основной линии
            ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals,
                            color=color, alpha=0.2 * coeff)



import matplotlib.ticker as mtick

formatter = mtick.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
# ax3.yaxis.set_major_formatter(formatter)

# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
fig, (ax1, ax2, ax3) = get_fig_set_style(1, (3,1), (10,19))

unique_groups = ['M-LCB','LimitedAdvice', 'SmoothCORRAL']
plot_regret(ax1, unique_groups)
plot_bar(ax2, "selection_for_decisions", unique_groups)
plot_bar(ax3, "counts", unique_groups)


ax1.set_title("(a) Cumulative Loss vs Steps")
ax1.set_xlabel(r"$\#$ steps")
ax1.set_ylabel("Cumulative Loss")
ax1.grid(True)

# Для оси X: 10000 станет 10k
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

# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


h, legend_ = ax1.get_legend_handles_labels()
ncol = 3



# 1. Определяем целевую сетку
rows = 4
cols = 3
total_slots = rows * cols # 12

import matplotlib.patches as mpatches
# 2. Создаем "невидимый" элемент
empty_handle = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)

# 3. Добиваем списки до 12 элементов
h_padded = list(h)
l_padded = list(legend_)

while len(h_padded) < total_slots:
    h_padded.append(empty_handle)
    l_padded.append("") # Пустая строка для текста

# 4. Выводим легенду (теперь 3 колонки по 4 строки)
leg = fig.legend(
    h_padded,
    l_padded,
    ncol=cols,
    bbox_to_anchor=(0.0, -0.02, 1, 0.1),
    loc="outside upper left",
    mode="expand",
    borderaxespad=0.0,
)


def save_plots(fig, filename="experiment_results.png"):
    """Save plots to file.

    Args:

        fig: matplotlib figure object
        filename: output filename
    """
    fig.savefig(filename,  format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

save_plots(fig, "bandit_experiment.pdf")
