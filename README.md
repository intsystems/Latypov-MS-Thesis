<table>
    <tr>
        <td align="center"> <b> Название исследуемой задачи </b> </td>
        <td> UCB-type Algorithm for Budget-Constrained Expert Learning </td>
    </tr>
    <tr>
        <td align="center"> <b> Тип научной работы </b> </td>
        <td> ВКР </td>
    </tr>
    <tr>
        <td align="center"> <b> Автор </b> </td>
        <td> Латыпов Ильгам Магданович </td>
    </tr>
    <tr>
        <td align="center"> <b> Научный руководитель </b> </td>
        <td> к.т.н. , Дорн Юрий Владимирович </td>
    </tr>
</table>


========================================================================================

Abstract
 In many modern applications, a system must dynamically choose between several adaptive learning algorithms that are trained online. Examples include model selection in streaming environments, switching between trading strategies in finance, and orchestrating multiple contextual bandit or reinforcement learning agents. At each round, a learner must select one predictor among $K$ adaptive experts to make a prediction, while being able to update at most $M \le K$ of them under a fixed training budget.

We address this problem in the \emph{stochastic setting} and introduce \algname{M-LCB}, a computationally efficient UCB-style meta-algorithm that provides \emph{anytime regret guarantees}. Its confidence intervals are built directly from realized losses, require no additional optimization, and seamlessly reflect the convergence properties of the underlying experts. If each expert achieves internal regret $\tilde O(T^\alpha)$, then \algname{M-LCB} ensures overall regret bounded by $\tilde O\!\Bigl(\sqrt{\tfrac{KT}{M}} \;+\; (K/M)^{1-\alpha}\,T^\alpha\Bigr)$.

To our knowledge, this is the first result establishing regret guarantees when multiple adaptive experts are trained simultaneously under per-round budget constraints. We illustrate the framework with two representative cases: (i) parametric models trained online with stochastic losses, and (ii) experts that are themselves multi-armed bandit algorithms. These examples highlight how \algname{M-LCB} extends the classical bandit paradigm to the more realistic scenario of coordinating stateful, self-learning experts under limited resources.


========================================================================================

Как использовать:
1) установить зависимости через `uv sync`
2) смотреть `notebooks/algorithm_testing.ipynb` для исходных экспериментов
3) запускать smoke/full эксперименты командами ниже
========================================================================================

## Development and Smoke Runs

This repository uses `uv` for dependency management and command execution.

Install or update the local environment:

```bash
uv sync
```

Run the basic import/syntax check:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m compileall stat_online experiments
```

Check experiment CLIs:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_bandits --help
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_glm --help
```

Run smoke experiments:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_bandits --preset smoke --seed 42 --output_path /private/tmp/bandit_smoke_uv.pdf --output_dir /private/tmp/bandit_smoke_uv_artifacts
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_glm --preset smoke --seed 42 --output_dir /private/tmp/glm_smoke_uv
```

Featured GLM smoke run:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_glm --preset smoke --seed 42 --output_dir /private/tmp/glm_smoke_uv_featured --is_featured --context_d 2
```

`MPLCONFIGDIR` is set because some local environments cannot write to the default Matplotlib cache directory.

Phase 3 uses shared runner/lifecycle helpers for both experiment families:

- `stat_online/experiments/runner.py` owns timing, task/repeat execution, and deterministic repeat seed allocation.
- `stat_online/experiments/lifecycle.py` owns output directory creation and artifact bundle writing.
- `stat_online/classical_bandits/runners.py` and `stat_online/lin_bandits/runners.py` convert domain outputs into shared `RunRecord` rows and primitive arrays.

The top-level `--seed` controls deterministic data/environment setup and per-repeat seeds where the current algorithm implementations still use NumPy's legacy global RNG.


Phase 2 artifact layout for smoke/full runs:

```text
<output_dir>/
  metadata.json
  config.json
  runs.csv
  timeseries.npz
```

GLM still renders `experiment_results_<best_arm_number>.pdf`; classical bandits render the plot path passed via `--output_path`. Debug pickle output is disabled by default and only available for GLM through `--save_debug_pickle`.

Regenerate a GLM cumulative-loss plot from saved primitive artifacts:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_glm --regenerate_plot_from /private/tmp/glm_smoke_uv
```

Regenerate a classical-bandit summary plot from saved primitive artifacts:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run python -m experiments.experiment_bandits --regenerate_plot_from /private/tmp/bandit_smoke_uv_artifacts
```

Run lightweight artifact tests:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run pytest tests/test_storage_artifacts.py
```

Run Phase 3 shared-runner tests:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig-codex uv run pytest tests/test_runner.py tests/test_storage_artifacts.py
```
