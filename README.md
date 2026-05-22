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

# Latypov-MS-Thesis

## Setup

Install dependencies:

```bash
uv sync
```

Basic validation:

```bash
uv run python -m compileall stat_online experiments tests
uv run pytest tests/test_runner.py tests/test_storage_artifacts.py
```

## Run Experiments

Paper-scale synthetic experiments:

```bash
scripts/experiments.sh exp_results/article
```

GLM / linear-bandit experiment:

```bash
uv run python -m experiments.experiment_glm --preset smoke --seed 42 --output_dir /private/tmp/glm_smoke_uv
```

Classical-bandit experiment:

```bash
uv run python -m experiments.experiment_bandits --preset smoke --seed 42 --output_dir /private/tmp/bandit_smoke_uv
```

For the full classical-bandit comparison, add `--include_smooth_corral` explicitly; it is kept opt-in because that branch is much heavier than the M-LCB and LimitedAdvice runs.

Featured GLM smoke run:

```bash
uv run python -m experiments.experiment_glm --preset smoke --seed 42 --is_featured --context_d 2 --output_dir /private/tmp/glm_smoke_featured_uv
```

Help:

```bash
uv run python -m experiments.experiment_glm --help
uv run python -m experiments.experiment_bandits --help
```

## Experiment Artifacts

Each run stores primitive artifacts in `--output_dir`:

```text
<output_dir>/
  metadata.json
  config.json
  runs.csv
  timeseries.npz
  .matplotlib/
```

Both experiment entry points save two plots:

- a live plot from the current run;
- a regenerated plot from saved artifacts.

GLM outputs:

- `experiment_results_<best_arm_number>.pdf`
- `regenerated_glm_summary.pdf`

Classical bandit outputs:

- `bandit_experiment.pdf`
- `regenerated_bandit_summary.pdf`

Regenerate plots from artifacts:

```bash
uv run python -m experiments.experiment_glm --regenerate_plot_from /private/tmp/glm_smoke_uv
uv run python -m experiments.experiment_bandits --regenerate_plot_from /private/tmp/bandit_smoke_uv
```

## Notes

- Use `--output_dir` for every experiment run.
- Use `--plot_config` to override the default plotting style.
- Use `--seed` for reproducible smoke runs.
