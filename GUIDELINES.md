# GUIDELINES.md

This file documents the repository structure and the current experiment pipeline.

## Repository Layout

- `stat_online/classical_bandits/`
  - Classical multi-armed bandits and meta-selection over bandit algorithms.
- `stat_online/lin_bandits/`
  - GLM / linear-contextual bandit experiments and optimizer models.
- `stat_online/core/`
  - Serializable experiment records.
- `stat_online/experiments/`
  - Shared runner, storage, plotting, lifecycle, and Matplotlib setup.
- `experiments/`
  - CLI entry points for experiments.
- `scripts/`
  - Shell entry points for reproducible multi-experiment runs.
- `configs/plotting/`
  - JSON configs for figure style.

## Shared Experiment Pipeline

The current structure is split into:

- `stat_online/experiments/runner.py`
  - timing, repeat execution, task execution, deterministic repeat seeds.
- `stat_online/experiments/lifecycle.py`
  - output directory creation and artifact bundle saving.
- `stat_online/experiments/storage.py`
  - JSON/CSV/NPZ artifact IO.
- `stat_online/experiments/plotting.py`
  - live and artifact-based plotting.
- `stat_online/experiments/matplotlib.py`
  - output-scoped Matplotlib setup.

Domain adapters live in:

- `stat_online/classical_bandits/runners.py`
- `stat_online/lin_bandits/runners.py`

These adapters convert domain-specific experiment objects into shared
`RunRecord` rows and primitive arrays.

## Plotting Style

Plot styling is configured in:

- `configs/plotting/classical_bandits.json`
- `configs/plotting/glm_bandits.json`

Both live and regenerated plots use a three-panel layout:

1. cumulative loss
2. selection count
3. optimization count

## Artifact Policy

Primary results should be stored as primitive artifacts:

- `metadata.json`
- `config.json`
- `runs.csv`
- `timeseries.npz`

Avoid pickle/cloudpickle for canonical experiment output.
Use `--save_debug_pickle` only for temporary GLM debugging.

## Reproducibility

- Prefer `uv run ...` commands from the repo root.
- Always pass `--output_dir`.
- Use `--seed` for smoke and batch runs.
- Matplotlib cache is stored inside the output directory, so `MPLCONFIGDIR`
  should not be required by users.

## Current CLI Shape

- `experiments.experiment_glm`
  - `--preset smoke`
  - `--output_dir`
  - `--regenerate_plot_from`
  - `--plot_config`
  - `--seed`
- `experiments.experiment_bandits`
  - `--preset smoke`
  - `--output_dir`
  - `--output_path`
  - `--regenerate_plot_from`
  - `--plot_config`
  - `--include_smooth_corral`
  - `--seed`

## Paper-Scale Run Script

Use `scripts/experiments.sh` for article-scale synthetic experiments:

```bash
scripts/experiments.sh exp_results/article
```

The script runs:

- GLM model selection with `K=10`, `M in {1,2,3}`, `num_repeats=30`, `T=2500`.
- MAB model selection with `K=10`, `K_env=2`, `M in {1,2,3,4}`, `T=250000`, `num_repeats=100`.

The second experiment will run for a very long time. If you want to try use $T=2500$ and less number of repeats.

The first positional argument is the output root. Environment variables can
override heavy-run settings, e.g. `SEED`, `GLM_T`, `GLM_DIM`, `GLM_JOBS`,
`BANDIT_T`, and `BANDIT_JOBS`.

For classical bandits, `SmoothCORRAL` is opt-in via `--include_smooth_corral`
because it is substantially slower than the other baselines.

## Maintenance Notes

- Keep algorithm logic in `stat_online/`.
- Keep CLI orchestration in `experiments/`.
- Keep new plot themes in `configs/plotting/`.
- Keep generated outputs out of version control unless explicitly needed.
