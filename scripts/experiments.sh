#!/usr/bin/env bash
set -euo pipefail

# Run paper-scale synthetic experiments from the repository root.
# Usage:
#   scripts/experiments.sh [output_root]
# Optional environment overrides:
#   SEED=42 GLM_JOBS=12 BANDIT_JOBS=-1 scripts/experiments.sh exp_results/article

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${1:-exp_results/article}"
SEED="${SEED:-42}"

GLM_OUTPUT_DIR="${OUTPUT_ROOT}/glm_model_selection"
BANDIT_OUTPUT_DIR="${OUTPUT_ROOT}/mab_model_selection"

# Appendix GLM experiment: K=10, M in {1,2,3}, 30 independent runs.
# The current CLI uses --num_optimize 3 to sweep M=1..3.
GLM_T="${GLM_T:-2500}"
GLM_DIM="${GLM_DIM:-5}"
GLM_JOBS="${GLM_JOBS:-12}"

# Main-text MAB model-selection experiment: K=10 experts, two Bernoulli arms per expert,
# M in {1,2,3,4}, T=25*10^4, averaged over 100 independent trials.
BANDIT_T="${BANDIT_T:-25000}"
BANDIT_JOBS="${BANDIT_JOBS:--1}"

mkdir -p "$OUTPUT_ROOT"

echo "Running GLM model-selection experiment -> ${GLM_OUTPUT_DIR}"
uv run python -m experiments.experiment_glm \
  --T "$GLM_T" \
  --dim "$GLM_DIM" \
  --K 10 \
  --num_optimize 3 \
  --best_arm_number 9 \
  --num_repeats 30 \
  --n_jobs "$GLM_JOBS" \
  --seed "$SEED" \
  --output_dir "$GLM_OUTPUT_DIR" \
  --plot_config configs/plotting/glm_bandits.json

echo "Running MAB model-selection experiment -> ${BANDIT_OUTPUT_DIR}"
uv run python -m experiments.experiment_bandits \
  --T "$BANDIT_T" \
  --K 10 \
  --K_env 2 \
  --num_repeats 101 \
  --n_jobs "$BANDIT_JOBS" \
  --include_smooth_corral \
  --seed "$SEED" \
  --output_dir "$BANDIT_OUTPUT_DIR" \
  --plot_config configs/plotting/classical_bandits.json

echo "Done. Results saved under ${OUTPUT_ROOT}"
