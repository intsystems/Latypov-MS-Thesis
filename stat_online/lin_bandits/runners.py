"""GLM/linear-bandit experiment artifact adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from stat_online.core.records import RunRecord
from stat_online.utils.visualization import AlgRes


def glm_results_to_artifacts(
    res_dict: dict[str, list[AlgRes]],
    *,
    run_id: str,
    config: dict[str, Any],
) -> tuple[list[RunRecord], dict[str, Any]]:
    run_records: list[RunRecord] = []
    arrays: dict[str, Any] = {}
    repeat_seed_values = config.get("repeat_seeds") or []
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
                seed=repeat_seed_values[repeat_idx] if repeat_idx < len(repeat_seed_values) else None,
                runtime_sec=run.runtime,
                final_loss=float(np.sum(losses)) if losses.size else 0.0,
                extra={"strategy_class": run.strategy_class.__name__},
            ))
    return run_records, arrays
