"""Classical-bandit experiment artifact adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from stat_online.core.records import RunRecord


def classical_results_to_artifacts(
    temp_map,
    *,
    run_id: str,
    config: dict[str, Any],
) -> tuple[list[RunRecord], dict[str, Any]]:
    run_records: list[RunRecord] = []
    arrays: dict[str, Any] = {}
    T = int(config["T"])
    K = int(config["K"])
    repeat_seeds = config.get("repeat_seeds", [])
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
            arrays[f"selection_count/{key_prefix}"] = np.asarray(
                getattr(exp.algorithm, "selection_for_decisions"),
                dtype=float,
            )
            if group_name == "SmoothCORRAL":
                optimization_counts = getattr(exp.algorithm, "selection_for_decisions")
            else:
                optimization_counts = getattr(exp.algorithm, "counts")
            arrays[f"optimization_count/{key_prefix}"] = np.asarray(optimization_counts, dtype=float)
            run_records.append(RunRecord(
                run_id=run_id,
                experiment_name="classical_bandits",
                repeat=repeat_idx,
                algorithm=algorithm,
                group=group_name,
                M=int(m_value),
                K=K,
                T=T,
                seed=repeat_seeds[repeat_idx] if repeat_idx < len(repeat_seeds) else None,
                final_regret=float(regret[-1]) if regret.size else 0.0,
                total_reward=float(np.sum(rewards)) if rewards.size else 0.0,
            ))
    return run_records, arrays
