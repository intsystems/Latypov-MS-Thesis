"""Shared experiment lifecycle helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from stat_online.core.records import RunRecord
from stat_online.experiments.storage import write_experiment_artifacts


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_experiment_bundle(
    output_dir: str | Path,
    experiment_name: str,
    config: Mapping[str, Any],
    run_records: list[RunRecord],
    arrays: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> Path:
    ensure_output_dir(output_dir)
    return write_experiment_artifacts(
        output_dir,
        experiment_name,
        config,
        run_records,
        arrays,
        run_id=run_id,
    )
