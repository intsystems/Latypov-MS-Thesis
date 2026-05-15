"""Primitive artifact storage for experiments.

Canonical outputs are JSON metadata/config, CSV run summaries, and NPZ numeric
arrays. Pickle is deliberately not used here.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stat_online.core.records import ExperimentMetadata, RunRecord


def make_run_id(experiment_name: str, created_at: datetime | None = None) -> str:
    created_at = created_at or datetime.now(timezone.utc)
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{sanitize_key(experiment_name)}"


def sanitize_key(value: str) -> str:
    """Return a filesystem/NPZ-friendly key."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "value"


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, type):
        return value.__name__
    return str(value)


def current_git_commit(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def package_versions(package_names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python >=3.12 in this project.
        return versions
    for name in package_names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def write_json(path: Path, payload: Mapping[str, Any] | Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=json_default, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_runs_csv(path: Path, records: Sequence[RunRecord | Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() if isinstance(record, RunRecord) else dict(record) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np_arrays = {sanitize_key(key): np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(path, **np_arrays)


def write_experiment_artifacts(
    output_dir: str | Path,
    experiment_name: str,
    config: Mapping[str, Any],
    run_records: Sequence[RunRecord | Mapping[str, Any]],
    arrays: Mapping[str, Any],
    *,
    run_id: str | None = None,
    command: str | None = None,
    cwd: str | Path | None = None,
) -> Path:
    """Write primitive experiment artifacts and return the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    run_id = run_id or make_run_id(experiment_name, created_at)
    metadata = ExperimentMetadata(
        experiment_name=experiment_name,
        run_id=run_id,
        created_at=created_at.isoformat(),
        command=command or " ".join(sys.argv),
        git_commit=current_git_commit(Path(cwd) if cwd is not None else None),
        python_version=platform.python_version(),
        package_versions=package_versions(["numpy", "matplotlib", "jax", "joblib", "scipy", "seaborn"]),
    )

    write_json(out / "metadata.json", metadata)
    write_json(out / "config.json", dict(config))
    write_runs_csv(out / "runs.csv", run_records)
    write_npz(out / "timeseries.npz", arrays)
    return out


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_runs_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_experiment_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Load primitive artifacts written by ``write_experiment_artifacts``."""
    out = Path(output_dir)
    return {
        "metadata": read_json(out / "metadata.json"),
        "config": read_json(out / "config.json"),
        "runs": read_runs_csv(out / "runs.csv"),
        "arrays": np.load(out / "timeseries.npz"),
    }
