"""Serializable experiment records.

These records intentionally contain only primitive values so experiment results
can be stored without pickling live algorithm objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    """One summary row for one algorithm/repeat run."""

    run_id: str
    experiment_name: str
    repeat: int
    algorithm: str
    group: str = ""
    M: int | None = None
    K: int | None = None
    T: int | None = None
    seed: int | None = None
    runtime_sec: float | None = None
    final_regret: float | None = None
    final_loss: float | None = None
    total_reward: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra") or {}
        row.update({f"extra_{key}": value for key, value in extra.items()})
        return row


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata needed to audit an experiment run."""

    experiment_name: str
    run_id: str
    created_at: str
    command: str
    git_commit: str | None = None
    python_version: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
