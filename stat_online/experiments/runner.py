"""Shared experiment runner utilities."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Generic, Iterable, TypeVar

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - joblib is a project dependency.
    Parallel = None
    delayed = None

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class TimedResult(Generic[R]):
    """Result wrapper with runtime measured by the shared runner."""

    result: R
    runtime_sec: float


def timed_call(fn: Callable[[], R]) -> TimedResult[R]:
    start = perf_counter()
    result = fn()
    return TimedResult(result=result, runtime_sec=perf_counter() - start)


def run_tasks(
    tasks: Iterable[T],
    worker: Callable[[T], R],
    *,
    n_jobs: int = 1,
    prefer: str | None = None,
) -> list[R]:
    """Run independent tasks sequentially or through joblib.

    This keeps experiment scripts from open-coding the joblib boilerplate. Use
    `n_jobs=1` for deterministic smoke/debug runs.
    """
    task_list = list(tasks)
    if n_jobs == 1 or len(task_list) <= 1 or Parallel is None or delayed is None:
        return [worker(task) for task in task_list]
    kwargs = {"n_jobs": n_jobs}
    if prefer is not None:
        kwargs["prefer"] = prefer
    return Parallel(**kwargs)(delayed(worker)(task) for task in task_list)


def run_repeats(
    num_repeats: int,
    worker: Callable[[int], R],
    *,
    n_jobs: int = 1,
    prefer: str | None = None,
) -> list[R]:
    """Run a repeat-indexed worker and return results in repeat order."""
    return run_tasks(range(num_repeats), worker, n_jobs=n_jobs, prefer=prefer)


def repeat_seeds(seed: int, num_repeats: int) -> list[int]:
    """Deterministically derive one uint32 seed per repeat."""
    import numpy as np

    seed_sequence = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(num_repeats)]


def seed_numpy(seed: int) -> None:
    """Seed legacy global NumPy RNG used by current algorithm implementations."""
    import numpy as np

    np.random.seed(int(seed) % (2**32))
