import numpy as np

from stat_online.experiments.runner import repeat_seeds, run_repeats, run_tasks, seed_numpy, timed_call


def test_run_tasks_sequential():
    assert run_tasks([1, 2, 3], lambda value: value + 1, n_jobs=1) == [2, 3, 4]


def test_run_repeats_order():
    assert run_repeats(3, lambda repeat: repeat * 2, n_jobs=1) == [0, 2, 4]


def test_timed_call():
    result = timed_call(lambda: "ok")
    assert result.result == "ok"
    assert result.runtime_sec >= 0


def test_repeat_seeds_are_deterministic_and_distinct():
    first = repeat_seeds(42, 3)
    second = repeat_seeds(42, 3)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3


def test_seed_numpy_reproducible_legacy_rng():
    seed_numpy(123)
    first = np.random.random(3)
    seed_numpy(123)
    second = np.random.random(3)

    np.testing.assert_array_equal(first, second)
