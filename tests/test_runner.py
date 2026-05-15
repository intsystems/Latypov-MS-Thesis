from stat_online.experiments.runner import run_repeats, run_tasks, timed_call


def test_run_tasks_sequential():
    assert run_tasks([1, 2, 3], lambda value: value + 1, n_jobs=1) == [2, 3, 4]


def test_run_repeats_order():
    assert run_repeats(3, lambda repeat: repeat * 2, n_jobs=1) == [0, 2, 4]


def test_timed_call():
    result = timed_call(lambda: "ok")
    assert result.result == "ok"
    assert result.runtime_sec >= 0
