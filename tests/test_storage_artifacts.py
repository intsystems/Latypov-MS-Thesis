from pathlib import Path

import numpy as np

from stat_online.core.records import RunRecord
from stat_online.experiments.plotting import save_classical_bandit_artifact_plot, save_glm_artifact_plot
from stat_online.experiments.storage import load_experiment_artifacts, write_experiment_artifacts


def test_write_and_load_experiment_artifacts(tmp_path: Path):
    records = [
        RunRecord(
            run_id="test_run",
            experiment_name="unit",
            repeat=0,
            algorithm="AlgA",
            K=2,
            T=3,
            final_loss=1.5,
        )
    ]
    arrays = {
        "loss/AlgA/repeat_0": np.array([0.5, 0.4, 0.6]),
        "cum_loss/AlgA/repeat_0": np.array([0.5, 0.9, 1.5]),
    }

    write_experiment_artifacts(
        tmp_path,
        "unit",
        {"T": 3, "K": 2},
        records,
        arrays,
        run_id="test_run",
        command="unit-test",
    )

    artifacts = load_experiment_artifacts(tmp_path)
    assert artifacts["metadata"]["experiment_name"] == "unit"
    assert artifacts["config"] == {"K": 2, "T": 3}
    assert artifacts["runs"][0]["algorithm"] == "AlgA"
    assert "loss_AlgA_repeat_0" in artifacts["arrays"].files


def test_glm_artifact_plot_regeneration(tmp_path: Path):
    records = [
        RunRecord(
            run_id="glm",
            experiment_name="glm_bandits",
            repeat=0,
            algorithm="UCBAlgorithm_1_no",
            M=1,
            K=4,
            T=3,
            final_loss=1.5,
        )
    ]
    arrays = {
        "cum_loss/UCBAlgorithm_1_no/repeat_0": np.array([0.5, 0.9, 1.5]),
    }
    write_experiment_artifacts(tmp_path, "glm_bandits", {}, records, arrays, run_id="glm")

    path = save_glm_artifact_plot(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_classical_bandit_artifact_plot_regeneration(tmp_path: Path):
    records = [
        RunRecord(
            run_id="classical",
            experiment_name="classical_bandits",
            repeat=0,
            algorithm="M-LCB_M1",
            group="M-LCB",
            M=1,
            K=2,
            T=3,
            final_regret=0.3,
            total_reward=2.0,
        )
    ]
    arrays = {
        "regret/M-LCB_M1/repeat_0": np.array([0.1, 0.2, 0.3]),
        "reward/M-LCB_M1/repeat_0": np.array([1.0, 0.0, 1.0]),
        "selected_expert/M-LCB_M1/repeat_0": np.array([0, 1, 1]),
        "selected_arm/M-LCB_M1/repeat_0": np.array([0, 0, 1]),
    }
    write_experiment_artifacts(tmp_path, "classical_bandits", {}, records, arrays, run_id="classical")

    path = save_classical_bandit_artifact_plot(tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
