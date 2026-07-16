from __future__ import annotations

from world_marl.scripts.eval_dmc_jepa import (
    compare_evaluations,
    return_tail_metrics,
)


def test_return_tail_metrics_matches_training_protocol():
    metrics = return_tail_metrics(
        [0.0, 50.0, 900.0, 950.0, 1000.0],
        failure_threshold=100.0,
        success_threshold=900.0,
    )

    assert metrics["failure_count"] == 2
    assert metrics["failure_rate"] == 0.4
    assert metrics["success_count"] == 3
    assert metrics["success_rate"] == 0.6
    assert metrics["return_cvar10"] == 0.0
    assert metrics["nonfailure_mean_return"] == 950.0


def test_compare_evaluations_identifies_new_and_recovered_failures():
    before = {
        "checkpoint": "before",
        "returns": [950.0, 0.0, 900.0, 800.0],
        "failure_return_threshold": 100.0,
    }
    after = {
        "checkpoint": "after",
        "returns": [0.0, 950.0, 920.0, 700.0],
        "failure_return_threshold": 100.0,
    }

    comparison = compare_evaluations(before, after)

    assert comparison["mean_return_delta"] == -20.0
    assert comparison["regressed_episode_indices"] == [0, 3]
    assert comparison["new_failure_indices"] == [0]
    assert comparison["recovered_failure_indices"] == [1]
