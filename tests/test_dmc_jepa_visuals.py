from __future__ import annotations

import json

from world_marl.scripts.plot_dmc_jepa_run import make_visual_report


def test_dmc_jepa_visual_report_from_summary(tmp_path):
    experiment_dir = tmp_path / "dmc_jepa_20260621T000000Z"
    experiment_dir.mkdir()
    summary = {
        "passed": True,
        "world_model_passed": True,
        "policy_training_enabled": True,
        "aggregate_policy_random_mean": 10.0,
        "aggregate_policy_initial_mean": 20.0,
        "aggregate_policy_trained_mean": 80.0,
        "aggregate_policy_improvement": 60.0,
        "aggregate_final_jepa_loss": 0.01,
        "aggregate_final_open_loop_loss": 0.03,
        "paired_control_differences": {
            "no-action-world-model": {
                "mean_policy_improvement_advantage": 12.0,
                "mean_open_loop_advantage": 0.2,
            },
        },
        "runs": [
            {
                "run_index": 0,
                "control": "none",
                "run_dir": "none/run_000",
                "passed": True,
                "policy_random_mean": 10.0,
                "policy_initial_mean": 20.0,
                "policy_trained_mean": 80.0,
                "policy_improvement": 60.0,
                "final_jepa_loss": 0.01,
                "final_open_loop_loss": 0.03,
                "final_model_metrics": {
                    "model/continuous_action_low_high_sensitivity": 4.5,
                },
                "online_history": [
                    {"actor_replay": {"mean_return": 50.0}},
                    {"actor_replay": {"mean_return": 75.0}},
                ],
            },
            {
                "run_index": 0,
                "control": "no-action-world-model",
                "run_dir": "no-action-world-model/run_000",
                "passed": True,
                "policy_random_mean": 10.0,
                "policy_initial_mean": 20.0,
                "policy_trained_mean": 65.0,
                "policy_improvement": 45.0,
                "final_jepa_loss": 0.04,
                "final_open_loop_loss": 0.2,
                "final_model_metrics": {
                    "model/continuous_action_low_high_sensitivity": 0.1,
                },
                "online_history": [],
            },
        ],
    }
    (experiment_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    output = make_visual_report(experiment_dir)

    assert output.exists()
    assert output.stat().st_size > 0


def test_dmc_jepa_visual_report_from_single_outcome(tmp_path):
    run_dir = tmp_path / "none" / "run_000"
    run_dir.mkdir(parents=True)
    outcome = {
        "run_index": 0,
        "control": "none",
        "passed": True,
        "policy_training_enabled": False,
        "final_jepa_loss": 0.02,
        "final_open_loop_loss": 0.05,
        "final_model_metrics": {},
    }
    (run_dir / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")

    output = make_visual_report(run_dir)

    assert output.exists()
    assert output.stat().st_size > 0
