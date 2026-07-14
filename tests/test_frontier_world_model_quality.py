from __future__ import annotations

import json

import pytest

from world_marl.scripts.frontier_world_model_quality import (
    _run_dmc,
    assess_quality,
    main,
    parse_args,
)


def _run(model: str, initial: float, final: float, return_: float) -> dict:
    return {
        "model": model,
        "initial_loss": initial,
        "final_loss": final,
        "learning_gate_passed": final <= initial,
        "evaluation_return": return_,
    }


def test_quality_gate_requires_learning_and_real_environment_quality() -> None:
    brax_runs = [
        _run("dreamer_v3_baseline", 10.0, 5.0, -300.0),
        _run("dreamer_v3_baseline", 8.0, 4.0, -280.0),
        _run("genie2_continuous_jax", 12.0, 6.0, -250.0),
        _run("genie2_continuous_jax", 10.0, 5.0, -230.0),
    ]
    dmc_rows = [
        {"model": "dreamer_v3_baseline", "mean_return": 140.0},
        {"model": "genie2_continuous_jax", "mean_return": 125.0},
        {"model": "random_action", "mean_return": 50.0},
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=2,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is True
    assert report["models"]["dreamer_v3_baseline"]["passed"] is True
    assert report["models"]["genie2_continuous_jax"]["passed"] is True
    assert report["models"]["dreamer_v3_baseline"]["brax_execution_passed"] is True
    assert report["models"]["dreamer_v3_baseline"]["dmc_reference_passed"] is True


def test_quality_gate_rejects_model_that_does_not_beat_comparators() -> None:
    brax_runs = [
        _run("dreamer_v3_baseline", 10.0, 9.5, -700.0),
        _run("genie2_continuous_jax", 10.0, 5.0, -250.0),
    ]
    dmc_rows = [
        {"model": "dreamer_v3_baseline", "mean_return": 55.0},
        {"model": "genie2_continuous_jax", "mean_return": 130.0},
        {"model": "random_action", "mean_return": 50.0},
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=1,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is False
    dreamer = report["models"]["dreamer_v3_baseline"]
    assert dreamer["learning_passed"] is False
    assert dreamer["brax_execution_passed"] is True
    assert dreamer["dmc_reference_passed"] is False
    assert dreamer["dmc_quality_passed"] is False


def test_quality_gate_requires_complete_random_dmc_comparator() -> None:
    brax_runs = [
        _run("dreamer_v3_baseline", 10.0, 5.0, -300.0),
        _run("genie2_continuous_jax", 10.0, 5.0, -250.0),
    ]
    dmc_rows = [
        {
            "model": "dreamer_v3_baseline",
            "mean_return": 140.0,
            "successful_seed_count": 1,
        },
        {
            "model": "genie2_continuous_jax",
            "mean_return": 125.0,
            "successful_seed_count": 1,
        },
        {
            "model": "random_action",
            "mean_return": 50.0,
            "successful_seed_count": 0,
        },
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=1,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is False
    assert report["dmc_random_seed_count"] == 0


def test_dmc_reference_is_not_applied_to_brax_returns() -> None:
    brax_runs = [
        _run("dreamer_v3_baseline", 10.0, 5.0, -10_000.0),
        _run("genie2_continuous_jax", 10.0, 5.0, -9_000.0),
    ]
    dmc_rows = [
        {
            "model": "dreamer_v3_baseline",
            "mean_return": 140.0,
            "successful_seed_count": 1,
        },
        {
            "model": "genie2_continuous_jax",
            "mean_return": 125.0,
            "successful_seed_count": 1,
        },
        {
            "model": "random_action",
            "mean_return": 50.0,
            "successful_seed_count": 1,
        },
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=1,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is True
    assert report["models"]["dreamer_v3_baseline"]["brax_execution_passed"] is True


def test_quality_gate_accepts_staged_model_without_fake_combined_loss() -> None:
    brax_runs = [
        {
            "model": "dreamer_v3_baseline",
            "learning_gate_passed": True,
            "evaluation_return": -300.0,
        },
        {
            "model": "genie2_continuous_jax",
            "learning_gate_passed": True,
            "evaluation_return": -250.0,
        },
    ]
    dmc_rows = [
        {
            "model": "dreamer_v3_baseline",
            "mean_return": 140.0,
            "successful_seed_count": 1,
        },
        {
            "model": "genie2_continuous_jax",
            "mean_return": 125.0,
            "successful_seed_count": 1,
        },
        {
            "model": "random_action",
            "mean_return": 50.0,
            "successful_seed_count": 1,
        },
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=1,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is True
    assert report["models"]["genie2_continuous_jax"]["loss_ratios"] == []


def test_quality_gate_requires_reference_matched_real_transition_budget() -> None:
    brax_runs = [
        _run("dreamer_v3_baseline", 10.0, 5.0, -300.0),
        _run("genie2_continuous_jax", 10.0, 5.0, -250.0),
    ]
    dmc_rows = [
        {
            "model": "dreamer_v3_baseline",
            "mean_return": 140.0,
            "successful_seed_count": 1,
            "real_env_transition_counts": [163_840],
        },
        {
            "model": "genie2_continuous_jax",
            "mean_return": 125.0,
            "successful_seed_count": 1,
            "real_env_transition_counts": [256],
        },
        {
            "model": "random_action",
            "mean_return": 50.0,
            "successful_seed_count": 1,
        },
    ]

    report = assess_quality(
        brax_runs=brax_runs,
        dmc_rows=dmc_rows,
        expected_seed_count=1,
        expected_real_env_transitions=163_840,
        reference_random_return=20.0,
        reference_trained_return=200.0,
        min_reference_fraction=0.5,
        max_loss_ratio=0.9,
        min_dmc_random_improvement=50.0,
    )

    assert report["passed"] is False
    assert report["models"]["dreamer_v3_baseline"]["transition_budget_passed"]
    assert not report["models"]["genie2_continuous_jax"]["transition_budget_passed"]


def test_quality_runner_uses_jax_native_mjx_dmc_commands(tmp_path) -> None:
    args = parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--reference-random-return",
            "-800",
            "--reference-trained-return",
            "-100",
            "--dry-run",
        ]
    )

    records = _run_dmc(args, (0,))

    assert args.dmc_task == "cartpole/swingup"
    assert args.reference_env == "dmc:cartpole/swingup"
    assert args.reference_label == "singlerl_jepa_dmc_cartpole_swingup"
    assert args.dmc_collect_steps * args.dmc_num_envs == 163_840
    assert args.dmc_train_steps == 3_000
    assert args.dmc_policy_train_steps == 1_500
    assert len(records) == 2
    records_by_model = {record["model"]: record for record in records}
    for record in records_by_model.values():
        command = record["command"]
        assert "dmc:cartpole/swingup" in command
        assert "world-marl-benchmark-dmc-pixels" not in command
    dreamer_command = records_by_model["dreamer_v3_baseline"]["command"]
    genie_command = records_by_model["genie2_continuous_jax"]["command"]
    assert dreamer_command[dreamer_command.index("--policy-train-steps") + 1] == "3000"
    assert genie_command[genie_command.index("--policy-train-steps") + 1] == "1500"
    assert "--policy-objective" not in dreamer_command
    assert genie_command[genie_command.index("--policy-objective") + 1] == (
        "candidate-distill"
    )
    assert genie_command[genie_command.index("--num-policy-candidates") + 1] == "64"
    assert genie_command[genie_command.index("--candidate-min-gap") + 1] == "0.0"


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--genie-num-policy-candidates", "1"),
        ("--genie-candidate-min-gap", "-0.1"),
    ),
)
def test_quality_runner_rejects_invalid_candidate_policy_config(
    tmp_path,
    flag: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--out-dir",
                str(tmp_path),
                "--reference-random-return",
                "-800",
                "--reference-trained-return",
                "-100",
                flag,
                value,
            ]
        )


def test_quality_dry_run_manifest_preserves_brax_and_dmc_commands(tmp_path) -> None:
    result = main(
        [
            "--out-dir",
            str(tmp_path),
            "--seed",
            "3",
            "--reference-random-return",
            "-800",
            "--reference-trained-return",
            "-100",
            "--dry-run",
        ]
    )

    commands = json.loads((tmp_path / "commands.json").read_text())
    assert result == 0
    assert len(commands) == 4
    assert {row["environment_family"] for row in commands} == {"brax", "dmc"}


def test_quality_runner_rejects_cross_environment_reference(tmp_path) -> None:
    with pytest.raises(ValueError, match="reference environment"):
        main(
            [
                "--out-dir",
                str(tmp_path),
                "--dmc-task",
                "walker/walk",
                "--reference-random-return",
                "20",
                "--reference-trained-return",
                "200",
                "--dry-run",
            ]
        )
