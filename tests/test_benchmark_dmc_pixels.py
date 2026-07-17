from __future__ import annotations

import json
import math

import pytest

from world_marl.scripts.benchmark_dmc_pixels import (
    DEFAULT_ARMS,
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
    aggregate_benchmark_summaries,
    build_benchmark_runs,
    main,
)


def test_build_benchmark_runs_covers_three_arms_four_tasks_five_seeds(tmp_path):
    calibration = tmp_path / "expert.npz"
    runs = build_benchmark_runs(
        arms=DEFAULT_ARMS,
        tasks=DEFAULT_TASKS,
        seeds=DEFAULT_SEEDS,
        out_dir=tmp_path,
        collect_steps=1000,
        num_envs=4,
        max_cycles=1000,
        train_steps=5000,
        policy_train_steps=3000,
        eval_episodes=32,
        image_size=64,
        dmc_camera_id=0,
        dmc_workers=4,
        allow_fail=False,
        expert_calibration=calibration,
    )

    assert len(runs) == 60
    assert len({run["summary_path"] for run in runs}) == 60
    assert {run["env"] for run in runs} == {
        f"dmc-pixels:{task}" for task in DEFAULT_TASKS
    }
    assert {run["seed"] for run in runs} == set(DEFAULT_SEEDS)
    assert {run["arm"] for run in runs} == set(DEFAULT_ARMS)
    for run in runs:
        command = run["command"]
        assert command[command.index("--seed") + 1] == str(run["seed"])
        assert command[command.index("--image-size") + 1] == "64"
        assert command[command.index("--dmc-camera-id") + 1] == "0"
        assert command[command.index("--dmc-workers") + 1] == "4"
        if run["arm"] in {"jafar", "jasmine"}:
            assert command[command.index("--expert-calibration") + 1] == str(
                calibration
            )


def test_aggregate_benchmark_summaries_reports_five_seed_interval(tmp_path):
    summaries = []
    for seed, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        path = tmp_path / f"seed_{seed}" / "summary.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "model": "dreamer_v3_baseline",
                    "env": "dmc-pixels:point_mass/easy",
                    "seed": seed,
                    "status": "ok",
                    "environment_backend": "dm_control",
                    "observation_mode": "pixels",
                    "real_env_return": value,
                    "real_env_transitions": 4000,
                    "model_updates": 5000,
                    "imagined_transitions": 45000,
                }
            )
        )
        summaries.append(path)

    rows = aggregate_benchmark_summaries(summaries)

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "dreamer_v3_baseline"
    assert row["successful_seed_count"] == 5
    assert row["seeds"] == [0, 1, 2, 3, 4]
    assert row["returns"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert row["mean_return"] == 3.0
    assert math.isclose(row["sample_std_return"], math.sqrt(2.5))
    expected_margin = 2.7764451051977987 * math.sqrt(2.5) / math.sqrt(5.0)
    assert math.isclose(row["ci95_low"], 3.0 - expected_margin)
    assert math.isclose(row["ci95_high"], 3.0 + expected_margin)
    assert row["real_env_transitions"] == 20000
    assert row["model_updates"] == 25000
    assert row["imagined_transitions"] == 225000


def test_aggregate_benchmark_summaries_rejects_synthetic_pixels(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "model": "dreamer_v3_baseline",
                "env": "pixels:pointmass",
                "seed": 0,
                "environment_backend": "synthetic",
                "observation_mode": "pixels",
                "real_env_return": 1.0,
            }
        )
    )

    with pytest.raises(ValueError, match="genuine dm_control pixel run"):
        aggregate_benchmark_summaries([path])


def test_privileged_state_baseline_is_explicit_and_never_pooled(tmp_path):
    pixel_path = tmp_path / "pixel.json"
    state_path = tmp_path / "state.json"
    shared = {
        "model": "dreamer_v3_baseline",
        "seed": 0,
        "environment_backend": "dm_control",
        "real_env_return": 1.0,
    }
    pixel_path.write_text(
        json.dumps(
            {
                **shared,
                "env": "dmc-pixels:point_mass/easy",
                "observation_mode": "pixels",
            }
        )
    )
    state_path.write_text(
        json.dumps(
            {
                **shared,
                "model": "dreamer_v3_privileged_state",
                "env": "dmc:point_mass/easy",
                "observation_mode": "vector",
            }
        )
    )

    with pytest.raises(ValueError, match="genuine dm_control pixel run"):
        aggregate_benchmark_summaries([state_path])

    rows = aggregate_benchmark_summaries(
        [pixel_path, state_path], allow_privileged_state=True
    )
    assert len(rows) == 2
    assert {row["observation_mode"] for row in rows} == {"pixels", "vector"}
    assert {row["task"] for row in rows} == {"point_mass/easy"}


def test_benchmark_dry_run_writes_default_matrix_without_launching(tmp_path):
    assert (
        main(
            [
                "--out-dir",
                str(tmp_path),
                "--expert-calibration",
                str(tmp_path / "expert.npz"),
                "--dry-run",
            ]
        )
        == 0
    )
    commands = json.loads((tmp_path / "commands.json").read_text())
    assert len(commands) == 60
    assert not (tmp_path / "aggregate.json").exists()


def test_aggregate_only_combines_separate_branch_summaries(tmp_path):
    summaries = []
    for model, observation_mode, env in (
        (
            "jafar",
            "pixels",
            "dmc-pixels:point_mass/easy",
        ),
        ("dreamer_v3_privileged_state", "vector", "dmc:point_mass/easy"),
    ):
        path = tmp_path / model / "summary.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "model": model,
                    "env": env,
                    "seed": 0,
                    "environment_backend": "dm_control",
                    "observation_mode": observation_mode,
                    "real_env_return": 1.0,
                }
            )
        )
        summaries.append(path)

    out_dir = tmp_path / "aggregate"
    assert (
        main(
            [
                "--aggregate-only",
                "--summary",
                str(summaries[0]),
                "--baseline-summary",
                str(summaries[1]),
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    assert json.loads((out_dir / "commands.json").read_text()) == []
    rows = json.loads((out_dir / "aggregate.json").read_text())
    assert len(rows) == 2
    assert {row["observation_mode"] for row in rows} == {"pixels", "vector"}
