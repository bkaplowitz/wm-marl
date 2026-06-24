import json

from world_marl.scripts.benchmark_policy import (
    loss_at_episode_checkpoints,
    summarize_run_artifacts,
)


def test_loss_at_episode_checkpoints_uses_first_row_at_or_after_budget():
    rows = [
        {
            "update": 1,
            "cumulative_real_episodes": 3,
            "ppo/total_loss": 9.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.1,
        },
        {
            "update": 2,
            "cumulative_real_episodes": 7,
            "ppo/total_loss": 5.0,
            "ppo/actor_loss": 0.5,
            "ppo/value_loss": 1.5,
            "ppo/entropy": 0.2,
        },
    ]

    result = loss_at_episode_checkpoints(rows, [1, 5, 10])

    assert result == {
        "1": {
            "checkpoint": 1,
            "actual_real_episodes": 3,
            "update": 1,
            "ppo/total_loss": 9.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.1,
        },
        "5": {
            "checkpoint": 5,
            "actual_real_episodes": 7,
            "update": 2,
            "ppo/total_loss": 5.0,
            "ppo/actor_loss": 0.5,
            "ppo/value_loss": 1.5,
            "ppo/entropy": 0.2,
        },
        "10": None,
    }


def test_summarize_run_artifacts_includes_runtime_updates_and_losses(tmp_path):
    run_dir = tmp_path / "run_000"
    run_dir.mkdir()
    rows = [
        {
            "update": 1,
            "real_env_steps": 4,
            "imagined_env_steps": 0,
            "completed_real_episodes": 0,
            "cumulative_real_episodes": 0,
            "ppo/total_loss": 3.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.5,
        },
        {
            "update": 2,
            "real_env_steps": 8,
            "imagined_env_steps": 0,
            "completed_real_episodes": 1,
            "cumulative_real_episodes": 1,
            "ppo/total_loss": 2.0,
            "ppo/actor_loss": 0.8,
            "ppo/value_loss": 1.2,
            "ppo/entropy": 0.4,
        },
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (run_dir / "timings.json").write_text(
        json.dumps({"runtime_seconds": 12.5}),
        encoding="utf-8",
    )
    (run_dir / "outcome.json").write_text(
        json.dumps({"trained_mean": 4.0, "real_env_steps": 8}),
        encoding="utf-8",
    )

    summary = summarize_run_artifacts(run_dir, [1])

    assert summary["runtime_seconds"] == 12.5
    assert summary["total_updates"] == 2
    assert summary["loss_per_update"] == [
        {
            "update": 1,
            "real_env_steps": 4,
            "imagined_env_steps": 0,
            "completed_real_episodes": 0,
            "cumulative_real_episodes": 0,
            "ppo/total_loss": 3.0,
            "ppo/actor_loss": 1.0,
            "ppo/value_loss": 2.0,
            "ppo/entropy": 0.5,
        },
        {
            "update": 2,
            "real_env_steps": 8,
            "imagined_env_steps": 0,
            "completed_real_episodes": 1,
            "cumulative_real_episodes": 1,
            "ppo/total_loss": 2.0,
            "ppo/actor_loss": 0.8,
            "ppo/value_loss": 1.2,
            "ppo/entropy": 0.4,
        },
    ]
    assert summary["loss_at_real_episode_checkpoints"]["1"]["update"] == 2
