"""End-to-end smoke test for the CoinGame + prefit-world-model training loop.

Runs the *whole* ``run_training`` pipeline at tiny scale: random/policy
transition collection, world-model fit, the imagined model-rollout PPO loop,
checkpoint save, and the coin-specific reload-evaluation subprocess. It asserts
the loop **completes** and writes its artifacts -- not that it beats the
``min_improvement`` gate. With a world model prefit on a single random + single
policy rollout and soft states fed back over the horizon, imagined returns are
expected to drift, so the run may legitimately *run yet fail* its success gate.
Completion is the contract under test here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from world_marl.scripts import train_e2e


def _tiny_coins_args(
    out_dir: Path,
    monkeypatch,
    *,
    algorithm: str = "ippo",
    prefit: bool = True,
) -> argparse.Namespace:
    argv = [
        "train_e2e",
        "--substrate",
        "coins",
        "--algorithm",
        algorithm,
        "--num-envs",
        "1",
        "--rollout-steps",
        "4",
        "--total-env-steps",
        "8",
        "--num-runs",
        "1",
        "--max-cycles",
        "4",
        "--negative-control",
        "none",
        "--eval-episodes",
        "1",
        "--eval-max-steps",
        "4",
        "--num-minibatches",
        "1",
        "--update-epochs",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    if prefit:
        argv += [
            "--prefit-world-model",
            "--wm-random-rollouts",
            "1",
            "--wm-initial-rollouts",
            "1",
            "--wm-fit-steps",
            "2",
            "--wm-integration-steps",
            "1",
            "--wm-hidden-dim",
            "8",
            "--wm-policy-warmup-updates",
            "1",
        ]
    monkeypatch.setattr(sys, "argv", argv)
    return train_e2e.parse_args()


def test_coins_prefit_run_training_completes_and_writes_artifacts(
    tmp_path, monkeypatch
):
    args = _tiny_coins_args(tmp_path, monkeypatch)
    run_dir = tmp_path / "run_000"

    outcome = train_e2e.run_training(
        args,
        run_dir=run_dir,
        name="run_000",
        run_index=0,
        control=None,
    )

    # Completed without raising and produced a finite trained-return estimate
    # (the reload subprocess parsed and returned a number).
    assert outcome.checkpoint_dir == str(run_dir / "checkpoint")
    assert (run_dir / "checkpoint").is_dir()
    assert (run_dir / "checkpoint" / "checkpoint.msgpack").exists()
    # The coin-specific reload-eval subprocess ran and its JSON was persisted.
    assert (run_dir / "reload_evaluation.json").exists()
    assert (run_dir / "outcome.json").exists()
    assert (run_dir / "world_model_prefit.json").exists()
    assert (run_dir / "world_model_policy_warmup.json").exists()

    timing = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
    assert timing["runtime_seconds"] > 0.0

    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metrics_rows
    final_row = metrics_rows[-1]
    assert final_row["real_env_steps"] == 6
    assert final_row["imagined_env_steps"] == 8
    assert final_row["cumulative_real_episodes"] == 1

    prefit = json.loads(
        (run_dir / "world_model_prefit.json").read_text(encoding="utf-8")
    )
    assert prefit["random_completed_episodes"] == 0
    assert prefit["initial_policy_completed_episodes"] == 0
    assert prefit["prefit_completed_episodes"] == 0


def test_coins_model_free_ippo_run_training_scan_path(tmp_path, monkeypatch):
    """Model-free coins now trains through one ``train_real_scan`` call; the
    per-update row schema written to ``metrics.jsonl`` must survive the switch.
    """
    args = _tiny_coins_args(tmp_path, monkeypatch, algorithm="ippo", prefit=False)
    run_dir = tmp_path / "run_000"

    outcome = train_e2e.run_training(
        args,
        run_dir=run_dir,
        name="run_000",
        run_index=0,
        control=None,
    )

    assert (run_dir / "checkpoint" / "checkpoint.msgpack").exists()
    assert (run_dir / "outcome.json").exists()
    assert outcome.imagined_env_steps == 0

    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # total_env_steps=8 / (num_envs=1 * rollout_steps=4) -> 2 updates, one row each.
    assert [row["update"] for row in metrics_rows] == [1, 2]
    final_row = metrics_rows[-1]
    assert final_row["real_env_steps"] == 8
    assert final_row["imagined_env_steps"] == 0
    # max_cycles=4 -> each 4-step update completes exactly one episode.
    assert final_row["cumulative_real_episodes"] == 2
    for key in (
        "rollout_mean_reward",
        "episode_return_mean",
        "completed_episodes",
        "control",
        "ppo/total_loss",
    ):
        assert key in final_row, key


def test_coins_prefit_mappo_run_training_scan_path(tmp_path, monkeypatch):
    """The MAPPO prefit pipeline (scan warmup + scan collection + imagined scan
    loop + checkpoint/reload-eval) must complete end-to-end on coins.
    """
    args = _tiny_coins_args(tmp_path, monkeypatch, algorithm="mappo", prefit=True)
    run_dir = tmp_path / "run_000"

    outcome = train_e2e.run_training(
        args,
        run_dir=run_dir,
        name="run_000",
        run_index=0,
        control=None,
    )

    assert (run_dir / "checkpoint" / "checkpoint.msgpack").exists()
    assert (run_dir / "reload_evaluation.json").exists()
    assert (run_dir / "world_model_policy_warmup.json").exists()
    assert outcome.imagined_env_steps == 8

    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["update"] for row in metrics_rows] == [1, 2]
    final_row = metrics_rows[-1]
    # warmup (1 update * 4 steps) + random (1) + policy (1) real steps, then
    # the main loop is fully imagined.
    assert final_row["real_env_steps"] == 6
    assert final_row["imagined_env_steps"] == 8
    assert final_row["cumulative_real_episodes"] == 1
    for key in (
        "model_rollout_mean_reward",
        "world_model/prefit_loss",
        "ppo/total_loss",
    ):
        assert key in final_row, key
