"""Compare model-free and world-model policy training runs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.logging import RunLogger, timestamp, to_jsonable
from world_marl.scripts import train_e2e


LOSS_KEYS = (
    "ppo/total_loss",
    "ppo/actor_loss",
    "ppo/value_loss",
    "ppo/entropy",
)
COUNTER_KEYS = (
    "real_env_steps",
    "imagined_env_steps",
    "completed_real_episodes",
    "cumulative_real_episodes",
)
MODEL_FREE_IGNORED_OPTIONS = {
    "--prefit-world-model": 0,
    "--wm-random-rollouts": 1,
    "--wm-initial-rollouts": 1,
    "--wm-fit-steps": 1,
    "--wm-learning-rate": 1,
    "--wm-hidden-dim": 1,
    "--wm-integration-steps": 1,
    "--wm-policy-warmup-updates": 1,
    "--wm-flow-type": 1,
    "--wm-num-categories": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/policy_benchmark")
    parser.add_argument(
        "--episode-checkpoints",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100],
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' are passed to world-marl-train-e2e.",
    )
    args = parser.parse_args()
    if args.train_args and args.train_args[0] == "--":
        args.train_args = args.train_args[1:]
    if any(checkpoint < 1 for checkpoint in args.episode_checkpoints):
        parser.error("--episode-checkpoints values must be >= 1")
    return args


def _parse_train_args(extra_args: list[str]) -> argparse.Namespace:
    old_argv = sys.argv
    try:
        sys.argv = ["world-marl-train-e2e", *extra_args]
        return train_e2e.parse_args()
    finally:
        sys.argv = old_argv


def _strip_model_free_ignored_options(args: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        option = args[index]
        if option in MODEL_FREE_IGNORED_OPTIONS:
            index += 1 + MODEL_FREE_IGNORED_OPTIONS[option]
            continue
        result.append(option)
        index += 1
    return result


def _arm_train_args(
    train_args: list[str],
    *,
    arm: str,
    out_dir: Path,
) -> argparse.Namespace:
    if arm == "model_free":
        args = _strip_model_free_ignored_options(train_args)
    elif arm == "model_based":
        args = list(train_args)
        if "--prefit-world-model" not in args:
            args.append("--prefit-world-model")
    else:
        raise ValueError(f"unknown arm {arm!r}")
    args.extend(["--negative-control", "none", "--out-dir", str(out_dir)])
    parsed = _parse_train_args(args)
    parsed.negative_control = "none"
    parsed.out_dir = str(out_dir)
    return parsed


def loss_at_episode_checkpoints(
    rows: list[dict[str, Any]],
    checkpoints: list[int],
) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for checkpoint in checkpoints:
        selected = None
        for row in rows:
            if int(row.get("cumulative_real_episodes", 0)) >= checkpoint:
                selected = row
                break
        if selected is None:
            result[str(checkpoint)] = None
            continue
        result[str(checkpoint)] = {
            "checkpoint": checkpoint,
            "actual_real_episodes": int(selected["cumulative_real_episodes"]),
            "update": int(selected["update"]),
            **{key: selected.get(key) for key in LOSS_KEYS},
        }
    return result


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run_artifacts(
    run_dir: Path,
    episode_checkpoints: list[int],
) -> dict[str, Any]:
    rows = read_metrics(run_dir / "metrics.jsonl")
    timing = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    loss_per_update = [
        {
            "update": int(row["update"]),
            **{key: row.get(key) for key in COUNTER_KEYS},
            **{key: row.get(key) for key in LOSS_KEYS},
        }
        for row in rows
    ]
    return {
        "run_dir": str(run_dir),
        "runtime_seconds": float(timing["runtime_seconds"]),
        "total_updates": len(rows),
        "outcome": outcome,
        "loss_per_update": loss_per_update,
        "loss_at_real_episode_checkpoints": loss_at_episode_checkpoints(
            rows,
            episode_checkpoints,
        ),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.asarray(values, dtype=float).mean()) if values else None


def summarize_arm(
    arm_dir: Path,
    outcomes: list[train_e2e.RunOutcome],
    episode_checkpoints: list[int],
) -> dict[str, Any]:
    runs = [
        summarize_run_artifacts(arm_dir / f"run_{run_index:03d}", episode_checkpoints)
        for run_index in range(len(outcomes))
    ]
    runtime_values = [float(run["runtime_seconds"]) for run in runs]
    trained_values = [float(run["outcome"]["trained_mean"]) for run in runs]
    real_steps = [float(run["outcome"]["real_env_steps"]) for run in runs]
    real_episodes = [
        float(run["outcome"]["cumulative_real_episodes"]) for run in runs
    ]
    total_updates = [float(run["total_updates"]) for run in runs]
    return {
        "arm_dir": str(arm_dir),
        "aggregate": {
            "runtime_seconds_mean": _mean(runtime_values),
            "trained_mean_mean": _mean(trained_values),
            "real_env_steps_mean": _mean(real_steps),
            "cumulative_real_episodes_mean": _mean(real_episodes),
            "total_updates_mean": _mean(total_updates),
        },
        "runs": runs,
    }


def run_arm(
    args: argparse.Namespace,
    *,
    arm: str,
    out_dir: Path,
    episode_checkpoints: list[int],
) -> dict[str, Any]:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    train_args = _arm_train_args(args.train_args, arm=arm, out_dir=arm_dir)
    outcomes = [
        train_e2e.run_training(
            train_args,
            run_dir=arm_dir / f"run_{run_index:03d}",
            name=f"{arm}_run_{run_index:03d}",
            run_index=run_index,
            control=None,
        )
        for run_index in range(train_args.num_runs)
    ]
    train_summary = train_e2e.summarize(
        outcomes,
        None,
        min_improvement=train_args.min_improvement,
    )
    RunLogger(arm_dir).write_json("summary.json", train_summary)
    arm_summary = summarize_arm(arm_dir, outcomes, episode_checkpoints)
    arm_summary["train_e2e_summary"] = train_summary
    return arm_summary


def compare_arms(model_free: dict[str, Any], model_based: dict[str, Any]) -> dict[str, Any]:
    free = model_free["aggregate"]
    based = model_based["aggregate"]

    def delta(key: str) -> float | None:
        if free.get(key) is None or based.get(key) is None:
            return None
        return float(based[key]) - float(free[key])

    return {
        "runtime_seconds_delta": delta("runtime_seconds_mean"),
        "trained_mean_delta": delta("trained_mean_mean"),
        "real_env_steps_delta": delta("real_env_steps_mean"),
        "cumulative_real_episodes_delta": delta("cumulative_real_episodes_mean"),
        "total_updates_delta": delta("total_updates_mean"),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) / f"policy_benchmark_{timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_free = run_arm(
        copy.deepcopy(args),
        arm="model_free",
        out_dir=out_dir,
        episode_checkpoints=args.episode_checkpoints,
    )
    model_based = run_arm(
        copy.deepcopy(args),
        arm="model_based",
        out_dir=out_dir,
        episode_checkpoints=args.episode_checkpoints,
    )
    report = {
        "out_dir": str(out_dir),
        "episode_checkpoints": args.episode_checkpoints,
        "model_free": model_free,
        "model_based": model_based,
        "comparison": compare_arms(model_free, model_based),
    }
    RunLogger(out_dir).write_json("policy_training_benchmark.json", report)
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
