"""Run and aggregate genuine dm_control pixel world-model evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.scripts.compare_visual_wm import ARM_COMMANDS, build_arm_command


DEFAULT_ARMS = tuple(ARM_COMMANDS)
DEFAULT_TASKS = (
    "point_mass/easy",
    "point_mass/hard",
    "cartpole/swingup",
    "finger/spin",
)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

_T_CRITICAL_95 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636305,
    6: 2.446911848791681,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571628540993,
    10: 2.2281388519649385,
    11: 2.200985160082949,
    12: 2.1788128296634177,
    13: 2.1603686564610127,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.109815577833181,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
    30: 2.042272456301238,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", choices=DEFAULT_ARMS)
    parser.add_argument("--task", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dmc_pixels"))
    parser.add_argument("--collect-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--policy-train-steps", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dmc-camera-id", type=int, default=0)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--random-baseline", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def build_benchmark_runs(
    *,
    arms: tuple[str, ...] | list[str],
    tasks: tuple[str, ...] | list[str],
    seeds: tuple[int, ...] | list[int],
    out_dir: Path,
    collect_steps: int,
    num_envs: int,
    max_cycles: int,
    train_steps: int,
    policy_train_steps: int,
    eval_episodes: int,
    image_size: int,
    dmc_camera_id: int,
    dmc_workers: int,
    allow_fail: bool,
) -> list[dict[str, Any]]:
    runs = []
    for task in tasks:
        _validate_task(task)
        task_slug = task.replace("/", "__")
        env = f"dmc-pixels:{task}"
        for arm in arms:
            for seed in seeds:
                run_out_dir = out_dir / task_slug / arm / f"seed_{seed}"
                command = build_arm_command(
                    arm,
                    env=env,
                    out_dir=run_out_dir,
                    collect_steps=collect_steps,
                    num_envs=num_envs,
                    max_cycles=max_cycles,
                    train_steps=train_steps,
                    policy_train_steps=policy_train_steps,
                    eval_episodes=eval_episodes,
                    allow_fail=allow_fail,
                    seed=seed,
                    image_size=image_size,
                    dmc_camera_id=dmc_camera_id,
                    dmc_workers=dmc_workers,
                )
                runs.append(
                    {
                        "arm": arm,
                        "task": task,
                        "env": env,
                        "seed": int(seed),
                        "out_dir": str(run_out_dir),
                        "summary_path": str(run_out_dir / "summary.json"),
                        "command": command,
                    }
                )
    return runs


def aggregate_benchmark_summaries(
    summary_paths: list[Path] | tuple[Path, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in summary_paths:
        payload = json.loads(Path(path).read_text())
        if (
            payload.get("environment_backend") != "dm_control"
            or payload.get("observation_mode") != "pixels"
        ):
            raise ValueError(f"{path} is not a genuine dm_control pixel run")
        model = str(payload.get("model", Path(path).parent.name))
        env = str(payload.get("env"))
        result = dict(payload)
        result["summary_path"] = str(path)
        result["evaluation_return"] = _evaluation_return(payload)
        grouped[(model, env)].append(result)

    rows = []
    for (model, env), runs in sorted(grouped.items()):
        successful = [run for run in runs if run["evaluation_return"] is not None]
        successful.sort(key=lambda run: int(run.get("seed", -1)))
        returns = [float(run["evaluation_return"]) for run in successful]
        seeds = [int(run["seed"]) for run in successful]
        mean_return = statistics.mean(returns) if returns else None
        sample_std = statistics.stdev(returns) if len(returns) >= 2 else None
        ci_low, ci_high = _confidence_interval_95(returns)
        rows.append(
            {
                "model": model,
                "env": env,
                "environment_backend": "dm_control",
                "observation_mode": "pixels",
                "successful_seed_count": len(successful),
                "requested_run_count": len(runs),
                "seeds": seeds,
                "returns": returns,
                "mean_return": mean_return,
                "sample_std_return": sample_std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "real_env_transitions": _sum_int(successful, "real_env_transitions"),
                "model_updates": _sum_int(successful, "model_updates"),
                "imagined_transitions": _sum_int(successful, "imagined_transitions"),
                "summary_paths": [run["summary_path"] for run in successful],
            }
        )
    return rows


def evaluate_random_policy(
    *,
    task: str,
    seed: int,
    episodes: int,
    max_cycles: int,
    image_size: int,
    camera_id: int,
) -> dict[str, Any]:
    from world_marl.envs.dmc_pixel_adapter import DMCPixelAdapter

    adapter = DMCPixelAdapter(
        task,
        num_envs=1,
        max_cycles=max_cycles,
        seed=seed,
        image_size=image_size,
        camera_id=camera_id,
    )
    rng = np.random.default_rng(seed)
    returns = []
    transitions = 0
    try:
        adapter.reset()
        while len(returns) < episodes:
            step = adapter.step(adapter.sample_actions(rng))
            transitions += 1
            returns.extend(float(value[0]) for value in step.completed_returns)
    finally:
        adapter.close()
    returns = returns[:episodes]
    return {
        "model": "random_action",
        "env": f"dmc-pixels:{task}",
        "seed": int(seed),
        "status": "ok",
        "environment_backend": "dm_control",
        "observation_mode": "pixels",
        "real_env_return": float(np.mean(returns)),
        "real_env_transitions": transitions,
        "model_updates": 0,
        "imagined_transitions": 0,
        "evaluation_episodes": episodes,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arms = tuple(args.arm or DEFAULT_ARMS)
    tasks = tuple(args.task or DEFAULT_TASKS)
    seeds = tuple(args.seed or DEFAULT_SEEDS)
    runs = build_benchmark_runs(
        arms=arms,
        tasks=tasks,
        seeds=seeds,
        out_dir=args.out_dir,
        collect_steps=args.collect_steps,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        train_steps=args.train_steps,
        policy_train_steps=args.policy_train_steps,
        eval_episodes=args.eval_episodes,
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
        dmc_workers=args.dmc_workers,
        allow_fail=args.allow_fail,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "commands.json", runs)
    if args.dry_run:
        return 0

    records = []
    summary_paths = list(args.summary)
    failed = False
    for run in runs:
        summary_path = Path(run["summary_path"])
        if args.resume and summary_path.exists():
            return_code = 0
            state = "existing"
        else:
            result = subprocess.run(run["command"], check=False)
            return_code = int(result.returncode)
            state = "completed" if return_code == 0 else "failed"
        record = {**run, "state": state, "return_code": return_code}
        records.append(record)
        failed = failed or return_code != 0 or not summary_path.exists()
        if summary_path.exists():
            summary_paths.append(summary_path)
        _write_json(args.out_dir / "runs.json", records)

    if args.random_baseline:
        for task in tasks:
            for seed in seeds:
                random_dir = (
                    args.out_dir
                    / task.replace("/", "__")
                    / "random_action"
                    / f"seed_{seed}"
                )
                summary_path = random_dir / "summary.json"
                if not (args.resume and summary_path.exists()):
                    random_dir.mkdir(parents=True, exist_ok=True)
                    payload = evaluate_random_policy(
                        task=task,
                        seed=seed,
                        episodes=args.eval_episodes,
                        max_cycles=args.max_cycles,
                        image_size=args.image_size,
                        camera_id=args.dmc_camera_id,
                    )
                    _write_json(summary_path, payload)
                summary_paths.append(summary_path)

    rows = aggregate_benchmark_summaries(summary_paths)
    _write_json(args.out_dir / "aggregate.json", rows)
    _write_csv(args.out_dir / "aggregate.csv", rows)
    return 1 if failed else 0


def _evaluation_return(payload: dict[str, Any]) -> float | None:
    value = payload.get("real_env_bridged_return")
    if value is None:
        value = payload.get("real_env_return")
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _confidence_interval_95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    degrees_of_freedom = len(values) - 1
    critical = _T_CRITICAL_95.get(degrees_of_freedom, 1.959963984540054)
    margin = critical * standard_error
    return mean - margin, mean + margin


def _sum_int(runs: list[dict[str, Any]], key: str) -> int:
    return sum(int(run.get(key, 0) or 0) for run in runs)


def _validate_task(task: str) -> None:
    if "/" not in task or task.startswith("/") or task.endswith("/"):
        raise ValueError("DMC tasks must be formatted as '<domain>/<task>'")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model",
        "env",
        "environment_backend",
        "observation_mode",
        "successful_seed_count",
        "requested_run_count",
        "mean_return",
        "sample_std_return",
        "ci95_low",
        "ci95_high",
        "real_env_transitions",
        "model_updates",
        "imagined_transitions",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
