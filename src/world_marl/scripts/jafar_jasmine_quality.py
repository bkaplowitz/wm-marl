"""Run equal-budget Jafar and Jasmine source-sized GPU quality evaluations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from world_marl.world_model_foundation.collect import write_json_artifact


ARMS = ("jafar", "jasmine")
WARP_VISION_RUNTIME = {
    "playground": "0.2.0",
    "mujoco": "3.6.0",
    "mujoco-mjx": "3.6.0",
    "warp-lang": "1.11.0",
    "jax": "0.4.36",
    "flax": "0.10.4",
}
REQUIRED_ARTIFACTS = (
    "config.json",
    "sources.json",
    "replay_metadata.json",
    "tokenizer_metrics.jsonl",
    "lam_metrics.jsonl",
    "dynamics_metrics.jsonl",
    "reward_continue_metrics.jsonl",
    "code_usage.json",
    "bridge.json",
    "ppo_metrics.jsonl",
    "rollout.png",
    "real_evaluation.jsonl",
    "outcome.json",
    "summary.json",
)
METRICS = (
    "random_return",
    "learned_simulator_return",
    "bridged_real_return",
    "final_tokenizer_loss",
    "final_lam_loss",
    "final_dynamics_loss",
    "final_reward_continue_loss",
    "updates_per_second",
)


@dataclass(frozen=True)
class QualityCommand:
    arm: str
    seed: int
    run_dir: Path
    command: tuple[str, ...]
    budgets: dict[str, int]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three fixed-seed, equal-budget Jafar/Jasmine GPU evaluations."
    )
    parser.add_argument("--expert-calibration", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--env", default="playground-vision:CartpoleBalance")
    parser.add_argument("--seeds", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--tokenizer-steps", type=int, default=1)
    parser.add_argument("--lam-steps", type=int, default=1)
    parser.add_argument("--dynamics-steps", type=int, default=1)
    parser.add_argument("--reward-continue-steps", type=int, default=1)
    parser.add_argument("--policy-train-steps", type=int, default=1)
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != 3:
        parser.error("--seeds must contain three distinct fixed seeds")
    for name in (
        "time_steps",
        "batch_size",
        "sequence_length",
        "image_size",
        "tokenizer_steps",
        "lam_steps",
        "dynamics_steps",
        "reward_continue_steps",
        "policy_train_steps",
        "imagination_horizon",
        "num_envs",
        "max_cycles",
        "eval_episodes",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def _budgets(args: argparse.Namespace) -> dict[str, int]:
    return {
        "tokenizer": args.tokenizer_steps,
        "lam": args.lam_steps,
        "dynamics": args.dynamics_steps,
        "reward_continue": args.reward_continue_steps,
        "ppo": args.policy_train_steps,
    }


def build_commands(args: argparse.Namespace) -> tuple[QualityCommand, ...]:
    budgets = _budgets(args)
    records: list[QualityCommand] = []
    for arm in ARMS:
        executable = str(Path(sys.executable).with_name(f"world-marl-train-{arm}"))
        for seed in args.seeds:
            run_dir = args.out_dir / arm / f"seed-{seed}"
            command = (
                executable,
                "--env",
                args.env,
                "--out-dir",
                str(run_dir),
                "--expert-calibration",
                str(args.expert_calibration),
                "--seed",
                str(seed),
                "--model-size",
                "source",
                "--time-steps",
                str(args.time_steps),
                "--batch-size",
                str(args.batch_size),
                "--sequence-length",
                str(args.sequence_length),
                "--image-size",
                str(args.image_size),
                "--tokenizer-steps",
                str(args.tokenizer_steps),
                "--lam-steps",
                str(args.lam_steps),
                "--dynamics-steps",
                str(args.dynamics_steps),
                "--reward-continue-steps",
                str(args.reward_continue_steps),
                "--policy-train-steps",
                str(args.policy_train_steps),
                "--imagination-horizon",
                str(args.imagination_horizon),
                "--num-envs",
                str(args.num_envs),
                "--max-cycles",
                str(args.max_cycles),
                "--eval-episodes",
                str(args.eval_episodes),
            )
            records.append(QualityCommand(arm, seed, run_dir, command, budgets))
    return tuple(records)


def _gpu_utilization() -> list[dict[str, float | int]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, float | int]] = []
    for line in completed.stdout.splitlines():
        index, utilization, memory_used, memory_total = (
            part.strip() for part in line.split(",")
        )
        rows.append(
            {
                "index": int(index),
                "utilization_percent": float(utilization),
                "memory_used_mib": float(memory_used),
                "memory_total_mib": float(memory_total),
            }
        )
    return rows


def _runtime_versions() -> dict[str, str]:
    import flax
    import jax

    return {
        "playground": importlib.metadata.version("playground"),
        "mujoco": importlib.metadata.version("mujoco"),
        "mujoco-mjx": importlib.metadata.version("mujoco-mjx"),
        "warp-lang": importlib.metadata.version("warp-lang"),
        "jax": jax.__version__,
        "flax": flax.__version__,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _source_sized_check(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    checkpoint_stages = (
        "tokenizer",
        "lam",
        "world_model",
        "reward_continue",
        "ppo",
    )
    missing_checkpoints = [
        stage
        for stage in checkpoint_stages
        if not (run_dir / "checkpoints" / stage).is_dir()
    ]
    passed = (
        config["model_size"] == "source"
        and config["runtime"]["image_size"] == 64
        and summary["metrics"]["jax_platform"] == "gpu"
        and not missing
        and not missing_checkpoints
    )
    return {
        "passed": passed,
        "source_configuration": config["model_size"] == "source",
        "image_shape": [64, 64, 3],
        "jax_platform": summary["metrics"]["jax_platform"],
        "forward_backward_metrics": {
            "tokenizer": "tokenizer_metrics.jsonl" not in missing,
            "lam": "lam_metrics.jsonl" not in missing,
            "dynamics": "dynamics_metrics.jsonl" not in missing,
        },
        "complete_sampler": "rollout.png" not in missing,
        "missing_artifacts": missing,
        "missing_checkpoints": missing_checkpoints,
    }


def aggregate_runs(run_dirs: Sequence[Path]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    statuses: list[str] = []
    for arm in ARMS:
        summaries = [
            _read_json(path / "summary.json")
            for path in run_dirs
            if _read_json(path / "summary.json")["model"] == arm
        ]
        summaries.sort(key=lambda item: item["seed"])
        statuses.extend(item["status"] for item in summaries)
        metric_summary: dict[str, dict[str, float]] = {}
        for name in METRICS:
            values = np.asarray(
                [item["metrics"][name] for item in summaries], dtype=np.float64
            )
            metric_summary[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        arm_dirs = [
            path
            for path in run_dirs
            if _read_json(path / "summary.json")["model"] == arm
        ]
        arm_dirs.sort(key=lambda path: _read_json(path / "summary.json")["seed"])
        code_coverage = [
            int(
                np.count_nonzero(
                    _read_json(path / "code_usage.json")["training_transition_counts"]
                )
            )
            for path in arm_dirs
        ]
        models[arm] = {
            "seeds": [item["seed"] for item in summaries],
            "metrics": metric_summary,
            "code_coverage": code_coverage,
            "gpu_runs": sum(
                item["metrics"]["jax_platform"] == "gpu" for item in summaries
            ),
        }
    return {
        "schema_version": 1,
        "status": "ok"
        if statuses and all(status == "ok" for status in statuses)
        else "failed",
        "models": models,
    }


def _dry_run_payload(
    args: argparse.Namespace, records: Sequence[QualityCommand]
) -> dict[str, Any]:
    return {
        "environment": args.env,
        "seeds": list(args.seeds),
        "warp_vision_runtime": WARP_VISION_RUNTIME,
        "equal_budgets": _budgets(args),
        "runs": [
            {
                **asdict(record),
                "run_dir": str(record.run_dir),
                "command": list(record.command),
            }
            for record in records
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_commands(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        write_json_artifact(
            args.out_dir / "dry_run.json", _dry_run_payload(args, records)
        )
        return 0
    if not args.expert_calibration.is_file():
        raise FileNotFoundError(
            f"expert calibration does not exist: {args.expert_calibration}"
        )
    runtime_versions = _runtime_versions()
    if runtime_versions != WARP_VISION_RUNTIME:
        raise RuntimeError(
            "Warp vision runtime does not match the validated versions: "
            f"expected {WARP_VISION_RUNTIME}, got {runtime_versions}"
        )
    write_json_artifact(args.out_dir / "runtime_versions.json", runtime_versions)

    run_dirs: list[Path] = []
    source_checks: list[dict[str, Any]] = []
    for record in records:
        before = _gpu_utilization()
        subprocess.run(record.command, check=True)
        after = _gpu_utilization()
        write_json_artifact(
            record.run_dir / "gpu_utilization.json",
            {"before": before, "after": after},
        )
        check = _source_sized_check(record.run_dir)
        check.update({"model": record.arm, "seed": record.seed})
        source_checks.append(check)
        run_dirs.append(record.run_dir)

    summary = aggregate_runs(run_dirs)
    summary.update(
        {
            "environment": args.env,
            "seeds": list(args.seeds),
            "equal_budgets": _budgets(args),
            "warp_vision_runtime": WARP_VISION_RUNTIME,
            "observed_runtime_versions": runtime_versions,
            "source_sized_checks_passed": all(
                check["passed"] for check in source_checks
            ),
        }
    )
    write_json_artifact(args.out_dir / "source_sized_checks.json", source_checks)
    write_json_artifact(
        args.out_dir / "quality_runs.json", _dry_run_payload(args, records)
    )
    write_json_artifact(args.out_dir / "summary.json", summary)
    return (
        0 if summary["status"] == "ok" and summary["source_sized_checks_passed"] else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
