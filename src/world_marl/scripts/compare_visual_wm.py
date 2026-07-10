from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

ARM_COMMANDS = {
    "dreamer_v3_baseline": "world-marl-train-dreamer-v3-baseline",
    "genie2_continuous_jax": "world-marl-train-genie2-continuous-jax",
}

FIELDS = (
    "model",
    "env",
    "seed",
    "status",
    "environment_backend",
    "observation_mode",
    "final_loss",
    "real_env_return",
    "real_env_bridged_return",
    "real_env_transitions",
    "model_updates",
    "imagined_transitions",
    "learning_gate_passed",
    "summary_path",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        help="Path to a summary.json artifact. Repeat once per arm.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=tuple(ARM_COMMANDS),
        help="Visual/world-model arm to launch before aggregation.",
    )
    parser.add_argument("--env", default="synthetic:image-grid")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/visual_wm_compare"))
    parser.add_argument("--collect-steps", type=int, default=6)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--policy-train-steps", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dmc-camera-id", type=int, default=0)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "model": payload.get("model", path.parent.name),
        "env": payload.get("env"),
        "seed": payload.get("seed"),
        "status": payload.get("status", "unknown"),
        "environment_backend": payload.get("environment_backend"),
        "observation_mode": payload.get("observation_mode"),
        "final_loss": payload.get("final_loss"),
        "real_env_return": payload.get("real_env_return"),
        "real_env_bridged_return": payload.get("real_env_bridged_return"),
        "real_env_transitions": payload.get("real_env_transitions"),
        "model_updates": payload.get("model_updates"),
        "imagined_transitions": payload.get("imagined_transitions"),
        "learning_gate_passed": payload.get("learning_gate_passed"),
        "summary_path": str(path),
    }


def write_comparison(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arm_command(
    arm: str,
    *,
    env: str,
    out_dir: Path,
    collect_steps: int,
    num_envs: int,
    max_cycles: int,
    train_steps: int,
    policy_train_steps: int,
    eval_episodes: int,
    allow_fail: bool,
    seed: int | None = None,
    image_size: int | None = None,
    dmc_camera_id: int | None = None,
    dmc_workers: int | None = None,
) -> list[str]:
    try:
        cli = ARM_COMMANDS[arm]
    except KeyError as exc:
        raise ValueError(f"unsupported visual world-model arm: {arm}") from exc

    command = [
        "uv",
        "run",
        cli,
        "--env",
        env,
        "--out-dir",
        str(out_dir),
        "--collect-steps",
        str(collect_steps),
        "--num-envs",
        str(num_envs),
        "--max-cycles",
        str(max_cycles),
        "--train-steps",
        str(train_steps),
        "--policy-train-steps",
        str(policy_train_steps),
        "--eval-episodes",
        str(eval_episodes),
    ]
    if allow_fail:
        command.append("--allow-fail")
    if seed is not None:
        command.extend(("--seed", str(seed)))
    if image_size is not None:
        command.extend(("--image-size", str(image_size)))
    if dmc_camera_id is not None:
        command.extend(("--dmc-camera-id", str(dmc_camera_id)))
    if dmc_workers is not None:
        command.extend(("--dmc-workers", str(dmc_workers)))
    return command


def _dispatch_arms(args: argparse.Namespace) -> list[Path]:
    summary_paths = []
    commands = []
    for arm in args.arm:
        arm_out_dir = args.out_dir / arm
        command = build_arm_command(
            arm,
            env=args.env,
            out_dir=arm_out_dir,
            collect_steps=args.collect_steps,
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            train_steps=args.train_steps,
            policy_train_steps=args.policy_train_steps,
            eval_episodes=args.eval_episodes,
            allow_fail=args.allow_fail,
            seed=args.seed,
            image_size=args.image_size,
            dmc_camera_id=args.dmc_camera_id,
            dmc_workers=args.dmc_workers,
        )
        commands.append({"arm": arm, "command": command})
        summary_paths.append(arm_out_dir / "summary.json")
        if not args.dry_run:
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

    if args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "commands.json").write_text(
            json.dumps(commands, indent=2) + "\n"
        )
    return summary_paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary:
        summary_paths = args.summary
    elif args.arm:
        summary_paths = _dispatch_arms(args)
        if args.dry_run:
            return 0
    else:
        raise SystemExit("provide --summary or at least one --arm")

    rows = [load_summary(path) for path in summary_paths]
    write_comparison(args.out_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
