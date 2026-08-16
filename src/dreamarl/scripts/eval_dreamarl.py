"""Evaluate the latest canonical DreaMARL checkpoint without selection."""

from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import absolute_path
from dreamarl.runtime import repository_root


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_checkpoint(experiment: Path) -> Path:
    root = experiment / "run" / "ckpt"
    latest = root / "latest"
    checkpoint = root / latest.read_text(encoding="utf-8").strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--policy-mode",
        choices=("deterministic", "stochastic"),
        default="deterministic",
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    manifest = json.loads((experiment / "launch.json").read_text(encoding="utf-8"))
    if manifest.get("implementation") != "first-party decoder-free DreaMARL":
        raise ValueError("use the ablation evaluator for non-canonical checkpoints")

    evaluation = experiment / "evaluation" / f"seed_{args.eval_seed}_{_timestamp()}"
    python = absolute_path(args.python or Path(manifest["python"]))
    outputs = ["jsonl", "scope"]
    if args.wandb_project:
        outputs.append("wandb")
    command = [
        str(python),
        "-m",
        "dreamarl.main",
        "--logdir",
        str(evaluation),
        "--configs",
        *manifest["configs"],
        "--task",
        manifest["task"],
        "--seed",
        str(args.eval_seed),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(_latest_checkpoint(experiment)),
        "--run.eval_eps",
        str(args.episodes),
        "--run.eval_policy_mode",
        "eval" if args.policy_mode == "deterministic" else "eval_sample",
        "--run.envs",
        str(min(args.envs, args.episodes)),
        "--jax.platform",
        args.platform or manifest["platform"],
        "--jax.precompile",
        "False",
        "--logger.outputs",
        *outputs,
    ]
    evaluation.parent.mkdir(parents=True, exist_ok=True)
    evaluation.with_suffix(".launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "episodes": args.episodes,
                "eval_seed": args.eval_seed,
                "policy_mode": args.policy_mode,
                "source_experiment": str(experiment),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(" ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "glfw" if host_platform.system() == "Darwin" else "egl")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(manifest["infrastructure_root"])),
            str(repository_root() / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONUNBUFFERED"] = "1"
    if args.wandb_project:
        env["WANDB_PROJECT"] = args.wandb_project
        env["WANDB_NAME"] = f"{experiment.name}_fixed_eval"
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    return subprocess.run(
        command, cwd=repository_root(), env=env, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
