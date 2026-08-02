"""Run a fixed deterministic evaluation of a DreaMARL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from world_marl.jepa_transformer.foundation import repository_root


def _timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    manifest_path = experiment / "launch.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = experiment / "evaluation" / f"seed_{args.eval_seed}_{_timestamp()}"
    python = (args.python or Path(manifest["python"])).expanduser().resolve()
    platform = args.platform or manifest["platform"]
    outputs = ["jsonl", "scope"]
    if args.wandb_project:
        outputs.append("wandb")
    command = [
        str(python),
        "-m",
        "world_marl.dreamarl.main",
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
        str(experiment / "run" / "ckpt"),
        "--run.eval_eps",
        str(args.episodes),
        "--run.envs",
        str(min(args.envs, args.episodes)),
        "--jax.platform",
        platform,
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
    return subprocess.run(command, cwd=repository_root(), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
