"""Evaluate a frozen all-action critic as a focal-action intervention policy."""

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


def _latest_checkpoint(experiment):
    latest = experiment / "run" / "ckpt" / "latest"
    checkpoint = latest.parent / latest.read_text().strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("probe_model", type=Path)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    model = args.probe_model.expanduser().resolve()
    manifest = json.loads((experiment / "launch.json").read_text())
    python = absolute_path(args.python or Path(manifest["python"]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = experiment / "counterfactual_evaluation" / f"seed_{args.seed}_{timestamp}"
    command = [
        str(python),
        "-m",
        "dreamarl.main",
        "--logdir",
        str(logdir),
        "--configs",
        *manifest["configs"],
        "--task",
        manifest["task"],
        "--seed",
        str(args.seed),
        "--agent.num_agents",
        str(manifest["num_agents"]),
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(_latest_checkpoint(experiment)),
        "--run.eval_eps",
        str(args.episodes),
        "--run.eval_policy_mode",
        "eval",
        "--run.envs",
        str(min(args.envs, args.episodes)),
        "--run.probe_controller",
        "True",
        "--run.probe_model",
        str(model),
        "--jax.platform",
        args.platform,
        "--jax.precompile",
        "False",
        "--logger.outputs",
        "jsonl",
        "scope",
    ]
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
    return subprocess.run(command, cwd=repository_root(), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
