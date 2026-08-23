"""Collect frozen SMAC trajectories with JEPA latents and combat outcomes."""

from __future__ import annotations

import argparse
import json
import os
import platform as host_platform
import subprocess
from pathlib import Path

from dreamarl.baselines.dreamerv3.config import absolute_path
from dreamarl.runtime import repository_root


def _latest_checkpoint(experiment: Path) -> Path:
    latest = experiment / "run" / "ckpt" / "latest"
    checkpoint = latest.parent / latest.read_text(encoding="utf-8").strip()
    if not (checkpoint / "done").is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint}")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument(
        "--policy-mode", choices=("deterministic", "stochastic"), default="stochastic"
    )
    parser.add_argument("--python", type=Path)
    parser.add_argument("--platform", choices=("cpu", "cuda", "tpu"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment = args.experiment_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = json.loads((experiment / "launch.json").read_text())
    if not str(manifest["task"]).startswith("smac_"):
        raise ValueError("SMAC probe collection requires a SMAC experiment")
    python = absolute_path(args.python or Path(manifest["python"]))
    logdir = output.parent / f"{output.stem}_collector"
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
        "smac_probe_collect",
        "--run.from_checkpoint",
        str(_latest_checkpoint(experiment)),
        "--run.eval_eps",
        str(args.episodes),
        "--run.eval_policy_mode",
        "eval" if args.policy_mode == "deterministic" else "eval_sample",
        "--run.envs",
        str(min(args.envs, args.episodes)),
        "--run.probe_output",
        str(output),
        "--jax.platform",
        args.platform or manifest["platform"],
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
