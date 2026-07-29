"""Run one failure-tolerant GPU lane of the official Phase 1 baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from world_marl.baselines.dreamerv3.config import repository_root


TASKS = ("dmc_walker_walk", "dmc_cheetah_run")
IMPLEMENTATIONS = ("dreamerv3", "nedreamer")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _completed(run_dir: Path) -> bool:
    outcome = run_dir / "outcome.json"
    if not outcome.exists():
        return False
    try:
        return bool(json.loads(outcome.read_text(encoding="utf-8"))["completed"])
    except (KeyError, json.JSONDecodeError):
        return False


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _command(
    implementation: str,
    *,
    run_dir: Path,
    task: str,
    seed: int,
    project: str,
    entity: str,
) -> list[str]:
    if implementation == "dreamerv3":
        return [
            sys.executable,
            "-m",
            "world_marl.scripts.train_dmc_dreamerv3",
            "--observation-mode",
            "vision",
            "--task",
            task,
            "--seed",
            str(seed),
            "--official-budget",
            "--platform",
            "cuda",
            "--python",
            str(repository_root() / ".venv-dreamerv3" / "bin" / "python"),
            "--experiment-dir",
            str(run_dir),
            "--save-every-seconds",
            "1800",
            "--wandb-project",
            project,
            "--wandb-entity",
            entity,
        ]
    if implementation == "nedreamer":
        return [
            sys.executable,
            "-m",
            "world_marl.scripts.train_dmc_nedreamer",
            "--task",
            task,
            "--seed",
            str(seed),
            "--total-env-steps",
            "1100000",
            "--device",
            "cuda:0",
            "--python",
            str(repository_root() / ".venv-nedreamer" / "bin" / "python"),
            "--experiment-dir",
            str(run_dir),
            "--wandb-project",
            project,
            "--wandb-entity",
            entity,
        ]
    raise ValueError(implementation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "jepatransformer" / "phase1",
    )
    parser.add_argument("--wandb-project", default="world-marl")
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument(
        "--order",
        nargs="+",
        choices=IMPLEMENTATIONS,
        default=list(IMPLEMENTATIONS),
    )
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / f"lane_gpu{args.gpu}_seed{args.seed}.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    failed = 0
    for implementation in args.order:
        for task in TASKS:
            run_dir = args.output_root / implementation / task / f"seed_{args.seed}"
            if _completed(run_dir):
                _append(
                    status_path,
                    {
                        "time": _timestamp(),
                        "event": "skip_completed",
                        "implementation": implementation,
                        "task": task,
                        "seed": args.seed,
                    },
                )
                continue
            command = _command(
                implementation,
                run_dir=run_dir,
                task=task,
                seed=args.seed,
                project=args.wandb_project,
                entity=args.wandb_entity,
            )
            _append(
                status_path,
                {
                    "time": _timestamp(),
                    "event": "start",
                    "implementation": implementation,
                    "task": task,
                    "seed": args.seed,
                    "command": command,
                },
            )
            returncode = subprocess.run(
                command, env=env, cwd=repository_root()
            ).returncode
            _append(
                status_path,
                {
                    "time": _timestamp(),
                    "event": "finish",
                    "implementation": implementation,
                    "task": task,
                    "seed": args.seed,
                    "returncode": returncode,
                },
            )
            failed += returncode != 0
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
