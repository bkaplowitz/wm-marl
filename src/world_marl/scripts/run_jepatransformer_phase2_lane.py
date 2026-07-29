"""Run one GPU lane of the pre-registered 250K Dreamer-CDP gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from world_marl.baselines.dreamerv3.config import repository_root


TASKS = (
    "dmc_reacher_easy",
    "dmc_walker_walk",
    "dmc_cheetah_run",
    "dmc_hopper_hop",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _completed(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8"))["completed"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def training_command(
    *, run_dir: Path, task: str, seed: int, project: str, entity: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "world_marl.scripts.train_dmc_dreamer_cdp",
        "--task",
        task,
        "--seed",
        str(seed),
        "--total-env-steps",
        "250000",
        "--platform",
        "cuda",
        "--python",
        str(repository_root() / ".venv-dreamer-cdp" / "bin" / "python"),
        "--experiment-dir",
        str(run_dir),
        "--wandb-project",
        project,
        "--wandb-entity",
        entity,
    ]


def evaluation_command(run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "world_marl.scripts.eval_dmc_dreamer_cdp",
        str(run_dir),
        "--episodes",
        "20",
        "--envs",
        "4",
        "--eval-seed",
        "10000",
        "--python",
        str(repository_root() / ".venv-dreamer-cdp" / "bin" / "python"),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "jepatransformer" / "phase2",
    )
    parser.add_argument("--wandb-project", default="world-marl")
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / f"lane_gpu{args.gpu}_seed{args.seed}.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    failed = 0
    for task in TASKS:
        run_dir = args.output_root / "dreamer_cdp" / task / f"seed_{args.seed}"
        if not _completed(run_dir / "outcome.json"):
            command = training_command(
                run_dir=run_dir,
                task=task,
                seed=args.seed,
                project=args.wandb_project,
                entity=args.wandb_entity,
            )
            _append(
                status_path,
                {"time": _timestamp(), "event": "train_start", "command": command},
            )
            returncode = subprocess.run(
                command, cwd=repository_root(), env=env
            ).returncode
            _append(
                status_path,
                {
                    "time": _timestamp(),
                    "event": "train_finish",
                    "task": task,
                    "returncode": returncode,
                },
            )
            if returncode:
                failed += 1
                continue
        eval_outcome = (
            run_dir / "evaluation" / "latest_20eps_seed10000" / "outcome.json"
        )
        if _completed(eval_outcome):
            continue
        command = evaluation_command(run_dir)
        _append(
            status_path,
            {"time": _timestamp(), "event": "eval_start", "command": command},
        )
        returncode = subprocess.run(command, cwd=repository_root(), env=env).returncode
        _append(
            status_path,
            {
                "time": _timestamp(),
                "event": "eval_finish",
                "task": task,
                "returncode": returncode,
            },
        )
        failed += returncode != 0
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
