"""Wait for a training lane, then run fixed latest-policy evaluations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from world_marl.baselines.dreamerv3.config import repository_root
from world_marl.scripts.run_jepatransformer_phase1_lane import IMPLEMENTATIONS, TASKS


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _training_completed(run_dir: Path) -> bool:
    try:
        return bool(
            json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))[
                "completed"
            ]
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def _evaluation_completed(run_dir: Path) -> bool:
    outcome = run_dir / "evaluation" / "latest_20eps_seed10000" / "outcome.json"
    try:
        return bool(json.loads(outcome.read_text(encoding="utf-8"))["completed"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def _command(implementation: str, run_dir: Path) -> list[str]:
    if implementation == "dreamerv3":
        return [
            sys.executable,
            "-m",
            "world_marl.scripts.eval_dmc_dreamerv3",
            str(run_dir),
            "--episodes",
            "20",
            "--envs",
            "4",
            "--eval-seed",
            "10000",
            "--python",
            str(repository_root() / ".venv-dreamerv3" / "bin" / "python"),
        ]
    if implementation == "nedreamer":
        return [
            sys.executable,
            "-m",
            "world_marl.scripts.eval_dmc_nedreamer",
            str(run_dir),
            "--episodes",
            "20",
            "--eval-seed",
            "10000",
            "--device",
            "cuda:0",
            "--python",
            str(repository_root() / ".venv-nedreamer" / "bin" / "python"),
        ]
    raise ValueError(implementation)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lane-pid-file", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repository_root() / "runs" / "jepatransformer" / "phase1",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    pid = int(args.lane_pid_file.read_text(encoding="utf-8").strip())
    while _alive(pid):
        time.sleep(args.poll_seconds)

    status_path = args.output_root / f"eval_gpu{args.gpu}_seed{args.seed}.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    failed = 0
    for implementation in IMPLEMENTATIONS:
        for task in TASKS:
            run_dir = args.output_root / implementation / task / f"seed_{args.seed}"
            if not _training_completed(run_dir):
                _append(
                    status_path,
                    {
                        "time": _timestamp(),
                        "event": "skip_incomplete_training",
                        "implementation": implementation,
                        "task": task,
                    },
                )
                failed += 1
                continue
            if _evaluation_completed(run_dir):
                continue
            command = _command(implementation, run_dir)
            _append(
                status_path,
                {
                    "time": _timestamp(),
                    "event": "start",
                    "implementation": implementation,
                    "task": task,
                    "command": command,
                },
            )
            returncode = subprocess.run(
                command, cwd=repository_root(), env=env
            ).returncode
            _append(
                status_path,
                {
                    "time": _timestamp(),
                    "event": "finish",
                    "implementation": implementation,
                    "task": task,
                    "returncode": returncode,
                },
            )
            failed += returncode != 0
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
