#!/usr/bin/env python3
"""Add a DMAWM-adjacent final evaluation to a running SMAC suite slot.

The suite worker writes ``final128/launch.json`` only after the final training
checkpoint has been validated. This sidecar watches for that marker and runs a
32-episode greedy evaluation from the same checkpoint with worker seed offset
zero, matching the released DMAWM evaluation convention as closely as our
environment interface permits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import run_annealed_smac_suite as suite


def argument(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def command(args, run: suite.RunSpec, logdir: Path, checkpoint: Path) -> list[str]:
    return suite.common_command(args, run, logdir) + [
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(checkpoint),
        "--run.eval_worker_offset",
        "0",
        "--run.eval_eps",
        "32",
        "--run.envs",
        "4",
        "--run.eval_policy_mode",
        "eval",
        "--jax.precompile",
        "False",
        "--jax.platform",
        "cuda",
        "--logger.outputs",
        "jsonl",
        "wandb",
        "--logger.filter",
        suite.LOGGER_FILTER,
    ]


def environment(args, run: suite.RunSpec, phase_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    identifier = (
        f"dmas-{suite.ALGORITHM_COMMIT[:4]}-{run.map_name}-"
        f"s{run.seed}-{run.budget_name}"
    )
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PORTSERVER_ADDRESS=args.portserver_address,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{args.source}/src:{args.external}",
        SC2PATH=str(args.sc2),
        WANDB_DIR=str(phase_root),
        WANDB_ENTITY=args.wandb_entity,
        WANDB_PROJECT=args.wandb_project,
        WANDB_RUN_GROUP=(
            f"ma-jepa-annealed-smac-suite-{suite.ALGORITHM_COMMIT[:7]}"
        ),
        WANDB_JOB_TYPE="dmawm32",
        WANDB_NAME=identifier,
        WANDB_RUN_ID=identifier,
        WANDB_RESUME="never",
        WANDB_MODE="online",
        WANDB_NOTES=(
            "DMAWM-adjacent final evaluation: same final MA-JEPA checkpoint; "
            "32 greedy episodes, four environments, worker seed offset zero."
        ),
    )
    env.pop("WANDB_FORK_FROM", None)
    return env


def evaluate(args, run: suite.RunSpec, checkpoint: Path) -> None:
    run_root = args.experiment_root / "runs" / run.run_name
    phase_root = run_root / "dmawm32"
    outcome = run_root / "dmawm32-outcome.json"
    if outcome.is_file() and json.loads(outcome.read_text()).get("completed"):
        return
    if phase_root.exists():
        raise FileExistsError(phase_root)
    phase_root.mkdir()
    invocation = command(args, run, phase_root / "run", checkpoint)
    suite.atomic_json(
        phase_root / "launch.json",
        {
            "phase": "dmawm32",
            "command": invocation,
            "started_at": time.time(),
            "protocol": {
                "episodes": 32,
                "envs": 4,
                "policy_mode": "eval",
                "worker_seed_offset": 0,
            },
        },
    )
    with (phase_root / "launch.log").open("x") as output:
        process = subprocess.Popen(
            invocation,
            cwd=args.source,
            env=environment(args, run, phase_root),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        (phase_root / "pid").write_text(f"{process.pid}\n")
        returncode = process.wait()
    if returncode:
        suite.atomic_json(
            outcome,
            {"completed": False, "returncode": returncode},
        )
        return
    summary_path = phase_root / "run" / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("evaluation_protocol", {}).get("episodes") != 32:
        raise RuntimeError("DMAWM-adjacent 32-episode protocol was not completed")
    suite.atomic_json(
        outcome,
        {
            "completed": True,
            "checkpoint": str(checkpoint),
            "summary": summary,
        },
    )


def wait_for_checkpoint(args, run: suite.RunSpec) -> Path | None:
    run_root = args.experiment_root / "runs" / run.run_name
    marker = run_root / "final128" / "launch.json"
    while True:
        if marker.is_file():
            launch = json.loads(marker.read_text())
            return Path(argument(launch["command"], "--run.from_checkpoint"))
        outcome = run_root / "outcome.json"
        if outcome.is_file() and not json.loads(outcome.read_text()).get("completed"):
            return None
        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--sc2", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--slot-index", type=int, choices=tuple(suite.SLOT_QUEUES), required=True)
    parser.add_argument("--portserver-address", required=True)
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-project", default="majepa-annealed-smac-suite")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.source = args.source.resolve()
    args.python = args.python.absolute()
    args.external = args.external.resolve()
    args.sc2 = args.sc2.resolve()

    for run in suite.SLOT_QUEUES[args.slot_index]:
        checkpoint = wait_for_checkpoint(args, run)
        if checkpoint is not None:
            evaluate(args, run, checkpoint)


if __name__ == "__main__":
    main()
