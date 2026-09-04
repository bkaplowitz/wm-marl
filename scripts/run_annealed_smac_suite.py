#!/usr/bin/env python3
"""Run one queue slot of the fixed MA-JEPA annealed SMAC suite."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import elements

from majepa.main import _load_configs, _resolve_config_profiles


ALGORITHM_COMMIT = "f4d763c5d83b8e49eb39e78fd0d1d7cb3c72da11"
SEEDS = (0, 100, 200)


@dataclass(frozen=True)
class RunSpec:
    map_name: str
    num_agents: int
    steps: int
    horizon: int
    seed: int

    @property
    def task(self) -> str:
        return f"smac_{self.map_name}"

    @property
    def budget_name(self) -> str:
        return f"{self.steps // 1000}k"

    @property
    def entropy_decay_steps(self) -> int:
        # Preserve the successful 2s3z schedule as a fraction of the budget.
        return int(0.8 * self.steps)

    @property
    def run_name(self) -> str:
        return f"{self.map_name}-seed{self.seed}-{self.budget_name}"


MAPS = {
    "3m": (3, 50_000, 15),
    "8m": (8, 50_000, 8),
    "MMM": (10, 50_000, 8),
    "2s_vs_1sc": (2, 50_000, 15),
    "3s_vs_4z": (3, 50_000, 15),
    "3s_vs_5z": (3, 200_000, 15),
    "2c_vs_64zg": (2, 200_000, 15),
    "5m_vs_6m": (5, 200_000, 15),
    "MMM2": (10, 200_000, 8),
    "corridor": (6, 200_000, 8),
}


def spec(map_name: str, seed: int) -> RunSpec:
    num_agents, steps, horizon = MAPS[map_name]
    return RunSpec(map_name, num_agents, steps, horizon, seed)


# Every slot starts with easy maps. The high-agent hard maps are assigned to
# slots with only two 200k jobs to reduce wall-clock imbalance.
SLOT_QUEUES = {
    0: (
        spec("3m", 0),
        spec("8m", 100),
        spec("MMM", 200),
        spec("MMM2", 0),
        spec("corridor", 100),
    ),
    1: (
        spec("8m", 0),
        spec("MMM", 100),
        spec("2s_vs_1sc", 200),
        spec("MMM2", 100),
        spec("corridor", 200),
    ),
    2: (
        spec("MMM", 0),
        spec("2s_vs_1sc", 100),
        spec("3s_vs_4z", 200),
        spec("MMM2", 200),
        spec("corridor", 0),
    ),
    3: (
        spec("2s_vs_1sc", 0),
        spec("3s_vs_4z", 100),
        spec("3s_vs_5z", 0),
        spec("2c_vs_64zg", 100),
        spec("5m_vs_6m", 200),
    ),
    4: (
        spec("3s_vs_4z", 0),
        spec("3m", 200),
        spec("3s_vs_5z", 100),
        spec("2c_vs_64zg", 200),
        spec("5m_vs_6m", 0),
    ),
    5: (
        spec("3m", 100),
        spec("8m", 200),
        spec("3s_vs_5z", 200),
        spec("2c_vs_64zg", 0),
        spec("5m_vs_6m", 100),
    ),
}

LOGGER_FILTER = (
    "score|return|length|fps|ratio|sample_age|replay/behavior_|schedule/|"
    "counters/|replay_views/|train/loss/|train/opt/|train/ctde/|"
    "report/ctde/|train/critic/|train/reploss/critic/|central_critic/|"
    "battle_won|win_rate|legacy_|corrected_|enemy_|ally_|timeout|action_|"
    "attack_target_|eval/|episode/|final_eval/"
)
CURRENT: subprocess.Popen[str] | None = None
PORTSERVER: subprocess.Popen[str] | None = None


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def handle_signal(_signum, _frame) -> None:
    stop_process(CURRENT)
    stop_process(PORTSERVER)
    raise KeyboardInterrupt


def validate_source(args) -> None:
    marker = args.source / "DEPLOYED_COMMIT"
    if not marker.is_file() or marker.read_text().strip() != ALGORITHM_COMMIT:
        raise RuntimeError("deployed algorithm commit does not match suite runner")
    for path in (args.python, args.external, args.sc2, args.portserver_script):
        if not path.exists():
            raise FileNotFoundError(path)


def treatment_flags(run: RunSpec) -> tuple[str, ...]:
    return (
        "--agent.num_agents",
        str(run.num_agents),
        "--agent.imag_length",
        str(run.horizon),
        "--agent.marl.ctde.death_masking.enabled",
        "True",
        "--agent.entropy_schedule.enabled",
        "True",
        "--agent.entropy_schedule.initial",
        "0.001",
        "--agent.entropy_schedule.final",
        "0.0003",
        "--agent.entropy_schedule.decay_steps",
        str(run.entropy_decay_steps),
        "--agent.entropy_schedule.schedule",
        "cosine",
        "--agent.entropy_schedule.normalize",
        "True",
        "--agent.collection_unimix",
        "0.0",
        "--agent.policy.unimix",
        "0.0",
        "--agent.marl.ctde.actor_lr",
        "1e-5",
        "--agent.marl.ctde.multistep_jepa.plan_aggregation",
        "mean",
    )


def validate_profile(run: RunSpec) -> dict[str, object]:
    config = _resolve_config_profiles(_load_configs(), ("smac_vector", "ma_jepa"))
    parsed = elements.Flags(config).parse(
        ["--task", run.task, "--seed", str(run.seed), *treatment_flags(run)]
    )
    ctde = parsed.agent.marl.ctde
    entropy = parsed.agent.entropy_schedule
    resolved = {
        "task": str(parsed.task),
        "seed": int(parsed.seed),
        "num_agents": int(parsed.agent.num_agents),
        "imag_length": int(parsed.agent.imag_length),
        "death_masking": bool(ctde.death_masking.enabled),
        "action_binding": bool(ctde.authoritative_action_binding.enabled),
        "support_preserving": bool(ctde.support_preserving.enabled),
        "actor_lr": float(ctde.actor_lr),
        "plan_aggregation": str(ctde.multistep_jepa.plan_aggregation),
        "collection_unimix": float(parsed.agent.collection_unimix),
        "policy_unimix": float(parsed.agent.policy.unimix),
        "world_model_unimix": float(parsed.agent.dyn.parallel_transformer.unimix),
        "entropy_enabled": bool(entropy.enabled),
        "entropy_initial": float(entropy.initial),
        "entropy_final": float(entropy.final),
        "entropy_decay_steps": int(entropy.decay_steps),
        "entropy_schedule": str(entropy.schedule),
        "entropy_normalize": bool(entropy.normalize),
    }
    expected = {
        "task": run.task,
        "seed": run.seed,
        "num_agents": run.num_agents,
        "imag_length": run.horizon,
        "death_masking": True,
        "action_binding": False,
        "support_preserving": False,
        "actor_lr": 1e-5,
        "plan_aggregation": "mean",
        "collection_unimix": 0.0,
        "policy_unimix": 0.0,
        "world_model_unimix": 0.01,
        "entropy_enabled": True,
        "entropy_initial": 0.001,
        "entropy_final": 0.0003,
        "entropy_decay_steps": run.entropy_decay_steps,
        "entropy_schedule": "cosine",
        "entropy_normalize": True,
    }
    if resolved != expected:
        raise RuntimeError(f"suite profile mismatch: {resolved} != {expected}")
    return resolved


def environment(args, run: RunSpec, phase_root: Path, phase: str) -> dict[str, str]:
    env = os.environ.copy()
    identifier = (
        f"mas-{ALGORITHM_COMMIT[:4]}-{run.map_name}-s{run.seed}-{run.budget_name}"
    )
    if phase == "final128":
        identifier = "f" + identifier
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
        WANDB_RUN_GROUP=f"ma-jepa-annealed-smac-suite-{ALGORITHM_COMMIT[:7]}",
        WANDB_JOB_TYPE=phase,
        WANDB_NAME=identifier,
        WANDB_RUN_ID=identifier,
        WANDB_RESUME="never",
        WANDB_MODE="online",
        WANDB_NOTES=(
            f"Fixed annealed MA-JEPA baseline at {ALGORITHM_COMMIT}; "
            f"{run.map_name} H{run.horizon} seed {run.seed}; "
            f"{run.budget_name} plus held-out fixed128."
        ),
    )
    env.pop("WANDB_FORK_FROM", None)
    return env


def common_command(args, run: RunSpec, logdir: Path) -> list[str]:
    return [
        str(args.python),
        "-m",
        "majepa.main",
        "--logdir",
        str(logdir),
        "--configs",
        "smac_vector",
        "ma_jepa",
        "--task",
        run.task,
        "--seed",
        str(run.seed),
        *treatment_flags(run),
    ]


def train_command(args, run: RunSpec, logdir: Path) -> list[str]:
    return common_command(args, run, logdir) + [
        "--script",
        "train",
        "--run.steps",
        str(run.steps),
        "--run.save_every",
        "900",
        "--run.final_save",
        "True",
        "--run.checkpoint_at_curve_eval",
        "False",
        "--run.curve_eval_interval",
        "5000",
        "--run.curve_eval_eps",
        "64",
        "--run.eval_envs",
        "4",
        "--run.curve_eval_seed_offset",
        "50000",
        "--run.curve_eval_policy_mode",
        "eval",
        "--jax.precompile",
        "True",
        "--jax.platform",
        "cuda",
        "--logger.outputs",
        "jsonl",
        "wandb",
        "--logger.filter",
        LOGGER_FILTER,
    ]


def eval_command(
    args, run: RunSpec, logdir: Path, checkpoint: Path
) -> list[str]:
    return common_command(args, run, logdir) + [
        "--script",
        "eval_only",
        "--run.from_checkpoint",
        str(checkpoint),
        "--run.eval_worker_offset",
        "100000",
        "--run.eval_eps",
        "128",
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
        LOGGER_FILTER,
    ]


def run_child(
    args, run: RunSpec, phase_root: Path, phase: str, command: list[str]
) -> None:
    global CURRENT
    phase_root.mkdir()
    atomic_json(
        phase_root / "launch.json",
        {"phase": phase, "command": command, "started_at": time.time()},
    )
    with (phase_root / "launch.log").open("x") as output:
        CURRENT = subprocess.Popen(
            command,
            cwd=args.source,
            env=environment(args, run, phase_root, phase),
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        (phase_root / "pid").write_text(f"{CURRENT.pid}\n")
        returncode = CURRENT.wait()
        CURRENT = None
    if returncode:
        raise RuntimeError(f"{phase} exited with status {returncode}")


def latest_checkpoint(train_root: Path, expected_step: int) -> Path:
    checkpoint_root = train_root / "run" / "ckpt"
    latest = checkpoint_root / "latest"
    if not latest.is_file():
        raise RuntimeError("training completed without a latest checkpoint")
    checkpoint = checkpoint_root / latest.read_text().strip()
    if not (checkpoint / "done").is_file():
        raise RuntimeError("latest checkpoint is incomplete")
    with (checkpoint / "step.pkl").open("rb") as stream:
        step = int(pickle.load(stream))
    if step != expected_step:
        raise RuntimeError(f"checkpoint step {step} != {expected_step}")
    return checkpoint


def start_portserver(args, run_root: Path) -> None:
    global PORTSERVER
    log = (run_root / "portserver.log").open("x")
    PORTSERVER = subprocess.Popen(
        [
            str(args.python),
            str(args.portserver_script),
            "--portserver_static_pool",
            args.portserver_pool,
            "--portserver_address",
            args.portserver_address,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    time.sleep(2)
    if PORTSERVER.poll() is not None:
        raise RuntimeError(f"portserver exited with status {PORTSERVER.returncode}")


def prune_heavy_training_state(train_root: Path) -> list[str]:
    removed = []
    for name in ("ckpt", "replay"):
        target = train_root / "run" / name
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed


def run_one(args, run: RunSpec) -> bool:
    global PORTSERVER
    run_root = args.experiment_root / "runs" / run.run_name
    if run_root.exists():
        outcome = run_root / "outcome.json"
        if outcome.is_file() and json.loads(outcome.read_text()).get("completed"):
            return True
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    resolved = validate_profile(run)
    atomic_json(
        run_root / "manifest.json",
        {
            "algorithm_commit": ALGORITHM_COMMIT,
            "gpu": args.gpu,
            "slot_index": args.slot_index,
            "run": asdict(run),
            "resolved_configuration": resolved,
            "curve_evaluation": {"interval": 5000, "episodes": 64, "envs": 4},
            "final_evaluation": {"episodes": 128, "envs": 4},
        },
    )
    outcome = run_root / "outcome.json"
    try:
        start_portserver(args, run_root)
        train_root = run_root / "train"
        run_child(
            args,
            run,
            train_root,
            "train",
            train_command(args, run, train_root / "run"),
        )
        checkpoint = latest_checkpoint(train_root, run.steps)
        final_root = run_root / "final128"
        run_child(
            args,
            run,
            final_root,
            "final128",
            eval_command(args, run, final_root / "run", checkpoint),
        )
        summary = json.loads(
            (final_root / "run" / "evaluation_summary.json").read_text()
        )
        if summary.get("evaluation_protocol", {}).get("episodes") != 128:
            raise RuntimeError("fixed-128 evaluation protocol was not completed")
        removed = prune_heavy_training_state(train_root)
        atomic_json(
            outcome,
            {
                "completed": True,
                "heavy_training_state_pruned": removed,
                "summary": summary,
            },
        )
        return True
    except Exception as error:
        atomic_json(outcome, {"completed": False, "error": repr(error)})
        return False
    finally:
        stop_process(PORTSERVER)
        PORTSERVER = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--sc2", type=Path, required=True)
    parser.add_argument("--portserver-script", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--slot-index", type=int, choices=tuple(SLOT_QUEUES), required=True)
    parser.add_argument("--portserver-address", required=True)
    parser.add_argument("--portserver-pool", required=True)
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-project", default="majepa-annealed-smac-suite")
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.source = args.source.resolve()
    args.python = args.python.absolute()
    args.external = args.external.resolve()
    args.sc2 = args.sc2.resolve()
    args.portserver_script = args.portserver_script.resolve()

    validate_source(args)
    queue = SLOT_QUEUES[args.slot_index]
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.experiment_root / f"slot-{args.slot_index}.json",
        {
            "algorithm_commit": ALGORITHM_COMMIT,
            "gpu": args.gpu,
            "queue": [asdict(run) for run in queue],
            "slot_index": args.slot_index,
        },
    )
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    results = []
    for run in queue:
        try:
            completed = run_one(args, run)
            results.append({"run": asdict(run), "completed": completed})
        except Exception as error:
            results.append({"run": asdict(run), "completed": False, "error": repr(error)})
        atomic_json(args.experiment_root / f"slot-{args.slot_index}-outcome.json", results)


if __name__ == "__main__":
    main()
