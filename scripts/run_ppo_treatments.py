#!/usr/bin/env python3
"""Run one fail-fast GPU queue for the MA-JEPA PPO treatment study."""

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


SEED = 0
STEPS = 50_000
HORIZON = 5
LOGGER_FILTER = (
    "score|return|length|fps|ratio|sample_age|replay/behavior_|schedule/|"
    "counters/|replay_views/|train/loss/|train/opt/|train/ppo/|train/ctde/|"
    "report/ctde/|central_critic/|battle_won|win_rate|legacy_|corrected_|"
    "enemy_|ally_|timeout|action_|attack_target_|eval/|episode/|final_eval/"
)
CURRENT: subprocess.Popen[str] | None = None
PORTSERVER: subprocess.Popen[str] | None = None


@dataclass(frozen=True)
class RunSpec:
    treatment: str
    map_name: str
    num_agents: int
    seed: int = SEED
    steps: int = STEPS
    horizon: int = HORIZON

    @property
    def task(self) -> str:
        return f"smac_{self.map_name}"

    @property
    def run_name(self) -> str:
        return f"{self.treatment}-{self.map_name}-seed{self.seed}"


def spec(treatment: str, map_name: str) -> RunSpec:
    return RunSpec(treatment, map_name, 3 if map_name == "3m" else 5)


# Treatments in each slot run sequentially. Each successor starts only after
# the preceding training run and its fixed final evaluation complete.
SLOT_QUEUES = {
    0: (spec("base", "3m"), spec("actor2x256_nopeerres", "3m")),
    1: (spec("base", "2s3z"), spec("actor2x256_nopeerres", "2s3z")),
    2: (spec("env16_total50k", "3m"), spec("critic1l8h", "3m")),
    3: (spec("env16_total50k", "2s3z"), spec("critic1l8h", "2s3z")),
    4: (spec("wm_start5k_nowarmup", "3m"), spec("entropy_annealed", "3m")),
    5: (spec("wm_start5k_nowarmup", "2s3z"), spec("entropy_annealed", "2s3z")),
    6: (spec("wm_start5k_env16_total50k", "2s3z"),),
}


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
    if not marker.is_file():
        raise RuntimeError("deployed source is missing DEPLOYED_COMMIT")
    args.algorithm_commit = marker.read_text().strip()
    if len(args.algorithm_commit) != 40:
        raise RuntimeError("DEPLOYED_COMMIT must contain a full Git commit")
    for path in (args.python, args.external, args.sc2, args.portserver_script):
        if not path.exists():
            raise FileNotFoundError(path)


def treatment_flags(run: RunSpec) -> tuple[str, ...]:
    common = (
        "--agent.num_agents",
        str(run.num_agents),
        "--agent.imag_length",
        str(run.horizon),
    )
    treatments = {
        "base": (),
        "wm_start5k_nowarmup": (
            "--run.world_model_start_step",
            "5000",
            "--agent.opt.warmup",
            "0",
            "--agent.marl.ctde.opt.warmup",
            "0",
        ),
        "env16_total50k": ("--run.envs", "16"),
        "wm_start5k_env16_total50k": (
            "--run.envs",
            "16",
            "--run.world_model_start_step",
            "5000",
            "--agent.opt.warmup",
            "0",
            "--agent.marl.ctde.opt.warmup",
            "0",
        ),
        "actor2x256_nopeerres": (
            "--agent.policy.layers",
            "2",
            "--agent.policy.units",
            "256",
            "--agent.marl.ctde.teammate_belief.actor_residual",
            "False",
        ),
        "entropy_annealed": (
            "--agent.collection_unimix",
            "0.0",
            "--agent.ppo.entropy_schedule.enabled",
            "True",
            "--agent.ppo.entropy_schedule.initial",
            "0.001",
            "--agent.ppo.entropy_schedule.final",
            "0.0003",
            "--agent.ppo.entropy_schedule.decay_steps",
            "40000",
            "--agent.ppo.entropy_schedule.schedule",
            "cosine",
            "--agent.ppo.entropy_schedule.normalize",
            "True",
        ),
        "critic1l8h": (
            "--agent.marl.ctde.critic.layers",
            "1",
            "--agent.marl.ctde.critic.heads",
            "8",
        ),
    }
    if run.treatment not in treatments:
        raise ValueError(f"unknown treatment: {run.treatment}")
    return (*common, *treatments[run.treatment])


def validate_profile(run: RunSpec) -> dict[str, object]:
    config = _resolve_config_profiles(_load_configs(), ("smac_vector", "ma_jepa"))
    parsed = elements.Flags(config).parse(
        [
            "--task",
            run.task,
            "--seed",
            str(run.seed),
            "--run.steps",
            str(run.steps),
            *treatment_flags(run),
        ]
    )
    belief = parsed.agent.marl.ctde.teammate_belief
    critic = parsed.agent.marl.ctde.critic
    entropy = parsed.agent.ppo.entropy_schedule
    resolved = {
        "task": str(parsed.task),
        "seed": int(parsed.seed),
        "steps": int(parsed.run.steps),
        "train_ratio": float(parsed.run.train_ratio),
        "training_envs": int(parsed.run.envs),
        "world_model_start_step": int(parsed.run.world_model_start_step),
        "ppo_start_step": int(parsed.run.ppo_start_step),
        "imag_length": int(parsed.agent.imag_length),
        "num_agents": int(parsed.agent.num_agents),
        "local_world_warmup": int(parsed.agent.opt.warmup),
        "joint_world_warmup": int(parsed.agent.marl.ctde.opt.warmup),
        "actor_layers": int(parsed.agent.policy.layers),
        "actor_units": int(parsed.agent.policy.units),
        "teammate_belief": bool(belief.enabled),
        "teammate_actor_residual": bool(belief.actor_residual),
        "critic_width": int(critic.width),
        "critic_layers": int(critic.layers),
        "critic_heads": int(critic.heads),
        "collection_legal_unimix": float(parsed.agent.collection_unimix),
        "policy_unimix": float(parsed.agent.policy.unimix),
        "fixed_entropy_coefficient": float(parsed.agent.ppo.entropy_coefficient),
        "entropy_schedule_enabled": bool(entropy.enabled),
        "entropy_initial": float(entropy.initial),
        "entropy_final": float(entropy.final),
        "entropy_decay_steps": int(entropy.decay_steps),
        "entropy_schedule": str(entropy.schedule),
        "entropy_normalize": bool(entropy.normalize),
    }
    expected = {
        "task": run.task,
        "seed": run.seed,
        "steps": run.steps,
        "train_ratio": 128.0,
        "training_envs": 1,
        "world_model_start_step": 0,
        "ppo_start_step": 5000,
        "imag_length": run.horizon,
        "num_agents": run.num_agents,
        "local_world_warmup": 1000,
        "joint_world_warmup": 1000,
        "actor_layers": 3,
        "actor_units": 1024,
        "teammate_belief": True,
        "teammate_actor_residual": True,
        "critic_width": 256,
        "critic_layers": 2,
        "critic_heads": 4,
        "collection_legal_unimix": 0.05,
        "policy_unimix": 0.01,
        "fixed_entropy_coefficient": 0.01,
        "entropy_schedule_enabled": False,
        "entropy_initial": 0.001,
        "entropy_final": 0.0003,
        "entropy_decay_steps": 40_000,
        "entropy_schedule": "cosine",
        "entropy_normalize": True,
    }
    if run.treatment == "wm_start5k_nowarmup":
        expected.update(
            world_model_start_step=5000,
            local_world_warmup=0,
            joint_world_warmup=0,
        )
    elif run.treatment == "env16_total50k":
        expected.update(training_envs=16)
    elif run.treatment == "wm_start5k_env16_total50k":
        expected.update(
            training_envs=16,
            world_model_start_step=5000,
            local_world_warmup=0,
            joint_world_warmup=0,
        )
    elif run.treatment == "actor2x256_nopeerres":
        expected.update(
            actor_layers=2,
            actor_units=256,
            teammate_actor_residual=False,
        )
    elif run.treatment == "entropy_annealed":
        expected.update(
            collection_legal_unimix=0.0,
            entropy_schedule_enabled=True,
        )
    elif run.treatment == "critic1l8h":
        expected.update(critic_layers=1, critic_heads=8)
    if resolved != expected:
        raise RuntimeError(f"treatment profile mismatch: {resolved} != {expected}")
    return resolved


def environment(args, run: RunSpec, phase_root: Path, phase: str) -> dict[str, str]:
    short = args.algorithm_commit[:4]
    identifier = f"mppo-{short}-{run.treatment}-{run.map_name}-s{run.seed}"
    if phase == "final128":
        identifier = f"f-{identifier}"
    env = os.environ.copy()
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
        WANDB_RUN_GROUP=f"ma-jepa-ppo-treatments-{args.algorithm_commit[:7]}",
        WANDB_JOB_TYPE=phase,
        WANDB_NAME=identifier,
        WANDB_RUN_ID=identifier,
        WANDB_RESUME="never",
        WANDB_MODE="online",
        WANDB_NOTES=(
            f"MA-JEPA PPO treatment {run.treatment} at "
            f"{args.algorithm_commit}; {run.map_name} H{run.horizon} "
            f"seed {run.seed}, 50k total environment transitions."
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
        "32",
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
    args,
    run: RunSpec,
    logdir: Path,
    checkpoint: Path,
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
    args,
    run: RunSpec,
    phase_root: Path,
    phase: str,
    command: list[str],
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
    output = (run_root / "portserver.log").open("x")
    PORTSERVER = subprocess.Popen(
        [
            str(args.python),
            str(args.portserver_script),
            "--portserver_static_pool",
            args.portserver_pool,
            "--portserver_address",
            args.portserver_address,
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    time.sleep(2)
    if PORTSERVER.poll() is not None:
        raise RuntimeError(f"portserver exited with status {PORTSERVER.returncode}")


def prune_replay(train_root: Path) -> str | None:
    replay = train_root / "run" / "replay"
    if not replay.is_dir():
        return None
    shutil.rmtree(replay)
    return str(replay)


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
            "algorithm_commit": args.algorithm_commit,
            "gpu": args.gpu,
            "slot_index": args.slot_index,
            "run": asdict(run),
            "resolved_configuration": resolved,
            "death_masking": "mandatory_present_and_controllable",
            "curve_evaluation": {"interval": 5000, "episodes": 32, "envs": 4},
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
        removed_replay = prune_replay(train_root)
        atomic_json(
            outcome,
            {
                "completed": True,
                "replay_pruned": removed_replay,
                "checkpoint_preserved": str(checkpoint),
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
    parser.add_argument(
        "--slot-index", type=int, choices=tuple(SLOT_QUEUES), required=True
    )
    parser.add_argument("--portserver-address", required=True)
    parser.add_argument("--portserver-pool", required=True)
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-project", default="majepa-ppo-treatments")
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
            "algorithm_commit": args.algorithm_commit,
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
            completed = False
            results.append(
                {"run": asdict(run), "completed": False, "error": repr(error)}
            )
        atomic_json(
            args.experiment_root / f"slot-{args.slot_index}-outcome.json", results
        )
        if not completed:
            break


if __name__ == "__main__":
    main()
