#!/usr/bin/env python3
"""Run an isolated PPO correction comparison after an assigned GPU is free.

This runner never stops another experiment. One process owns one assigned GPU
and one result directory. It waits for the preceding experiment's supervisors
and all GPU compute processes, then performs training and fixed final evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import time


@dataclass(frozen=True)
class RunSpec:
    arm: str
    map_name: str
    seed: int
    replay_value_scale: float | None

    @property
    def num_agents(self):
        return {"2s3z": 5, "3m": 3}[self.map_name]

    @property
    def name(self):
        return f"{self.arm}-{self.map_name}-seed{self.seed}"


SLOTS = {
    0: RunSpec("team_return", "2s3z", 0, 0.0),
    1: RunSpec("team_return_anchor", "2s3z", 0, 0.3),
    2: RunSpec("team_return", "2s3z", 1, 0.0),
    3: RunSpec("team_return_anchor", "2s3z", 1, 0.3),
    4: RunSpec("team_return_anchor", "3m", 0, 0.3),
    5: RunSpec("original_base", "2s3z", 1, None),
}

LOGGER_FILTER = (
    "score|return|length|fps|ratio|sample_age|replay/behavior_|schedule/|"
    "counters/|replay_views/|train/loss/|train/opt/|train/ppo/|train/ctde/|"
    "report/ctde/|central_critic/|battle_won|win_rate|legacy_|corrected_|"
    "enemy_|ally_|timeout|action_|attack_target_|eval/|episode/|final_eval/"
)


def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def source_fingerprint(source):
    """Content hash of executable package/config files, independent of mtimes."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in (source / "src" / "majepa").rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".yml"}
        and not path.name.startswith("._")
    )
    if not files:
        raise ValueError(f"No MA-JEPA package files in {source}")
    for path in files:
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def common_command(args, run, logdir):
    command = [
        str(args.python),
        "-m",
        "majepa.main",
        "--logdir",
        str(logdir),
        "--configs",
        "smac_vector",
        "ma_jepa",
        "--task",
        f"smac_{run.map_name}",
        "--seed",
        str(run.seed),
        "--agent.num_agents",
        str(run.num_agents),
        "--agent.imag_length",
        "5",
    ]
    if run.replay_value_scale is not None:
        command += ["--agent.ppo.replay_value_scale", str(run.replay_value_scale)]
    return command


def logger_outputs(args):
    return ["jsonl", "wandb"] if getattr(args, "wandb_project", "") else ["jsonl"]


def train_command(args, run, logdir):
    return common_command(args, run, logdir) + [
        "--script",
        "train",
        "--run.steps",
        "50000",
        "--run.envs",
        "1",
        "--run.world_model_start_step",
        "0",
        "--run.ppo_start_step",
        "5000",
        "--run.train_ratio",
        "128",
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
        *logger_outputs(args),
        "--logger.filter",
        LOGGER_FILTER,
    ]


def eval_command(args, run, logdir, checkpoint):
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
        *logger_outputs(args),
        "--logger.filter",
        LOGGER_FILTER,
    ]


def execution_environment(args):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PORTSERVER_ADDRESS=args.portserver_address,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=f"{args.source}/src:{args.external}",
        SC2PATH=str(args.sc2),
        WANDB_MODE="online" if getattr(args, "wandb_project", "") else "disabled",
    )
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_FORK_FROM", "WANDB_RESUME"):
        env.pop(key, None)
    return env


def phase_environment(args, phase_root, phase, env):
    env = env.copy()
    if getattr(args, "wandb_project", ""):
        run = SLOTS[args.slot]
        env.update(
            WANDB_ENTITY=args.wandb_entity,
            WANDB_PROJECT=args.wandb_project,
            WANDB_RUN_GROUP=args.wandb_group,
            WANDB_RUN_ID=f"{args.wandb_run_prefix}-s{args.slot}-{phase}",
            WANDB_NAME=f"{args.wandb_run_prefix}-{run.name}-{phase}",
            WANDB_JOB_TYPE=phase,
            WANDB_DIR=str(phase_root),
            WANDB_RESUME="never",
            WANDB_NOTES=(
                f"Correction screen {run.name}; package SHA256 "
                f"{args.expected_source_sha256}; 50k total environment transitions."
            ),
        )
    return env


def gpu_processes(gpu):
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        int(row.split(",")[1].strip())
        for row in rows.splitlines()
        if row.split(",")[0].strip() == uuid
    ]


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_gpu(args, run_root):
    idle_since = None
    while True:
        predecessor_pids = [pid for pid in args.wait_pid if pid_alive(pid)]
        compute_pids = gpu_processes(args.gpu)
        now = time.monotonic()
        idle_since = None if predecessor_pids or compute_pids else idle_since or now
        atomic_json(
            run_root / "status.json",
            {
                "status": "waiting_for_gpu",
                "gpu": args.gpu,
                "predecessor_pids": predecessor_pids,
                "compute_pids": compute_pids,
                "idle_seconds": 0 if idle_since is None else now - idle_since,
                "updated_at": time.time(),
            },
        )
        if idle_since is not None and now - idle_since >= args.idle_seconds:
            return
        time.sleep(args.poll_seconds)


def latest_checkpoint(train_root):
    root = train_root / "run" / "ckpt"
    checkpoint = root / (root / "latest").read_text().strip()
    if not (checkpoint / "done").is_file():
        raise RuntimeError("Final checkpoint is incomplete")
    with (checkpoint / "step.pkl").open("rb") as file:
        step = int(pickle.load(file))
    if step != 50000:
        raise RuntimeError(f"Expected final checkpoint at 50000, got {step}")
    return checkpoint


def validate_profile(args, run, env):
    """Validate configuration with the chosen source/runtime, before GPU waiting."""
    flags = common_command(args, run, Path("unused"))[3:]
    # common_command starts [python, -m, majepa.main]; parse only config flags.
    code = """
import json, pathlib, sys
import elements, majepa
from majepa.main import _load_configs, _resolve_config_profiles
expected=pathlib.Path(sys.argv[1]).resolve()
if not pathlib.Path(majepa.__file__).resolve().is_relative_to(expected):
    raise RuntimeError('Package import did not resolve to the selected snapshot')
parsed,other=elements.Flags(configs=['smac_vector','ma_jepa']).parse_known(json.loads(sys.argv[2]))
config=_resolve_config_profiles(_load_configs(), parsed.configs)
config=elements.Flags(config).parse(other)
print(json.dumps({
 'task':str(config.task),'seed':int(config.seed),'envs':int(config.run.envs),
 'train_ratio':float(config.run.train_ratio),'ppo_start_step':int(config.run.ppo_start_step),
 'world_model_start_step':int(config.run.world_model_start_step),
 'actor_layers':int(config.agent.policy.layers),'actor_units':int(config.agent.policy.units),
 'actor_lr':float(config.agent.ppo.actor_lr),'critic_lr':float(config.agent.ppo.critic_lr),
 'local_world_warmup':int(config.agent.opt.warmup),
 'joint_world_warmup':int(config.agent.marl.ctde.opt.warmup),
 'collection_unimix':float(config.agent.collection_unimix),
 'entropy_coefficient':float(config.agent.ppo.entropy_coefficient),
 'entropy_schedule_enabled':bool(config.agent.ppo.entropy_schedule.enabled),
 'replay_value_scale':float(config.agent.ppo.replay_value_scale) if hasattr(config.agent.ppo,'replay_value_scale') else None,
}))
"""
    output = subprocess.check_output(
        [str(args.python), "-c", code, str(args.source), json.dumps(flags)],
        env={**env, "JAX_PLATFORMS": "cpu"},
        cwd=args.source,
        text=True,
    )
    resolved = json.loads(output.strip().splitlines()[-1])
    expected = {
        "task": f"smac_{run.map_name}",
        "seed": run.seed,
        "envs": 1,
        "train_ratio": 128.0,
        "ppo_start_step": 5000,
        "world_model_start_step": 0,
        "actor_layers": 3,
        "actor_units": 1024,
        "actor_lr": 3e-5,
        "critic_lr": 3e-5,
        "local_world_warmup": 1000,
        "joint_world_warmup": 1000,
        "collection_unimix": 0.05,
        "entropy_coefficient": 0.01,
        "entropy_schedule_enabled": False,
        "replay_value_scale": run.replay_value_scale,
    }
    if resolved != expected:
        raise RuntimeError(
            f"Correction screen configuration mismatch: {resolved} != {expected}"
        )
    return resolved


class OwnedChildren:
    """Only processes started here are eligible for cleanup."""

    def __init__(self):
        self.processes = []

    def start(self, *args, **kwargs):
        process = subprocess.Popen(*args, **kwargs, start_new_session=True)
        self.processes.append(process)
        return process

    def close(self):
        for process in reversed(self.processes):
            if process.poll() is not None:
                continue
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run_child(args, children, run_root, phase, command, env):
    phase_root = run_root / phase
    phase_root.mkdir()
    env = phase_environment(args, phase_root, phase, env)
    atomic_json(
        phase_root / "launch.json", {"command": command, "started_at": time.time()}
    )
    atomic_json(run_root / "status.json", {"status": phase, "updated_at": time.time()})
    with (phase_root / "launch.log").open("x") as log:
        process = children.start(
            command,
            cwd=args.source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (phase_root / "pid").write_text(f"{process.pid}\n")
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{phase} exited with status {returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, choices=SLOTS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--sc2", type=Path, required=True)
    parser.add_argument("--portserver-script", type=Path, required=True)
    parser.add_argument("--portserver-address", required=True)
    parser.add_argument("--portserver-pool", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--idle-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--wandb-run-prefix", default="")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for field in ("source", "python", "external", "sc2", "portserver_script"):
        path = getattr(args, field).absolute()
        if not path.exists():
            raise FileNotFoundError(path)
        setattr(args, field, path)
    if args.idle_seconds < 0 or not 0 < args.poll_seconds <= 60:
        raise ValueError("Require idle_seconds >= 0 and poll_seconds in (0, 60]")
    if args.wandb_project and (
        not args.wandb_group
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,45}", args.wandb_run_prefix)
    ):
        raise ValueError("W&B logging requires a group and a 1–45 character run prefix")
    actual = source_fingerprint(args.source)
    if actual != args.expected_source_sha256:
        raise RuntimeError(f"Source fingerprint mismatch: {actual}")
    run = SLOTS[args.slot]
    env = execution_environment(args)
    resolved = validate_profile(args, run, env)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "source_sha256": actual,
                    "run": asdict(run),
                    "configuration": resolved,
                },
                sort_keys=True,
            )
        )
        return

    args.experiment_root = args.experiment_root.absolute()
    run_root = args.experiment_root / "runs" / run.name
    run_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        run_root / "manifest.json",
        {
            "run": asdict(run),
            "source": str(args.source),
            "source_sha256": actual,
            "source_commit": (args.source / "DEPLOYED_COMMIT").read_text().strip()
            if (args.source / "DEPLOYED_COMMIT").is_file()
            else None,
            "slot": args.slot,
            "gpu": args.gpu,
            "configuration": resolved,
            "wait_pids": args.wait_pid,
            "logging": {
                "outputs": logger_outputs(args),
                "wandb_entity": args.wandb_entity if args.wandb_project else None,
                "wandb_project": args.wandb_project or None,
                "wandb_group": args.wandb_group or None,
                "wandb_run_prefix": args.wandb_run_prefix or None,
            },
            "created_at": time.time(),
            "protocol": {
                "train_steps": 50000,
                "curve_interval": 5000,
                "curve_episodes": 32,
                "curve_seed_offset": 50000,
                "final_episodes": 128,
                "final_seed_offset": 100000,
            },
        },
    )
    children = OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        wait_for_gpu(args, run_root)
        if source_fingerprint(args.source) != actual:
            raise RuntimeError("Frozen source changed while waiting for GPU")
        with (run_root / "portserver.log").open("x") as portlog:
            portserver = children.start(
                [
                    str(args.python),
                    str(args.portserver_script),
                    "--portserver_static_pool",
                    args.portserver_pool,
                    "--portserver_address",
                    args.portserver_address,
                ],
                env=env,
                stdout=portlog,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time.sleep(2)
            if portserver.poll() is not None:
                raise RuntimeError("Portserver failed to start")
            run_child(
                args,
                children,
                run_root,
                "train",
                train_command(args, run, run_root / "train/run"),
                env,
            )
            checkpoint = latest_checkpoint(run_root / "train")
            run_child(
                args,
                children,
                run_root,
                "final128",
                eval_command(args, run, run_root / "final128/run", checkpoint),
                env,
            )
        summary = json.loads(
            (run_root / "final128/run/evaluation_summary.json").read_text()
        )
        if summary.get("evaluation_protocol", {}).get("episodes") != 128:
            raise RuntimeError("Fixed-128 evaluation was not completed")
        atomic_json(
            run_root / "outcome.json",
            {
                "completed": True,
                "checkpoint": str(checkpoint),
                "summary": summary,
                "finished_at": time.time(),
            },
        )
        atomic_json(
            run_root / "status.json", {"status": "complete", "updated_at": time.time()}
        )
    except BaseException as error:
        atomic_json(
            run_root / "outcome.json",
            {"completed": False, "error": repr(error), "finished_at": time.time()},
        )
        raise
    finally:
        children.close()


if __name__ == "__main__":
    main()
