#!/usr/bin/env python3
"""Run one isolated 2s3z fix treatment through 50k and fixed-128 eval."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path

import elements

from majepa.main import _load_configs, _resolve_config_profiles


ALGORITHM_COMMIT = "3583b59dee145a1854c91e92e57063844e7ca500"
TREATMENTS = {
    "death_masking": ("--agent.marl.ctde.death_masking.enabled", "True"),
    "action_binding": (
        "--agent.marl.ctde.authoritative_action_binding.enabled",
        "True",
    ),
    "support_preserving": (
        "--agent.marl.ctde.support_preserving.enabled",
        "True",
    ),
    "entropy_keep_unimix": (
        "--agent.marl.ctde.death_masking.enabled",
        "True",
        "--agent.entropy_schedule.enabled",
        "True",
    ),
    "entropy_no_death_masking": (
        "--agent.entropy_schedule.enabled",
        "True",
    ),
    "entropy_no_collection": (
        "--agent.marl.ctde.death_masking.enabled",
        "True",
        "--agent.entropy_schedule.enabled",
        "True",
        "--agent.collection_unimix",
        "0.0",
    ),
    "entropy_no_action_unimix": (
        "--agent.marl.ctde.death_masking.enabled",
        "True",
        "--agent.entropy_schedule.enabled",
        "True",
        "--agent.collection_unimix",
        "0.0",
        "--agent.policy.unimix",
        "0.0",
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
        raise RuntimeError("deployed algorithm commit does not match treatment runner")
    for path in (args.python, args.external, args.sc2, args.portserver_script):
        if not path.exists():
            raise FileNotFoundError(path)


def validate_profile(args) -> dict[str, object]:
    config = _resolve_config_profiles(_load_configs(), ("smac_vector", "ma_jepa"))
    parsed = elements.Flags(config).parse(
        [
            "--task",
            "smac_2s3z",
            "--seed",
            str(args.seed),
            "--agent.num_agents",
            "5",
            "--agent.imag_length",
            "5",
            *TREATMENTS[args.treatment],
        ]
    )
    ctde = parsed.agent.marl.ctde
    selected = {
        "death_masking": bool(ctde.death_masking.enabled),
        "action_binding": bool(ctde.authoritative_action_binding.enabled),
        "support_preserving": bool(ctde.support_preserving.enabled),
    }
    legacy = {"death_masking", "action_binding", "support_preserving"}
    if args.treatment in legacy:
        expected = {name: name == args.treatment for name in legacy}
    else:
        expected = {
            "death_masking": args.treatment != "entropy_no_death_masking",
            "action_binding": False,
            "support_preserving": False,
        }
    if selected != expected:
        raise RuntimeError(f"treatment isolation failed: {selected} != {expected}")
    entropy = parsed.agent.entropy_schedule
    entropy_expected = args.treatment.startswith("entropy_")
    if bool(entropy.enabled) != entropy_expected:
        raise RuntimeError("entropy schedule does not match the selected treatment")
    expected_collection = (
        0.0
        if args.treatment in {"entropy_no_collection", "entropy_no_action_unimix"}
        else 0.05
    )
    expected_policy = 0.0 if args.treatment == "entropy_no_action_unimix" else 0.01
    if float(parsed.agent.collection_unimix) != expected_collection:
        raise RuntimeError("collection unimix does not match the selected treatment")
    if float(parsed.agent.policy.unimix) != expected_policy:
        raise RuntimeError("policy unimix does not match the selected treatment")
    if float(parsed.agent.dyn.parallel_transformer.unimix) != 0.01:
        raise RuntimeError("world-model latent unimix must remain at 0.01")
    return {
        **selected,
        "seed": int(parsed.seed),
        "imag_length": int(parsed.agent.imag_length),
        "collection_unimix": float(parsed.agent.collection_unimix),
        "policy_unimix": float(parsed.agent.policy.unimix),
        "world_model_unimix": float(parsed.agent.dyn.parallel_transformer.unimix),
        "entropy_schedule": {
            "enabled": bool(entropy.enabled),
            "initial": float(entropy.initial),
            "final": float(entropy.final),
            "decay_steps": int(entropy.decay_steps),
            "schedule": str(entropy.schedule),
            "normalize": bool(entropy.normalize),
        },
        "action_binding_anchors": int(ctde.authoritative_action_binding.anchors),
        "action_binding_margin": float(ctde.authoritative_action_binding.margin),
        "support_probability_floor": float(ctde.support_preserving.probability_floor),
    }


def environment(args, run_root: Path, phase: str) -> dict[str, str]:
    env = os.environ.copy()
    identifier = f"mfix-{ALGORITHM_COMMIT[:4]}-2s3z-{args.treatment}-s{args.seed}"
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
        WANDB_DIR=str(run_root),
        WANDB_ENTITY=args.wandb_entity,
        WANDB_PROJECT=args.wandb_project,
        WANDB_RUN_GROUP=f"ma-jepa-fixes-2s3z-isolated-{ALGORITHM_COMMIT[:7]}",
        WANDB_JOB_TYPE=phase,
        WANDB_NAME=identifier,
        WANDB_RUN_ID=identifier,
        WANDB_RESUME="never",
        WANDB_MODE="online",
        WANDB_NOTES=(
            f"Isolated {args.treatment} treatment at {ALGORITHM_COMMIT}; "
            f"2s3z H5 seed {args.seed}; 50k plus held-out fixed128."
        ),
    )
    env.pop("WANDB_FORK_FROM", None)
    return env


def common_command(args, logdir: Path) -> list[str]:
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
        "smac_2s3z",
        "--seed",
        str(args.seed),
        "--agent.num_agents",
        "5",
        "--agent.imag_length",
        "5",
        *TREATMENTS[args.treatment],
    ]


def train_command(args, logdir: Path) -> list[str]:
    return common_command(args, logdir) + [
        "--script",
        "train",
        "--run.steps",
        str(args.steps),
        "--run.save_every",
        "900",
        "--run.final_save",
        "True",
        "--run.checkpoint_at_curve_eval",
        "False",
        "--run.curve_eval_interval",
        "1000",
        "--run.curve_eval_eps",
        "16",
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


def eval_command(args, logdir: Path, checkpoint: Path) -> list[str]:
    return common_command(args, logdir) + [
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


def run_child(args, phase_root: Path, phase: str, command: list[str]) -> None:
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
            env=environment(args, phase_root, phase),
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


def start_portserver(args, experiment_root: Path) -> None:
    global PORTSERVER
    port_root = experiment_root / "portservers"
    port_root.mkdir(exist_ok=True)
    log = (port_root / f"{args.slot}.log").open("x")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--sc2", type=Path, required=True)
    parser.add_argument("--portserver-script", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--portserver-address", required=True)
    parser.add_argument("--portserver-pool", required=True)
    parser.add_argument("--treatment", choices=tuple(TREATMENTS), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--wandb-entity", default="osaze-obahor")
    parser.add_argument("--wandb-project", default="majepa-fixes-2s3z")
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.source = args.source.resolve()
    # Resolving a venv Python symlink selects the base interpreter and drops
    # the venv's site-packages. Keep the absolute symlink path intact.
    args.python = args.python.absolute()
    args.external = args.external.resolve()
    args.sc2 = args.sc2.resolve()
    args.portserver_script = args.portserver_script.resolve()

    validate_source(args)
    selected = validate_profile(args)
    run_root = (
        args.experiment_root / "runs" / (f"2s3z-{args.treatment}-seed{args.seed}-50k")
    )
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    atomic_json(
        run_root / "manifest.json",
        {
            "algorithm_commit": ALGORITHM_COMMIT,
            "treatment": args.treatment,
            "seed": args.seed,
            "gpu": args.gpu,
            "slot": args.slot,
            "resolved_treatment": selected,
        },
    )
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    outcome = run_root / "outcome.json"
    try:
        start_portserver(args, args.experiment_root)
        train_root = run_root / "train"
        run_child(args, train_root, "train", train_command(args, train_root / "run"))
        checkpoint = latest_checkpoint(train_root, args.steps)
        final_root = run_root / "final128"
        run_child(
            args,
            final_root,
            "final128",
            eval_command(args, final_root / "run", checkpoint),
        )
        summary = json.loads(
            (final_root / "run" / "evaluation_summary.json").read_text()
        )
        if summary.get("evaluation_protocol", {}).get("episodes") != 128:
            raise RuntimeError("fixed-128 evaluation protocol was not completed")
        atomic_json(outcome, {"completed": True, "summary": summary})
    except BaseException as error:
        atomic_json(outcome, {"completed": False, "error": repr(error)})
        raise
    finally:
        stop_process(PORTSERVER)


if __name__ == "__main__":
    main()
