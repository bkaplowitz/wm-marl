#!/usr/bin/env python3
"""Durable six-GPU BPTT2 matrix; all maps get seeds 0/1 before seed 2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_ppo_correction_screen as base
import run_ppo_recurrent_extensions as recurrent


SOURCE = Path("/workspace/ma_jepa_recurrent_extensions_f2273e9")
SOURCE_HASH = "af6e6d23ab98e60d893071054942889ca73c14d0d9071505c4a9ee886df46b5a"
ROOT = Path("/workspace/majepa_bptt2_matrix_20260905")
PYTHON = Path("/workspace/majepa-runtime/bin/python")
GROUP = "ma-jepa-bptt2-matrix-20260905"
PROJECT_URL = "https://wandb.ai/osaze-obahor/majepa-ppo-treatments"
# Start the two largest action/agent shapes early, while four GPUs cover 50k maps.
MAPS = (
    ("3m", 3, 50000),
    ("8m", 8, 50000),
    ("2s3z", 5, 50000),
    ("MMM", 10, 50000),
    ("2c_vs_64zg", 2, 200000),
    ("MMM2", 10, 200000),
    ("3s_vs_3z", 3, 50000),
    ("3s_vs_4z", 3, 50000),
    ("so_many_baneling", 7, 50000),
    ("2s_vs_1sc", 2, 50000),
    ("2m_vs_1z", 2, 50000),
    ("3s_vs_5z", 3, 200000),
    ("5m_vs_6m", 5, 200000),
    ("corridor", 6, 200000),
)


@dataclass(frozen=True)
class RunSpec(recurrent.RunSpec):
    steps: int = 50000
    agents: int = 5
    final_episodes: int = 100

    @property
    def num_agents(self):
        return self.agents


def run_spec(index):
    if not 0 <= index < 42:
        raise ValueError(index)
    name, agents, steps = MAPS[index % 14]
    return RunSpec(
        arm="bptt2",
        map_name=name,
        seed=index // 14,
        replay_value_scale=0.3,
        imag_length=5,
        recurrent=True,
        self_fed_scale=0.1,
        slowvalue_rate=1.0,
        envs=1,
        fresh_history=False,
        bptt_steps=2,
        steps=steps,
        agents=agents,
    )


def job_id(run):
    return f"bptt2-20260905-{run.map_name}-s{run.seed}"


def job_args(index, gpu, root):
    run = run_spec(index)
    return argparse.Namespace(
        slot=gpu,
        gpu=gpu,
        source=SOURCE,
        expected_source_sha256=SOURCE_HASH,
        experiment_root=root,
        python=PYTHON,
        external=Path("/workspace/external/dreamerv3"),
        sc2=Path("/workspace/StarCraftII"),
        portserver_script=Path("/workspace/majepa-runtime/bin/portserver.py"),
        portserver_address=f"@majepa-bptt2-matrix-g{gpu}",
        portserver_pool=f"{45000 + gpu * 500}-{45499 + gpu * 500}",
        wait_pid=[],
        idle_seconds=5,
        poll_seconds=5,
        wandb_project="majepa-ppo-treatments",
        wandb_entity="osaze-obahor",
        wandb_group=GROUP,
        wandb_run_prefix=job_id(run),
        wandb_job_id=job_id(run),
        validate_only=False,
        screen_label="BPTT2 multi-map matrix",
        run_spec=run,
    )


def resolve_configuration(args, run):
    import elements
    import majepa
    from majepa.main import _load_configs, _resolve_config_profiles
    from smac.env.starcraft2.maps import get_map_params

    if not Path(majepa.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError("Package import escaped the selected snapshot")
    if get_map_params(run.map_name)["n_agents"] != run.num_agents:
        raise RuntimeError("Map agent count mismatch")
    if not (args.sc2 / "Maps/SMAC_Maps" / f"{run.map_name}.SC2Map").is_file():
        raise FileNotFoundError(run.map_name)
    out = {}
    commands = {
        "train": base.train_command(args, run, Path("unused")),
        "final100": base.eval_command(args, run, Path("unused"), Path("checkpoint")),
    }
    for phase, command in commands.items():
        parsed, other = elements.Flags(configs=["smac_vector", "ma_jepa"]).parse_known(
            command[3:]
        )
        c = elements.Flags(
            _resolve_config_profiles(_load_configs(), parsed.configs)
        ).parse(other)
        actual = dict(
            task=str(c.task),
            seed=int(c.seed),
            num_agents=int(c.agent.num_agents),
            imag_length=int(c.agent.imag_length),
            self_fed=json.loads(json.dumps(dict(c.agent.marl.ctde.self_fed))),
            slowvalue_rate=float(c.agent.slowvalue.rate),
            encoder_rate=float(c.agent.target_encoder.rate),
            world_lr=float(c.agent.opt.lr),
            joint_world_lr=float(c.agent.marl.ctde.opt.lr),
            world_warmup=int(c.agent.opt.warmup),
            joint_world_warmup=int(c.agent.marl.ctde.opt.warmup),
            actor_layers=int(c.agent.policy.layers),
            actor_units=int(c.agent.policy.units),
            actor_lr=float(c.agent.ppo.actor_lr),
            critic_lr=float(c.agent.ppo.critic_lr),
            replay_value_scale=float(c.agent.ppo.replay_value_scale),
        )
        expected = dict(
            task=f"smac_{run.map_name}",
            seed=run.seed,
            num_agents=run.num_agents,
            imag_length=5,
            self_fed=dict(
                enabled=True,
                horizons=[2, 4, 5],
                anchors=8,
                scale=0.1,
                consumer_kl_scale=0.0,
                fresh_history=False,
                bptt_steps=2,
            ),
            slowvalue_rate=1.0,
            encoder_rate=0.01,
            world_lr=4e-5,
            joint_world_lr=4e-5,
            world_warmup=0,
            joint_world_warmup=0,
            actor_layers=3,
            actor_units=1024,
            actor_lr=3e-5,
            critic_lr=3e-5,
            replay_value_scale=0.3,
        )
        if phase == "train":
            actual.update(
                steps=int(c.run.steps),
                envs=int(c.run.envs),
                train_ratio=float(c.run.train_ratio),
                prefill=int(c.run.world_model_start_step),
                ppo_start=int(c.run.ppo_start_step),
                curve_interval=int(c.run.curve_eval_interval),
                curve_eps=int(c.run.curve_eval_eps),
                final_save=bool(c.run.final_save),
                curve_seed_offset=int(c.run.curve_eval_seed_offset),
            )
            expected.update(
                steps=run.steps,
                envs=1,
                train_ratio=128.0,
                prefill=5000,
                ppo_start=5000,
                curve_interval=5000,
                curve_eps=32,
                final_save=True,
                curve_seed_offset=50000,
            )
        else:
            actual.update(
                envs=int(c.run.envs),
                episodes=int(c.run.eval_eps),
                seed_offset=int(c.run.eval_worker_offset),
                policy_mode=str(c.run.eval_policy_mode),
            )
            expected.update(
                envs=4, episodes=100, seed_offset=100000, policy_mode="eval"
            )
        if actual != expected:
            raise RuntimeError(f"{run.name}/{phase}: {actual} != {expected}")
        out[phase] = actual
    return out


def validate_profile(args, run, env):
    index = run.seed * 14 + [row[0] for row in MAPS].index(run.map_name)
    output = subprocess.check_output(
        [str(PYTHON), str(Path(__file__).resolve()), "check", "--index", str(index)],
        env={**env, "JAX_PLATFORMS": "cpu"},
        cwd=SOURCE,
        text=True,
    )
    return {"phases": json.loads(output.strip().splitlines()[-1])}


@contextmanager
def queue_state(root):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "queue.json").read_text())
        yield state
        state["updated_at"] = time.time()
        base.atomic_json(root / "queue.json", state)


def initialize(root):
    if base.source_fingerprint(SOURCE) != SOURCE_HASH:
        raise RuntimeError("Source fingerprint mismatch")
    validations = {}
    jobs = []
    for index in range(42):
        run = run_spec(index)
        validations[run.name] = resolve_configuration(job_args(index, 0, root), run)
        jobs.append(
            dict(
                index=index,
                name=run.name,
                map=run.map_name,
                seed=run.seed,
                steps=run.steps,
                stage=int(run.seed == 2),
                status="pending",
                wandb_train=f"{PROJECT_URL}/runs/{job_id(run)}-train",
                wandb_final=f"{PROJECT_URL}/runs/{job_id(run)}-final100",
            )
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "workers").mkdir()
    base.atomic_json(root / "validated_configurations.json", validations)
    base.atomic_json(
        root / "queue.json",
        dict(
            source=str(SOURCE),
            source_sha256=SOURCE_HASH,
            jobs=jobs,
            created_at=time.time(),
            wandb_group=GROUP,
            rule="All 28 seed-0/1 jobs finish before any seed-2 job starts; failed jobs pause seed 2.",
        ),
    )
    print(
        json.dumps(
            dict(
                initialized=len(jobs),
                first_stage=28,
                third_seed=14,
                steps=sum(j["steps"] for j in jobs),
                root=str(root),
            )
        ),
        flush=True,
    )


def claim(root, gpu):
    with queue_state(root) as state:
        # Explicit supervisor handoff preserves the already running training PID.
        for job in state["jobs"]:
            if job["status"] != "running" or job.get("gpu") != gpu:
                continue
            if job.get("handoff") and process_identity(job["worker_pid"]) is None:
                job.update(worker_pid=os.getpid(), adopted_at=time.time())
                return dict(job)
        first = [j for j in state["jobs"] if j["stage"] == 0]
        stage = 1 if all(j["status"] == "complete" for j in first) else 0
        candidates = [
            j for j in state["jobs"] if j["stage"] == stage and j["status"] == "pending"
        ]
        if candidates:
            job = candidates[0]
            job.update(
                status="running",
                gpu=gpu,
                worker_pid=os.getpid(),
                started_at=time.time(),
            )
            return dict(job)
        if all(j["status"] in ("complete", "failed") for j in state["jobs"]):
            return "done"
        return None


def worker(root, gpu):
    lock = (root / "workers" / f"gpu{gpu}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "workers" / f"gpu{gpu}.pid").write_text(f"{os.getpid()}\n")
    children = base.OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            job = claim(root, gpu)
            if job == "done":
                return
            if job is None:
                time.sleep(15)
                continue
            print(json.dumps(dict(event="starting", **job)), flush=True)
            command = [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "adopt" if job.get("handoff") else "job",
                "--root",
                str(root),
                "--index",
                str(job["index"]),
                "--gpu",
                str(gpu),
            ]
            try:
                suffix = "-adopt" if job.get("handoff") else ""
                with (root / "workers" / f"{job['name']}{suffix}.log").open(
                    "x"
                ) as logfile:
                    process = children.start(
                        command, stdout=logfile, stderr=subprocess.STDOUT
                    )
                    with queue_state(root) as state:
                        state["jobs"][job["index"]]["supervisor_pid"] = process.pid
                    code = process.wait()
                outcome_path = root / "runs" / job["name"] / "outcome.json"
                outcome = (
                    json.loads(outcome_path.read_text())
                    if outcome_path.exists()
                    else {}
                )
                success = code == 0 and outcome.get("completed") is True
                with queue_state(root) as state:
                    state["jobs"][job["index"]].update(
                        status="complete" if success else "failed",
                        returncode=code,
                        finished_at=time.time(),
                        error=outcome.get("error"),
                    )
                print(
                    json.dumps(
                        dict(event="finished", name=job["name"], success=success)
                    ),
                    flush=True,
                )
            except BaseException as error:
                with queue_state(root) as state:
                    state["jobs"][job["index"]].update(
                        status="failed", error=repr(error), finished_at=time.time()
                    )
                raise
    finally:
        children.close()
        lock.close()


def process_identity(pid):
    """Match Linux process start time as well as PID; zombies are finished."""
    try:
        root = Path(f"/proc/{pid}")
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return dict(
            pid=pid,
            start_ticks=int(fields[19]),
            command=(root / "cmdline").read_bytes().decode().split("\0")[:-1],
        )
    except FileNotFoundError:
        return None


def same_process(record):
    return process_identity(record["pid"]) == record


def stop_adopted(record):
    if not same_process(record):
        return
    os.killpg(record["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while same_process(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if same_process(record):
        os.killpg(record["pid"], signal.SIGKILL)


def adopt_training(root, index, gpu):
    """Replace only orchestration; the training process and its data stay intact."""
    with queue_state(root) as state:
        handoff = state["jobs"][index]["handoff"]
    args, run = job_args(index, gpu, root), run_spec(index)
    run_root = root / "runs" / run.name
    if base.source_fingerprint(SOURCE) != SOURCE_HASH:
        raise RuntimeError("Source fingerprint mismatch during handoff")
    env = base.execution_environment(args)
    resolved = validate_profile(args, run, env)
    manifest = json.loads((run_root / "manifest.json").read_text())
    manifest["configuration"] = resolved
    manifest["protocol"]["final_episodes"] = run.final_episodes
    manifest["run"]["final_episodes"] = run.final_episodes
    manifest["evaluation_amendment"] = dict(
        old_final_episodes=128,
        final_episodes=100,
        curve_episodes=32,
        training_pid_preserved=handoff["train"]["pid"],
        changed_at=time.time(),
        reason="User requested final-only 100-episode evaluation before any final evaluation began.",
    )
    base.atomic_json(run_root / "manifest.json", manifest)
    children = base.OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while same_process(handoff["train"]):
            time.sleep(5)
        checkpoint = base.latest_checkpoint(run_root / "train", run.steps)
        if not same_process(handoff["portserver"]):
            raise RuntimeError("Adopted portserver exited before final evaluation")
        phase = f"final{run.final_episodes}"
        base.run_child(
            args,
            children,
            run_root,
            phase,
            base.eval_command(args, run, run_root / phase / "run", checkpoint),
            env,
        )
        summary = json.loads(
            (run_root / phase / "run/evaluation_summary.json").read_text()
        )
        if summary.get("evaluation_protocol", {}).get("episodes") != run.final_episodes:
            raise RuntimeError("Fixed-100 evaluation was not completed")
        base.atomic_json(
            run_root / "outcome.json",
            dict(
                completed=True,
                checkpoint=str(checkpoint),
                final_phase=phase,
                summary=summary,
                finished_at=time.time(),
            ),
        )
        base.atomic_json(
            run_root / "status.json", dict(status="complete", updated_at=time.time())
        )
    except BaseException as error:
        base.atomic_json(
            run_root / "outcome.json",
            dict(
                completed=False,
                error=repr(error),
                finished_at=time.time(),
            ),
        )
        raise
    finally:
        children.close()
        stop_adopted(handoff["train"])
        stop_adopted(handoff["portserver"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("init", "check", "worker", "job", "adopt", "status")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, choices=range(6), default=0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "init":
        initialize(args.root)
    elif args.mode == "check":
        print(
            json.dumps(
                resolve_configuration(
                    job_args(args.index, args.gpu, args.root), run_spec(args.index)
                )
            )
        )
    elif args.mode == "worker":
        worker(args.root, args.gpu)
    elif args.mode == "adopt":
        adopt_training(args.root, args.index, args.gpu)
    elif args.mode == "job":
        base.run_screen(
            job_args(args.index, args.gpu, args.root),
            run_spec(args.index),
            profile_validator=validate_profile,
            extra_manifest=dict(
                matrix=GROUP,
                queue_index=args.index,
                seed_stage=int(args.index >= 28),
                wandb_job_id=job_id(run_spec(args.index)),
            ),
        )
    else:
        print((args.root / "queue.json").read_text())


if __name__ == "__main__":
    main()
