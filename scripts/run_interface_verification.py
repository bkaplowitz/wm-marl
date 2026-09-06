#!/usr/bin/env python3
"""Six-GPU fixed-data factorial followed by paired fresh online verification."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_ppo_correction_screen as base
import run_ppo_recurrent_extensions as recurrent

SOURCE = Path(__file__).resolve().parents[1]
ROOT = Path("/workspace/majepa_interface_verify_20260906")
MATRIX = Path("/workspace/majepa_bptt2_matrix_20260905/runs")
BANK = Path("/workspace/majepa_seed_diagnostics_20260906")
PYTHON = Path("/workspace/majepa-runtime/bin/python")
GROUP = "ma-jepa-interface-verify-20260906"
URL = "https://wandb.ai/osaze-obahor/majepa-ppo-treatments"
MAPS = {"3s_vs_3z": 3, "3s_vs_4z": 3, "2s3z": 5, "MMM": 10}
ORIGINAL_TRAIN_COMMAND = base.train_command


@dataclass(frozen=True)
class RunSpec(recurrent.RunSpec):
    trajectory_kl_scale: float = 0.0
    steps: int = 50000
    final_episodes: int = 100

    @property
    def num_agents(self):
        return MAPS[self.map_name]

    @property
    def configuration_flags(self):
        return super().configuration_flags + (
            "--run.isolate_report_rng", "True",
            "--agent.ppo.actor_epochs", "5", "--agent.ppo.critic_epochs", "5",
            "--agent.marl.ctde.self_fed.trajectory_kl_scale", str(self.trajectory_kl_scale),
        )


def run_spec(arm, map_name, seed):
    return RunSpec(arm=arm, map_name=map_name, seed=seed, replay_value_scale=0.3,
                   imag_length=5, recurrent=True, self_fed_scale=0.1,
                   slowvalue_rate=1.0, envs=1, fresh_history=False, bptt_steps=2,
                   trajectory_kl_scale=0.1 if arm == "align" else 0.0)


def train_command(args, run, logdir):
    command = ORIGINAL_TRAIN_COMMAND(args, run, logdir)
    # Preserve the divergence history on the primary stability map for all seeds.
    if run.map_name == "3s_vs_3z":
        command[command.index("--run.checkpoint_at_curve_eval") + 1] = "True"
    return command


base.train_command = train_command


def run_id(run):
    return f"ifv-{run.arm}-{run.map_name}-s{run.seed}"


def job_args(run, gpu, root):
    return argparse.Namespace(slot=gpu, gpu=gpu, source=SOURCE,
        expected_source_sha256=(SOURCE / "SOURCE_SHA256").read_text().strip(),
        experiment_root=root, python=PYTHON, external=Path("/workspace/external/dreamerv3"),
        sc2=Path("/workspace/StarCraftII"),
        portserver_script=Path("/workspace/majepa-runtime/bin/portserver.py"),
        portserver_address=f"@majepa-interface-verify-g{gpu}",
        portserver_pool=f"{57000 + gpu * 500}-{57499 + gpu * 500}",
        wait_pid=[], idle_seconds=5, poll_seconds=5,
        wandb_project="majepa-ppo-treatments", wandb_entity="osaze-obahor",
        wandb_group=GROUP, wandb_run_prefix=run_id(run), wandb_job_id=run_id(run),
        validate_only=False, screen_label="Interface and coverage verification", run_spec=run)


def resolve(run, root):
    import elements
    from majepa.main import _load_configs, _resolve_config_profiles
    from smac.env.starcraft2.maps import get_map_params

    if get_map_params(run.map_name)["n_agents"] != run.num_agents:
        raise ValueError("Map roster mismatch")
    args = job_args(run, 0, root)
    result = {}
    for phase, command in {
        "train": train_command(args, run, Path("unused")),
        "final100": base.eval_command(args, run, Path("unused"), Path("checkpoint")),
    }.items():
        parsed, other = elements.Flags(configs=["smac_vector", "ma_jepa"]).parse_known(command[3:])
        c = elements.Flags(_resolve_config_profiles(_load_configs(), parsed.configs)).parse(other)
        sf = c.agent.marl.ctde.self_fed
        actual = dict(task=c.task, seed=c.seed, agents=c.agent.num_agents,
            recurrent=dict(sf), world_lr=c.agent.opt.lr, joint_lr=c.agent.marl.ctde.opt.lr,
            actor_lr=c.agent.ppo.actor_lr, critic_lr=c.agent.ppo.critic_lr,
            actor_layers=c.agent.policy.layers, actor_units=c.agent.policy.units,
            actor_epochs=c.agent.ppo.actor_epochs, critic_epochs=c.agent.ppo.critic_epochs,
            entropy=c.agent.ppo.entropy_coefficient, advantage=c.agent.ppo.advantage_mode,
            factual=c.agent.ppo.factual_value.enabled, critic_slowreg=c.agent.ppo.critic_slowreg,
            encoder_rate=c.agent.target_encoder.rate, critic_rate=c.agent.slowvalue.rate,
            imag_length=c.agent.imag_length, isolate_report_rng=c.run.isolate_report_rng,
            world_warmup=c.agent.opt.warmup, joint_warmup=c.agent.marl.ctde.opt.warmup)
        expected = dict(task=f"smac_{run.map_name}", seed=run.seed, agents=run.num_agents,
            recurrent=dict(enabled=True, horizons=(2, 4, 5), anchors=8, scale=0.1,
                consumer_kl_scale=0.0, trajectory_kl_scale=run.trajectory_kl_scale,
                fresh_history=False, bptt_steps=2),
            world_lr=4e-5, joint_lr=4e-5, actor_lr=3e-5, critic_lr=3e-5,
            actor_layers=3, actor_units=1024, actor_epochs=5, critic_epochs=5,
            entropy=0.01, advantage="batch", factual=False, critic_slowreg=0.0,
            encoder_rate=0.01, critic_rate=1.0, imag_length=5, isolate_report_rng=True,
            world_warmup=0, joint_warmup=0)
        if json.loads(json.dumps(actual)) != json.loads(json.dumps(expected)):
            raise ValueError(f"Resolved algorithm mismatch: {actual} != {expected}")
        if phase == "train":
            assert c.run.steps == 50000 and c.run.envs == 1
            assert c.run.world_model_start_step == c.run.ppo_start_step == 5000
            assert c.run.train_ratio == 128 and c.run.curve_eval_eps == 32
            assert c.run.curve_eval_interval == 5000
            assert c.run.checkpoint_at_curve_eval == (run.map_name == "3s_vs_3z")
        else:
            assert c.run.eval_eps == 100 and c.run.envs == 4
            assert c.run.eval_worker_offset == 100000 and c.run.eval_policy_mode == "eval"
        result[phase] = dict(config=c.flat, verified_algorithm=actual)
    return result


@contextmanager
def state_lock(root):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "queue.json").read_text())
        yield state
        state["updated_at"] = time.time()
        base.atomic_json(root / "queue.json", state)


def initialize(root):
    expected = (SOURCE / "SOURCE_SHA256").read_text().strip()
    if base.source_fingerprint(SOURCE) != expected:
        raise ValueError("Source changed before initialization")
    old = json.loads(Path("/workspace/majepa_value_sweep_20260906/queue.json").read_text())
    jobs = []
    for map_name, seed, strong in (("3s_vs_3z", 0, 1), ("3s_vs_3z", 2, 1), ("3s_vs_5z", 0, 2)):
        for coverage in ("own", "mixed"):
            for arm in ("base", "align"):
                name = f"off-{map_name}-s{seed}-{arm}-{coverage}"
                jobs.append(dict(index=len(jobs), kind="offline", name=name,
                    map=map_name, seed=seed, strong_seed=strong, arm=arm, coverage=coverage,
                    status="pending", wandb=f"{URL}/runs/ifv-{name}"))
    resolved = {}
    for map_name in MAPS:
        for seed in (0, 1, 2):
            for arm in ("base", "align"):
                run = run_spec(arm, map_name, seed)
                resolved[run.name] = resolve(run, root)
                jobs.append(dict(index=len(jobs), kind="online", name=run.name,
                    map=map_name, seed=seed, arm=arm, status="pending", spec=asdict(run),
                    wandb=f"{URL}/runs/{run_id(run)}-train",
                    wandb_final=f"{URL}/runs/{run_id(run)}-final100"))
    root.mkdir(exist_ok=False)
    (root / "workers").mkdir()
    (root / "offline").mkdir()
    base.atomic_json(root / "resolved.json", resolved)
    base.atomic_json(root / "queue.json", dict(source=str(SOURCE), source_sha256=expected,
        group=GROUP, created_at=time.time(), updated_at=time.time(), jobs=jobs,
        phase="offline", maximum_jobs=36, offline_jobs=12, online_jobs=24,
        new_environment_transitions=1200000, decision_deadline=old["decision_deadline"],
        claim_deadline=old["superseded_by"]["previous_claim_deadline"],
        protocol="Complete the fixed-data factorial, then all predeclared fresh paired seeds/maps. No winner selected from partial curves. Imported data never enters online jobs."))
    print(json.dumps(dict(root=str(root), initialized=len(jobs), group=GROUP)), flush=True)


def execute(children, command, log, env):
    with log.open("x") as stream:
        process = children.start(command, cwd=SOURCE, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        code = process.wait()
    if code:
        raise RuntimeError(f"Child exited {code}: {log}")


def offline_job(root, job, gpu, children):
    output = root / "offline" / job.get("output_name", job["name"])
    attempt = int(job.get("attempt", 0))
    wandb_id = job.get("wandb_id", f"ifv-{job['name']}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": f"{SOURCE}/src:/workspace/external/dreamerv3",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    run = MATRIX / f"bptt2-{job['map']}-seed{job['seed']}/train/run"
    if job.get("resume_stage") == "audit":
        status = json.loads((output / "status.json").read_text())
        if status["phase"] != "complete" or status["completed_updates"] != 500:
            raise ValueError("Cannot resume audits without a verified completed model")
    else:
        execute(children, [str(PYTHON), str(SOURCE / "scripts/verify_interface_offline.py"),
            "--run", str(run), "--own", str(BANK / f"calibration-{job['map']}-s{job['seed']}/episodes.npz"),
            "--strong", str(BANK / f"calibration-{job['map']}-s{job['strong_seed']}/episodes.npz"),
            "--output", str(output), "--arm", job["arm"], "--coverage", job["coverage"],
            "--updates", "500", "--wandb-id", wandb_id],
            root / "workers" / f"{job['name']}.a{attempt}.training.log", env)
    for phase, model_run in (("before", run), ("after", output / "run")):
        if (output / f"{phase}.json").exists():
            if len(json.loads((output / f"{phase}.json").read_text())["records"]) != 96:
                raise ValueError("Incomplete prior audit must be preserved before retry")
            continue
        execute(children, [str(PYTHON), str(SOURCE / "scripts/audit_causal_simulator.py"),
            "--run", str(model_run), "--replay", str(output / "data/heldout"),
            "--output", str(output / f"{phase}.json"), "--external", "/workspace/external/dreamerv3",
            "--platform", "cuda", "--roots", "96", "--max-episodes", "32", "--max-chunks", "2",
            "--horizons", "1", "2", "4", "5", "8", "--seed", "2718"],
            root / "workers" / f"{job['name']}.a{attempt}.{phase}.log", env)
    before, after = [json.loads((output / f"{phase}.json").read_text()) for phase in ("before", "after")]
    def identity(row):
        return row["episode"], row["source"], row["root"]
    if list(map(identity, before["records"])) != list(map(identity, after["records"])):
        raise ValueError("Held-out roots differed across pre/post intervention")
    result = {"before": before["summary"], "after": after["summary"],
              "training": json.loads((output / "status.json").read_text())}
    base.atomic_json(output / "result.json", result)
    import wandb
    remote_run = wandb.Api().run(f"osaze-obahor/majepa-ppo-treatments/{wandb_id}")
    remote_run.summary["heldout_before"] = result["before"]
    remote_run.summary["heldout_after"] = result["after"]
    remote_run.summary["verification_complete"] = True
    remote_run.summary.update()
    return dict(heldout_result=str(output / "result.json"))


def worker(root, gpu):
    lock = (root / "workers" / f"gpu{gpu}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "workers" / f"gpu{gpu}.pid").write_text(str(os.getpid()) + "\n")
    children = base.OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            if base.gpu_processes(gpu):
                time.sleep(10)
                continue
            with state_lock(root) as state:
                if time.time() >= state["claim_deadline"] or state.get("claims_closed"):
                    return
                offline = [j for j in state["jobs"] if j["kind"] == "offline" and j["status"] in ("pending", "running")]
                kind = "offline" if offline else "online"
                state["phase"] = kind
                pending = [j for j in state["jobs"] if j["kind"] == kind and j["status"] == "pending"]
                if kind == "online" and not state.get("storage_ready", False):
                    state["phase"] = "waiting_for_verified_storage_archive"
                    job = None
                elif not pending:
                    if not any(j["status"] == "running" for j in state["jobs"]):
                        state["phase"] = "complete"
                        return
                    job = None
                else:
                    job = pending[0]
                    job.update(status="running", gpu=gpu, worker_pid=os.getpid(), started_at=time.time())
                    job = dict(job)
            if job is None:
                time.sleep(15)
                continue
            print(json.dumps(dict(event="starting", **job)), flush=True)
            if base.source_fingerprint(SOURCE) != (SOURCE / "SOURCE_SHA256").read_text().strip():
                raise RuntimeError("Frozen source changed")
            try:
                if job["kind"] == "offline":
                    result = offline_job(root, job, gpu, children)
                else:
                    command = [str(PYTHON), str(Path(__file__).resolve()), "job", "--root", str(root),
                               "--index", str(job["index"]), "--gpu", str(gpu)]
                    execute(children, command, root / "workers" / f"{job['name']}.log", os.environ.copy())
                    result = json.loads((root / "runs" / job["name"] / "outcome.json").read_text())
                    if not result.get("completed"):
                        raise RuntimeError(str(result))
                with state_lock(root) as state:
                    state["jobs"][job["index"]].update(status="complete", finished_at=time.time(), result=result)
            except Exception as error:
                with state_lock(root) as state:
                    state["jobs"][job["index"]].update(status="failed", finished_at=time.time(), error=repr(error))
                print(json.dumps(dict(event="failed", name=job["name"], error=repr(error))), flush=True)
                # Stop this worker on a defect; preserve all outputs for diagnosis.
                return
    finally:
        children.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "worker", "job"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, choices=range(6), default=0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "init":
        initialize(args.root)
    elif args.mode == "worker":
        worker(args.root, args.gpu)
    else:
        state = json.loads((args.root / "queue.json").read_text())
        job = state["jobs"][args.index]
        run = RunSpec(**job["spec"])
        def profile(_args, _run, _env):
            return json.loads((args.root / "resolved.json").read_text())[run.name]
        base.run_screen(job_args(run, args.gpu, args.root), run, profile,
            extra_manifest=dict(verification=GROUP, queue_index=job["index"], fresh_online=True,
                                imported_experience=False))


if __name__ == "__main__":
    main()
