#!/usr/bin/env python3
"""Bounded, adaptive value-learning sweep on the existing six-GPU pod."""

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
ROOT = Path("/workspace/majepa_value_sweep_20260906")
MATRIX = Path("/workspace/majepa_bptt2_matrix_20260905/queue.json")
PYTHON = Path("/workspace/majepa-runtime/bin/python")
GROUP = "ma-jepa-value-sweep-20260906"
PROJECT_URL = "https://wandb.ai/osaze-obahor/majepa-ppo-treatments"
MAPS = {"2s3z": 5, "3s_vs_3z": 3, "3s_vs_4z": 3, "MMM": 10, "8m": 8}
ARMS = {
    "fact": dict(factual=True),
    "factrep": dict(factual=True, representation_scale=0.1),
    "base": {},
    "actor1": dict(actor_epochs=1),
    "fact-a1": dict(factual=True, actor_epochs=1),
    "frep-a1": dict(factual=True, representation_scale=0.1, actor_epochs=1),
    "entlow": dict(entropy=3e-4),
    "retperc": dict(advantage_mode="return_percentile"),
    "ret-ent": dict(advantage_mode="return_percentile", entropy=3e-4),
    "vreg": dict(critic_slowreg=1.0),
    "fact-vreg": dict(factual=True, critic_slowreg=1.0),
    "frep-ent": dict(factual=True, representation_scale=0.1, entropy=3e-4),
}


@dataclass(frozen=True)
class RunSpec(recurrent.RunSpec):
    factual: bool = False
    representation_scale: float = 0.0
    actor_epochs: int = 5
    critic_epochs: int = 5
    entropy: float = 0.01
    advantage_mode: str = "batch"
    critic_slowreg: float = 0.0
    steps: int = 50000
    final_episodes: int = 100

    @property
    def num_agents(self):
        return MAPS[self.map_name]

    @property
    def configuration_flags(self):
        changes = {
            "run.isolate_report_rng": True,
            "agent.ppo.factual_value.enabled": self.factual,
            "agent.ppo.factual_value.representation_scale": self.representation_scale,
            "agent.ppo.factual_value.rho_clip": 1.0,
            "agent.ppo.factual_value.c_clip": 1.0,
            "agent.ppo.actor_epochs": self.actor_epochs,
            "agent.ppo.critic_epochs": self.critic_epochs,
            "agent.ppo.entropy_coefficient": self.entropy,
            "agent.ppo.advantage_mode": self.advantage_mode,
            "agent.ppo.critic_slowreg": self.critic_slowreg,
        }
        return super().configuration_flags + tuple(
            part for key, value in changes.items() for part in (f"--{key}", str(value))
        )


def run_spec(arm, map_name, seed):
    return RunSpec(
        arm=arm,
        map_name=map_name,
        seed=seed,
        replay_value_scale=0.3,
        imag_length=5,
        recurrent=True,
        self_fed_scale=0.1,
        slowvalue_rate=1.0,
        envs=1,
        fresh_history=False,
        bptt_steps=2,
        **ARMS[arm],
    )


def job_id(run):
    return f"vs1-{run.arm}-{run.map_name}-s{run.seed}"


def job_args(run, gpu, root):
    return argparse.Namespace(
        slot=gpu,
        gpu=gpu,
        source=SOURCE,
        expected_source_sha256=(SOURCE / "SOURCE_SHA256").read_text().strip(),
        experiment_root=root,
        python=PYTHON,
        external=Path("/workspace/external/dreamerv3"),
        sc2=Path("/workspace/StarCraftII"),
        portserver_script=Path("/workspace/majepa-runtime/bin/portserver.py"),
        portserver_address=f"@majepa-value-sweep-g{gpu}",
        portserver_pool=f"{52000 + gpu * 500}-{52499 + gpu * 500}",
        wait_pid=[],
        idle_seconds=5,
        poll_seconds=5,
        wandb_project="majepa-ppo-treatments",
        wandb_entity="osaze-obahor",
        wandb_group=GROUP,
        wandb_run_prefix=job_id(run),
        wandb_job_id=job_id(run),
        validate_only=False,
        screen_label="Factual value and PPO learning sweep",
        run_spec=run,
    )


def resolve_configuration(args, run):
    import elements
    import majepa
    from majepa.main import _load_configs, _resolve_config_profiles
    from smac.env.starcraft2.maps import get_map_params

    if not Path(majepa.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError("Package import escaped the frozen sweep source")
    if get_map_params(run.map_name)["n_agents"] != run.num_agents:
        raise RuntimeError("Map agent count mismatch")
    if not (args.sc2 / "Maps/SMAC_Maps" / f"{run.map_name}.SC2Map").is_file():
        raise FileNotFoundError(run.map_name)
    commands = {
        "train": base.train_command(args, run, Path("unused")),
        "final100": base.eval_command(args, run, Path("unused"), Path("checkpoint")),
    }
    resolved = {}
    for phase, command in commands.items():
        parsed, other = elements.Flags(configs=["smac_vector", "ma_jepa"]).parse_known(
            command[3:]
        )
        c = elements.Flags(
            _resolve_config_profiles(_load_configs(), parsed.configs)
        ).parse(other)
        actual = {
            "task": str(c.task),
            "seed": int(c.seed),
            "agents": int(c.agent.num_agents),
            "horizon": int(c.agent.imag_length),
            "world_lr": float(c.agent.opt.lr),
            "joint_lr": float(c.agent.marl.ctde.opt.lr),
            "world_warmup": int(c.agent.opt.warmup),
            "joint_warmup": int(c.agent.marl.ctde.opt.warmup),
            "actor_lr": float(c.agent.ppo.actor_lr),
            "critic_lr": float(c.agent.ppo.critic_lr),
            "actor_layers": int(c.agent.policy.layers),
            "actor_units": int(c.agent.policy.units),
            "actor_epochs": int(c.agent.ppo.actor_epochs),
            "critic_epochs": int(c.agent.ppo.critic_epochs),
            "entropy": float(c.agent.ppo.entropy_coefficient),
            "advantage_mode": str(c.agent.ppo.advantage_mode),
            "critic_slowreg": float(c.agent.ppo.critic_slowreg),
            "factual": bool(c.agent.ppo.factual_value.enabled),
            "representation_scale": float(
                c.agent.ppo.factual_value.representation_scale
            ),
            "rho_clip": float(c.agent.ppo.factual_value.rho_clip),
            "c_clip": float(c.agent.ppo.factual_value.c_clip),
            "replay_value_scale": float(c.agent.ppo.replay_value_scale),
            "slowvalue_rate": float(c.agent.slowvalue.rate),
            "encoder_rate": float(c.agent.target_encoder.rate),
            "self_fed": dict(c.agent.marl.ctde.self_fed),
            "isolate_report_rng": bool(c.run.isolate_report_rng),
        }
        expected = dict(
            task=f"smac_{run.map_name}",
            seed=run.seed,
            agents=run.num_agents,
            horizon=5,
            world_lr=4e-5,
            joint_lr=4e-5,
            world_warmup=0,
            joint_warmup=0,
            actor_lr=3e-5,
            critic_lr=3e-5,
            actor_layers=3,
            actor_units=1024,
            actor_epochs=run.actor_epochs,
            critic_epochs=5,
            entropy=run.entropy,
            advantage_mode=run.advantage_mode,
            critic_slowreg=run.critic_slowreg,
            factual=run.factual,
            representation_scale=run.representation_scale,
            rho_clip=1.0,
            c_clip=1.0,
            replay_value_scale=0.3,
            slowvalue_rate=1.0,
            encoder_rate=0.01,
            isolate_report_rng=True,
            self_fed=dict(
                enabled=True,
                horizons=(2, 4, 5),
                anchors=8,
                scale=0.1,
                consumer_kl_scale=0.0,
                trajectory_kl_scale=0.0,
                fresh_history=False,
                bptt_steps=2,
            ),
        )
        # Elements Config preserves lists as tuples; normalize for provenance.
        actual = json.loads(json.dumps(actual))
        expected = json.loads(json.dumps(expected))
        if phase == "train":
            actual.update(
                steps=int(c.run.steps),
                envs=int(c.run.envs),
                train_ratio=float(c.run.train_ratio),
                prefill=int(c.run.world_model_start_step),
                ppo_start=int(c.run.ppo_start_step),
                curve_eps=int(c.run.curve_eval_eps),
                curve_interval=int(c.run.curve_eval_interval),
                final_save=bool(c.run.final_save),
            )
            expected.update(
                steps=50000,
                envs=1,
                train_ratio=128.0,
                prefill=5000,
                ppo_start=5000,
                curve_eps=32,
                curve_interval=5000,
                final_save=True,
            )
        else:
            actual.update(
                envs=int(c.run.envs),
                episodes=int(c.run.eval_eps),
                offset=int(c.run.eval_worker_offset),
                policy=str(c.run.eval_policy_mode),
            )
            expected.update(envs=4, episodes=100, offset=100000, policy="eval")
        if actual != expected:
            raise RuntimeError(f"{run.name}/{phase}: {actual} != {expected}")
        resolved[phase] = actual
    return resolved


def make_job(run, index, stage):
    return dict(
        index=index,
        stage=stage,
        name=run.name,
        arm=run.arm,
        map=run.map_name,
        seed=run.seed,
        status="pending",
        spec=asdict(run),
        wandb_train=f"{PROJECT_URL}/runs/{job_id(run)}-train",
        wandb_final=f"{PROJECT_URL}/runs/{job_id(run)}-final100",
    )


@contextmanager
def queue_state(root):
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((root / "queue.json").read_text())
        yield state
        state["updated_at"] = time.time()
        base.atomic_json(root / "queue.json", state)


def initialize(root):
    expected = (SOURCE / "SOURCE_SHA256").read_text().strip()
    if base.source_fingerprint(SOURCE) != expected:
        raise RuntimeError("Source fingerprint mismatch")
    jobs, validations = [], {}
    # Prepare validation configurations too; promotion never invents new flags.
    for arm in ARMS:
        for map_name in MAPS:
            for seed in (0, 1, 2):
                run = run_spec(arm, map_name, seed)
                validations[run.name] = resolve_configuration(
                    job_args(run, 3, root), run
                )
    for seed in (0, 1):
        for map_name in ("2s3z", "3s_vs_3z"):
            for arm in ARMS:
                jobs.append(make_job(run_spec(arm, map_name, seed), len(jobs), 0))
    root.mkdir(parents=True, exist_ok=False)
    (root / "workers").mkdir()
    now = time.time()
    base.atomic_json(root / "resolved.json", validations)
    base.atomic_json(
        root / "queue.json",
        dict(
            group=GROUP,
            source=str(SOURCE),
            source_sha256=expected,
            created_at=now,
            updated_at=now,
            phase="screen",
            jobs=jobs,
            decision_deadline=now + 48 * 3600,
            claim_deadline=now + 45 * 3600,
            screen_runs=48,
            maximum_runs=72,
            promotion_rule="At most two complete candidates: macro win gain >=5pp and no map mean loss >5pp; rank macro win, then worst map. Validate with fresh reference.",
        ),
    )
    print(
        json.dumps(dict(initialized=len(jobs), maximum_runs=72, root=str(root))),
        flush=True,
    )


def select_candidates(jobs):
    scores = {}
    for arm in ARMS:
        rows = [j for j in jobs if j["stage"] == 0 and j["arm"] == arm]
        if len(rows) != 4 or any(j["status"] != "complete" for j in rows):
            continue
        means = {
            m: sum(j["summary"]["win_rate"] for j in rows if j["map"] == m) / 2
            for m in ("2s3z", "3s_vs_3z")
        }
        scores[arm] = dict(maps=means, mean=sum(means.values()) / 2)
    if "base" not in scores:
        return [], scores, "Reference incomplete; review infrastructure failures"
    reference = scores["base"]
    eligible = [
        arm
        for arm, score in scores.items()
        if arm != "base"
        and score["mean"] >= reference["mean"] + 0.05 - 1e-9
        and all(
            score["maps"][m] >= reference["maps"][m] - 0.05 - 1e-9
            for m in reference["maps"]
        )
    ]
    eligible.sort(
        key=lambda arm: (-scores[arm]["mean"], -min(scores[arm]["maps"].values()), arm)
    )
    return (
        eligible[:2],
        scores,
        "Candidates selected for replication"
        if eligible
        else "No candidate passed; revise hypotheses before more runs",
    )


def advance(state):
    active = [j for j in state["jobs"] if j["status"] in ("pending", "running")]
    if active or state["phase"] not in ("screen", "validation"):
        return
    if state["phase"] == "validation":
        state.update(phase="complete", needs_review=True)
        return
    chosen, scores, reason = select_candidates(state["jobs"])
    state["selection"] = dict(arms=chosen, scores=scores, reason=reason, at=time.time())
    if not chosen:
        state.update(phase="review", needs_review=True)
        return
    for seed, maps in (
        (0, ("3s_vs_4z", "MMM", "8m")),
        (1, ("3s_vs_4z", "MMM", "8m")),
        (2, ("2s3z", "3s_vs_3z")),
    ):
        for map_name in maps:
            for arm in ("base", *chosen):
                state["jobs"].append(
                    make_job(run_spec(arm, map_name, seed), len(state["jobs"]), 1)
                )
    state["phase"] = "validation"


def matrix_has_work(gpu):
    if not MATRIX.exists():
        raise FileNotFoundError(
            "Matrix queue unavailable; refusing to assume GPU ownership"
        )
    jobs = json.loads(MATRIX.read_text())["jobs"]
    return any(
        j["status"] == "pending" or (j["status"] == "running" and j.get("gpu") == gpu)
        for j in jobs
    )


def claim(root, gpu):
    with queue_state(root) as state:
        advance(state)
        if time.time() >= state["claim_deadline"]:
            state.update(needs_review=True, claims_closed=True)
            return "done"
        if state["phase"] in ("complete", "review"):
            return "done"
        pending = [j for j in state["jobs"] if j["status"] == "pending"]
        if not pending:
            return None
        job = pending[0]
        job.update(
            status="running", gpu=gpu, worker_pid=os.getpid(), started_at=time.time()
        )
        return dict(job)


def validate_profile(args, run, env):
    del env
    return json.loads((args.experiment_root / "resolved.json").read_text())[run.name]


def worker(root, gpu):
    lock = (root / "workers" / f"gpu{gpu}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "workers" / f"gpu{gpu}.pid").write_text(f"{os.getpid()}\n")
    children = base.OwnedChildren()

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            if matrix_has_work(gpu) or base.gpu_processes(gpu):
                base.atomic_json(
                    root / "workers" / f"gpu{gpu}.status.json",
                    dict(status="waiting_for_gpu", at=time.time(), gpu=gpu),
                )
                time.sleep(15)
                continue
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
                "job",
                "--root",
                str(root),
                "--index",
                str(job["index"]),
                "--gpu",
                str(gpu),
            ]
            try:
                with (root / "workers" / f"{job['name']}.log").open("x") as logfile:
                    process = children.start(
                        command, stdout=logfile, stderr=subprocess.STDOUT
                    )
                    with queue_state(root) as state:
                        state["jobs"][job["index"]]["supervisor_pid"] = process.pid
                    code = process.wait()
                path = root / "runs" / job["name"] / "outcome.json"
                outcome = json.loads(path.read_text()) if path.exists() else {}
                complete = code == 0 and outcome.get("completed") is True
                with queue_state(root) as state:
                    state["jobs"][job["index"]].update(
                        status="complete" if complete else "failed",
                        returncode=code,
                        finished_at=time.time(),
                        error=outcome.get("error"),
                        summary=outcome.get("summary"),
                    )
                    advance(state)
                print(
                    json.dumps(
                        dict(event="finished", name=job["name"], complete=complete)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "worker", "job", "status"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gpu", type=int, choices=range(6), default=3)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "init":
        initialize(args.root)
    elif args.mode == "worker":
        worker(args.root, args.gpu)
    elif args.mode == "status":
        print((args.root / "queue.json").read_text())
    else:
        state = json.loads((args.root / "queue.json").read_text())
        job = state["jobs"][args.index]
        run = RunSpec(**job["spec"])
        base.run_screen(
            job_args(run, args.gpu, args.root),
            run,
            validate_profile,
            extra_manifest=dict(
                sweep=GROUP, stage=job["stage"], queue_index=job["index"]
            ),
        )


if __name__ == "__main__":
    main()
