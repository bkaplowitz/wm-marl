"""Bounded, restartable six-run RunPod campaign; no account keys leave this host."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import netrc
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
JOBS = tuple(f"samples{samples}-seed{seed}" for samples in (2, 1) for seed in range(3))
REPO = Path(__file__).resolve().parents[2]
OVERRIDES = {
    "task": "smac_2s3z",
    "agent.num_agents": 5,
    "run.envs": 1,
    "run.steps": 50000,
    "run.train_ratio": 128,
    "run.replay_stream_mode": "snapshot_staggered",
    "run.replay_startup_behavior_min_starts": 4,
    "run.isolate_report_rng": True,
    "agent.dyn.parallel_transformer.deter": 4096,
    "agent.dyn.parallel_transformer.hidden": 512,
    "agent.dyn.parallel_transformer.stoch": 32,
    "agent.dyn.parallel_transformer.classes": 64,
    "agent.dyn.parallel_transformer.unimix": 0.01,
    "agent.policy.layers": 3,
    "agent.policy.units": 512,
    "agent.policy.unimix": 0.0,
    "agent.collection_unimix": 0.0,
    "agent.rewhead.units": 512,
    "agent.conhead.units": 512,
    "agent.maskhead.units": 512,
    "agent.marl.ctde.critic.width": 256,
    "agent.marl.ctde.critic.value_layers": 2,
    "agent.marl.ctde.critic.value_units": 256,
    "agent.loss_scales.ctde_posterior_alignment": 0.05,
    "agent.loss_scales.ctde_multistep_jepa_action": 0.1,
    "agent.marl.ctde.self_fed.enabled": True,
    "agent.marl.ctde.self_fed.horizons": [2, 4, 5],
    "agent.marl.ctde.self_fed.anchors": 8,
    "agent.marl.ctde.self_fed.scale": 0.1,
    "agent.marl.ctde.self_fed.trajectory_kl_scale": 0.1,
    "agent.marl.ctde.self_fed.consumer_kl_scale": 0.0,
    "agent.marl.ctde.self_fed.fresh_history": False,
    "agent.marl.ctde.self_fed.local_feedback_gradient": False,
    "agent.marl.ctde.self_fed.bptt_steps": 2,
    "agent.imag_length": 5,
    "agent.opt.lr": 1e-4,
    "agent.opt.warmup": 0,
    "agent.marl.ctde.opt.lr": 1e-4,
    "agent.marl.ctde.opt.warmup": 0,
    "agent.ppo.actor_lr": 3e-5,
    "agent.ppo.critic_lr": 3e-5,
    "agent.ppo.clip_epsilon": 0.2,
    "agent.ppo.actor_epochs": 5,
    "agent.ppo.critic_epochs": 5,
    "agent.ppo.entropy_coefficient": 0.003,
    "agent.ppo.entropy_schedule.enabled": False,
    "agent.ppo.replay_value_scale": 0.3,
    "agent.slowvalue.rate": 1.0,
    "agent.ppo.factual_value.enabled": False,
    "agent.marl.ctde.direct_latent": False,
    "agent.marl.ctde.imagination_mask_sampling": "bernoulli",
    "agent.action_mask_reduction": "balanced",
    "replay.size": 250000,
    "replay.sampling": "recent_world_uniform_behavior",
    "replay.world_uniform_mix": 0.5,
    "replay.recency_decay": 0.9998,
    "run.world_model_start_step": 5000,
    "run.ppo_start_step": 5000,
    "agent.marl.ctde.teammate_belief.enabled": False,
    "agent.marl.ctde.multistep_jepa.belief_context": False,
    "run.curve_eval_interval": 5000,
    "run.curve_eval_eps": 32,
    "run.eval_envs": 4,
    "run.curve_eval_seed_offset": 50000,
    "run.curve_eval_policy_mode": "eval",
    "run.checkpoint_at_curve_eval": True,
    "run.final_save": True,
    "jax.policy_devices": [0],
    "jax.train_devices": [0],
    "logger.outputs": ["jsonl", "wandb"],
}


def training_args(*, seed, samples, logdir, checkpoint=None):
    if seed not in range(3) or samples not in (1, 2):
        raise ValueError("campaign requires seeds 0/1/2 and one/two samples")
    config = dict(OVERRIDES, seed=seed, logdir=str(logdir))
    config["agent.imag_action_samples"] = samples
    if checkpoint:
        config.update(
            {
                "script": "eval_only",
                "run.from_checkpoint": str(checkpoint),
                "run.envs": 4,
                "run.eval_eps": 100,
                "run.eval_worker_offset": 100000,
                "run.eval_policy_mode": "eval",
            }
        )
    args = []
    for key, value in config.items():
        args.append("--" + key)
        args.extend(str(x) for x in value) if isinstance(value, list) else args.append(
            str(value)
        )
    return args


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


@contextmanager
def locked_manifest(directory):
    with (directory / "manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads((directory / "manifest.json").read_text())
        yield manifest
        write_json(directory / "manifest.json", manifest)


def save_job(directory, job):
    with locked_manifest(directory) as manifest:
        saved = next(item for item in manifest["jobs"] if item["name"] == job["name"])
        status = saved.get("status")
        saved.update(job)
        if status in ("stop_requested", "stopped"):
            saved["status"] = status
        stopped = saved["status"] in ("stop_requested", "stopped")
    if stopped and job["status"] != "failed":
        raise RuntimeError("campaign monitor stopped this job")


def accounted_cost(manifest, now):
    for job in manifest["jobs"]:
        elapsed = max(0, job.get("stopped_at", now) - job["created_at"])
        rate = max(manifest["rate"], job.get("actual_rate", 0))
        job["observed_cost"] = max(job.get("observed_cost", 0), rate * elapsed / 3600)
    return sum(
        max(job["reserved_cost"], job["observed_cost"]) for job in manifest["jobs"]
    )


def reserve(manifest, name, hours, *, now=None):
    if name not in ("smoke", *JOBS):
        raise ValueError("unknown campaign job")
    if any(job["name"] == name for job in manifest["jobs"]):
        raise ValueError(
            "job already reserved; reconcile manifest rather than recreate"
        )
    if not 0 < hours <= 8:
        raise ValueError("hours must be positive and at most eight")
    if sum("stopped_at" not in job for job in manifest["jobs"]) >= 4:
        raise ValueError("maximum four concurrent campaign pods")
    now = time.time() if now is None else now
    # Reserve the full allocation even when the local monitor disappears.
    reserved = accounted_cost(manifest, now)
    cost = hours * manifest["rate"]
    if reserved + cost > manifest["budget"] - 0.5:
        raise ValueError("GPU budget exhausted (including shutdown allowance)")
    job = {
        "name": name,
        "created_at": now,
        "deadline": now + hours * 3600,
        "reserved_cost": cost,
        "status": "creating",
        "wandb_id": uuid.uuid4().hex[:12],
    }
    manifest["jobs"].append(job)
    return job


def api(path, *, method="GET"):
    parts = path.split("/")
    if len(parts) not in (2, 3) or parts[0] != "pods":
        raise ValueError("only exact pod reads and stops are supported")
    command = ["runpodctl", "--output", "json", "pod"]
    if method == "GET" and len(parts) == 2:
        return json.loads(
            subprocess.check_output(command + ["get", parts[1]], text=True)
        )
    if method == "POST" and parts[2:] == ["stop"]:
        subprocess.run(command + ["stop", parts[1]], check=True)
        return {}
    raise ValueError("unsupported pod operation")


def ssh(pod, directory):
    return [
        "ssh",
        "-i",
        str(Path.home() / ".ssh/runpod_key"),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={directory / 'known_hosts'}",
        "-o",
        "ConnectTimeout=10",
        "-p",
        str(pod["ssh"]["port"]),
        "root@" + pod["ssh"]["ip"],
    ]


def remote(command, script, **kwargs):
    return subprocess.run(
        [*command, "bash -lc " + shlex.quote(script)], check=True, text=True, **kwargs
    )


def freeze(directory):
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/majepa/campaign.py"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "diff", "HEAD", "--name-only"], cwd=REPO, text=True
    )
    if dirty.strip():
        raise ValueError("commit implementation before freezing source")
    submodule = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO / "external/dreamerv3", text=True
    ).strip()
    pinned = subprocess.check_output(
        ["git", "rev-parse", "HEAD:external/dreamerv3"],
        cwd=REPO,
        text=True,
    ).strip()
    if submodule != pinned:
        raise ValueError("DreamerV3 checkout does not match pinned gitlink")
    with tarfile.open(directory / "source.tar.gz", "w:gz") as target:
        for cwd, prefix in [
            (REPO, ""),
            (REPO / "external/dreamerv3", "external/dreamerv3/"),
        ]:
            data = subprocess.check_output(["git", "archive", "HEAD"], cwd=cwd)
            with tarfile.open(fileobj=io.BytesIO(data)) as source:
                for member in source:
                    member.name = prefix + member.name
                    target.addfile(
                        member, source.extractfile(member) if member.isfile() else None
                    )
    return {
        "revision": revision,
        "dreamerv3_revision": submodule,
        "source_sha256": hashlib.sha256(
            (directory / "source.tar.gz").read_bytes()
        ).hexdigest(),
    }


def initialize(directory):
    directory.mkdir(parents=True, exist_ok=False)
    source = freeze(directory)
    manifest = {
        "campaign": directory.name,
        "image": IMAGE,
        "volume": "e8ishvsuf7",
        "region": "US-KS-2",
        "rate": 1.59,
        "budget": 50.0,
        "jobs": [],
        **source,
    }
    write_json(directory / "manifest.json", manifest)


def startup_command(deadline):
    code = (
        "import os,subprocess,time\n"
        f"time.sleep(max(0,{deadline!r}-time.time()-30))\n"
        "while True:\n"
        ' for command in (["runpodctl","stop","pod",os.environ["RUNPOD_POD_ID"]],'
        '["runpodctl","pod","stop",os.environ["RUNPOD_POD_ID"]]):\n'
        "  try:\n"
        "   if subprocess.run(command,timeout=30).returncode == 0: raise SystemExit(0)\n"
        "  except (OSError,subprocess.TimeoutExpired): pass\n"
        " time.sleep(5)\n"
    )
    script = f"nohup python -c {shlex.quote(code)} >/tmp/majepa-runtime-limit.log 2>&1 </dev/null & exec /start.sh"
    return shlex.join(["bash", "-lc", script])


def bootstrap_script(out):
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(out)}",
            "trap "
            + shlex.quote(
                "printf '%s\n' "
                + shlex.quote('{"completed":false,"reason":"bootstrap failed"}')
                + " > "
                + shlex.quote(out + "/outcome.json")
            )
            + " ERR",
            "mkdir repo; tar -xzf source.tar.gz -C repo",
            "python -m pip install uv",
            "cd repo",
            "uv sync --locked --python 3.11 --extra dev --extra smac --extra cuda12",
            'export PYTHONPATH="$PWD/src:$PWD/external/dreamerv3"',
            f"exec .venv/bin/python -m majepa.campaign run --directory {shlex.quote(out)}",
        ]
    )


def launch(directory, manifest, name, hours):
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if not auth:
        raise ValueError("W&B netrc credentials are required")
    public_key = (Path.home() / ".ssh/runpod_key.pub").read_text().strip()
    if (
        "source_sha256" in manifest
        and hashlib.sha256((directory / "source.tar.gz").read_bytes()).hexdigest()
        != manifest["source_sha256"]
    ):
        raise ValueError("frozen source archive checksum mismatch")
    with locked_manifest(directory) as current:
        job = reserve(current, name, hours).copy()
        manifest = current
    command = [
        "runpodctl",
        "--output",
        "json",
        "pod",
        "create",
        "--name",
        manifest["campaign"] + "-" + name,
        "--image",
        IMAGE,
        "--gpu-id",
        "NVIDIA A100-SXM4-80GB",
        "--gpu-count",
        "1",
        "--cloud-type",
        "SECURE",
        "--network-volume-id",
        manifest["volume"],
        "--data-center-ids",
        manifest["region"],
        "--container-disk-in-gb",
        "30",
        "--volume-mount-path",
        "/workspace",
        "--ports",
        "22/tcp",
        "--docker-args",
        startup_command(job["deadline"]),
        "--env",
        json.dumps({"PUBLIC_KEY": public_key}),
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
        job["pod_id"] = payload["id"] if "id" in payload else payload["pod"]["id"]
        job["status"] = "provisioning"
        save_job(directory, job)
        print(f"Created {job['pod_id']}; waiting for SSH", file=sys.stderr, flush=True)
        wait_until = min(job["deadline"], time.time() + 900)
        while time.time() < wait_until:
            pod = api("pods/" + job["pod_id"])
            actual_rate = pod.get("costPerHr")
            if (
                actual_rate is None
                or float(actual_rate) > manifest["rate"]
                or float(actual_rate) <= 0
            ):
                raise ValueError(
                    f"pod GPU hourly rate exceeds reservation or is unavailable: {actual_rate}"
                )
            job["actual_rate"] = float(actual_rate)
            if (pod.get("ssh") or {}).get("ip") and pod["ssh"].get("port"):
                try:
                    remote(ssh(pod, directory), "true", capture_output=True)
                    break
                except subprocess.CalledProcessError:
                    pass
            time.sleep(10)
        else:
            raise TimeoutError("pod SSH unavailable after 15 minutes")
        connection = ssh(pod, directory)
        out = f"/workspace/{manifest['campaign']}/{name}"
        job["remote_dir"] = out
        save_job(directory, job)
        remote(
            connection,
            f"mkdir -p {shlex.quote(out)}; test ! -e {shlex.quote(out + '/job.json')}",
        )
        data = {**manifest, "job": job}
        remote(
            connection,
            f"cat > {shlex.quote(out + '/job.json')}",
            input=json.dumps(data),
        )
        with tarfile.open(directory / "source.tar.gz") as source:
            runner = source.extractfile("src/majepa/campaign.py").read().decode()
        remote(connection, f"cat > {shlex.quote(out + '/campaign.py')}", input=runner)
        remote(
            connection,
            f"nohup python {shlex.quote(out + '/campaign.py')} watchdog --directory {shlex.quote(out)} > {shlex.quote(out + '/watchdog.log')} 2>&1 < /dev/null &",
        )
        login, _, password = auth
        remote(
            connection,
            "umask 077; cat > /root/.netrc",
            input=f"machine api.wandb.ai\nlogin {login}\npassword {password}\n",
        )
        with (directory / "source.tar.gz").open("rb") as source:
            subprocess.run(
                [*connection, "cat > " + shlex.quote(out + "/source.tar.gz")],
                stdin=source,
                check=True,
            )
        bootstrap = bootstrap_script(out)
        remote(
            connection, f"cat > {shlex.quote(out + '/bootstrap.sh')}", input=bootstrap
        )
        remote(
            connection,
            f"nohup bash {shlex.quote(out + '/bootstrap.sh')} > {shlex.quote(out + '/job.log')} 2>&1 < /dev/null & echo $!",
        )
        job["status"] = "running"
        save_job(directory, job)
    except BaseException:
        job["status"] = "failed" if job.get("pod_id") else "creation_uncertain"
        save_job(directory, job)
        if job.get("pod_id"):
            api("pods/" + job["pod_id"] + "/stop", method="POST")
        raise


def monitor(directory, manifest):
    budget_exhausted = accounted_cost(manifest, time.time()) >= manifest["budget"] - 0.5
    for job in manifest["jobs"]:
        if not job.get("pod_id") or "stopped_at" in job:
            continue
        pod = api("pods/" + job["pod_id"])
        if pod.get("desiredStatus") == "EXITED":
            job["stopped_at"] = time.time()
            job["status"] = "stopped"
        elif budget_exhausted or time.time() >= job["deadline"] - 30:
            api("pods/" + job["pod_id"] + "/stop", method="POST")
            job["status"] = "stop_requested"
    accounted_cost(manifest, time.time())
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def own_stop(pod_id):
    if not re.fullmatch("[a-zA-Z0-9]+", pod_id):
        raise ValueError("invalid own pod ID")
    subprocess.run(
        [
            "bash",
            "-ec",
            ". /etc/rp_environment; "
            'test "$RUNPOD_POD_ID" = "$1"; '
            'runpodctl stop pod "$1" || runpodctl pod stop "$1"',
            "own-stop",
            pod_id,
        ],
        check=True,
        timeout=60,
    )


def watchdog(directory):
    job = json.loads((directory / "job.json").read_text())["job"]
    while (
        time.time() < job["deadline"] - 30 and not (directory / "outcome.json").exists()
    ):
        time.sleep(min(5, max(0, job["deadline"] - 30 - time.time())))
    if not (directory / "outcome.json").exists():
        write_json(
            directory / "outcome.json", {"completed": False, "reason": "runtime limit"}
        )
    while True:
        try:
            own_stop(job["pod_id"])
            return
        except Exception as exc:
            print(f"self-stop failed: {type(exc).__name__}", flush=True)
            time.sleep(5)


def discover_sc2():
    candidates = [Path(os.environ["SC2PATH"])] if os.environ.get("SC2PATH") else []
    for root, dirs, files in os.walk("/workspace"):
        if "StarCraftII" in dirs:
            candidates.append(Path(root) / "StarCraftII")
            dirs.remove("StarCraftII")
        dirs[:] = [
            name
            for name in dirs
            if name not in (".venv", ".git", "wandb", "replay", "ckpt")
        ]
        if len(Path(root).parts) >= 6:
            dirs.clear()
    valid = [
        p
        for p in candidates
        if (p / "Versions/Base75689/SC2_x64").is_file()
        and (p / "Maps/SMAC_Maps/2s3z.SC2Map").is_file()
    ]
    if not valid:
        raise FileNotFoundError(
            "existing StarCraft II 4.10 Base75689 and SMAC 2s3z map not found"
        )
    versions = [
        int(p.name[4:])
        for p in (valid[0] / "Versions").glob("Base*")
        if p.name[4:].isdigit()
    ]
    if max(versions) != 75689:
        raise ValueError(
            "SC2 installation has a newer default version than required 4.10"
        )
    return valid[0]


def smoke(directory):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import wandb
    from majepa.envs.smac import SMACEnv

    assert any(device.platform == "gpu" for device in jax.devices()), jax.devices()
    result = float(
        (jnp.ones((128, 128)) @ jnp.ones((128, 128))).sum().block_until_ready()
    )
    assert np.isfinite(result)
    env = SMACEnv("2s3z", seed=0)
    try:
        obs = env.step({"reset": True, "action": np.zeros(5, np.int32)})
        actions = np.asarray(obs["action_mask"]).argmax(-1)
        obs = env.step({"reset": False, "action": actions})
        assert np.isfinite(obs["reward"]).all()
    finally:
        env.close()
    with wandb.init(
        job_type="smoke",
        config={"gpu": str(jax.devices()), "sc2path": os.environ["SC2PATH"]},
    ) as run:
        run.log({"smoke/gpu_sum": result, "smoke/smac_step": 1}, step=1)
        tiny = directory / "smoke.txt"
        tiny.write_text("MA-JEPA GPU + SMAC + W&B smoke\n")
        artifact = wandb.Artifact("smoke-" + run.id, type="smoke")
        artifact.add_file(str(tiny))
        uploaded = run.log_artifact(artifact).wait()
        downloaded = Path(
            wandb.Api()
            .artifact(f"{run.entity}/{run.project}/{uploaded.name}")
            .download(root=str(directory / "smoke-download"))
        )
        assert (downloaded / tiny.name).read_bytes() == tiny.read_bytes()
        write_json(
            directory / "smoke_evidence.json",
            {"gpu_sum": result, "artifact": uploaded.name, "roundtrip_equal": True},
        )


def run_job(directory):
    manifest = json.loads((directory / "job.json").read_text())
    job = manifest["job"]
    try:
        if shutil.disk_usage(directory).free < 20 * 1024**3:
            raise RuntimeError("less than 20GiB free on persistent volume")
        os.environ.update(
            RUNPOD_POD_ID=job["pod_id"],
            SC2PATH=str(discover_sc2()),
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
            WANDB_ENTITY="osaze-obahor",
            WANDB_PROJECT="majepa-multi-sample-treatments",
            WANDB_RUN_ID=job["wandb_id"],
            WANDB_NAME=manifest["campaign"] + "-" + job["name"],
            WANDB_RESUME="never",
            WANDB_MODE="online",
            PYTHONUNBUFFERED="1",
            MAJEPA_CAMPAIGN_MANIFEST=str(directory / "job.json"),
            MAJEPA_SOURCE_ARCHIVE=str(directory / "source.tar.gz"),
        )
        if job["name"] == "smoke":
            smoke(directory)
        else:
            match = re.fullmatch(r"samples([12])-seed([012])", job["name"])
            samples, seed = map(int, match.groups())
            command = [sys.executable, "-m", "majepa.main"]
            subprocess.run(
                command
                + training_args(seed=seed, samples=samples, logdir=directory / "run"),
                check=True,
            )
            ckpt_root = directory / "run/ckpt"
            checkpoint = ckpt_root / (ckpt_root / "latest").read_text().strip()
            if not (checkpoint / "done").is_file():
                raise FileNotFoundError("final checkpoint is incomplete")
            os.environ["WANDB_RUN_ID"] = job["wandb_id"] + "-eval"
            os.environ["WANDB_NAME"] += "-eval"
            subprocess.run(
                command
                + training_args(
                    seed=seed,
                    samples=samples,
                    logdir=directory / "evaluation",
                    checkpoint=checkpoint,
                ),
                check=True,
            )
        write_json(
            directory / "outcome.json", {"completed": True, "finished_at": time.time()}
        )
    except BaseException as exc:
        write_json(
            directory / "outcome.json",
            {"completed": False, "error": str(exc), "finished_at": time.time()},
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["init", "launch", "monitor", "run", "watchdog"]
    )
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--job", choices=["smoke", *JOBS])
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "jobs": list(JOBS),
                    "image": IMAGE,
                    "max_concurrent": 4,
                    "budget": 50,
                    "smoke_hours": 0.5,
                    "train_hours": 5,
                    "overrides": OVERRIDES,
                    "bootstrap": bootstrap_script(
                        f"/workspace/{args.directory.name}/<job>"
                    ),
                    "training_commands": {
                        name: [".venv/bin/python", "-m", "majepa.main"]
                        + training_args(
                            seed=int(name[-1]),
                            samples=int(name[7]),
                            logdir=f"/workspace/{args.directory.name}/{name}/run",
                        )
                        for name in JOBS
                    },
                },
                indent=2,
            )
        )
        return
    directory = args.directory.expanduser().resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", directory.name):
        raise ValueError("campaign directory basename must be a simple identifier")
    if args.action == "init":
        initialize(directory)
    elif args.action == "run":
        run_job(directory)
    elif args.action == "watchdog":
        watchdog(directory)
    elif args.action == "launch":
        if not args.job:
            parser.error("launch requires --job")
        hours = (
            args.max_hours
            if args.max_hours is not None
            else (0.5 if args.job == "smoke" else 5)
        )
        manifest = json.loads((directory / "manifest.json").read_text())
        launch(directory, manifest, args.job, hours)
    else:
        while True:
            with locked_manifest(directory) as manifest:
                monitor(directory, manifest)
            if args.once:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
