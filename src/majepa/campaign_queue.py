"""Persistent per-pod campaign queue; independent jobs use one GPU each."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

from .campaign import write_json


@contextmanager
def locked_state(directory):
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "queue.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        yield state
        write_json(path, state)


def initialize(directory, manifest):
    runs = {item["name"]: item for item in manifest["runs"]}
    names = manifest["job"]["run_names"]
    if not names or len(set(names)) != len(names):
        raise ValueError("run_names must be nonempty and unique")
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid run name: {name!r}")
        for dependency in runs[name].get("depends_on", []):
            if dependency not in names or dependency == name:
                raise ValueError(f"invalid dependency for {name}: {dependency}")
    remaining = set(names)
    while remaining:
        ready = {
            n
            for n in remaining
            if not remaining.intersection(runs[n].get("depends_on", []))
        }
        if not ready:
            raise ValueError("cyclic run dependencies")
        remaining -= ready
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with locked_state(directory) as state:
        if state:
            if state.get("manifest_sha256") != identity:
                raise ValueError("job.json changed after queue initialization")
            return state
        state.update(
            manifest_sha256=identity,
            created_at=time.time(),
            deadline=manifest["job"]["deadline"],
            workers={},
            jobs=[
                dict(
                    runs[name],
                    status="pending",
                    gpu=None,
                    wandb_id=runs[name].get("wandb_id") or uuid.uuid4().hex[:12],
                    evaluation_wandb_id=runs[name].get("evaluation_wandb_id")
                    or uuid.uuid4().hex[:12],
                )
                for name in names
            ],
        )
        return state


def update_worker(directory, gpu, **values):
    with locked_state(directory) as state:
        state["workers"].setdefault(str(gpu), {}).update(
            pid=os.getpid(), heartbeat=time.time(), **values
        )


def update_job(directory, name, **values):
    with locked_state(directory) as state:
        entry = next(item for item in state["jobs"] if item["name"] == name)
        entry.update(**values)


def claim(directory, gpu):
    with locked_state(directory) as state:
        statuses = {item["name"]: item["status"] for item in state["jobs"]}
        for item in state["jobs"]:
            if item["status"] != "pending":
                continue
            dependencies = item.get("depends_on", [])
            if any(statuses[name] == "failed" for name in dependencies):
                item.update(
                    status="failed", error="dependency failed", finished_at=time.time()
                )
                continue
            if any(statuses[name] != "complete" for name in dependencies):
                continue
            if time.time() >= state["deadline"]:
                return None
            item.update(
                status="running",
                gpu=gpu,
                worker_pid=os.getpid(),
                started_at=time.time(),
            )
            return item.copy()
    return None


def predecessor_wait(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("predecessor queue path must be absolute")
    if not path.exists():
        return f"predecessor queue not present: {path}"
    state = json.loads(path.read_text())
    jobs = state.get("jobs")
    if not jobs:
        raise ValueError(f"predecessor has no jobs: {path}")
    if any(item["status"] in ("failed", "interrupted") for item in jobs):
        raise RuntimeError(f"predecessor failed: {path}")
    if any(item["status"] != "complete" for item in jobs):
        return f"predecessor not complete: {path}"
    return None


def storage_check(directory, storage):
    root = Path(storage.get("workspace_root", "/workspace")).resolve(strict=True)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("queue directory must be inside storage.workspace_root")
    used = (
        int(
            subprocess.check_output(
                ["du", "-sk", "--exclude=*.tmp", str(root)], text=True
            ).split()[0]
        )
        * 1024
    )
    free = shutil.disk_usage(root).free
    quota = int(float(storage["quota_gb"]) * 1e9)
    minimum = int(float(storage["min_free_gb"]) * 1e9)
    if quota <= 0 or minimum <= 0 or minimum >= quota:
        raise ValueError(
            "storage quota and minimum headroom must be positive and ordered"
        )
    record = dict(
        at=time.time(),
        root=str(root),
        used_bytes=used,
        filesystem_free_bytes=free,
        quota_bytes=quota,
        headroom_bytes=min(free, quota - used),
        minimum_bytes=minimum,
    )
    write_json(directory / "storage.json", record)
    if record["headroom_bytes"] < minimum:
        raise RuntimeError(
            f"storage guard: {record['headroom_bytes']} bytes headroom < {minimum}"
        )
    return record


def verify_source(directory, manifest):
    archive = directory / "source.tar.gz"
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("frozen source archive checksum mismatch")
    source = (directory / "repo").resolve(strict=True)
    if Path(__file__).resolve() != source / "src/majepa/campaign_queue.py":
        raise ValueError("queue is not running from the frozen source")
    with tarfile.open(archive) as handle:
        for member in handle:
            target = source / member.name
            if not target.resolve().is_relative_to(source):
                raise ValueError(f"source path escapes frozen source: {member.name}")
            if member.isfile():
                with (
                    handle.extractfile(member) as expected,
                    target.open("rb") as actual,
                ):
                    if (
                        hashlib.file_digest(expected, "sha256").digest()
                        != hashlib.file_digest(actual, "sha256").digest()
                    ):
                        raise ValueError(f"extracted source mismatch: {member.name}")
            elif not member.isdir():
                raise ValueError(f"unsupported frozen source member: {member.name}")
    return digest


def gpu_processes(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


@contextmanager
def interruptible():
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    previous = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def stop_child(process, grace=10):
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


@contextmanager
def port_server(directory, gpu):
    address = f"@majepa-{os.getpid()}-gpu{gpu}"
    first = 59000 + 300 * gpu
    if first + 299 > 65535:
        raise ValueError("GPU count exceeds available SC2 port ranges")
    script = Path(sys.executable).with_name("portserver.py")
    if not script.is_file():
        raise FileNotFoundError(f"locked SMAC runtime is missing {script}")
    with (directory / f"portserver-{gpu}.log").open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--portserver_address",
                address,
                "--portserver_static_pool",
                f"{first}-{first + 299}",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.time() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"portserver failed; see {log.name}")
                try:
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.connect(address.replace("@", "\0", 1))
                    break
                except OSError:
                    if time.time() >= deadline:
                        raise TimeoutError("SC2 portserver did not start")
                    time.sleep(0.1)
            yield address
        finally:
            stop_child(process)


def phase_config(directory, entry, checkpoint=None):
    config = dict(entry["config"])
    if config["logdir"] != f"/RUN/{entry['name']}/train":
        raise ValueError("expected canonical /RUN/<name>/train logdir")
    for key in ("jax.policy_devices", "jax.train_devices"):
        if list(config[key]) != [0]:
            raise ValueError(f"one-GPU worker requires {key}=[0]")
    root = directory / "jobs" / entry["name"]
    config["logdir"] = str(root / ("final100" if checkpoint else "train"))
    if checkpoint:
        config.update(
            script="eval_only",
            **{
                "run.from_checkpoint": str(checkpoint),
                "run.eval_eps": 100,
                "run.eval_policy_mode": "eval",
            },
        )
    return config


def phase_command(config, path):
    import elements
    from .main import _load_configs, _resolve_config_profiles

    elements.Config({"defaults": dict(elements.Config(config))}).save(path)
    resolved = _resolve_config_profiles(_load_configs(path), ["defaults"])
    parsed = elements.Flags(resolved).parse([])
    if json.loads(json.dumps(parsed.flat)) != json.loads(json.dumps(config)):
        raise ValueError("frozen configuration does not resolve to the intended flags")
    code = "import sys; from majepa.main import main; main(['--configs', 'defaults'], extra_config_path=sys.argv[1])"
    return [sys.executable, "-c", code, str(path)]


def execute_phase(directory, manifest, entry, gpu, address, phase, checkpoint=None):
    config = phase_config(directory, entry, checkpoint)
    root = directory / "jobs" / entry["name"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{phase}-resolved.yaml"
    command = phase_command(config, path)
    run_id = entry["evaluation_wandb_id"] if checkpoint else entry["wandb_id"]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PORTSERVER_ADDRESS=address,
        WANDB_ENTITY=manifest["wandb"]["entity"],
        WANDB_PROJECT=manifest["wandb"]["project"],
        WANDB_RUN_ID=run_id,
        WANDB_NAME=f"{manifest['campaign']}-{entry['name']}-{phase}",
        WANDB_RUN_GROUP=manifest["campaign"],
        WANDB_RESUME="never",
        WANDB_MODE="online",
        PYTHONUNBUFFERED="1",
        MAJEPA_CAMPAIGN_MANIFEST=str(directory / "job.json"),
        MAJEPA_SOURCE_ARCHIVE=str(directory / "source.tar.gz"),
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
    )
    env.pop("JOB_COMPLETION_INDEX", None)
    with (root / f"{phase}.log").open("a") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            update_job(directory, entry["name"], phase=phase, child_pid=process.pid)
            checked_storage_at = 0
            while process.poll() is None:
                if time.time() >= manifest["job"]["deadline"]:
                    raise TimeoutError("pod runtime deadline reached")
                update_worker(
                    directory, gpu, status="running", phase=phase, run=entry["name"]
                )
                if time.time() - checked_storage_at >= 30:
                    storage_check(directory, manifest["storage"])
                    checked_storage_at = time.time()
                time.sleep(min(5, max(0, manifest["job"]["deadline"] - time.time())))
            if process.returncode:
                raise RuntimeError(
                    f"{phase} exited {process.returncode}; see {log.name}"
                )
        finally:
            stop_child(process)
            update_job(directory, entry["name"], child_pid=None)
    import elements

    saved = elements.Config.load(Path(config["logdir"]) / "config.yaml").flat
    if json.loads(json.dumps(saved)) != json.loads(json.dumps(config)):
        raise ValueError(f"{phase} saved config differs from intended config")
    receipts = json.loads(
        (Path(config["logdir"]) / "artifact_verification.json").read_text()
    )
    if any(
        receipts.get(kind, {}).get("verified") is not True
        for kind in ("config", "source")
    ):
        raise ValueError(f"{phase} source/config artifact upload is not verified")
    return Path(config["logdir"])


def final_checkpoint(logdir, expected_updates, expected_steps):
    root = logdir / "ckpt"
    name = (root / "latest").read_text().strip()
    path = root / name
    if not name or Path(name).name != name or path.is_symlink():
        raise ValueError("invalid final checkpoint pointer")
    if not (path / "done").is_file():
        raise ValueError("final checkpoint is incomplete")
    for filename, expected in (
        ("learner_update_calls.pkl", expected_updates),
        ("step.pkl", expected_steps),
    ):
        with (path / filename).open("rb") as handle:
            actual = int(pickle.load(handle))
        if actual != expected:
            raise ValueError(f"{filename}: expected {expected}, got {actual}")
    return path


def prune_checkpoints(directory, entry, checkpoint, receipt):
    if entry["status"] != "complete" or receipt.get("verified") is not True:
        raise ValueError(
            "pruning requires completed job and verified final checkpoint upload"
        )
    root = directory / "jobs" / entry["name"] / "train" / "ckpt"
    if root.is_symlink() or not root.resolve().is_relative_to(directory.resolve()):
        raise ValueError("checkpoint root escapes this campaign")
    if checkpoint.parent != root or not (checkpoint / "done").is_file():
        raise ValueError("final checkpoint does not belong to this job")
    for path in root.iterdir():
        if path == checkpoint or path.is_symlink() or not path.is_dir():
            continue
        if (path / "done").is_file():
            shutil.rmtree(path)


def run_job(directory, manifest, entry, gpu, address):
    from .tracking import publish_run_artifact
    from .evaluation import _validate_standalone_evaluation

    storage_check(directory, manifest["storage"])
    train = execute_phase(directory, manifest, entry, gpu, address, "train")
    checkpoint = final_checkpoint(
        train, entry["expected_updates"], int(entry["config"]["run.steps"])
    )
    evaluation = execute_phase(
        directory, manifest, entry, gpu, address, "final100", checkpoint
    )
    summary = json.loads((evaluation / "evaluation_summary.json").read_text())
    records = [
        json.loads(line)
        for line in (evaluation / "evaluation_episodes.jsonl").read_text().splitlines()
    ]
    _validate_standalone_evaluation(summary, records, 100)
    if summary.get("policy_mode") != "eval":
        raise ValueError("final evaluation must use greedy policy mode")
    metadata = dict(
        source_sha256=manifest["source_sha256"],
        base_commit=manifest["base_commit"],
        learner_update_calls=entry["expected_updates"],
    )
    destination = manifest["wandb"]
    update_worker(
        directory, gpu, status="running", phase="artifacts", run=entry["name"]
    )
    receipt = publish_run_artifact(
        destination["entity"],
        destination["project"],
        entry["wandb_id"],
        "checkpoint",
        [checkpoint],
        metadata,
    )
    evaluation_receipt = publish_run_artifact(
        destination["entity"],
        destination["project"],
        entry["evaluation_wandb_id"],
        "evaluation",
        [evaluation],
        metadata,
    )
    if (
        receipt.get("verified") is not True
        or evaluation_receipt.get("verified") is not True
    ):
        raise ValueError("artifact publication was not verified")
    update_job(
        directory,
        entry["name"],
        status="complete",
        finished_at=time.time(),
        checkpoint=str(checkpoint),
        checkpoint_artifact=receipt,
        evaluation_artifact=evaluation_receipt,
    )
    prune_checkpoints(directory, dict(entry, status="complete"), checkpoint, receipt)


def worker(directory, gpu):
    manifest = json.loads((directory / "job.json").read_text())
    if not 0 <= gpu < manifest["job"]["gpu_count"]:
        raise ValueError("worker GPU outside this pod's allocation")
    with (
        interruptible(),
        (Path(tempfile.gettempdir()) / f"majepa-gpu-{gpu}.lock").open("a") as lock,
    ):
        update_worker(directory, gpu, status="starting")
        try:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    update_worker(
                        directory, gpu, status="waiting", reason="GPU worker lock held"
                    )
                    if time.time() >= manifest["job"]["deadline"]:
                        raise TimeoutError("deadline waiting for GPU lock")
                    time.sleep(5)
            verify_source(directory, manifest)
            from .campaign import discover_sc2

            if manifest.get("sc2path"):
                os.environ["SC2PATH"] = str(manifest["sc2path"])
            maps = [
                entry["config"]["task"].removeprefix("smac_")
                for entry in manifest["runs"]
            ]
            os.environ["SC2PATH"] = str(discover_sc2(map_names=maps))
            with port_server(directory, gpu) as address:
                while time.time() < manifest["job"]["deadline"]:
                    reason = predecessor_wait(manifest.get("predecessor"))
                    pids = gpu_processes(gpu)
                    if reason or pids:
                        update_worker(
                            directory,
                            gpu,
                            status="waiting",
                            reason=reason or f"GPU busy: {pids}",
                        )
                        time.sleep(5)
                        continue
                    entry = claim(directory, gpu)
                    if entry is None:
                        state = json.loads((directory / "queue.json").read_text())
                        if not any(
                            item["status"] == "pending" for item in state["jobs"]
                        ):
                            break
                        update_worker(
                            directory, gpu, status="waiting", reason="run dependencies"
                        )
                        time.sleep(5)
                        continue
                    try:
                        run_job(directory, manifest, entry, gpu, address)
                    except BaseException as exc:
                        update_job(
                            directory,
                            entry["name"],
                            status="failed",
                            error=str(exc),
                            interrupted=isinstance(
                                exc, (InterruptedError, KeyboardInterrupt, TimeoutError)
                            ),
                            finished_at=time.time(),
                        )
                        if not isinstance(exc, Exception) or isinstance(
                            exc, (InterruptedError, TimeoutError)
                        ):
                            raise
            update_worker(directory, gpu, status="finished", reason=None)
        except BaseException as exc:
            update_worker(directory, gpu, status="failed", reason=str(exc))
            raise


def run(directory, dry_run=False):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "job.json").read_text())
    if dry_run:
        return {
            "gpu_count": manifest["job"]["gpu_count"],
            "runs": manifest["job"]["run_names"],
            "deadline": manifest["job"]["deadline"],
            "spawned": False,
        }
    with (directory / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = initialize(directory, manifest)
        if any(entry["status"] == "running" for entry in state["jobs"]):
            raise RuntimeError(
                "queue contains running jobs; reconcile interrupted work explicitly before restart"
            )
        children = []
        try:
            verify_source(directory, manifest)
            storage_check(directory, manifest["storage"])
            with interruptible():
                for gpu in range(manifest["job"]["gpu_count"]):
                    log = (directory / f"worker-{gpu}.log").open("a")
                    try:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "majepa.campaign_queue",
                                "--directory",
                                str(directory),
                                "--gpu",
                                str(gpu),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        children.append(process)
                    finally:
                        log.close()
                while any(process.poll() is None for process in children):
                    if time.time() >= manifest["job"]["deadline"]:
                        raise TimeoutError("pod runtime deadline reached")
                    with locked_state(directory) as state:
                        state.update(supervisor_pid=os.getpid(), heartbeat=time.time())
                    time.sleep(1)
                state = json.loads((directory / "queue.json").read_text())
                if any(process.returncode for process in children) or any(
                    item["status"] != "complete" for item in state["jobs"]
                ):
                    raise RuntimeError(
                        "queue did not complete successfully; inspect queue.json and worker logs"
                    )
            outcome = {"completed": True, "finished_at": time.time()}
        except BaseException as exc:
            outcome = {
                "completed": False,
                "error": str(exc),
                "finished_at": time.time(),
            }
            raise
        finally:
            for process in children:
                stop_child(process, grace=30)
            write_json(directory / "outcome.json", outcome)
        return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu is None or args.dry_run:
        print(json.dumps(run(args.directory, dry_run=args.dry_run)))
    else:
        worker(args.directory.resolve(), args.gpu)


if __name__ == "__main__":
    main()
