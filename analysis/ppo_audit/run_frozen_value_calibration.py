"""Inspect or run one frozen calibration after final128 and GPU release.

Default is read-only inspection. --execute is only for an assigned, idle GPU;
it preserves the exact final checkpoint/configuration in a new directory first.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def final_run(root, name, alive=pid_alive):
    run = root / "runs" / name
    if not run.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("Run escapes experiment root")
    manifest = read_json(run / "manifest.json")
    outcome = read_json(run / "outcome.json")
    if (
        not outcome.get("completed")
        or read_json(run / "status.json").get("status") != "complete"
    ):
        raise ValueError(f"{name}: no successful completed outcome")
    if outcome.get("summary", {}).get("evaluation_protocol", {}).get("episodes") != 128:
        raise ValueError(f"{name}: final128 has not completed")
    supervisor = int((root / f"slot-{manifest['slot']}-supervisor.pid").read_text())
    if alive(supervisor):
        raise ValueError(f"{name}: supervisor {supervisor} has not exited")
    checkpoint = Path(outcome["checkpoint"])
    checkpoint_root = run / "train/run/ckpt"
    if not checkpoint.resolve().is_relative_to(checkpoint_root.resolve()):
        raise ValueError("Outcome checkpoint is outside its own training run")
    if not (checkpoint / "done").is_file():
        raise ValueError("Final checkpoint lacks done marker")
    # Only the named, trusted training run's scalar counter is unpickled here.
    with (checkpoint / "step.pkl").open("rb") as stream:
        step = int(pickle.load(stream))
    if step != 50000:
        raise ValueError(f"Expected final step 50000, got {step}")
    if not (checkpoint / "agent.pkl").is_file():
        raise ValueError("Final checkpoint lacks agent payload")
    return {
        "name": name,
        "run": str(run),
        "checkpoint": str(checkpoint),
        "step": step,
        "manifest": manifest,
        "supervisor_pid": supervisor,
    }


def gpu_state(gpu):
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--id",
            str(gpu),
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, memory, utilization = [value.strip() for value in line.split(",")]
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = [
        int(row.split(",")[1])
        for row in rows.splitlines()
        if row.split(",")[0].strip() == uuid
    ]
    return {
        "gpu": gpu,
        "uuid": uuid,
        "memory_mib": int(memory),
        "utilization_percent": int(utilization),
        "compute_pids": processes,
        "idle": not processes and int(memory) <= 64 and int(utilization) == 0,
    }


def verify_stage(stage, training):
    manifest = read_json(stage / "calibration_stage_manifest.json")
    actual_source = Path(training["manifest"]["source"])
    if not manifest.get("source_files_sha256"):
        raise ValueError("Calibration stage lacks source hashes")
    for name, expected in manifest["source_files_sha256"].items():
        if digest(stage / name) != expected or digest(actual_source / name) != expected:
            raise ValueError(f"Frozen model-source mismatch: {name}")
    for name, expected in manifest["diagnostic_scripts_sha256"].items():
        if digest(stage / "scripts" / name) != expected:
            raise ValueError(f"Frozen diagnostic-script mismatch: {name}")
    return manifest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def preserve(training, destination, stage_manifest):
    run, checkpoint = Path(training["run"]), Path(training["checkpoint"])
    frozen = destination / "frozen"
    target_checkpoint = frozen / "ckpt" / checkpoint.name
    target_checkpoint.mkdir(parents=True)
    files = []
    for source in sorted(checkpoint.iterdir()):
        if source.is_file():
            target = target_checkpoint / source.name
            os.link(source, target)
            files.append(
                {
                    "source": str(source),
                    "path": str(target.relative_to(destination)),
                    "bytes": target.stat().st_size,
                    "sha256": digest(target),
                    "preservation": "hardlink",
                }
            )
    for source, name in (
        (run / "train/run/config.yaml", "config.yaml"),
        (run / "manifest.json", "training_manifest.json"),
        (run / "outcome.json", "training_outcome.json"),
    ):
        target = frozen / name
        shutil.copy2(source, target)
        files.append(
            {
                "source": str(source),
                "path": str(target.relative_to(destination)),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "preservation": "copy",
            }
        )
    write_json(
        destination / "snapshot_manifest.json",
        {
            "created_utc": utc(),
            "training": training,
            "stage": stage_manifest,
            "files": files,
        },
    )
    return frozen, target_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        choices=(
            "original_base-2s3z-seed1",
            "team_return_anchor-2s3z-seed1",
            "team_return-2s3z-seed1",
        ),
    )
    parser.add_argument(
        "--release-run",
        required=True,
        help="Completed final128 run whose supervisor released the assigned GPU",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("/workspace/majepa_ppo_correction_2663ae5_20260905"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/workspace/majepa_value_calibration_20260905"),
    )
    parser.add_argument(
        "--python", type=Path, default=Path("/workspace/majepa-runtime/bin/python")
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    training = final_run(args.experiment_root, args.run)
    released = final_run(args.experiment_root, args.release_run)
    if released["manifest"]["gpu"] != args.gpu:
        raise ValueError("Release run did not own the assigned GPU")
    stage_manifest = verify_stage(args.stage, training)
    state = gpu_state(args.gpu)
    if not state["idle"]:
        raise ValueError(f"Assigned GPU is not idle: {state}")
    ready = {
        "checked_utc": utc(),
        "training": training,
        "release": released,
        "gpu": state,
        "stage": str(args.stage),
        "execute": args.execute,
    }
    print(json.dumps(ready), flush=True)
    if not args.execute:
        return 0
    time.sleep(5)
    if not gpu_state(args.gpu)["idle"]:
        raise ValueError("GPU became occupied before calibration launch")
    destination = args.destination / args.run
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "launch_readiness.json", ready)
    frozen, checkpoint = preserve(training, destination, stage_manifest)
    # Recheck after checkpoint hashing, immediately before allocating a device.
    if not gpu_state(args.gpu)["idle"]:
        raise ValueError("GPU became occupied during checkpoint preservation")
    cpus = sorted(os.sched_getaffinity(0))[-8:]
    os.sched_setaffinity(0, cpus)
    address = f"@majepa-value-calibration-20260905-gpu{args.gpu}"
    pool = f"{18000 + 200 * args.gpu}-{18199 + 200 * args.gpu}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        JAX_PLATFORMS="cuda",
        PYTHONPATH=f"{args.stage}/src:/workspace/external/dreamerv3",
        SC2PATH="/workspace/StarCraftII",
        PORTSERVER_ADDRESS=address,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OPENBLAS_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        OMP_NUM_THREADS="8",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        WANDB_MODE="disabled",
    )
    result = destination / "result32"
    command = [
        str(args.python),
        str(args.stage / "scripts/evaluate_value_calibration.py"),
        "--config",
        str(frozen / "config.yaml"),
        "--checkpoint",
        str(checkpoint),
        "--training-manifest",
        str(frozen / "training_manifest.json"),
        "--output",
        str(result),
        "--episodes",
        "32",
        "--envs",
        "1",
        "--eval-seed",
        "70001",
        "--worker-offset",
        "200000",
        "--diagnostic-seed",
        "9001",
        "--policy-mode",
        "eval_sample",
        "--max-driver-steps",
        "20000",
        "--platform",
        "cuda",
    ]
    started = time.monotonic()
    execution = {
        "started_utc": utc(),
        "command": command,
        "cpu_affinity": cpus,
        "gpu": args.gpu,
        "policy": "eval_sample",
        "environment_seed": 270001,
    }
    write_json(destination / "execution.json", execution)
    children = []

    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (destination / "portserver.log").open("x") as log:
            portserver = subprocess.Popen(
                [
                    str(args.python),
                    "/workspace/majepa-runtime/bin/portserver.py",
                    "--portserver_static_pool",
                    pool,
                    "--portserver_address",
                    address,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(portserver)
            time.sleep(2)
            if portserver.poll() is not None:
                raise RuntimeError("Diagnostic portserver failed")
        with (destination / "calibration.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=args.stage,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(process)
            (destination / "calibration.pid").write_text(str(process.pid) + "\n")
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        execution.update(
            exit_code=process.returncode,
            elapsed_seconds=time.monotonic() - started,
            completed_utc=utc(),
            user_seconds=usage.ru_utime,
            system_seconds=usage.ru_stime,
            maximum_rss_kib=usage.ru_maxrss,
        )
        write_json(destination / "execution.json", execution)
        if process.returncode:
            raise RuntimeError(
                f"Calibration exited {process.returncode}; inspect calibration.log"
            )
        summary = read_json(result / "summary.json")
        if not (result / "done.json").is_file() or summary.get("episodes") != 32:
            raise RuntimeError("Calibration did not complete 32 episodes")
        if summary["frozen_state_before"] != summary["frozen_state_after"]:
            raise RuntimeError("Frozen calibration state changed")
        if read_json(result / "provenance.json")["source"]["root"] != str(args.stage):
            raise RuntimeError("Calibration imported the wrong source package")
        shutil.copy2(
            args.stage / "calibration_stage_manifest.json",
            result / "calibration_stage_manifest.json",
        )
        print(
            json.dumps(
                {"completed": True, "output": str(result), "execution": execution}
            ),
            flush=True,
        )
    finally:
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
