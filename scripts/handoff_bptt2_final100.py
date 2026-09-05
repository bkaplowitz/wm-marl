#!/usr/bin/env python3
"""One-time, owned-process handoff to final100 without restarting training."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import run_ppo_bptt2_matrix as matrix


def main():
    root = matrix.ROOT
    launcher = Path(__file__).resolve().parent
    previous_launcher = "/workspace/majepa_bptt2_matrix_launcher_20260905/scripts"
    validations = json.loads(
        (root / "validated_configurations_final100.json").read_text()
    )
    assert len(validations) == 42
    assert all(
        v["final100"]["episodes"] == 100 and v["train"]["curve_eps"] == 32
        for v in validations.values()
    )
    records = []
    frozen = []
    try:
        with matrix.queue_state(root) as state:
            active = [j for j in state["jobs"] if j["status"] == "running"]
            assert len(active) == 6 and len({j["gpu"] for j in active}) == 6
            for job in active:
                run_root = root / "runs" / job["name"]
                assert (
                    json.loads((run_root / "status.json").read_text())["status"]
                    == "train"
                )
                assert not list(run_root.glob("final*"))
                worker = matrix.process_identity(job["worker_pid"])
                supervisor = matrix.process_identity(job["supervisor_pid"])
                train = matrix.process_identity(
                    int((run_root / "train/pid").read_text())
                )
                assert (
                    previous_launcher + "/run_ppo_bptt2_matrix.py" in worker["command"]
                )
                assert "worker" in worker["command"]
                assert (
                    previous_launcher + "/run_ppo_bptt2_matrix.py"
                    in supervisor["command"]
                )
                assert "job" in supervisor["command"]
                assert (
                    "majepa.main" in train["command"]
                    and str(run_root / "train/run") in train["command"]
                )
                servers = []
                for path in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = path.read_text().rsplit(")", 1)[1].split()
                        if int(fields[1]) != supervisor["pid"]:
                            continue
                        process = matrix.process_identity(int(path.parent.name))
                        if (
                            process
                            and "/workspace/majepa-runtime/bin/portserver.py"
                            in process["command"]
                        ):
                            servers.append(process)
                    except FileNotFoundError:
                        continue
                assert len(servers) == 1
                assert os.getpgid(train["pid"]) == train["pid"]
                assert os.getpgid(servers[0]["pid"]) == servers[0]["pid"]
                assert worker["pid"] not in (train["pid"], servers[0]["pid"])
                assert supervisor["pid"] not in (train["pid"], servers[0]["pid"])
                records.append(
                    dict(
                        index=job["index"],
                        name=job["name"],
                        gpu=job["gpu"],
                        worker=worker,
                        supervisor=supervisor,
                        train=train,
                        portserver=servers[0],
                    )
                )
            # Freeze only control processes before replacing their ownership records.
            for record in records:
                for key in ("worker", "supervisor"):
                    process = record[key]
                    assert matrix.same_process(process)
                    os.kill(process["pid"], signal.SIGSTOP)
                    frozen.append(process)
            for record in records:
                assert matrix.same_process(record["train"])
                state["jobs"][record["index"]]["handoff"] = record
            for job in state["jobs"]:
                job["final_episodes"] = 100
                job["wandb_final"] = (
                    job["wandb_final"].removesuffix("final128") + "final100"
                )
            state["evaluation_amendment"] = dict(
                final_episodes=100, curve_episodes=32, changed_at=time.time()
            )
        matrix.base.atomic_json(
            root / "final100_handoff.json",
            dict(records=records, changed_at=time.time()),
        )
        # SIGTERM would run the old supervisors' cleanup and stop training.
        # Kill only the frozen supervisors; their separately sessioned children survive.
        for process in frozen:
            assert matrix.same_process(process)
            os.kill(process["pid"], signal.SIGKILL)
        frozen.clear()
        assert all(
            matrix.same_process(r["train"]) and matrix.same_process(r["portserver"])
            for r in records
        )
        env = os.environ.copy()
        for key in (
            "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES",
            "WANDB_RUN_ID",
            "WANDB_NAME",
            "WANDB_RESUME",
        ):
            env.pop(key, None)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        workers = []
        for gpu in range(6):
            command = [
                str(matrix.PYTHON),
                str(launcher / "run_ppo_bptt2_matrix.py"),
                "worker",
                "--gpu",
                str(gpu),
            ]
            with (root / "workers" / f"gpu{gpu}-final100.log").open("xb") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            workers.append(dict(gpu=gpu, pid=process.pid, command=command))
        amendment = dict(
            final_episodes=100,
            curve_episodes=32,
            workers=workers,
            training_pids_preserved=[r["train"]["pid"] for r in records],
            launcher=str(launcher.parent),
            changed_at=time.time(),
        )
        matrix.base.atomic_json(root / "final100_deployment.json", amendment)
        print(json.dumps(amendment), flush=True)
    finally:
        for process in frozen:
            if matrix.same_process(process):
                os.kill(process["pid"], signal.SIGCONT)


if __name__ == "__main__":
    main()
