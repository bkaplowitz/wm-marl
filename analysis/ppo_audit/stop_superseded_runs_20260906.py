"""Preserve and stop only the two superseded experiment process trees."""

import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import time

ROOTS = [Path("/workspace/majepa_value_sweep_20260906"),
         Path("/workspace/majepa_bptt2_matrix_20260905")]
ARCHIVE = Path("/workspace/majepa_interrupted_for_interface_20260906")
MARKERS = ("/workspace/ma_jepa_value_sweep_47af44e/scripts/run_ppo_value_sweep.py",
           "/workspace/majepa_bptt2_final100_launcher_20260905/scripts/run_ppo_bptt2_matrix.py")


def processes():
    rows = {}
    for line in subprocess.check_output(["ps", "-eo", "pid,ppid,pgid,args"], text=True).splitlines()[1:]:
        pid, parent, group, command = line.strip().split(None, 3)
        rows[int(pid)] = dict(parent=int(parent), group=int(group), command=command)
    return rows


def start_id(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    ARCHIVE.mkdir(exist_ok=False)
    jobs = []
    for root in ROOTS:
        queue = json.loads((root / "queue.json").read_text())
        (ARCHIVE / f"{root.name}.queue-before.json").write_text(json.dumps(queue, indent=2) + "\n")
        for job in queue["jobs"]:
            if job["status"] != "running":
                continue
            record = dict(root=str(root), name=job["name"], index=job["index"], original_job=job)
            ckpt = root / "runs" / job["name"] / "train/run/ckpt"
            if (ckpt / "latest").exists():
                source = ckpt / (ckpt / "latest").read_text().strip()
                if (source / "done").exists():
                    target = ARCHIVE / job["name"] / source.name
                    target.mkdir(parents=True)
                    for path in source.iterdir():
                        if path.is_file():
                            os.link(path, target / path.name)
                    with (target / "step.pkl").open("rb") as stream:
                        record["preserved_step"] = int(pickle.load(stream))
                    record.update(checkpoint=str(target), source_checkpoint=str(source))
            jobs.append(record)
    rows = processes()
    workers = [pid for pid, row in rows.items()
               if any(marker in row["command"] for marker in MARKERS)
               and " worker " in row["command"]]
    owned = set(workers)
    while True:
        extra = {pid for pid, row in rows.items() if row["parent"] in owned} - owned
        if not extra:
            break
        owned.update(extra)
    identities = {pid: start_id(pid) for pid in owned}
    manifest = dict(at=time.time(), authorization="User requested all six GPUs for the new verification",
        workers=workers, processes={pid: rows[pid] for pid in owned}, jobs=jobs)
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for pid in workers:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # The supervisors perform their own process-group cleanup and W&B shutdown.
    deadline = time.time() + 60
    while time.time() < deadline:
        remaining = [pid for pid in owned if start_id(pid) == identities[pid]
                     and identities[pid] is not None]
        if not remaining:
            break
        time.sleep(2)
    forced = []
    for pid in owned:
        if identities[pid] is not None and start_id(pid) == identities[pid]:
            try:
                os.kill(pid, signal.SIGKILL)
                forced.append(pid)
            except ProcessLookupError:
                pass
    # The superseded keeper has no selected jobs left to watch.
    for pid, row in processes().items():
        if row["command"].endswith("/workspace/retain_diagnostic_checkpoints_20260906.py"):
            os.kill(pid, signal.SIGTERM)
    for root in ROOTS:
        with (root / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = root / "queue.json"
            queue = json.loads(path.read_text())
            original_running = {j["name"]: j for j in jobs if j["root"] == str(root)}
            for job in queue["jobs"]:
                if job["name"] in original_running and job["status"] != "complete":
                    record = original_running[job["name"]]
                    job.update(status="interrupted", interruption_reason="user_reallocated_all_six_gpus",
                        interrupted_at=time.time(), preserved_checkpoint=record.get("checkpoint"),
                        preserved_step=record.get("preserved_step"))
                elif job["status"] == "pending":
                    job.update(status="cancelled", cancellation_reason="user_reallocated_all_six_gpus")
            queue.update(phase="superseded", claims_closed=True, updated_at=time.time(),
                         replacement_root="/workspace/majepa_interface_verify_20260906")
            temp = root / "queue.json.tmp"
            temp.write_text(json.dumps(queue, indent=2) + "\n")
            temp.replace(path)
    manifest.update(finished_at=time.time(), forced_pids=forced)
    (ARCHIVE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(archive=str(ARCHIVE), preserved_jobs=len(jobs), forced_pids=forced)), flush=True)


if __name__ == "__main__":
    main()
