"""Run frozen campaign batches, releasing every predecessor pod first."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from wandb.errors import CommError

from . import campaign


def step(directory, client):
    client.flush()
    config = json.loads((directory / "sequence.json").read_text())
    budget = campaign.positive_finite(config["budget"])
    state_path = directory / "sequence-state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "completed": {},
            "pod_stopped_at": {},
            "stage": 0,
        }
    )
    pods = {p["id"]: p for p in campaign.list_pods()}

    def save(status, **details):
        state.update(status=status, checked_at=time.time(), **details)
        campaign.write_json(state_path, state)
        print(
            json.dumps({"status": status, "stage": state["stage"], **details}),
            flush=True,
        )
        return status

    def finish(paths):
        done = True
        for value in paths:
            path = Path(value).resolve()
            key = str(path)
            if key in state["completed"]:
                continue
            manifest = json.loads((path / "manifest.json").read_text())
            jobs = [j for j in manifest["jobs"] if j.get("pod_id")]
            if not jobs:
                done = False
                continue
            active = any(
                pods.get(j["pod_id"], {}).get("desiredStatus") == "RUNNING"
                for j in jobs
            )
            report_path = path / "sequence-results.json"
            if not report_path.exists():
                try:
                    report = campaign.collect_results(manifest, client)
                except (ValueError, CommError):
                    if active:
                        done = False
                        continue
                    raise
                campaign.write_json(report_path, report)
            remaining = False
            for job in jobs:
                pod_id = job["pod_id"]
                pod = pods.get(pod_id)
                if pod is None:
                    state["pod_stopped_at"].setdefault(
                        pod_id, job.get("stopped_at", time.time())
                    )
                    continue
                remaining = True
                if pod.get("desiredStatus") == "RUNNING":
                    campaign.api("pods/" + pod_id + "/stop", method="POST")
                elif pod.get("desiredStatus") == "EXITED":
                    state["pod_stopped_at"].setdefault(
                        pod_id, job.get("stopped_at", time.time())
                    )
                    with campaign.locked_manifest(path) as current:
                        for entry in current["jobs"]:
                            if entry.get("pod_id") == pod_id:
                                entry.update(
                                    stopped_at=state["pod_stopped_at"][pod_id],
                                    status="stopped",
                                )
                    campaign.write_json(state_path, state)
                    subprocess.run(["runpodctl", "pod", "delete", pod_id], check=True)
                else:
                    raise RuntimeError(
                        f"unknown pod state for {pod_id}: {pod.get('desiredStatus')}"
                    )
            if remaining:
                done = False
                continue
            cost = sum(
                max(0, state["pod_stopped_at"][j["pod_id"]] - j["created_at"])
                / 3600
                * float(j["actual_rate"])
                for j in jobs
            )
            state["completed"][key] = {
                "gpu_cost": cost,
                "pod_ids": [j["pod_id"] for j in jobs],
            }
            campaign.write_json(state_path, state)
        return done

    if not finish(config["predecessors"]):
        return save("waiting", reason="predecessor completion and pod teardown")
    stages = config["stages"]
    if state["stage"] >= len(stages):
        return save("complete")
    stage = stages[state["stage"]]
    if finish(stage):
        state["stage"] += 1
        return save("waiting", reason="batch complete; pods removed")
    spent = float(config.get("spent_before", 0)) + sum(
        c["gpu_cost"] for c in state["completed"].values()
    )
    manifests = [json.loads((Path(p) / "manifest.json").read_text()) for p in stage]
    reservation = sum(
        m["rate"] * m["max_hours"] * sum(a["gpu_count"] for a in m["allocations"])
        for p, m in zip(stage, manifests)
        if str(Path(p).resolve()) not in state["completed"]
    )
    state.update(spent=spent, next_batch_reservation=reservation, budget=budget)
    if spent + reservation > budget - 0.5:
        return save(
            "budget_blocked", reason="cumulative cap cannot cover the next batch"
        )
    for value in stage:
        path = Path(value).resolve()
        if (
            str(path) not in state["completed"]
            and not (path / "sequence-results.json").exists()
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(path / "controller.py"),
                    "launch",
                    "--directory",
                    str(path),
                    "--once",
                ],
                check=True,
            )
    return save("running")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    config = json.loads((directory / "sequence.json").read_text())
    paths = config["predecessors"] + [p for stage in config["stages"] for p in stage]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate campaign directory in sequence")
    for stage in config["stages"]:
        if not 1 <= len(stage) <= 2:
            raise ValueError("a batch must contain one or two campaign pods")
        for value in stage:
            manifest = json.loads((Path(value) / "manifest.json").read_text())
            campaign.verify_manifest(manifest)
            if (
                len(manifest["allocations"]) != 1
                or manifest["allocations"][0]["gpu_count"] != 4
            ):
                raise ValueError("each campaign must allocate exactly one four-GPU pod")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    import wandb

    with (directory / "sequence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = wandb.Api(timeout=30)
        while True:
            try:
                status = step(directory, client)
            except Exception as exc:
                campaign.write_json(
                    directory / "sequence-error.json",
                    {
                        "at": time.time(),
                        "error": str(exc),
                        "type": type(exc).__name__,
                    },
                )
                raise
            if args.once or status == "complete":
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
