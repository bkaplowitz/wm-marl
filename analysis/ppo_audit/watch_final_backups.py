"""Back up each successful mirrored final128 outcome, independently of peers.

Runs one bandwidth-limited transfer at a time. Unfinished and failed experiments
cannot block completed peers. No source data is removed. Diagnostics exports can
be archived later with backup_final_artifacts.py --exports-only.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def candidates(mirror, destination, attempts, max_attempts):
    outcomes = {}
    ready = []
    for path in sorted((mirror / "runs").glob("*/outcome.json")):
        name = path.parent.name
        outcome = read(path)
        archive = read(destination / name / "backup_manifest.json")
        outcomes[name] = {
            "successful": bool(outcome.get("completed")),
            "backed_up": bool(archive.get("completed_utc")),
            "attempts": attempts.get(name, 0),
        }
        if (
            outcomes[name]["successful"]
            and not outcomes[name]["backed_up"]
            and attempts.get(name, 0) < max_attempts
        ):
            ready.append(name)
    return outcomes, ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    )
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=6)
    parser.add_argument("--bandwidth-kbps", type=int, default=64000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if (
        min(args.expected_runs, args.bandwidth_kbps, args.max_attempts) < 1
        or args.interval < 10
    ):
        raise ValueError("Require positive counts and interval >= 10 seconds")
    args.destination.mkdir(parents=True, exist_ok=True)
    status_path = args.destination / "watcher_status.json"
    previous = read(status_path)
    attempts = previous.get("attempts", {})
    next_try = {}
    active = None
    log = None
    while True:
        if active and active[1].poll() is not None:
            name, process, command = active
            log.close()
            print(
                json.dumps({"run": name, "exit_code": process.returncode}), flush=True
            )
            if process.returncode:
                next_try[name] = time.time() + 300
            active = None
        outcomes, ready = candidates(
            args.mirror, args.destination, attempts, args.max_attempts
        )
        ready = sorted(ready, key=lambda name: (attempts.get(name, 0), name))
        ready = [name for name in ready if time.time() >= next_try.get(name, 0)]
        if active is None and ready:
            name = ready[0]
            command = [
                sys.executable,
                str(Path(__file__).with_name("backup_final_artifacts.py")),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--identity",
                str(args.identity),
                "--remote-root",
                args.remote_root,
                "--run",
                name,
                "--destination",
                str(args.destination),
                "--bandwidth-kbps",
                str(args.bandwidth_kbps),
                "--execute",
            ]
            log = (args.destination / (name + "-backup.log")).open("ab", buffering=0)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            attempts[name] = attempts.get(name, 0) + 1
            active = name, process, command
            print(
                json.dumps(
                    {"run": name, "pid": process.pid, "attempt": attempts[name]}
                ),
                flush=True,
            )
        exhausted = [
            name
            for name, state in outcomes.items()
            if state["successful"]
            and not state["backed_up"]
            and attempts.get(name, 0) >= args.max_attempts
            and (not active or active[0] != name)
        ]
        settled = (
            len(outcomes) >= args.expected_runs
            and active is None
            and all(
                not state["successful"] or state["backed_up"] or name in exhausted
                for name, state in outcomes.items()
            )
        )
        atomic_json(
            status_path,
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "state": "finished_with_errors"
                if settled and exhausted
                else "finished"
                if settled
                else "watching",
                "active": {"run": active[0], "pid": active[1].pid, "command": active[2]}
                if active
                else None,
                "attempts": attempts,
                "outcomes": outcomes,
                "exhausted": exhausted,
            },
        )
        if settled:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
