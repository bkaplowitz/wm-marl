"""Incrementally mirror compact experiment evidence; never copy model weights.

Run with --watch beside a training screen. Each successful cycle saves new
complete JSONL records and atomically replaces manifests/outcomes. Connection
failures do not delete local evidence. The watcher stops once all expected
runs have outcomes and every selected file is caught up.
"""

import argparse
import base64
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time


REMOTE = r"""
import base64,gzip,json,pathlib,sys
request=json.load(sys.stdin);root=pathlib.Path(request['root'])
if not root.is_dir():raise FileNotFoundError(root)
offsets=request.get('offsets',{});files=[];outcomes=[];budget=16*1024*1024
patterns=[
 'runs/*/manifest.json','runs/*/status.json','runs/*/outcome.json',
 'runs/*/train/launch.json','runs/*/final128/launch.json',
 'runs/*/train/run/config.yaml','runs/*/final128/run/config.yaml',
 'runs/*/train/run/metrics.jsonl','runs/*/train/run/scores.jsonl',
 'runs/*/final128/run/metrics.jsonl','runs/*/final128/run/scores.jsonl',
 'runs/*/final128/run/evaluation_summary.json','runs/*/final128/run/evaluation_episodes.jsonl',
]
for pattern in patterns:
 for path in sorted(root.glob(pattern)):
  if not path.is_file() or path.is_symlink():continue
  relative=str(path.relative_to(root));size=path.stat().st_size
  is_lines=path.suffix=='.jsonl';offset=int(offsets.get(relative,0)) if is_lines else 0
  if offset>size:offset=0
  with path.open('rb') as stream:
   stream.seek(offset);chunk=stream.read(min(1024*1024,budget))
  if is_lines:
   end=chunk.rfind(b'\n')
   chunk=chunk[:end+1] if end>=0 else b''
  budget-=len(chunk)
  files.append({'path':relative,'offset':offset,'size':size,'jsonl':is_lines,'data':base64.b64encode(gzip.compress(chunk)).decode()})
  if path.name=='outcome.json':
   try:outcomes.append(json.loads(chunk))
   except json.JSONDecodeError:pass
  if budget<=0:break
 if budget<=0:break
checkpoints=[]
for path in sorted(root.glob('runs/*/train/run/ckpt/latest')):
 try:
  latest=path.read_text().strip();folder=path.parent/latest
  checkpoints.append({'run':str(path.relative_to(root)).split('/')[1],'latest':latest,'complete':(folder/'done').is_file(),'files':[{'name':p.name,'bytes':p.stat().st_size} for p in folder.iterdir() if p.is_file()]})
 except OSError:pass
print(json.dumps({'files':files,'outcomes':outcomes,'checkpoint_inventory':checkpoints,'budget_exhausted':budget<=0},separators=(',',':')))
"""


def failure_detail(error):
    """Retain connection diagnoses without echoing commands or arbitrary stderr."""
    detail = {"error_type": type(error).__name__}
    if isinstance(error, subprocess.CalledProcessError):
        detail["returncode"] = error.returncode
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        signatures = (
            "Connection timed out",
            "Connection refused",
            "Connection reset",
            "No route to host",
            "Could not resolve hostname",
            "Host key verification failed",
            "Permission denied",
            "Broken pipe",
            "Connection closed",
            "kex_exchange_identification",
        )
        detail["connection_errors"] = [text for text in signatures if text in stderr]
    if isinstance(error, subprocess.TimeoutExpired):
        detail["timeout_seconds"] = error.timeout
    return detail


def safe_target(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Remote evidence path must remain inside local mirror")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Local mirror symlink escapes its root")
    return target


def apply_file(root, record):
    target = safe_target(root, record["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    content = gzip.decompress(base64.b64decode(record["data"]))
    if record["jsonl"]:
        if record["offset"] == 0:
            if target.is_file() and target.stat().st_size:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                target.rename(
                    target.with_name(f"{target.stem}.before-reset-{stamp}.jsonl")
                )
            target.write_bytes(content)
        else:
            if not target.is_file() or target.stat().st_size != record["offset"]:
                raise ValueError("Local JSONL offset changed during collection")
            with target.open("ab") as stream:
                stream.write(content)
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return len(content), record["offset"] + len(content) == record["size"]


def collect(args):
    root = args.local_root
    offsets = {
        str(path.relative_to(root)): path.stat().st_size
        for path in root.rglob("*.jsonl")
    }
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-i",
        str(args.identity),
        "-p",
        str(args.port),
        args.host,
        "python3 -c " + shlex.quote(REMOTE),
    ]
    proc = subprocess.run(
        command,
        input=json.dumps({"root": args.remote_root, "offsets": offsets}),
        text=True,
        capture_output=True,
        check=True,
        timeout=45,
    )
    result = json.loads(proc.stdout)
    copied = 0
    caught_up = not result.get("budget_exhausted", False)
    for record in result["files"]:
        count, complete = apply_file(root, record)
        copied += count
        caught_up &= complete
    status = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "remote_host": args.host,
        "remote_root": args.remote_root,
        "files": len(result["files"]),
        "bytes_copied": copied,
        "caught_up": caught_up,
        "outcomes": len(result["outcomes"]),
        "completed": sum(bool(x.get("completed")) for x in result["outcomes"]),
        "checkpoint_inventory": result["checkpoint_inventory"],
    }
    (root / "mirror_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    )
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=6)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=120)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args()
    if args.interval < 10 or args.expected_runs < 1 or args.max_errors < 1:
        raise ValueError("Require interval>=10 seconds and positive run/error counts")
    args.local_root = args.local_root.absolute()
    args.local_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    while True:
        try:
            status = collect(args)
            error_path = args.local_root / "mirror_error.json"
            if failures and error_path.is_file():
                previous = json.loads(error_path.read_text())
                previous["resolved_utc"] = status["captured_utc"]
                error_path.write_text(json.dumps(previous, indent=2) + "\n")
            failures = 0
            print(
                json.dumps(
                    {k: v for k, v in status.items() if k != "checkpoint_inventory"}
                ),
                flush=True,
            )
            if not args.watch or (
                status["outcomes"] >= args.expected_runs and status["caught_up"]
            ):
                return
        except Exception as error:
            failures += 1
            # Avoid echoing command inputs, headers, or credential-bearing errors.
            status = {
                "captured_utc": datetime.now(timezone.utc).isoformat(),
                **failure_detail(error),
                "consecutive_errors": failures,
            }
            (args.local_root / "mirror_error.json").write_text(
                json.dumps(status, indent=2) + "\n"
            )
            print(json.dumps(status), flush=True)
            if not args.watch or failures >= args.max_errors:
                raise SystemExit(1) from None
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
