"""Plan or copy one completed run's checkpoint, configuration, and replay.

Default mode only inspects file metadata. --execute requires a successful
final128 outcome for this run. Successful peers are independent of unfinished
or failed runs. --exports-only copies later diagnostic NPZ exports separately.
Model weights remain separate from the frequent metrics mirror.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


REMOTE_PLAN = r"""
import json,pathlib,sys
r=json.load(sys.stdin);root=pathlib.Path(r['root']);run=root/'runs'/r['run']
if not run.resolve().is_relative_to((root/'runs').resolve()):raise ValueError('Run outside experiment root')
def read(p):return json.loads(p.read_text()) if p.is_file() else {}
outcome=read(run/'outcome.json');reasons=[];files=[]
ready_marker=pathlib.Path(r['ready_marker']) if r.get('ready_marker') else None
if not outcome.get('completed'):reasons.append('This run has no successful final128 outcome')
latest=run/'train/run/ckpt/latest'
def add(path,relative):
 if not path.is_file():reasons.append('Missing '+str(path));return
 if path.is_symlink():reasons.append('Refusing symlink '+str(path));return
 stat=path.stat();files.append({'source':str(path),'relative':relative,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
if not r.get('exports_only') and latest.is_file():
 folder=pathlib.Path(outcome['checkpoint']) if outcome.get('checkpoint') else latest.parent/latest.read_text().strip()
 if not folder.resolve().is_relative_to(latest.parent.resolve()):raise ValueError('Checkpoint outside run')
 for name in ['agent.pkl','step.pkl','done']:add(folder/name,'checkpoint/'+folder.name+'/'+name)
 for path in sorted(folder.iterdir()):
  if path.is_file() and path.name not in ['agent.pkl','step.pkl','done']:add(path,'checkpoint/'+folder.name+'/'+path.name)
 add(latest,'checkpoint/latest')
elif not r.get('exports_only'):reasons.append('Final checkpoint pointer is missing')
for phase in ['train','final128']:add(run/phase/'run/config.yaml',phase+'/config.yaml')
for name in ['manifest.json','outcome.json']:add(run/name,name)
add(run/'final128/run/evaluation_summary.json','final128/evaluation_summary.json')
replay_files=[] if r.get('exports_only') else sorted((run/'train/run/replay').rglob('*.npz'))
if not r.get('exports_only') and not replay_files:reasons.append('Completed run has no raw replay NPZ files')
for path in replay_files:
 if path.is_file() and not path.is_symlink():add(path,'raw_replay/'+str(path.relative_to(run/'train/run/replay')))
for export in r['exports']:
 path=pathlib.Path(export)
 if path.suffix!='.npz' or not path.resolve().is_relative_to(pathlib.Path('/workspace')):raise ValueError('Expected an explicit workspace NPZ export')
 add(path,'replay_exports/'+path.name)
if ready_marker and ready_marker.is_file():add(ready_marker,'diagnostics_ready/'+ready_marker.name)
relative=[x['relative'] for x in files]
if len(set(relative))!=len(relative):raise ValueError('Backup destination collision')
print(json.dumps({'ready':not reasons,'reasons':reasons,'files':files,'raw_replay_files':len(replay_files),'source_manifest':read(run/'manifest.json'),'diagnostics_ready':str(ready_marker) if ready_marker else None},separators=(',',':')))
"""


def local_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--identity", type=Path, default=Path.home() / ".runpod/ssh/runpodctl-ssh-key"
    )
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--diagnostics-ready")
    parser.add_argument("--replay-export", action="append", default=[])
    parser.add_argument("--exports-only", action="store_true")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "/Users/osaze/MARL/remote_archives/majepa_ppo_correction_20260905"
        ),
    )
    parser.add_argument("--bandwidth-kbps", type=int, default=256000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    ssh = [
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
    ]
    request = {
        "root": args.remote_root,
        "run": args.run,
        "ready_marker": args.diagnostics_ready,
        "exports": args.replay_export,
        "exports_only": args.exports_only,
    }
    plan = json.loads(
        subprocess.check_output(
            ssh + ["python3 -c " + shlex.quote(REMOTE_PLAN)],
            input=json.dumps(request),
            text=True,
            timeout=30,
        )
    )
    total = sum(x["bytes"] for x in plan["files"])
    ancestor = args.destination.absolute()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    print(
        json.dumps(
            {
                "ready": plan["ready"],
                "reasons": plan["reasons"],
                "files": len(plan["files"]),
                "bytes": total,
                "local_free_bytes": free,
                "execute": args.execute,
            }
        ),
        flush=True,
    )
    if not args.execute:
        return
    if not plan["ready"]:
        raise RuntimeError("Final artifact backup is not ready")
    if free < total * 1.1 + 2 * 1024**3:
        raise RuntimeError("Insufficient free local disk with reserve")
    destination = args.destination.absolute() / args.run
    if not destination.resolve().is_relative_to(args.destination.resolve()):
        raise ValueError("Run name escaped local archive root")
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "remote_host": args.host,
        "remote_root": args.remote_root,
        "run": args.run,
        "plan": plan,
        "copied": [],
    }
    suffix = (
        "diagnostic_backup_manifest.json"
        if args.exports_only
        else "backup_manifest.json"
    )
    manifest = destination / suffix
    for file in plan["files"]:
        code = f"""
import hashlib,json,pathlib
p=pathlib.Path({file["source"]!r});h=hashlib.sha256();size=0
with p.open('rb') as stream:
 for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk);size+=len(chunk)
print(json.dumps({{'sha256':h.hexdigest(),'bytes':size}}))
"""
        fingerprint = json.loads(
            subprocess.check_output(
                ssh + ["python3 -c " + shlex.quote(code)], text=True, timeout=1800
            )
        )
        if fingerprint["bytes"] != file["bytes"]:
            raise RuntimeError("Remote artifact changed since planning")
        target = destination / file["relative"]
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Backup destination escaped its root")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or local_digest(target) != fingerprint["sha256"]:
            temporary = target.with_suffix(target.suffix + ".partial")
            command = [
                "scp",
                "-q",
                "-i",
                str(args.identity),
                "-P",
                str(args.port),
                "-l",
                str(args.bandwidth_kbps),
                args.host + ":" + file["source"],
                str(temporary),
            ]
            subprocess.run(command, check=True, timeout=3600)
            if local_digest(temporary) != fingerprint["sha256"]:
                raise RuntimeError("Transferred artifact checksum mismatch")
            temporary.replace(target)
        record["copied"].append({**file, **fingerprint, "local": str(target)})
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print("verified", file["relative"], fingerprint["bytes"], flush=True)
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
