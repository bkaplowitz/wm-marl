"""Deploy the explicitly approved, tested 2663ae5 snapshot and six waiters."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMIT = "2663ae57ab80482ad201f9c3bdaa9fe2c732dbe9"
SOURCE = "/workspace/ma_jepa_ppo_2663ae5"
RESULTS = "/workspace/majepa_ppo_correction_2663ae5_20260904"
ORIGINAL = "/workspace/ma_jepa_ppo_92014f1"
EXTERNAL = "/workspace/ma_jepa_ppo_cf97200/external/dreamerv3"
HOSTS = {
    "jnin": {
        "address": "216.249.100.66",
        "port": "22309",
        "python": "/workspace/dreamarl/.venv/bin/python",
        "sc2": "/workspace/StarCraftII",
        "slots": [(0, 0, 3425918), (1, 1, 3425925), (2, 2, 3425920), (3, 3, 3425922)],
    },
    "jinn": {
        "address": "154.54.102.30",
        "port": "15924",
        "python": "/opt/dreamarl-smac/.venv/bin/python",
        "sc2": "/opt/StarCraftII",
        "slots": [(4, 0, 1661531), (5, 1, 1661529)],
    },
}


def ssh(host, command, input=None):
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-i",
            str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            "-p",
            host["port"],
            "root@" + host["address"],
            command,
        ],
        input=input,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout.decode()


def remote_python(host, code):
    return json.loads(ssh(host, "python3 -", code.encode()))


def fingerprint(source):
    digest = hashlib.sha256()
    for path in sorted((source / "src/majepa").rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".py", ".yaml", ".yml"}
            or path.name.startswith("._")
        ):
            continue
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def save(record):
    (ROOT / "deployment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )


def main():
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", COMMIT], cwd=REPO
    )
    with tempfile.TemporaryDirectory(prefix="majepa-tested-") as temporary:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(temporary, filter="data")
        expected_sha = fingerprint(Path(temporary))
    record = {
        "approved_commit": COMMIT,
        "source_sha256": expected_sha,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "source": SOURCE,
        "results": RESULTS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "full_suite": "83 passed in 205 seconds",
            "subsequent_truncation_replay_and_learner_tests": "5 passed in 51 seconds",
            "ruff": "passed",
        },
        "hosts": {},
    }
    save(record)
    for name, host in HOSTS.items():
        extraction = f"""
import io,json,pathlib,sys,tarfile
dest=pathlib.Path({SOURCE!r})
if dest.exists():raise RuntimeError('Refusing existing source directory')
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())) as tar:
    for member in tar.getmembers():
        target=(dest/member.name).resolve()
        if not target.is_relative_to(dest):raise RuntimeError('Archive path outside source')
    tar.extractall(dest)
(dest/'DEPLOYED_COMMIT').write_text({COMMIT!r}+'\\n')
(dest/'SOURCE_SHA256').write_text({expected_sha!r}+'\\n')
print(json.dumps({{'source':str(dest),'commit':{COMMIT!r}}}))
"""
        staged = json.loads(ssh(host, "python3 -c " + shlex.quote(extraction), archive))
        hashes = remote_python(
            host,
            f"""
import json,pathlib,runpy
namespace=runpy.run_path({(SOURCE + "/scripts/run_ppo_correction_screen.py")!r})
fn=namespace['source_fingerprint']
print(json.dumps({{'corrected':fn(pathlib.Path({SOURCE!r})),'original':fn(pathlib.Path({ORIGINAL!r})),'original_commit':(pathlib.Path({ORIGINAL!r})/'DEPLOYED_COMMIT').read_text().strip()}}))
""",
        )
        if hashes["corrected"] != expected_sha:
            raise RuntimeError(f"Staged source hash mismatch on {name}: {hashes}")
        host_record = {"staged": staged, "hashes": hashes, "slots": []}
        record["hosts"][name] = host_record
        save(record)
        commands = []
        for slot, gpu, predecessor in host["slots"]:
            source = ORIGINAL if slot == 5 else SOURCE
            source_sha = hashes["original"] if slot == 5 else expected_sha
            start_port = 16000 + 200 * gpu
            command = [
                host["python"],
                SOURCE + "/scripts/run_ppo_correction_screen.py",
                "--slot",
                str(slot),
                "--source",
                source,
                "--expected-source-sha256",
                source_sha,
                "--experiment-root",
                RESULTS,
                "--python",
                host["python"],
                "--external",
                EXTERNAL,
                "--sc2",
                host["sc2"],
                "--portserver-script",
                str(Path(host["python"]).parent / "portserver.py"),
                "--portserver-address",
                f"@majepa-correction-2663ae5-s{slot}",
                "--portserver-pool",
                f"{start_port}-{start_port + 199}",
                "--gpu",
                str(gpu),
                "--wait-pid",
                str(predecessor),
            ]
            validation = json.loads(
                ssh(host, shlex.join(command + ["--validate-only"]))
                .strip()
                .splitlines()[-1]
            )
            slot_record = {
                "slot": slot,
                "gpu": gpu,
                "predecessor_pid": predecessor,
                "command": command,
                "validation": validation,
            }
            host_record["slots"].append(slot_record)
            commands.append(slot_record)
            save(record)
        for slot_record in commands:
            launched = remote_python(
                host,
                f"""
import json,pathlib,subprocess
root=pathlib.Path({RESULTS!r});root.mkdir(exist_ok=True)
slot={slot_record["slot"]!r}
with (root/f'slot-{{slot}}-supervisor.log').open('x') as output:
    process=subprocess.Popen({slot_record["command"]!r},cwd={SOURCE!r},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
(root/f'slot-{{slot}}-supervisor.pid').write_text(str(process.pid)+'\\n')
print(json.dumps({{'supervisor_pid':process.pid,'log':str(root/f'slot-{{slot}}-supervisor.log')}}))
""",
            )
            slot_record["launched"] = launched
            save(record)
        print(
            name,
            "staged",
            expected_sha,
            "queued slots",
            [r["slot"] for r in commands],
            flush=True,
        )


if __name__ == "__main__":
    main()
