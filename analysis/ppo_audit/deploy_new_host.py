"""Launch the authorized 2026-09-05 rerun with remote and local evidence copies."""

from datetime import datetime, timezone
import hashlib
import json
import netrc
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
HOST = "root@154.54.102.56"
PORT = "14808"
SOURCE = "/workspace/ma_jepa_ppo_2663ae5"
ORIGINAL = "/workspace/ma_jepa_ppo_92014f1"
RESULTS = "/workspace/majepa_ppo_correction_2663ae5_20260905"
RUNTIME = "/workspace/majepa-runtime/bin/python"
GROUP = "ma-jepa-ppo-correction-2663ae5-20260905"
PREFIX = "corr-2663-20260905"
MIRROR = ROOT / "correction_20260905_mirror"


def ssh(command, payload=None):
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-i",
            str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            "-p",
            PORT,
            HOST,
            command,
        ],
        input=payload,
        capture_output=True,
        check=True,
        timeout=60,
    ).stdout.decode()


def py(code, payload=None):
    return json.loads(ssh("python3 -c " + shlex.quote(code), payload))


def main():
    record_path = ROOT / "deployment_20260905.json"
    if record_path.exists():
        earlier = json.loads(record_path.read_text())
        if earlier.get("mirror") or any(
            slot.get("launched") for slot in earlier.get("slots", [])
        ):
            raise RuntimeError(
                "A launch already began; inspect it rather than relaunching"
            )
    previous = json.loads((ROOT / "deployment.json").read_text())
    expected = previous["source_sha256"]
    expected_original = previous["hosts"]["jinn"]["hashes"]["original"]
    launcher = (REPO / "scripts/run_ppo_correction_screen.py").read_bytes()
    launcher_hash = hashlib.sha256(launcher).hexdigest()
    old_launcher = subprocess.check_output(
        ["git", "show", "2663ae5:scripts/run_ppo_correction_screen.py"], cwd=REPO
    )
    old_launcher_hash = hashlib.sha256(old_launcher).hexdigest()
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": HOST,
        "port": int(PORT),
        "source": SOURCE,
        "original_source": ORIGINAL,
        "results": RESULTS,
        "algorithm_commit": previous["approved_commit"],
        "source_sha256": expected,
        "original_commit": previous["hosts"]["jinn"]["hashes"]["original_commit"],
        "original_source_sha256": expected_original,
        "launcher_sha256": launcher_hash,
        "wandb_project": "osaze-obahor/majepa-ppo-treatments",
        "wandb_group": GROUP,
        "wandb_run_prefix": PREFIX,
        "local_mirror": str(MIRROR),
        "slots": [],
        "validation": {
            "local_launcher_and_mirror": "11 passed",
            "ruff": "passed",
            "remote_gpu_affected_tests": "10 passed (parent verified)",
            "remote_random_smac": "3m and 2s3z completed (parent verified)",
        },
    }

    def save():
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    # Transfer only the specifically authorized W&B authenticator, over SSH stdin.
    credentials = netrc.netrc(str(Path.home() / ".netrc")).authenticators(
        "api.wandb.ai"
    )
    if not credentials:
        raise RuntimeError("The authorized local W&B authenticator is unavailable")
    login, account, password = credentials
    minimal = (
        "machine api.wandb.ai login "
        + shlex.quote(login)
        + " password "
        + shlex.quote(password)
        + "\n"
    )
    auth = py(
        """
import json,netrc,os,pathlib,sys
path=pathlib.Path.home()/'.netrc'
present=bool(netrc.netrc(str(path)).authenticators('api.wandb.ai')) if path.exists() else False
if not present:
 existing=path.read_bytes() if path.exists() else b''
 incoming=sys.stdin.buffer.read()
 temporary=path.with_name('.netrc.majepa-tmp')
 fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as stream:stream.write(existing+b'\\n'+incoming)
 temporary.replace(path)
os.chmod(path,0o600)
print(json.dumps({'wandb_auth_present':True,'credential_file_mode':'0600'}))
""",
        minimal.encode(),
    )
    del minimal, credentials, login, account, password
    record["authentication"] = auth
    save()
    staged = py(
        f"""
import hashlib,json,pathlib,sys
path=pathlib.Path({(SOURCE + "/scripts/run_ppo_correction_screen.py")!r})
previous=hashlib.sha256(path.read_bytes()).hexdigest()
if previous not in {{{old_launcher_hash!r},{launcher_hash!r}}}:raise RuntimeError('Unexpected remote launcher content')
content=sys.stdin.buffer.read()
if hashlib.sha256(content).hexdigest()!={launcher_hash!r}:raise RuntimeError('Launcher transfer digest mismatch')
temporary=path.with_suffix('.py.tmp');temporary.write_bytes(content);temporary.replace(path)
print(json.dumps({{'launcher_sha256':{launcher_hash!r}}}))
""",
        launcher,
    )
    record["staged_launcher"] = staged
    hashes = py(f"""
import json,pathlib,runpy
fn=runpy.run_path({(SOURCE + "/scripts/run_ppo_correction_screen.py")!r})['source_fingerprint']
for source,expected,commit in [({SOURCE!r},{expected!r},{record["algorithm_commit"]!r}),({ORIGINAL!r},{expected_original!r},{record["original_commit"]!r})]:
 root=pathlib.Path(source)
 if fn(root)!=expected:raise RuntimeError('Package fingerprint mismatch before writing provenance')
 for filename,value in [('DEPLOYED_COMMIT',commit),('SOURCE_SHA256',expected)]:
  marker=root/filename
  if marker.exists() and marker.read_text().strip()!=value:raise RuntimeError('Conflicting source provenance marker')
  if not marker.exists():marker.write_text(value+'\\n')
print(json.dumps({{'corrected':fn(pathlib.Path({SOURCE!r})),'original':fn(pathlib.Path({ORIGINAL!r})),'corrected_commit':(pathlib.Path({SOURCE!r})/'DEPLOYED_COMMIT').read_text().strip(),'original_commit':(pathlib.Path({ORIGINAL!r})/'DEPLOYED_COMMIT').read_text().strip()}}))
""")
    if hashes["corrected"] != expected or hashes["original"] != expected_original:
        raise RuntimeError("Algorithm package fingerprint changed")
    if (
        hashes["corrected_commit"] != record["algorithm_commit"]
        or hashes["original_commit"] != record["original_commit"]
    ):
        raise RuntimeError("Algorithm commit markers do not match")
    record["verified_package_hashes"] = hashes
    auth_check = ssh(
        shlex.join(
            [
                RUNTIME,
                "-c",
                "import json,wandb; api=wandb.Api(timeout=30); print(json.dumps({'authenticated':bool(api.viewer)}))",
            ]
        )
    )
    record["wandb_api_check"] = json.loads(auth_check.strip().splitlines()[-1])
    save()
    for slot in range(6):
        source = ORIGINAL if slot == 5 else SOURCE
        source_hash = expected_original if slot == 5 else expected
        start_port = 16000 + 200 * slot
        command = [
            RUNTIME,
            SOURCE + "/scripts/run_ppo_correction_screen.py",
            "--slot",
            str(slot),
            "--source",
            source,
            "--expected-source-sha256",
            source_hash,
            "--experiment-root",
            RESULTS,
            "--python",
            RUNTIME,
            "--external",
            "/workspace/external/dreamerv3",
            "--sc2",
            "/workspace/StarCraftII",
            "--portserver-script",
            "/workspace/majepa-runtime/bin/portserver.py",
            "--portserver-address",
            f"@majepa-correction-20260905-s{slot}",
            "--portserver-pool",
            f"{start_port}-{start_port + 199}",
            "--gpu",
            str(slot),
            "--wandb-project",
            "majepa-ppo-treatments",
            "--wandb-entity",
            "osaze-obahor",
            "--wandb-group",
            GROUP,
            "--wandb-run-prefix",
            PREFIX,
        ]
        validation = json.loads(
            ssh(shlex.join(command + ["--validate-only"])).strip().splitlines()[-1]
        )
        record["slots"].append(
            {"slot": slot, "gpu": slot, "command": command, "validation": validation}
        )
        save()
    py(
        f"import json,pathlib; p=pathlib.Path({RESULTS!r}); p.mkdir(exist_ok=False); print(json.dumps({{'created':str(p)}}))"
    )
    # Start the independent local mirror before any training supervisor.
    MIRROR.mkdir(parents=True, exist_ok=False)
    mirror_command = [
        sys.executable,
        str(ROOT / "mirror_remote_results.py"),
        "--host",
        HOST,
        "--port",
        PORT,
        "--remote-root",
        RESULTS,
        "--local-root",
        str(MIRROR),
        "--expected-runs",
        "6",
        "--watch",
        "--interval",
        "120",
    ]
    with (MIRROR / "mirror_supervisor.log").open("x") as output:
        process = subprocess.Popen(
            mirror_command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record["mirror"] = {"pid": process.pid, "command": mirror_command}
    save()
    for slot in record["slots"]:
        launched = py(f"""
import json,pathlib,subprocess
root=pathlib.Path({RESULTS!r});slot={slot["slot"]}
with (root/f'slot-{{slot}}-supervisor.log').open('x') as output:
 process=subprocess.Popen({slot["command"]!r},cwd={SOURCE!r},stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
(root/f'slot-{{slot}}-supervisor.pid').write_text(str(process.pid)+'\\n')
print(json.dumps({{'pid':process.pid,'log':str(root/f'slot-{{slot}}-supervisor.log')}}))
""")
        slot["launched"] = launched
        save()
        print("queued slot", slot["slot"], "pid", launched["pid"], flush=True)


if __name__ == "__main__":
    main()
