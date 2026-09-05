"""Take a compact read-only snapshot of the correction screen and its waiters."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from deploy_screen import HOSTS, ROOT

REMOTE = r"""
import datetime,json,pathlib,os
root=pathlib.Path('/workspace/majepa_ppo_correction_2663ae5_20260904')
keys={
 'counters/environment_steps','counters/learner_update_calls','counters/ppo_update_calls',
 'fps/policy','fps/train','train/ppo/batch_reward','train/ppo/batch_return',
 'train/ppo/batch_target_value','train/ppo/critic/explained_variance',
 'train/ppo/critic/rmse','train/ppo/actor/final_exact_kl','train/ppo/actor/final_clip_fraction',
 'train/ppo/actor/entropy','train/ppo/epochs','train/ppo/active',
 'train/opt/actor/active','train/opt/actor/updates','train/opt/critic/active',
 'train/opt/critic/updates','train/opt/local_world/updates','train/opt/joint_world/updates',
 'train/opt/actor/skipped','train/opt/critic/skipped','train/opt/local_world/skipped',
 'train/ctde/team_signal_valid_fraction','train/ctde/controllable_alive_fraction',
 'train/ppo/batch_critic_valid_fraction','train/ppo/batch_valid_fraction',
}
def read(p):return json.loads(p.read_text()) if p.is_file() else {}
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':[]}
for d in sorted((root/'runs').glob('*')):
    latest={};curve=[];laststep=0
    m=d/'train/run/metrics.jsonl'
    if m.is_file():
        for line in m.open():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            laststep=max(laststep,r.get('step',0))
            if 'eval/win_rate' in r:curve.append({k:v for k,v in r.items() if k in {'step','eval/win_rate','eval/return_mean','eval/timeout_rate'}})
            latest.update({k:v for k,v in r.items() if k in keys or k.startswith(('train/ppo/replay','train/ppo/critic/replay','train/replay_value/'))})
    manifest=read(d/'manifest.json');status=read(d/'status.json');outcome=read(d/'outcome.json')
    out['runs'].append({'name':d.name,'slot':manifest.get('slot'),'gpu':manifest.get('gpu'),'status':status,'last_step':laststep,'latest':latest,'curve':curve,'completed':outcome.get('completed'),'error':outcome.get('error'),'final':{k:v for k,v in outcome.get('summary',{}).items() if k in {'win_rate','wins','episodes','return_mean','timeout_rate'}},'source_commit':manifest.get('source_commit'),'source_sha256':manifest.get('source_sha256')})
print(json.dumps(out,separators=(',',':')))
"""


def collect(item):
    name, host = item
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-i",
            str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            "-p",
            host["port"],
            "root@" + host["address"],
            "python3 -",
        ],
        input=REMOTE,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return name, json.loads(proc.stdout)


if __name__ == "__main__":
    data = dict(ThreadPoolExecutor(2).map(collect, HOSTS.items()))
    path = ROOT / "correction_status.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    deployment = ROOT / "deployment.json"
    if deployment.is_file():
        record = json.loads(deployment.read_text())
        record["latest_status_utc"] = datetime.now(timezone.utc).isoformat()
        for host, value in data.items():
            record["hosts"][host]["last_observed_runs"] = value["runs"]
        deployment.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for host, value in data.items():
        print(host, value["captured_utc"])
        for run in value["runs"]:
            print(
                run["slot"],
                run["name"],
                run["status"].get("status"),
                "step",
                run["last_step"],
                "ppo_updates",
                run["latest"].get("counters/ppo_update_calls"),
                "latest_eval",
                run["curve"][-1:] or None,
                "error",
                run["error"],
            )
