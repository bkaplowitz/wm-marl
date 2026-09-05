"""Read historical local summaries without loading checkpoints or changing runs."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from collect_remote import HOSTS, ROOT, SUMMARY_KEYS

REMOTE = r"""
import datetime,json,os,pathlib
ws=pathlib.Path('/workspace')
roots=[p for p in ws.iterdir() if p.is_dir() and p.name.startswith(('dreamarl','ma_jepa_entropy_screen')) and 'archive' not in p.name.lower()]
pruned={'ckpt','replay','.venv','wandb','external','.git','StarCraftII','source','source_base','__pycache__','STAGE_FAILED_APPLEDOUBLE_20260831T1951Z'}
out=[]
for root in roots:
    for dp,dirs,files in os.walk(root):
        dirs[:]=[x for x in dirs if x not in pruned and not x.startswith('._')]
        if len(pathlib.Path(dp).relative_to(root).parts)>7:
            dirs[:]=[]
        if 'evaluation_summary.json' in files:
            p=pathlib.Path(dp)/'evaluation_summary.json'
            try:d=json.loads(p.read_text())
            except Exception:continue
            out.append({'source':str(p),'mtime':p.stat().st_mtime,'summary':{k:v for k,v in d.items() if not isinstance(v,list)}})
print(json.dumps({'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluations':out},separators=(',',':')))
"""


def collect(item):
    name, (host, port) = item
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-i",
            str(Path.home() / ".runpod/ssh/runpodctl-ssh-key"),
            "-p",
            port,
            "root@" + host,
            "python3 -",
        ],
        input=REMOTE,
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    for evaluation in data["evaluations"]:
        evaluation["summary"] = {
            k: v for k, v in evaluation["summary"].items() if k in SUMMARY_KEYS
        }
    (ROOT / f"{name}_historical.json").write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n"
    )
    return name, len(data["evaluations"])


if __name__ == "__main__":
    for item in ThreadPoolExecutor(2).map(collect, HOSTS.items()):
        print(item)
