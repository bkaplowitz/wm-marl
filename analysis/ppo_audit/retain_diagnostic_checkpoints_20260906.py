"""Retain existing completed checkpoints without changing training or its RNG."""

import json
import os
from pathlib import Path
import pickle
import shutil
import time

SWEEP = Path("/workspace/majepa_value_sweep_20260906")
DEST = Path("/workspace/majepa_seed_checkpoint_archive_20260906")
ARMS = {"base", "actor1", "ret-ent", "frep-a1"}
WINDOWS = [(5000, 15000), (15000, 25000), (25000, 35000), (35000, 45000)]
CAP = 64 * 1024**3


def main():
    DEST.mkdir(exist_ok=False)
    manifest = {"created_at": time.time(), "arms": sorted(ARMS), "map": "3s_vs_3z",
                "seeds": [0, 1], "windows": WINDOWS, "maximum_checkpoints": 32,
                "maximum_bytes": CAP, "training_mutations": False,
                "checkpoints": [], "errors": []}
    seen, used = set(), 0
    while True:
        queue = json.loads((SWEEP / "queue.json").read_text())
        if time.time() >= queue["decision_deadline"]:
            break
        jobs = [j for j in queue["jobs"] if j["map"] == "3s_vs_3z"
                and j["seed"] in (0, 1) and j["arm"] in ARMS]
        for job in jobs:
            root = SWEEP / "runs" / job["name"] / "train/run/ckpt"
            for source in sorted(root.glob("*/done")):
                source = source.parent
                try:
                    with (source / "step.pkl").open("rb") as stream:
                        step = int(pickle.load(stream))
                    window = next((i for i, (lo, hi) in enumerate(WINDOWS)
                                   if lo <= step < hi), None)
                    key = (job["name"], window)
                    if window is None or key in seen:
                        continue
                    files = [p for p in source.iterdir() if p.is_file()]
                    size = sum(p.stat().st_size for p in files)
                    if used + size > CAP:
                        raise RuntimeError("Diagnostic checkpoint archive reached its byte cap")
                    target = DEST / job["name"] / f"step{step}-{source.name}"
                    staging = target.with_name(target.name + ".partial")
                    staging.mkdir(parents=True)
                    try:
                        for path in files:
                            os.link(path, staging / path.name)
                        staging.rename(target)
                    except Exception:
                        shutil.rmtree(staging)
                        raise
                    manifest["checkpoints"].append({"run": job["name"], "step": step,
                        "window": window, "source": str(source), "archive": str(target),
                        "bytes": size, "captured_at": time.time()})
                    used += size
                    seen.add(key)
                except FileNotFoundError:
                    # The normal checkpoint cleaner may win this read-only race.
                    continue
                except Exception as error:
                    manifest["errors"].append({"at": time.time(), "run": job["name"],
                                               "error": repr(error)})
        manifest.update(updated_at=time.time(), retained_bytes=used)
        tmp = DEST / "manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(DEST / "manifest.json")
        if jobs and all(j["status"] in ("complete", "failed") for j in jobs):
            break
        if manifest["errors"]:
            break
        time.sleep(30)
    (DEST / "done.json").write_text(json.dumps({"at": time.time(), "count": len(seen),
        "errors": manifest["errors"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
