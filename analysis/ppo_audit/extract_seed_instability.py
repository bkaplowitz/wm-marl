"""Read-only extraction of scalar learning timelines from the experiment pod."""

import gzip
import json
from pathlib import Path
import sys
import time


PREFIXES = (
    "eval/", "episode/", "counters/", "train/ppo/", "train/opt/",
    "train/imagined_action/", "train/replay_views/", "train/loss/",
    "train/ctde/self_fed_train_h", "report/ctde/self_fed_h",
    "train/ctde/teammate_belief", "report/ctde/teammate_belief",
    "report/ctde/multistep_jepa_h", "report/sigreg/", "train/sigreg/",
    "report/dynamics_jepa/", "report/posterior_jepa/",
)
EXACT = {
    "step", "train/dyn_ent", "train/rep_ent", "report/dyn_ent", "report/rep_ent",
    "train/ctde/attack_mask_positive_recall", "train/ctde/attack_mask_target_rate",
    "train/ctde/attack_mask_prediction_rate", "train/ctde/controllable_alive_fraction",
    "report/ctde/multistep_jepa_action_counterfactual_enabled",
    "report/ctde/multistep_jepa_action_counterfactual_all_legal_enabled",
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def extract(root):
    queue = read_json(root / "queue.json")
    output = {"root": str(root), "queue": queue, "runs": []}
    for job in queue["jobs"]:
        if job["status"] == "pending":
            continue
        directory = root / "runs" / job["name"]
        metrics = directory / "train/run/metrics.jsonl"
        summary = read_json(directory / "final100/run/evaluation_summary.json")
        rows, malformed = [], 0
        if metrics.exists():
            with metrics.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    selected = {
                        key: value for key, value in row.items()
                        if isinstance(value, (int, float))
                        and (key in EXACT or key.startswith(PREFIXES))
                    }
                    if len(selected) > 1:
                        rows.append(selected)
        ckpt = directory / "train/run/ckpt"
        checkpoints = []
        if ckpt.exists():
            for path in sorted(ckpt.iterdir()):
                if path.is_dir():
                    checkpoints.append({
                        "path": str(path), "complete": (path / "done").exists(),
                        "bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file()),
                    })
        replay = directory / "train/run/replay"
        replay_files = list(replay.glob("*.npz")) if replay.exists() else []
        output["runs"].append({
            "job": job, "status": read_json(directory / "status.json"),
            "final": {k: v for k, v in summary.items() if not isinstance(v, (list, dict))}
            if summary else None,
            "rows": rows, "malformed_lines": malformed,
            "checkpoints": checkpoints, "replay_chunks": len(replay_files),
            "replay_bytes": sum(p.stat().st_size for p in replay_files),
            "metrics_bytes": metrics.stat().st_size if metrics.exists() else 0,
        })
    return output


if __name__ == "__main__":
    result = {
        "extracted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matrix": extract(Path("/workspace/majepa_bptt2_matrix_20260905")),
        "sweep": extract(Path("/workspace/majepa_value_sweep_20260906")),
    }
    with gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb") as stream:
        stream.write(json.dumps(result, allow_nan=True).encode())
