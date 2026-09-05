"""Recover the completed second-wave baseline evidence using read-only W&B API."""

from datetime import datetime, timezone
import json
from pathlib import Path

import wandb

from collect_remote import DIAGNOSTIC_KEYS, SUMMARY_KEYS


ROOT = Path(__file__).resolve().parent
PROJECT = "osaze-obahor/majepa-ppo-treatments"
GROUP = "ma-jepa-ppo-treatments-92014f1"


if __name__ == "__main__":
    api = wandb.Api(timeout=30)
    result = {
        "recovered_utc": datetime.now(timezone.utc).isoformat(),
        "project": PROJECT,
        "group": GROUP,
        "runs": [],
    }
    runs = list(api.runs(PROJECT, filters={"group": GROUP}, per_page=30))
    finals = {run.id.removeprefix("f-"): run for run in runs if run.id.startswith("f-")}
    for run in runs:
        if run.id.startswith("f-"):
            continue
        summary = dict(run.summary)
        final = finals.get(run.id)
        final_summary = dict(final.summary) if final else {}
        curve = list(
            run.scan_history(
                keys=[
                    "_step",
                    "eval/win_rate",
                    "eval/return_mean",
                    "eval/timeout_rate",
                ],
                page_size=50001,
            )
        )
        data = {
            "train_id": run.id,
            "train_url": run.url,
            "train_state": run.state,
            "train_created_utc": run.created_at,
            "train_last_environment_step": summary.get("_step"),
            "last_logged_metrics": {
                k: v for k, v in summary.items() if k in DIAGNOSTIC_KEYS
            },
            "final_id": final.id if final else None,
            "final_url": final.url if final else None,
            "final_state": final.state if final else None,
            "final_created_utc": final.created_at if final else None,
            "final": {
                k.removeprefix("final_eval/"): v
                for k, v in final_summary.items()
                if k.startswith("final_eval/")
                and k.removeprefix("final_eval/") in SUMMARY_KEYS
            },
            "curve": curve,
        }
        result["runs"].append(data)
        (ROOT / "second_wave_recovered.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(
            run.id,
            "step",
            data["train_last_environment_step"],
            "final",
            data["final"].get("wins"),
            "/",
            data["final"].get("episodes"),
            "curve_points",
            len(curve),
            flush=True,
        )
