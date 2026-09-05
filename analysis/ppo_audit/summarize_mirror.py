"""Read compact learner diagnostics and evaluation curves from local evidence.

This performs no network requests. Each retained metric includes its actual
measurement step, so an old report is not confused with the latest environment
step. Use --json for structured output and --output to save a compact snapshot.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


KEYS = {
    "counters/environment_steps",
    "counters/learner_update_calls",
    "counters/ppo_update_calls",
    "fps/policy",
    "fps/train",
    "train/ppo/batch_reward",
    "train/ppo/batch_return",
    "train/ppo/batch_target_value",
    "train/ppo/critic/explained_variance",
    "train/ppo/critic/rmse",
    "train/ppo/actor/final_exact_kl",
    "train/ppo/actor/final_clip_fraction",
    "train/ppo/actor/entropy",
    "train/ppo/epochs",
    "train/ppo/active",
    "train/ctde/team_signal_valid_fraction",
    "train/ctde/controllable_alive_fraction",
    "train/ppo/batch_critic_valid_fraction",
    "train/ppo/batch_valid_fraction",
    "replay/behavior_sample_age",
    "replay/sample_age",
}
PREFIXES = (
    "train/opt/",
    "train/ppo/replay",
    "train/ppo/critic/replay",
    "train/replay_value/",
)
CURVE_KEYS = {"step", "eval/win_rate", "eval/return_mean", "eval/timeout_rate"}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root):
    runs = []
    for directory in sorted((root / "runs").glob("*")):
        manifest = read(directory / "manifest.json")
        outcome = read(directory / "outcome.json")
        latest, curves = {}, {}
        first_learner, last_step, malformed = None, None, 0
        path = directory / "train/run/metrics.jsonl"
        if path.is_file():
            for line in path.open():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                step = row.get("step")
                if step is not None:
                    last_step = step if last_step is None else max(last_step, step)
                if first_learner is None and any(
                    key.startswith("train/opt/") for key in row
                ):
                    first_learner = step
                latest.update(
                    {
                        key: {"step": step, "value": value}
                        for key, value in row.items()
                        if key in KEYS or key.startswith(PREFIXES)
                    }
                )
                if "eval/win_rate" in row:
                    curves[step] = {
                        key: value for key, value in row.items() if key in CURVE_KEYS
                    }
        final = outcome.get("summary") or read(
            directory / "final128/run/evaluation_summary.json"
        )
        runs.append(
            {
                "name": directory.name,
                "slot": manifest.get("slot"),
                "run": manifest.get("run"),
                "status": read(directory / "status.json"),
                "environment_step": last_step,
                "first_learner_metric_step": first_learner,
                "latest": latest,
                "curve": [curves[step] for step in sorted(curves)],
                "final": final,
                "completed": outcome.get("completed"),
                "error": outcome.get("error"),
                "source_commit": manifest.get("source_commit"),
                "malformed_records": malformed,
            }
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mirror": read(root / "mirror_status.json"),
        "runs": sorted(
            runs, key=lambda run: (run["slot"] is None, run["slot"] or 0, run["name"])
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "correction_20260905_mirror",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(serialized)
    if args.json:
        print(serialized, end="")
        return
    print(
        "Evidence captured:",
        result["mirror"].get("captured_utc"),
        "caught up:",
        result["mirror"].get("caught_up"),
    )
    for run in result["runs"]:
        ppo = run["latest"].get("counters/ppo_update_calls", {}).get("value")
        curve = run["curve"][-1] if run["curve"] else None
        print(
            run["slot"],
            run["name"],
            run["status"].get("status"),
            "step",
            run["environment_step"],
            "first learner",
            run["first_learner_metric_step"],
            "PPO updates",
            ppo,
            "latest eval",
            json.dumps(curve),
        )
        if run["error"]:
            print("  error:", run["error"])


if __name__ == "__main__":
    main()
