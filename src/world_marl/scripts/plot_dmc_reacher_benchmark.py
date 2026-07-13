"""Create a DMC Reacher benchmark report for JEPA against paper references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


DREAMERV3_DMC_REACHER_EASY_REFERENCES = [
    {
        "method": "DreamerV3",
        "return": 947.1,
        "source": "DreamerV3 paper, Table O.1, Proprio Control",
        "notes": "Published DMC Reacher Easy score at 500K environment steps.",
    },
    {
        "method": "D4PG",
        "return": 941.5,
        "source": "DreamerV3 paper, Table O.1, Proprio Control",
        "notes": "Published model-free baseline in the DreamerV3 DMC table.",
    },
    {
        "method": "DMPO",
        "return": 965.1,
        "source": "DreamerV3 paper, Table O.1, Proprio Control",
        "notes": "Published model-free baseline in the DreamerV3 DMC table.",
    },
    {
        "method": "MPO",
        "return": 954.4,
        "source": "DreamerV3 paper, Table O.1, Proprio Control",
        "notes": "Published model-free baseline in the DreamerV3 DMC table.",
    },
    {
        "method": "DDPG",
        "return": 921.8,
        "source": "DreamerV3 paper, Table O.1, Proprio Control",
        "notes": "Published model-free baseline in the DreamerV3 DMC table.",
    },
]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve_rows, summary_rows = load_jepa_rows(
        args.jepa_root,
        env=args.env,
        step_limit=args.step_limit,
    )
    reference_rows = reference_scores(args)

    write_csv(args.out_dir / "dmc_reacher_return_curve.csv", curve_rows)
    write_csv(args.out_dir / "dmc_reacher_benchmark_summary.csv", summary_rows)
    write_csv(args.out_dir / "dmc_reacher_reference_scores.csv", reference_rows)
    plot_return_curve(
        args.out_dir / "dmc_reacher_return_vs_env_steps.png",
        curve_rows,
        reference_rows,
        title=args.title,
        step_limit=args.step_limit,
    )
    write_report(
        args.out_dir / "dmc_reacher_benchmark.md",
        curve_rows,
        summary_rows,
        reference_rows,
        env=args.env,
        step_limit=args.step_limit,
    )

    print(f"Wrote DMC benchmark report to {args.out_dir}")
    print(f"- {args.out_dir / 'dmc_reacher_return_vs_env_steps.png'}")
    print(f"- {args.out_dir / 'dmc_reacher_return_curve.csv'}")
    print(f"- {args.out_dir / 'dmc_reacher_benchmark_summary.csv'}")
    print(f"- {args.out_dir / 'dmc_reacher_reference_scores.csv'}")
    print(f"- {args.out_dir / 'dmc_reacher_benchmark.md'}")
    if not curve_rows:
        print("WARNING: no JEPA online actor-replay curve rows were found.")
    if args.ppo_reference_return is None:
        print(
            "NOTE: DreamerV3's DMC proprioceptive Reacher/easy table does not "
            "report PPO. Pass --ppo-reference-return or add a matched PPO run "
            "before claiming a PPO comparison."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jepa-root",
        type=Path,
        required=True,
        help="Root containing JEPA task/seed run directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for PNG, CSV, and markdown artifacts.",
    )
    parser.add_argument(
        "--env",
        default="reacher/easy",
        help="DMC task name. The built-in paper references are for reacher/easy.",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=500_000,
        help="Training-replay environment-step budget for the comparison plot.",
    )
    parser.add_argument(
        "--title",
        default="Reacher Easy",
        help="Plot title.",
    )
    parser.add_argument(
        "--ppo-reference-return",
        type=float,
        default=None,
        help=(
            "Optional PPO score to include as a reference line. Use only for a "
            "matched DMC Reacher/easy PPO run or a clearly cited external source."
        ),
    )
    parser.add_argument(
        "--ppo-reference-source",
        default="user-provided PPO reference",
        help="Source label for --ppo-reference-return.",
    )
    parser.add_argument(
        "--no-paper-baselines",
        action="store_true",
        help="Do not include DreamerV3 Table O.1 reference scores.",
    )
    return parser.parse_args()


def load_jepa_rows(
    root: Path,
    *,
    env: str,
    step_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        summary = load_json(summary_path)
        job = summary_path.parts[-3]
        for run_index, run in enumerate(summary.get("runs", [])):
            label = f"JEPA {job}"
            online_history = run.get("online_history", [])
            curve_rows.extend(
                online_curve_rows(
                    online_history,
                    label=label,
                    job=job,
                    run_index=run_index,
                    summary_path=summary_path,
                    step_limit=step_limit,
                )
            )
            summary_rows.append(
                summary_row(
                    summary,
                    run,
                    label=label,
                    job=job,
                    run_index=run_index,
                    env=env,
                    summary_path=summary_path,
                )
            )
    return curve_rows, summary_rows


def online_curve_rows(
    online_history: list[dict[str, Any]],
    *,
    label: str,
    job: str,
    run_index: int,
    summary_path: Path,
    step_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in online_history:
        actor_replay = item.get("actor_replay", {})
        finish_steps = actor_replay.get("episode_finish_train_env_steps") or []
        returns = actor_replay.get("returns") or []
        if finish_steps and returns:
            in_budget_returns = [
                float(value)
                for value, step in zip(returns, finish_steps)
                if int(step) <= step_limit
            ]
            in_budget_steps = [
                int(step) for step in finish_steps if int(step) <= step_limit
            ]
            if in_budget_returns:
                rows.append(
                    {
                        "source": "jepa_actor_replay",
                        "label": label,
                        "job": job,
                        "run_index": run_index,
                        "online_iteration": item.get("iteration"),
                        "step": max(in_budget_steps),
                        "return": mean(in_budget_returns),
                        "std_return": std(in_budget_returns),
                        "episodes": len(in_budget_returns),
                        "phase": "online_actor_replay",
                        "summary_path": str(summary_path),
                    }
                )
            continue

        step = actor_replay.get("train_env_step_offset")
        env_steps = actor_replay.get("env_steps")
        value = actor_replay.get("mean_return")
        if value is None:
            continue
        if step is None and env_steps is not None:
            step = int(env_steps)
        if step is None or int(step) > step_limit:
            continue
        rows.append(
            {
                "source": "jepa_actor_replay",
                "label": label,
                "job": job,
                "run_index": run_index,
                "online_iteration": item.get("iteration"),
                "step": int(step),
                "return": float(value),
                "std_return": actor_replay.get("std_return"),
                "episodes": actor_replay.get("completed_episodes"),
                "phase": "online_actor_replay",
                "summary_path": str(summary_path),
            }
        )
    return rows


def summary_row(
    summary: dict[str, Any],
    run: dict[str, Any],
    *,
    label: str,
    job: str,
    run_index: int,
    env: str,
    summary_path: Path,
) -> dict[str, Any]:
    score = run.get("dreamer_style_training_score") or {}
    final_eval = run.get("final_policy_eval") or {}
    return {
        "source": "jepa",
        "label": label,
        "job": job,
        "run_index": run_index,
        "env": env,
        "dreamer_style_mean_return": score.get("mean_return"),
        "dreamer_style_std_return": score.get("std_return"),
        "dreamer_style_episodes": score.get("episodes"),
        "dreamer_style_budget_reached": score.get("budget_reached"),
        "dreamer_style_window_start_env_step": score.get("window_start_env_step"),
        "dreamer_style_window_end_env_step": score.get("window_end_env_step"),
        "final_eval_mean_return": final_eval.get("mean_return")
        or run.get("final_policy_eval_mean")
        or summary.get("aggregate_final_policy_eval_mean"),
        "final_eval_std_return": final_eval.get("std_return")
        or run.get("final_policy_eval_std")
        or summary.get("aggregate_final_policy_eval_std"),
        "final_eval_episodes": final_eval.get("episodes")
        or run.get("final_policy_eval_episodes")
        or summary.get("aggregate_final_policy_eval_episodes"),
        "final_eval_failure_rate": final_eval.get("failure_rate")
        or run.get("final_policy_eval_failure_rate")
        or summary.get("aggregate_final_policy_eval_failure_rate"),
        "final_eval_success_rate": final_eval.get("success_rate")
        or run.get("final_policy_eval_success_rate")
        or summary.get("aggregate_final_policy_eval_success_rate"),
        "real_train_replay_env_steps": run.get("real_train_replay_env_steps")
        or summary.get("aggregate_real_train_replay_env_steps"),
        "real_train_plus_validation_env_steps": run.get(
            "real_train_plus_validation_env_steps"
        )
        or summary.get("aggregate_real_train_plus_validation_env_steps"),
        "real_total_env_steps": run.get("real_total_env_steps")
        or summary.get("aggregate_real_total_env_steps"),
        "summary_path": str(summary_path),
    }


def reference_scores(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not args.no_paper_baselines:
        rows.extend(DREAMERV3_DMC_REACHER_EASY_REFERENCES)
    if args.ppo_reference_return is not None:
        rows.append(
            {
                "method": "PPO",
                "return": float(args.ppo_reference_return),
                "source": args.ppo_reference_source,
                "notes": "User-provided PPO reference; verify matched env, horizon, and step budget.",
            }
        )
    return rows


def plot_return_curve(
    path: Path,
    curve_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    title: str,
    step_limit: int,
) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        by_label[str(row["label"])].append(row)
    for label, rows in sorted(by_label.items()):
        rows = sorted(rows, key=lambda item: int(item["step"]))
        xs = [int(row["step"]) for row in rows]
        ys = [float(row["return"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2.4, markersize=3.5, label=label)

    reference_styles = {
        "DreamerV3": {"color": "#0b63ce", "linewidth": 2.2},
        "PPO": {"color": "#ff4f6d", "linewidth": 2.0},
        "D4PG": {"color": "#30c7b5", "linewidth": 1.4},
        "DMPO": {"color": "#6f4bd8", "linewidth": 1.4},
        "MPO": {"color": "#999999", "linewidth": 1.2},
        "DDPG": {"color": "#bbbbbb", "linewidth": 1.2},
    }
    for row in reference_rows:
        method = str(row["method"])
        style = reference_styles.get(method, {"color": "#666666", "linewidth": 1.2})
        ax.axhline(
            float(row["return"]),
            linestyle="--",
            alpha=0.85,
            label=f"{method} ref",
            **style,
        )

    ax.set_title(title)
    ax.set_xlabel("Env steps")
    ax.set_ylabel("Return")
    ax.set_xlim(0, step_limit)
    ax.set_ylim(0, 1020)
    ax.grid(True, color="#e8e8e8", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(format_step_tick))
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    curve_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    env: str,
    step_limit: int,
) -> None:
    best_curve = max(
        curve_rows,
        key=lambda row: none_low(row.get("return")),
        default=None,
    )
    dreamer = next(
        (row for row in reference_rows if row.get("method") == "DreamerV3"),
        None,
    )
    lines = [
        "# DMC Reacher Benchmark",
        "",
        f"- Environment: `{env}`",
        f"- Step limit: `{step_limit}` training-replay env steps",
        "- JEPA curve source: online actor replay episodes, not best checkpoint selection",
        "- Final evaluation source: deterministic latest policy after the fixed schedule",
        "",
        "## Best JEPA Curve Point",
        "",
    ]
    if best_curve is None:
        lines.append("No JEPA curve rows found yet.")
    else:
        lines.extend(
            [
                f"- Label: `{best_curve['label']}`",
                f"- Step: `{best_curve['step']}`",
                f"- Return: `{format_number(best_curve['return'])}`",
                f"- Episodes: `{best_curve.get('episodes')}`",
            ]
        )
        if dreamer is not None:
            gap = float(best_curve["return"]) - float(dreamer["return"])
            lines.append(f"- Gap to DreamerV3 reference: `{format_number(gap)}`")

    lines.extend(["", "## JEPA Summary Rows", ""])
    if summary_rows:
        lines.append(
            "| label | dreamer-style mean | final eval mean | final eval episodes | train replay steps |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in summary_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["label"]),
                        format_number(row.get("dreamer_style_mean_return")),
                        format_number(row.get("final_eval_mean_return")),
                        format_number(row.get("final_eval_episodes")),
                        format_number(row.get("real_train_replay_env_steps")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No completed JEPA summaries found yet.")

    lines.extend(["", "## Reference Scores", ""])
    lines.append("| method | return | source |")
    lines.append("| --- | ---: | --- |")
    for row in reference_rows:
        lines.append(
            f"| {row['method']} | {format_number(row['return'])} | {row['source']} |"
        )
    lines.extend(
        [
            "",
            "Note: DreamerV3 Table O.1 reports DDPG, MPO, DMPO, D4PG, and "
            "DreamerV3 for DMC proprioceptive Reacher/easy. It does not provide "
            "a PPO number for that table, so PPO must come from a matched local "
            "run or a separate clearly cited source.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    avg = mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / len(values)))


def none_low(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.3f}"


def format_step_tick(value: float, _pos: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return str(int(value))


if __name__ == "__main__":
    main()
