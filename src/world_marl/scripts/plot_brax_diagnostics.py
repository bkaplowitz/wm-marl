"""Create paper-style Brax JEPA plots.

The command intentionally writes only two figures:

* ``paper_return_vs_env_steps.png``: Dreamer-style return vs real env steps.
* ``paper_train_loss.png``: JEPA train loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    return_rows = build_return_rows(
        args.jepa_root,
        args.ppo_root,
        env=args.env,
        step_limit=args.step_limit,
    )
    add_normalized_returns(return_rows)
    loss_rows = build_train_loss_rows(
        args.jepa_root,
        env=args.env,
        metric=args.loss_metric,
    )

    write_csv(args.out_dir / "paper_return_vs_env_steps.csv", return_rows)
    write_csv(args.out_dir / "paper_train_loss.csv", loss_rows)
    plot_return_vs_env_steps(
        args.out_dir / "paper_return_vs_env_steps.png",
        return_rows,
        title=args.title or pretty_env_title(args.env),
        step_limit=args.step_limit,
        normalize_returns=args.normalize_returns,
    )
    plot_train_loss(
        args.out_dir / "paper_train_loss.png",
        loss_rows,
        title="Train Loss",
        ylabel=args.loss_metric,
    )

    print(f"Wrote paper plots to {args.out_dir}")
    print(f"- {args.out_dir / 'paper_return_vs_env_steps.png'}")
    print(f"- {args.out_dir / 'paper_return_vs_env_steps.csv'}")
    print(f"- {args.out_dir / 'paper_train_loss.png'}")
    print(f"- {args.out_dir / 'paper_train_loss.csv'}")
    counts = source_counts(return_rows)
    print(f"Return rows: {counts}")
    if not any(row.get("source") == "jepa" for row in return_rows):
        print("WARNING: no JEPA actor-replay collection points were found.")
    if not any(row.get("source") == "ppo" for row in return_rows):
        print(
            "WARNING: no PPO evaluation points were found inside the step limit; "
            "only the dashed full-run PPO best reference is available."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jepa-root",
        type=Path,
        required=True,
        help="Root containing JEPA run directories.",
    )
    parser.add_argument(
        "--ppo-root",
        type=Path,
        required=True,
        help="Root containing PPO baseline summaries or logs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the two PNGs and their CSVs.",
    )
    parser.add_argument(
        "--env",
        "--sample-env",
        dest="env",
        default="reacher",
        help="Environment name to plot.",
    )
    parser.add_argument(
        "--step-limit",
        "--sample-step-limit",
        dest="step_limit",
        type=int,
        default=500_000,
        help="Maximum real training-replay environment steps on the return plot.",
    )
    parser.add_argument(
        "--title",
        "--paper-title",
        dest="title",
        default=None,
        help="Title for the return plot.",
    )
    parser.add_argument(
        "--loss-metric",
        "--paper-loss-metric",
        dest="loss_metric",
        default="model/total_loss",
        help="JEPA model-training metric to plot.",
    )
    parser.add_argument(
        "--normalize-returns",
        action="store_true",
        help=(
            "Plot min-max normalized returns while keeping raw returns in the CSV. "
            "Useful when raw negative-control returns make the first panel unreadable."
        ),
    )
    parser.add_argument(
        "--paper-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_return_rows(
    jepa_root: Path,
    ppo_root: Path,
    *,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    rows = load_ppo_return_rows(ppo_root, env, step_limit)
    for label, run_dir in latest_jepa_env_run_dirs(jepa_root, env):
        rows.extend(
            load_jepa_return_rows(
                label=label,
                run_dir=run_dir,
                env=env,
                step_limit=step_limit,
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("source", "")),
            str(row.get("label", "")),
            maybe_int(row.get("step")) or -1,
        ),
    )


def add_normalized_returns(rows: list[dict[str, Any]]) -> None:
    values = [
        value
        for row in rows
        if (value := maybe_float(row.get("return"))) is not None
    ]
    if not values:
        return

    baseline = min(values)
    target = max(values)
    denominator = target - baseline
    for row in rows:
        value = maybe_float(row.get("return"))
        row["normalization_baseline_return"] = baseline
        row["normalization_target_return"] = target
        if value is None or abs(denominator) < 1e-12:
            row["normalized_return"] = None
        else:
            row["normalized_return"] = 100.0 * (value - baseline) / denominator


def load_ppo_return_rows(
    root: Path,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    summary_path = root / env / "summary.json"
    points = load_ppo_history_points(root, env)
    if not points:
        return []

    rows: list[dict[str, Any]] = []
    for step, value in points:
        if step <= step_limit:
            rows.append(
                {
                    "source": "ppo",
                    "label": "PPO",
                    "env": env,
                    "step": step,
                    "return": value,
                    "phase": "eval",
                    "run_dir": str(summary_path.parent),
                }
            )

    boundary = interpolate_boundary_point(points, step_limit)
    if boundary is not None and not any(row["step"] == step_limit for row in rows):
        rows.append(
            {
                "source": "ppo",
                "label": "PPO",
                "env": env,
                "step": step_limit,
                "return": boundary,
                "phase": "eval_interpolated_to_limit",
                "run_dir": str(summary_path.parent),
            }
        )

    best_step, best_return = max(points, key=lambda item: item[1])
    rows.append(
        {
            "source": "ppo_reference",
            "label": "PPO full-run best",
            "env": env,
            "step": step_limit,
            "return": best_return,
            "phase": "full_run_best_reference",
            "reference_step": best_step,
            "run_dir": str(summary_path.parent),
        }
    )
    return rows


def load_ppo_history_points(root: Path, env: str) -> list[tuple[int, float]]:
    rows: list[dict[str, Any]] = []
    summary_path = root / env / "summary.json"
    if summary_path.exists():
        payload = load_json_dict(summary_path)
        history = payload.get("history", [])
        if isinstance(history, list):
            rows.extend(row for row in history if isinstance(row, dict))

    for path in sorted(
        [
            *root.glob(f"{env}*.log"),
            *root.glob(f"{env}*.jsonl"),
            *(root / env).glob("*.log"),
            *(root / env).glob("*.jsonl"),
        ]
    ):
        rows.extend(read_jsonl(path))

    points_by_step: dict[int, float] = {}
    for item in rows:
        step = maybe_int(item.get("num_steps"))
        value = maybe_float(item.get("eval/episode_reward"))
        if step is not None and value is not None:
            points_by_step[step] = value
    return sorted(points_by_step.items())


def load_jepa_return_rows(
    *,
    label: str,
    run_dir: Path,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    run_path = run_dir / "none" / "run_000"
    outcome = load_json_dict(run_path / "outcome.json")
    config = load_json_dict(run_path / "config.json")
    args = config.get("args", {})

    num_envs = maybe_int(args.get("num_envs")) or 1
    collect_steps = maybe_int(args.get("collect_steps")) or 0
    online_collect_steps = maybe_int(args.get("online_collect_steps")) or collect_steps
    initial_steps = (
        maybe_int(outcome.get("real_initial_train_replay_env_steps"))
        or maybe_int(load_json_dict(run_path / "train_replay.json").get("env_steps"))
        or collect_steps * num_envs
    )

    history = outcome.get("online_history")
    if not isinstance(history, list):
        history = load_json_list(run_path / "online_history.json")

    rows: list[dict[str, Any]] = []

    initial_policy_return = maybe_float(outcome.get("policy_initial_mean"))
    if initial_policy_return is not None:
        rows.append(
            {
                "source": "jepa",
                "label": label,
                "env": env,
                "step": 0,
                "return": initial_policy_return,
                "phase": "initial_policy",
                "iteration": 0,
                "run_dir": str(run_dir),
            }
        )

    first_trained_return = maybe_float(outcome.get("policy_pre_online_trained_mean"))
    if first_trained_return is None:
        first_trained_return = maybe_float(outcome.get("policy_trained_mean"))
    if first_trained_return is not None and initial_steps <= step_limit:
        rows.append(
            {
                "source": "jepa",
                "label": label,
                "env": env,
                "step": initial_steps,
                "return": first_trained_return,
                "phase": "initial_world_model_policy",
                "iteration": 0,
                "run_dir": str(run_dir),
            }
        )

    cumulative_steps = initial_steps
    for index, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            continue
        actor_replay = item.get("actor_replay", {})
        if not isinstance(actor_replay, dict):
            actor_replay = {}

        added_steps = maybe_int(actor_replay.get("env_steps"))
        if added_steps is None:
            added_steps = online_collect_steps * num_envs
        cumulative_steps += added_steps

        value = maybe_float(actor_replay.get("mean_return"))
        if value is None or cumulative_steps > step_limit:
            continue
        rows.append(
            {
                "source": "jepa",
                "label": label,
                "env": env,
                "step": cumulative_steps,
                "return": value,
                "phase": "actor_replay_collection",
                "iteration": maybe_int(item.get("iteration")) or index,
                "run_dir": str(run_dir),
            }
        )
    return rows


def build_train_loss_rows(
    root: Path,
    *,
    env: str,
    metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, run_dir in latest_jepa_env_run_dirs(root, env):
        point_index = 0
        for metrics_path in sorted(run_dir.glob("none/run_*/metrics.jsonl")):
            for item in read_jsonl(metrics_path):
                value = maybe_float(item.get(metric))
                if value is None:
                    continue
                point_index += 1
                rows.append(
                    {
                        "label": label,
                        "env": env,
                        "point": point_index,
                        "metric": metric,
                        "value": value,
                        "phase": item.get("phase"),
                        "update": maybe_int(item.get("update")),
                        "run_dir": str(run_dir),
                    }
                )
    return rows


def latest_jepa_env_run_dirs(root: Path, env: str) -> list[tuple[str, Path]]:
    if not root.exists():
        return []

    bases: list[Path] = []
    if any(root.glob("brax_jepa_*")):
        bases.append(root)
    bases.extend(
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path.name == env or path.name.startswith(f"{env}_"))
    )

    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for base in bases:
        run_dirs = sorted(path for path in base.glob("brax_jepa_*") if path.is_dir())
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        label = env if base == root / env else base.name
        out.append((label, run_dir))
    return out


def plot_return_vs_env_steps(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    step_limit: int,
    normalize_returns: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    y_key = "normalized_return" if normalize_returns else "return"
    jepa = aggregate_curve(
        [row for row in rows if row.get("source") == "jepa"],
        x_key="step",
        y_key=y_key,
    )
    ppo = aggregate_curve(
        [row for row in rows if row.get("source") == "ppo"],
        x_key="step",
        y_key=y_key,
    )
    ppo_reference = [row for row in rows if row.get("source") == "ppo_reference"]

    plot_aggregate_curve(ax, ppo, color="#ff4f6d", label="PPO", shade=False)
    for row in ppo_reference[:1]:
        value = maybe_float(row.get(y_key))
        if value is not None:
            ax.axhline(
                value,
                color="#ff4f6d",
                linestyle="--",
                linewidth=1.8,
                alpha=0.75,
                label="PPO best",
            )
    plot_aggregate_curve(ax, jepa, color="#0b63ce", label="JEPA", shade=True)

    ylabel = "Normalized Return (%)" if normalize_returns else "Return"
    style_paper_axis(ax, title=title, xlabel="Env steps", ylabel=ylabel)
    ax.set_xlim(0, step_limit)
    if normalize_returns:
        ax.set_ylim(-5, 105)
    ax.xaxis.set_major_formatter(FuncFormatter(format_k_tick))
    if rows:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=3,
            frameon=False,
            handlelength=2.4,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_train_loss(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    curve = aggregate_curve(rows, x_key="point", y_key="value")
    plot_aggregate_curve(ax, curve, color="#0b63ce", label="JEPA", shade=True)
    style_paper_axis(
        ax,
        title=title,
        xlabel="Logged train checkpoint",
        ylabel=ylabel,
    )
    if curve:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.36),
            ncol=1,
            frameon=False,
            handlelength=2.4,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def aggregate_curve(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
) -> list[dict[str, float]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        x_value = maybe_float(row.get(x_key))
        y_value = maybe_float(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        grouped.setdefault(x_value, []).append(y_value)

    out: list[dict[str, float]] = []
    for x_value, values in sorted(grouped.items()):
        if not values:
            continue
        out.append(
            {
                "x": x_value,
                "mean": sum(values) / len(values),
                "low": min(values),
                "high": max(values),
            }
        )
    return out


def plot_aggregate_curve(
    ax: plt.Axes,
    curve: list[dict[str, float]],
    *,
    color: str,
    label: str,
    shade: bool,
) -> None:
    if not curve:
        return
    xs = [row["x"] for row in curve]
    means = [row["mean"] for row in curve]
    lows = [row["low"] for row in curve]
    highs = [row["high"] for row in curve]
    if shade:
        ax.fill_between(xs, lows, highs, color=color, alpha=0.18, linewidth=0)
    ax.plot(xs, means, color=color, linewidth=2.8, label=label)


def style_paper_axis(
    ax: plt.Axes,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, color="#ececec", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.tick_params(axis="both", labelsize=11, width=1.1)


def interpolate_boundary_point(
    points: list[tuple[int, float]],
    step_limit: int,
) -> float | None:
    before = [(step, value) for step, value in points if step < step_limit]
    after = [(step, value) for step, value in points if step > step_limit]
    if any(step == step_limit for step, _ in points):
        return next(value for step, value in points if step == step_limit)
    if not before or not after:
        return None
    lo_step, lo_value = before[-1]
    hi_step, hi_value = after[0]
    if hi_step == lo_step:
        return hi_value
    weight = (step_limit - lo_step) / (hi_step - lo_step)
    return lo_value + weight * (hi_value - lo_value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_json_list(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", ""))
        counts[source] = counts.get(source, 0) + 1
    return counts


def format_k_tick(value: float, _: int) -> str:
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    return f"{value / 1_000:g}K"


def pretty_env_title(env: str) -> str:
    return env.replace("_", " ").title()


if __name__ == "__main__":
    main()
