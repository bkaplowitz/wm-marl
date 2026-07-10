"""Plot per-arm world-model and PPO training losses for comparison runs.

Reads one or more ``compare_single_wm`` output directories (``wm_comparison_*``
folders, repeatable via ``--comparison-dir``; each dir is treated as one seed)
and writes paper-style loss figures in the ``plot_brax_diagnostics`` idiom:
mean across seeds with min/max shading, per-arm panels.

Loss sources per arm:

- jepa: ``model/total_loss`` (or ``--jepa-metric``) from
  ``none/run_*/metrics.jsonl``, indexed by logged train checkpoint — the same
  rows ``plot_brax_diagnostics`` plots.
- genwm arms / model-free: ``metrics.jsonl`` records written by
  ``train_single_genwm`` through the shared ``RunLogger`` into each run dir
  (``phase``/``step``/``total`` plus the trainer's metric values), with the
  step axis accumulated across the offline fit and online refit phases.
  ``wm_loss`` records feed the wm_loss figure; policy/model-free
  ``total_loss`` records feed ppo_loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from world_marl.scripts.plot_brax_diagnostics import (
    aggregate_curve,
    plot_aggregate_curve,
    style_paper_axis,
)

JEPA_ARM = "jepa"


def genwm_loss_points(metrics_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Cumulative-step (x, loss) curves from a run's metrics.jsonl records."""
    losses: dict[str, list[tuple[int, float]]] = {"wm_loss": [], "ppo_loss": []}
    phases: dict[str, tuple[str, int, int]] = {}
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        phase = record.get("phase")
        step = record.get("step")
        total = record.get("total")
        if phase is None or step is None or total is None:
            continue
        if "wm_loss" in record:
            metric, value = "wm_loss", record["wm_loss"]
        elif "total_loss" in record:
            metric, value = "ppo_loss", record["total_loss"]
        else:
            continue
        previous_phase, offset, previous_total = phases.get(metric, ("", 0, 0))
        if phase != previous_phase:
            offset += previous_total
        phases[metric] = (phase, offset, int(total))
        losses[metric].append((offset + int(step), float(value)))
    return losses


def jepa_loss_points(job_dir: Path, *, metric: str) -> list[tuple[int, float]]:
    """Sequential logged-checkpoint (index, loss) points across a jepa job dir."""
    points: list[tuple[int, float]] = []
    index = 0
    for metrics_path in sorted(job_dir.glob("*/none/run_*/metrics.jsonl")):
        for line in metrics_path.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            index += 1
            points.append((index, float(value)))
    return points


def collect_loss_rows(
    comparison_dirs: list[Path],
    env_filter: list[str] | None,
    *,
    jepa_metric: str,
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return {env_slug: {metric_kind: {arm: [row, ...]}}} across seeds."""
    rows: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for comparison_dir in comparison_dirs:
        seed = comparison_dir.parent.name
        for env_dir in sorted(
            path for path in comparison_dir.iterdir() if path.is_dir()
        ):
            if env_filter and env_dir.name not in env_filter:
                continue
            for arm_dir in sorted(path for path in env_dir.iterdir() if path.is_dir()):
                arm = arm_dir.name
                if arm == JEPA_ARM:
                    curves = {"wm_loss": jepa_loss_points(arm_dir, metric=jepa_metric)}
                else:
                    metrics_paths = sorted(arm_dir.rglob("metrics.jsonl"))
                    if not metrics_paths:
                        print(f"warning: no metrics.jsonl under {arm_dir}", flush=True)
                        continue
                    curves = {"wm_loss": [], "ppo_loss": []}
                    for metrics_path in metrics_paths:
                        for kind, points in genwm_loss_points(metrics_path).items():
                            curves[kind].extend(points)
                for metric_kind, points in curves.items():
                    if not points:
                        continue
                    rows.setdefault(env_dir.name, {}).setdefault(
                        metric_kind, {}
                    ).setdefault(arm, []).extend(
                        {"seed": seed, "point": x, "value": value}
                        for x, value in points
                    )
    return rows


def loss_axis_scale(
    arms: dict[str, list[dict[str, Any]]],
) -> tuple[str, float, float]:
    """Shared y-scale and limits spanning every arm's loss values."""
    values = [row["value"] for rows in arms.values() for row in rows]
    low, high = min(values), max(values)
    if low > 0:
        return "log", low / 1.5, high * 1.5
    margin = 0.05 * (high - low) or 1.0
    return "symlog", low - margin, high + margin


def build_loss_figure(
    arms: dict[str, list[dict[str, Any]]],
    *,
    env: str,
    metric_kind: str,
    jepa_metric: str,
) -> tuple[Any, list[Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(arms)
    colors = plt.get_cmap("tab10")
    scale, low, high = loss_axis_scale(arms)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False, sharey=True
    )
    for index, (axis, arm) in enumerate(zip(axes[0], names)):
        curve = aggregate_curve(arms[arm], x_key="point", y_key="value")
        plot_aggregate_curve(
            axis, curve, color=colors(index % 10), label=arm, shade=True
        )
        if arm == JEPA_ARM:
            xlabel, ylabel = "Logged train checkpoint", jepa_metric
        elif metric_kind == "wm_loss":
            xlabel, ylabel = "WM train step", "wm_loss"
        else:
            xlabel, ylabel = "PPO update", "ppo_loss"
        style_paper_axis(axis, title=arm, xlabel=xlabel, ylabel=ylabel)
        axis.set_yscale(scale)
        axis.set_ylim(low, high)
    fig.suptitle(f"{env} — {metric_kind} (mean over seeds, min/max shaded)", y=1.02)
    fig.tight_layout()
    return fig, list(axes[0])


def write_loss_figure(
    arms: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    env: str,
    metric_kind: str,
    jepa_metric: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, _ = build_loss_figure(
        arms, env=env, metric_kind=metric_kind, jepa_metric=jepa_metric
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_loss_csv(arms: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "point", "value"])
        for arm in sorted(arms):
            for row in arms[arm]:
                writer.writerow([arm, row["seed"], row["point"], row["value"]])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        action="append",
        required=True,
        help="A wm_comparison_* directory (repeatable; one seed each).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: first comparison dir).",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Optional env-slug filter (e.g. brax_reacher).",
    )
    parser.add_argument(
        "--jepa-metric",
        default="model/total_loss",
        help="metrics.jsonl key plotted for the jepa arm.",
    )
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = args.comparison_dir[0]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for comparison_dir in args.comparison_dir:
        if not comparison_dir.is_dir():
            print(f"error: {comparison_dir} is not a directory", file=sys.stderr)
            return 2
    rows = collect_loss_rows(
        args.comparison_dir, args.envs, jepa_metric=args.jepa_metric
    )
    if not rows:
        print("error: no loss curves found", file=sys.stderr)
        return 1
    for env, metrics in rows.items():
        for metric_kind, arms in metrics.items():
            stem = f"paper_{metric_kind}_{env}"
            write_loss_figure(
                arms,
                args.out_dir / f"{stem}.png",
                env=env,
                metric_kind=metric_kind,
                jepa_metric=args.jepa_metric,
            )
            write_loss_csv(arms, args.out_dir / f"{stem}.csv")
            print(f"wrote {args.out_dir / stem}.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
