"""Plot per-arm world-model and PPO training losses for comparison runs.

Reads one or more ``compare_single_wm`` output directories (``wm_comparison_*``
folders, repeatable via ``--comparison-dir``; each dir is treated as one seed)
and writes paper-style loss figures in the ``plot_brax_diagnostics`` idiom:
mean across seeds with min/max shading, per-arm panels.

Loss sources per arm:

- jepa: ``model/total_loss`` (or ``--jepa-metric``) from
  ``none/run_*/metrics.jsonl``, indexed by logged train checkpoint — the same
  rows ``plot_brax_diagnostics`` plots.
- genwm arms: ``[run N ... fit] step i/S wm_loss=...`` and
  ``[run N ... policy] step i/S ppo_loss=...`` lines captured in each job's
  ``console.log``, with the step axis accumulated across the offline fit and
  online refit segments.
- model-free: ``[run N model-free] update i/U ppo_loss=...`` console lines.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from world_marl.scripts.plot_brax_diagnostics import (
    aggregate_curve,
    plot_aggregate_curve,
    style_paper_axis,
)

JEPA_ARM = "jepa"

CONSOLE_LOSS_PATTERN = re.compile(
    r"\[run \d+(?P<segment>(?: online \d+)? (?:fit|policy)| model-free)\]"
    r" (?:step|update) (?P<step>\d+)/(?P<total>\d+)"
    r" (?P<metric>wm_loss|ppo_loss)=(?P<value>[-+0-9.eE]+)"
)


def parse_console_losses(console_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Extract cumulative-step (x, loss) points from a job's console.log."""
    losses: dict[str, list[tuple[int, float]]] = {"wm_loss": [], "ppo_loss": []}
    segments: dict[str, tuple[str, int, int]] = {}
    for match in CONSOLE_LOSS_PATTERN.finditer(
        console_path.read_text(errors="replace")
    ):
        metric = match.group("metric")
        segment = match.group("segment")
        total = int(match.group("total"))
        previous_segment, offset, previous_total = segments.get(metric, ("", 0, 0))
        if segment != previous_segment:
            offset += previous_total
        segments[metric] = (segment, offset, total)
        losses[metric].append(
            (offset + int(match.group("step")), float(match.group("value")))
        )
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
                    console_path = arm_dir / "console.log"
                    if not console_path.is_file():
                        print(f"warning: no console.log in {arm_dir}", flush=True)
                        continue
                    curves = parse_console_losses(console_path)
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


def write_loss_figure(
    arms: dict[str, list[dict[str, Any]]],
    out_path: Path,
    *,
    env: str,
    metric_kind: str,
    jepa_metric: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(arms)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.2 * len(names), 3.4), squeeze=False
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
    fig.suptitle(f"{env} — {metric_kind} (mean over seeds, min/max shaded)", y=1.02)
    fig.tight_layout()
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
