"""Plot wall-clock runtime per arm for comparison runs.

Reads one or more ``compare_single_wm`` output directories (``wm_comparison_*``
folders, repeatable via ``--comparison-dir``; each dir is treated as one seed)
and, per environment, writes ``paper_runtime_<env>.png`` with two panels:
mean runtime bars (min/max whiskers, fastest first) and trained return
against runtime per seed. Rows come from each dir's ``comparison.json``
(``runtime_seconds`` / ``policy_trained_mean``), the same file the summary
table prints from.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from world_marl.scripts.compare_single_wm import env_slug
from world_marl.scripts.plot_brax_diagnostics import style_paper_axis


def collect_runtime_rows(
    comparison_dirs: list[Path],
    env_filter: list[str] | None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return {env_slug: {arm: [{seed, runtime_seconds, trained_mean}, ...]}}."""
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for comparison_dir in comparison_dirs:
        seed = comparison_dir.parent.name
        comparison_path = comparison_dir / "comparison.json"
        if not comparison_path.is_file():
            print(f"warning: no comparison.json under {comparison_dir}", flush=True)
            continue
        for row in json.loads(comparison_path.read_text())["rows"]:
            env = env_slug(row["env"])
            if env_filter and env not in env_filter:
                continue
            runtime = row.get("runtime_seconds")
            if runtime is None:
                continue
            trained = row.get("policy_trained_mean")
            rows.setdefault(env, {}).setdefault(row["arm"], []).append(
                {
                    "seed": seed,
                    "runtime_seconds": float(runtime),
                    "trained_mean": None if trained is None else float(trained),
                }
            )
    return rows


def write_runtime_figure(
    arms: dict[str, list[dict[str, Any]]], out_path: Path, *, env: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = {
        arm: sum(row["runtime_seconds"] for row in rows) / len(rows)
        for arm, rows in arms.items()
    }
    order = sorted(means, key=lambda arm: means[arm])
    colors = plt.get_cmap("tab10")
    color_index = {arm: index for index, arm in enumerate(sorted(arms))}
    fig, (bar_axis, scatter_axis) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    for position, arm in enumerate(order):
        runtimes = [row["runtime_seconds"] for row in arms[arm]]
        mean = means[arm]
        bar_axis.bar(position, mean, color=colors(color_index[arm] % 10), width=0.7)
        bar_axis.errorbar(
            position,
            mean,
            yerr=[[mean - min(runtimes)], [max(runtimes) - mean]],
            color="black",
            capsize=4,
            linewidth=1.2,
        )
        bar_axis.annotate(
            f"{mean / 60:.1f} min",
            xy=(position, mean),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    bar_axis.set_xticks(range(len(order)))
    bar_axis.set_xticklabels(order, rotation=20, ha="right")
    style_paper_axis(
        bar_axis,
        title=f"{env} — runtime per arm (mean, min/max)",
        xlabel="",
        ylabel="wall-clock seconds",
    )

    for arm in order:
        points = [
            (row["runtime_seconds"], row["trained_mean"])
            for row in arms[arm]
            if row["trained_mean"] is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        scatter_axis.scatter(
            xs, ys, color=colors(color_index[arm] % 10), label=arm, s=42, zorder=3
        )
    style_paper_axis(
        scatter_axis,
        title=f"{env} — trained return vs runtime",
        xlabel="wall-clock seconds",
        ylabel="trained policy return",
    )
    scatter_axis.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_runtime_csv(arms: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "runtime_seconds", "trained_mean"])
        for arm in sorted(arms):
            for row in arms[arm]:
                writer.writerow(
                    [arm, row["seed"], row["runtime_seconds"], row["trained_mean"]]
                )


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
    rows = collect_runtime_rows(args.comparison_dir, args.envs)
    if not rows:
        print("error: no runtime rows found", file=sys.stderr)
        return 1
    for env, arms in rows.items():
        stem = f"paper_runtime_{env}"
        write_runtime_figure(arms, args.out_dir / f"{stem}.png", env=env)
        write_runtime_csv(arms, args.out_dir / f"{stem}.csv")
        print(f"wrote {args.out_dir / stem}.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
