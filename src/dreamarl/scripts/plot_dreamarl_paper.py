"""Create DreamerV3-compatible DreaMARL learning-curve figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dreamarl.baselines.dreamerv3.artifacts import (
    EpisodeScore,
    bin_episode_scores_by_count,
    extract_episode_scores,
    load_official_binned_reference,
    read_jsonl,
)


TASK_LABELS = {
    "dmc_reacher_easy": "Reacher Easy",
    "dmc_hopper_hop": "Hopper Hop",
    "dmc_walker_walk": "Walker Walk",
    "dmc_cheetah_run": "Cheetah Run",
}


@dataclass(frozen=True)
class RunArtifact:
    task: str
    seed: int
    root: Path
    scores: tuple[EpisodeScore, ...]
    evaluation: dict[str, float]


def _find_run_root(path: Path) -> Path | None:
    if (path / "launch.json").is_file() and (path / "run" / "scores.jsonl").is_file():
        return path
    if (path / "scores.jsonl").is_file() and (path.parent / "launch.json").is_file():
        return path.parent
    return None


def discover_runs(inputs: Iterable[str | Path]) -> list[RunArtifact]:
    roots: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        direct = _find_run_root(path)
        if direct is not None:
            roots.add(direct)
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        for launch_path in path.rglob("launch.json"):
            candidate = launch_path.parent
            if (candidate / "run" / "scores.jsonl").is_file():
                roots.add(candidate)

    runs = []
    identities: set[tuple[str, int]] = set()
    for root in sorted(roots):
        launch = json.loads((root / "launch.json").read_text(encoding="utf-8"))
        task = str(launch["task"])
        seed = int(launch["seed"])
        identity = (task, seed)
        if identity in identities:
            raise ValueError(f"duplicate run for task={task!r}, seed={seed}")
        identities.add(identity)
        scores = tuple(
            extract_episode_scores(read_jsonl(root / "run" / "scores.jsonl"))
        )
        evaluation = _final_evaluation(root / "run" / "metrics.jsonl")
        runs.append(RunArtifact(task, seed, root, scores, evaluation))
    return runs


def _final_evaluation(path: Path) -> dict[str, float]:
    evaluations = [
        record for record in read_jsonl(path) if "eval/return_mean" in record
    ]
    if not evaluations:
        return {}
    record = evaluations[-1]
    return {
        key.removeprefix("eval/"): float(value)
        for key, value in record.items()
        if key.startswith("eval/")
    }


def aggregate_curves(
    runs: Iterable[RunArtifact],
    *,
    bins: int,
    max_steps: int,
) -> dict[str, list[dict[str, float | int]]]:
    grouped: dict[str, list[dict[int, float]]] = defaultdict(list)
    for run in runs:
        curve = bin_episode_scores_by_count(
            run.scores,
            bins=bins,
            max_steps=max_steps,
        )
        grouped[run.task].append(
            {int(row["env_steps"]): float(row["episode_return_mean"]) for row in curve}
        )

    result = {}
    for task, seed_curves in grouped.items():
        common_steps = sorted(set.intersection(*(set(curve) for curve in seed_curves)))
        rows = []
        for step in common_steps:
            values = np.asarray([curve[step] for curve in seed_curves], np.float64)
            rows.append(
                {
                    "env_steps": step,
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "seeds": len(values),
                }
            )
        result[task] = rows
    return result


def write_outputs(
    runs: list[RunArtifact],
    *,
    upstream_root: Path,
    output_dir: Path,
    bins: int,
    max_steps: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    curves = aggregate_curves(runs, bins=bins, max_steps=max_steps)
    references = {
        task: load_official_binned_reference(
            upstream_root,
            task=task,
            observation_mode="vision",
            bins=bins,
            max_steps=max_steps,
        )
        for task in curves
    }
    payload = {
        "protocol": {
            "bins": bins,
            "max_environment_steps": max_steps,
            "aggregation": "per-seed bin mean, followed by mean and population std across seeds",
            "reference": "danijar/dreamerv3 official dmc_vision score archive",
        },
        "dreamarl": curves,
        "dreamerv3": references,
        "runs": [
            {
                "task": run.task,
                "seed": run.seed,
                "root": str(run.root),
                "episodes": len(run.scores),
                "evaluation": run.evaluation,
            }
            for run in runs
        ],
    }
    (output_dir / "curves.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary_csv(output_dir / "summary.csv", runs, curves, references)
    _plot_curves(output_dir / "learning_curves", curves, references)


def _write_summary_csv(
    path: Path,
    runs: list[RunArtifact],
    curves: dict[str, list[dict[str, float | int]]],
    references: dict[str, list[dict[str, Any]]],
) -> None:
    evaluations = defaultdict(list)
    for run in runs:
        if run.evaluation:
            evaluations[run.task].append(run.evaluation)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "dreamarl_seeds",
                "dreamarl_auc",
                "dreamerv3_seeds",
                "dreamerv3_auc",
                "dreamarl_final_training_bin",
                "dreamerv3_final_training_bin",
                "dreamarl_fixed_eval_mean",
                "dreamarl_fixed_eval_std_across_seeds",
            ),
        )
        writer.writeheader()
        for task in sorted(curves):
            ours = curves[task]
            reference = references[task]
            fixed = np.asarray(
                [item["return_mean"] for item in evaluations[task]], np.float64
            )
            writer.writerow(
                {
                    "task": task,
                    "dreamarl_seeds": ours[-1]["seeds"] if ours else 0,
                    "dreamarl_auc": np.mean([row["mean"] for row in ours])
                    if ours
                    else "",
                    "dreamerv3_seeds": reference[-1]["seeds"] if reference else 0,
                    "dreamerv3_auc": (
                        np.mean([row["episode_return_mean"] for row in reference])
                        if reference
                        else ""
                    ),
                    "dreamarl_final_training_bin": ours[-1]["mean"] if ours else "",
                    "dreamerv3_final_training_bin": (
                        reference[-1]["episode_return_mean"] if reference else ""
                    ),
                    "dreamarl_fixed_eval_mean": fixed.mean() if len(fixed) else "",
                    "dreamarl_fixed_eval_std_across_seeds": (
                        fixed.std() if len(fixed) else ""
                    ),
                }
            )


def _plot_curves(
    stem: Path,
    curves: dict[str, list[dict[str, float | int]]],
    references: dict[str, list[dict[str, Any]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    tasks = [task for task in TASK_LABELS if task in curves]
    tasks.extend(sorted(set(curves) - set(tasks)))
    columns = len(tasks) if len(tasks) <= 3 else 2
    rows = (len(tasks) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2, 2.5 * rows),
        sharex=True,
        squeeze=False,
    )
    green = "#087E5B"
    gray = "#6B7280"
    for axis, task in zip(axes.flat, tasks, strict=False):
        ours = curves[task]
        reference = references[task]
        if reference:
            xref = np.asarray([row["env_steps"] for row in reference])
            mean = np.asarray([row["episode_return_mean"] for row in reference])
            std = np.asarray([row["episode_return_std"] for row in reference])
            axis.fill_between(
                xref,
                mean - std,
                mean + std,
                color=gray,
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(xref, mean, color=gray, linewidth=1.7, label="DreamerV3")
        if ours:
            x = np.asarray([row["env_steps"] for row in ours])
            mean = np.asarray([row["mean"] for row in ours])
            std = np.asarray([row["std"] for row in ours])
            seeds = int(ours[-1]["seeds"])
            if seeds > 1:
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=green,
                    alpha=0.2,
                    linewidth=0,
                )
            axis.plot(x, mean, color=green, linewidth=2.1, label="DreaMARL")
            axis.text(
                0.98,
                0.04,
                f"DreaMARL n={seeds}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color=green,
            )
        axis.set_title(TASK_LABELS.get(task, task), fontsize=10, fontweight="semibold")
        axis.set_ylim(0, 1000)
        axis.set_xlim(0, 500_000)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 1000:.0f}k")
        )

    for axis in axes.flat[len(tasks) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        if axis.get_visible():
            axis.set_xlabel("Environment transitions", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Episode return", fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Run directories or roots to scan")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("external/dreamerv3"),
        help="Pinned DreamerV3 checkout containing the official score archives",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=500_000)
    args = parser.parse_args()
    runs = discover_runs(args.inputs)
    if not runs:
        raise SystemExit("no completed or active DreaMARL run artifacts found")
    write_outputs(
        runs,
        upstream_root=args.upstream_root,
        output_dir=args.output_dir,
        bins=args.bins,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
