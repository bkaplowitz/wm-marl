"""Plot policy return against real environment transitions for comparison runs.

Reads one or more ``compare_single_wm`` output directories (``wm_comparison_*``
folders, mixable across runs via repeated ``--comparison-dir``) and draws, per
environment, mean episode return against cumulative real-env transitions for
every arm found. Also writes ``reward_curves.json`` with the extracted points.

Curve sources per arm:

- genwm arms / model-free: ``eval_points`` in each run's ``outcome.json``.
  Older outcomes lack ``eval_points``; those curves are reconstructed from
  ``policy_initial_mean`` + ``policy_iteration_returns`` and the experiment
  ``config.json`` budgets, in that run's own step units (pre-fix runs counted
  total transitions; post-fix runs count per-env steps x num_envs).
- jepa: ``online_policy_champion_returns`` from ``outcome.json`` against the
  training-replay schedule ``(collect + (i+1) x online_collect) x num_envs``.
  JEPA's extra validation/selection/eval interaction is not counted, which
  flatters jepa on the x axis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from world_marl.scripts.plot_brax_diagnostics import (
    aggregate_curve,
    plot_aggregate_curve,
    style_paper_axis,
)

JEPA_ARM = "jepa"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        action="append",
        required=True,
        help="A wm_comparison_* directory (repeatable; curves are merged).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <first comparison dir>/reward_curves.png).",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Optional env-slug filter (e.g. gymnax_CartPole-v1 brax_reacher).",
    )
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = args.comparison_dir[0] / "reward_curves.png"
    return args


def _first_number(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _find_config(outcome_path: Path, stop_dir: Path) -> dict[str, Any]:
    for parent in outcome_path.parents:
        candidate = parent / "config.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
        if parent == stop_dir:
            break
    return {}


def genwm_curve(
    outcome: dict[str, Any], config: dict[str, Any]
) -> list[tuple[float, float]]:
    points = outcome.get("eval_points")
    if points:
        return [(float(p["real_env_steps"]), float(p["return"])) for p in points]
    # Legacy outcome: rebuild the schedule in that run's own budget units.
    collect = float(config.get("collect_steps", 0))
    online = float(config.get("online_collect_steps", 0))
    curve: list[tuple[float, float]] = []
    initial = _first_number(outcome, "policy_initial_mean")
    if initial is not None:
        curve.append((0.0, initial))
    for index, value in enumerate(outcome.get("policy_iteration_returns") or []):
        curve.append((collect + (index + 1) * online, float(value)))
    return curve


def jepa_curve(
    outcome: dict[str, Any], config: dict[str, Any]
) -> list[tuple[float, float]]:
    # train_dmc_jepa nests the argparse namespace under "args" in config.json.
    nested_args = config.get("args")
    if isinstance(nested_args, dict):
        config = {**nested_args, **{k: v for k, v in config.items() if k != "args"}}
    num_envs = float(config.get("num_envs", 16))
    collect = float(config.get("collect_steps", 8192))
    online = float(config.get("online_collect_steps", 4096))
    curve: list[tuple[float, float]] = []
    initial = _first_number(outcome, "policy_initial_mean", "initial_policy_eval_mean")
    if initial is not None:
        curve.append((0.0, initial))
    champions = outcome.get("online_policy_champion_returns") or []
    for index, value in enumerate(champions):
        curve.append(((collect + (index + 1) * online) * num_envs, float(value)))
    final = _first_number(outcome, "final_policy_eval_mean", "policy_trained_mean")
    if final is not None:
        curve.append(((collect + len(champions) * online) * num_envs, final))
    return curve


def collect_curves(
    comparison_dirs: list[Path], env_filter: list[str] | None
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return {env_slug: {arm: [run record, ...]}} across all comparison dirs."""
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for comparison_dir in comparison_dirs:
        for env_dir in sorted(
            path for path in comparison_dir.iterdir() if path.is_dir()
        ):
            if env_filter and env_dir.name not in env_filter:
                continue
            for arm_dir in sorted(path for path in env_dir.iterdir() if path.is_dir()):
                arm = arm_dir.name
                for outcome_path in sorted(arm_dir.rglob("outcome.json")):
                    outcome = json.loads(outcome_path.read_text())
                    config = _find_config(outcome_path, arm_dir)
                    build = jepa_curve if arm == JEPA_ARM else genwm_curve
                    curve = build(outcome, config)
                    if not curve:
                        print(f"warning: no curve from {outcome_path}", flush=True)
                        continue
                    curves.setdefault(env_dir.name, {}).setdefault(arm, []).append(
                        {
                            "source": str(outcome_path),
                            "points": curve,
                            "random_return": _first_number(
                                outcome, "policy_random_mean"
                            ),
                        }
                    )
    return curves


def arm_aggregate(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, float]], float | None]:
    """Mean/min/max curve over an arm's seed records plus mean final return."""
    rows = [
        {"point": float(x), "value": float(y)}
        for record in records
        for x, y in record["points"]
    ]
    curve = aggregate_curve(rows, x_key="point", y_key="value")
    finals = [float(record["points"][-1][1]) for record in records if record["points"]]
    final_mean = sum(finals) / len(finals) if finals else None
    return curve, final_mean


def sorted_arm_summaries(
    arms: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, list[dict[str, float]], float | None]]:
    """(arm, aggregate curve, final mean) tuples, best final return first."""
    summaries = [(arm, *arm_aggregate(records)) for arm, records in arms.items()]
    summaries.sort(
        key=lambda item: (
            item[2] is None,
            -item[2] if item[2] is not None else 0.0,
            item[0],
        )
    )
    return summaries


def write_plot(
    curves: dict[str, dict[str, list[dict[str, Any]]]], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    envs = list(curves)
    fig, axes = plt.subplots(1, len(envs), figsize=(7.5 * len(envs), 5), squeeze=False)
    colors = plt.get_cmap("tab10")
    for axis, env in zip(axes[0], envs):
        arms = curves[env]
        color_index = {arm: index for index, arm in enumerate(sorted(arms))}
        summaries = sorted_arm_summaries(arms)
        band_values: list[float] = []
        for arm, curve, final_mean in summaries:
            color = colors(color_index[arm] % 10)
            label = arm if final_mean is None else f"{arm} (final {final_mean:.1f})"
            plot_aggregate_curve(axis, curve, color=color, label=label, shade=True)
            xs = [row["x"] for row in curve]
            means = [row["mean"] for row in curve]
            axis.plot(
                xs, means, linestyle="none", marker="o", markersize=4, color=color
            )
            band_values.extend(row["low"] for row in curve)
            band_values.extend(row["high"] for row in curve)
        low, high = min(band_values), max(band_values)
        margin = 0.05 * (high - low) or 1.0
        low, high = low - margin, high + margin
        axis.set_ylim(low, high)
        randoms = [
            record["random_return"]
            for records in arms.values()
            for record in records
            if record["random_return"] is not None
        ]
        if randoms:
            random_mean = float(np.mean(randoms))
            if low <= random_mean <= high:
                axis.axhline(
                    random_mean,
                    color="grey",
                    linestyle="--",
                    linewidth=1,
                    label=f"random ({random_mean:.1f})",
                )
            else:
                axis.annotate(
                    f"random = {random_mean:.1f} (below axis)",
                    xy=(0.02, 0.03),
                    xycoords="axes fraction",
                    fontsize=9,
                    color="grey",
                )
        style_paper_axis(
            axis,
            title=f"{env} (mean over seeds, min/max shaded)",
            xlabel="real env transitions",
            ylabel="mean episode return",
        )
        axis.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for comparison_dir in args.comparison_dir:
        if not comparison_dir.is_dir():
            print(f"error: {comparison_dir} is not a directory", file=sys.stderr)
            return 2
    curves = collect_curves(args.comparison_dir, args.envs)
    if not curves:
        print("error: no outcome.json curves found", file=sys.stderr)
        return 1
    data_path = args.out.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(curves, indent=2))
    write_plot(curves, args.out)
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {data_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
