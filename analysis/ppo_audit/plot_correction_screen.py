"""Render fixed-budget 2s3z correction curves from the local evidence mirror."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from summarize_mirror import summarize


COLORS = {
    "team_return": "#0072B2",
    "team_return_anchor": "#CC79A7",
    "original_base": "#666666",
}
LABELS = {
    "team_return": "Cooperative returns",
    "team_return_anchor": "Returns + factual critic",
    "original_base": "Original PPO",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror",
        type=Path,
        default=Path(__file__).parent / "correction_20260905_mirror",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "ppo_correction_curves.png",
    )
    args = parser.parse_args()
    result = summarize(args.mirror)
    runs = [
        run
        for run in result["runs"]
        if (run.get("run") or {}).get("map_name") == "2s3z"
    ]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    for run in runs:
        arm, seed = run["run"]["arm"], run["run"]["seed"]
        color = COLORS[arm]
        style = "--" if seed == 1 and arm != "original_base" else "-"
        for axis, metric, final_metric in zip(
            axes, ["eval/win_rate", "eval/return_mean"], ["win_rate", "return_mean"]
        ):
            values = [row for row in run["curve"] if metric in row]
            axis.plot(
                [row["step"] / 1000 for row in values],
                [row[metric] for row in values],
                color=color,
                linestyle=style,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=f"{LABELS[arm]} · seed {seed}",
            )
            if run["completed"] and final_metric in run["final"]:
                axis.scatter(
                    [50],
                    [run["final"][final_metric]],
                    color=color,
                    marker="D",
                    s=42,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )
    for axis in axes:
        axis.set(xlabel="Environment transitions (thousands)", xlim=(4, 52))
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].set(ylabel="Evaluation win rate", ylim=(-0.025, 1.035))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(ylabel="Evaluation return", ylim=(0, 21))
    complete = len(runs) == 5 and all(run["completed"] for run in runs)
    fig.suptitle(
        "2s3z correction screen"
        + (" · final checkpoints" if complete else " · training ongoing"),
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    timestamp = result["mirror"].get("captured_utc", "unavailable")
    fig.text(
        0.08,
        0.895,
        "Curves: 32 fixed episodes every 5k · diamonds: separate final 128 episodes",
        color="#555555",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.845,
        "Original seed-1 control uses the same runtime; both corrected training seeds are shown.",
        color="#555555",
        fontsize=9,
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.08,
        0.018,
        "Local evidence captured: " + timestamp,
        color="#666666",
        fontsize=8,
    )
    fig.subplots_adjust(top=0.76, bottom=0.27, left=0.075, right=0.98, wspace=0.25)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
