"""Plot the fixed-cohort simulator audit and episode-cluster uncertainty."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    root = Path(__file__).parent / "frozen_corrected_2s3z_10530"
    summary = json.loads((root / "summary.json").read_text())
    intervals = json.loads((root / "paired_confidence.json").read_text())[
        "paired_simulator"
    ]["horizons"]
    horizons = [1, 2, 4, 5, 8]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.3))
    paths = [
        ("teacher_factual", "Factual histories", "#167a91"),
        ("self_fed", "Predicted histories", "#cf6828"),
    ]
    for index, metric in enumerate(
        ("action_mask_false_positive", "deployed_alive_brier")
    ):
        for path, label, color in paths:
            values = [summary["paired"][path][f"h{h}"][metric] for h in horizons]
            axes[index].plot(horizons, values, "o-", color=color, label=label)
    axes[0].set_title("Unavailable actions predicted legal", loc="left", fontsize=11)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylim(0, 0.5)
    axes[1].set_title("Liveness prediction error", loc="left", fontsize=11)
    axes[1].set_ylabel("Deployed-path Brier score")
    axes[1].set_ylim(bottom=0)
    key = "self_feed_excess_return_mse"
    points = np.array([intervals[f"h{h}"][key]["point"] for h in horizons])
    bounds = np.array(
        [intervals[f"h{h}"][key]["episode_cluster_percentile_95"] for h in horizons]
    )
    axes[2].errorbar(
        horizons,
        points,
        yerr=np.stack([points - bounds[:, 0], bounds[:, 1] - points]),
        fmt="o",
        capsize=4,
        color="#cf6828",
    )
    axes[2].axhline(0, color="#666666", linewidth=0.8)
    axes[2].set_title("Added cumulative-return error", loc="left", fontsize=11)
    axes[2].set_ylabel("Predicted − factual histories: MSE")
    for axis in axes:
        axis.set_xticks(horizons)
        axis.set_xlabel("Prediction horizon")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "The simulator deteriorates when it consumes its own predictions",
        x=0.065,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.86,
        "2s3z · corrected checkpoint at 10,530 steps · identical roots, actions and categorical draws",
        fontsize=9,
        color="#555555",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.065,
        0.025,
        "64 replay roots / 42 episodes. Error bars: 95% episode-cluster intervals; checkpoint-conditional, not training-seed uncertainty.",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.27, top=0.73, wspace=0.42)
    fig.savefig(root / "simulator_drift.png", dpi=180)
    fig.savefig(root / "simulator_drift.pdf")


if __name__ == "__main__":
    main()
