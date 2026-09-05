"""Plot the fixed first-wave PPO protocol without checkpoint selection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
TREATMENTS = {
    "base": ("Original PPO", "#0072B2"),
    "wm_start5k_nowarmup": ("Delayed world model, no warmup", "#009E73"),
    "actor2x256_nopeerres": ("Small actor, no peer residual", "#E69F00"),
    "env16_total50k": ("16 environments", "#777777"),
    "entropy_annealed": ("Annealed entropy", "#D55E00"),
    "critic1l8h": ("Simplified critic", "#CC79A7"),
}


if __name__ == "__main__":
    runs = sum(
        [json.loads(p.read_text())["runs"] for p in ROOT.glob("*_treatments.json")],
        [],
    )
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for axis, map_name in zip(axes, ["3m", "2s3z"]):
        for treatment, (label, color) in TREATMENTS.items():
            run = next(r for r in runs if r["name"] == f"{treatment}-{map_name}-seed0")
            curve = run["curve"]
            axis.plot(
                [r["step"] / 1000 for r in curve],
                [r["eval/win_rate"] for r in curve],
                color=color,
                linewidth=1.9,
                label=label,
            )
            axis.scatter(
                [50],
                [run["final"]["win_rate"]],
                marker="D",
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=38,
                zorder=5,
            )
        axis.set(
            title=map_name,
            xlabel="Environment transitions (thousands)",
            xlim=(4, 52),
            ylim=(-0.025, 1.04),
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Evaluation win rate")
    fig.suptitle(
        "Existing PPO treatment results",
        x=0.09,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.09,
        0.89,
        "One training seed · curves: 32 episodes · diamonds: separate final 128 episodes",
        color="#555555",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.79, bottom=0.27, left=0.085, right=0.98, wspace=0.13)
    fig.savefig(ROOT / "ppo_baseline_curves.png", dpi=170)
    fig.savefig(ROOT / "ppo_baseline_curves.svg")
