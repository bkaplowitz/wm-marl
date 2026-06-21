"""Build visual dashboards for DMC JEPA experiment artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a visual dashboard from DMC JEPA summary/outcome JSON.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Experiment directory with summary.json, run directory with outcome.json, "
        "or a direct JSON path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <run-dir>/visual_report.png.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    output = make_visual_report(args.run_dir, output_path=args.out, dpi=args.dpi)
    print(output)
    return 0


def make_visual_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> Path:
    """Render a PNG dashboard for a DMC JEPA experiment or single run."""

    payload, artifact_dir = load_payload(run_dir)
    if output_path is None:
        output = artifact_dir / "visual_report.png"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = payload.get("runs", [])
    controls = _control_order(runs)
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    fig.suptitle(_title(payload, artifact_dir), fontsize=15, fontweight="bold")

    _plot_policy_returns(axes[0, 0], runs)
    _plot_policy_improvement(axes[0, 1], runs, controls)
    _plot_model_losses(axes[1, 0], runs, controls)
    _plot_online_returns(axes[1, 1], runs)
    _plot_action_diagnostics(axes[2, 0], runs, controls)
    _plot_summary_text(axes[2, 1], payload, runs)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def load_payload(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load summary.json or outcome.json and return a summary-shaped payload."""

    source = Path(path)
    if source.is_dir():
        summary_path = source / "summary.json"
        outcome_path = source / "outcome.json"
        if summary_path.exists():
            return _read_json(summary_path), source
        if outcome_path.exists():
            outcome = _read_json(outcome_path)
            return _single_run_summary(outcome), source
        raise FileNotFoundError(
            f"expected {summary_path} or {outcome_path} to exist",
        )

    payload = _read_json(source)
    if source.name == "outcome.json" or "run_dir" in payload:
        return _single_run_summary(payload), source.parent
    return payload, source.parent


def _single_run_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": outcome.get("passed", False),
        "world_model_passed": outcome.get(
            "world_model_passed",
            outcome.get("passed", False),
        ),
        "policy_training_enabled": outcome.get("policy_training_enabled", False),
        "aggregate_final_jepa_loss": outcome.get("final_jepa_loss"),
        "aggregate_final_open_loop_loss": outcome.get("final_open_loop_loss"),
        "aggregate_policy_random_mean": outcome.get("policy_random_mean"),
        "aggregate_policy_initial_mean": outcome.get("policy_initial_mean"),
        "aggregate_policy_trained_mean": outcome.get("policy_trained_mean"),
        "aggregate_policy_improvement": outcome.get("policy_improvement"),
        "runs": [outcome],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _title(payload: dict[str, Any], artifact_dir: Path) -> str:
    name = artifact_dir.name
    status = "passed" if payload.get("passed", False) else "not passed"
    return f"DMC JEPA Visual Report: {name} ({status})"


def _control_order(runs: list[dict[str, Any]]) -> list[str]:
    controls = sorted({str(run.get("control", "none")) for run in runs})
    return ["none", *[control for control in controls if control != "none"]]


def _values(
    runs: list[dict[str, Any]],
    control: str,
    key: str,
) -> list[float]:
    values = []
    for run in runs:
        if str(run.get("control", "none")) != control:
            continue
        value = run.get(key)
        if _finite(value):
            values.append(float(value))
    return values


def _metric_values(
    runs: list[dict[str, Any]],
    control: str,
    key: str,
) -> list[float]:
    values = []
    for run in runs:
        if str(run.get("control", "none")) != control:
            continue
        metrics = run.get("final_model_metrics", {})
        value = metrics.get(key)
        if _finite(value):
            values.append(float(value))
    return values


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _plot_policy_returns(ax, runs: list[dict[str, Any]]) -> None:
    main = [run for run in runs if str(run.get("control", "none")) == "none"]
    phases = [
        ("random", "policy_random_mean"),
        ("initial", "policy_initial_mean"),
        ("trained", "policy_trained_mean"),
    ]
    grouped = [[], [], []]
    for run in main:
        for index, (_, key) in enumerate(phases):
            value = run.get(key)
            if _finite(value):
                grouped[index].append(float(value))

    if not any(grouped):
        _empty(ax, "Policy returns", "No policy-training returns in this run.")
        return

    labels = [phase for phase, _ in phases]
    means = [_mean(values) if values else np.nan for values in grouped]
    xs = np.arange(len(labels))
    ax.bar(xs, means, color=["#8d99ae", "#457b9d", "#2a9d8f"], alpha=0.85)
    _scatter_points(ax, xs, grouped)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("real-env return")
    ax.set_title("Main Policy Returns")
    ax.grid(True, axis="y", alpha=0.25)


def _plot_policy_improvement(
    ax,
    runs: list[dict[str, Any]],
    controls: list[str],
) -> None:
    grouped = [_values(runs, control, "policy_improvement") for control in controls]
    if not any(grouped):
        _empty(ax, "Policy improvement", "Policy training was disabled.")
        return

    xs = np.arange(len(controls))
    means = [_mean(values) if values else np.nan for values in grouped]
    colors = ["#2a9d8f" if control == "none" else "#e76f51" for control in controls]
    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.bar(xs, means, color=colors, alpha=0.85)
    _scatter_points(ax, xs, grouped)
    ax.set_xticks(xs, [_short_label(control) for control in controls], rotation=15)
    ax.set_ylabel("trained - initial return")
    ax.set_title("Policy Improvement vs Controls")
    ax.grid(True, axis="y", alpha=0.25)


def _plot_model_losses(
    ax,
    runs: list[dict[str, Any]],
    controls: list[str],
) -> None:
    jepa = [_values(runs, control, "final_jepa_loss") for control in controls]
    open_loop = [
        _values(runs, control, "final_open_loop_loss") for control in controls
    ]
    if not any(jepa) and not any(open_loop):
        _empty(ax, "World-model losses", "No model-loss fields found.")
        return

    xs = np.arange(len(controls))
    width = 0.38
    jepa_means = [_mean(values) if values else np.nan for values in jepa]
    open_means = [_mean(values) if values else np.nan for values in open_loop]
    ax.bar(xs - width / 2, jepa_means, width, label="JEPA")
    ax.bar(xs + width / 2, open_means, width, label="open-loop")
    ax.set_xticks(xs, [_short_label(control) for control in controls], rotation=15)
    ax.set_ylabel("loss, lower is better")
    ax.set_title("Latent Prediction Fit")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    if _all_positive(jepa_means + open_means):
        ax.set_yscale("log")


def _plot_online_returns(ax, runs: list[dict[str, Any]]) -> None:
    main = [run for run in runs if str(run.get("control", "none")) == "none"]
    plotted = False
    for run in main:
        history = run.get("online_history", [])
        returns = []
        for item in history:
            actor_replay = item.get("actor_replay", {})
            value = actor_replay.get("mean_return")
            if _finite(value):
                returns.append(float(value))
        if not returns:
            continue
        plotted = True
        xs = np.arange(1, len(returns) + 1)
        label = f"run {run.get('run_index', len(ax.lines))}"
        ax.plot(xs, returns, marker="o", label=label)

    if not plotted:
        _empty(ax, "Online actor replay", "No online actor-replay returns found.")
        return

    ax.set_xlabel("online iteration")
    ax.set_ylabel("real-env return")
    ax.set_title("Fresh Data From Selected Actor")
    ax.legend()
    ax.grid(True, alpha=0.25)


def _plot_action_diagnostics(
    ax,
    runs: list[dict[str, Any]],
    controls: list[str],
) -> None:
    candidates = [
        ("sensitivity", "model/continuous_action_low_high_sensitivity"),
        ("Q gap", "model/action_value_gap"),
        ("terminal recall", "model/terminal_recall"),
    ]
    key = None
    label = None
    grouped: list[list[float]] = []
    for label_candidate, key_candidate in candidates:
        candidate_values = [
            _metric_values(runs, control, key_candidate) for control in controls
        ]
        if any(candidate_values):
            key = key_candidate
            label = label_candidate
            grouped = candidate_values
            break

    if key is None or label is None:
        _empty(ax, "Action diagnostics", "No action-diagnostic metric found.")
        return

    xs = np.arange(len(controls))
    means = [_mean(values) if values else np.nan for values in grouped]
    colors = ["#2a9d8f" if control == "none" else "#e76f51" for control in controls]
    ax.bar(xs, means, color=colors, alpha=0.85)
    _scatter_points(ax, xs, grouped)
    ax.set_xticks(xs, [_short_label(control) for control in controls], rotation=15)
    ax.set_ylabel(key)
    ax.set_title(f"Action-Conditioning Diagnostic: {label}")
    ax.grid(True, axis="y", alpha=0.25)


def _plot_summary_text(
    ax,
    payload: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    ax.axis("off")
    main = [run for run in runs if str(run.get("control", "none")) == "none"]
    controls = [run for run in runs if str(run.get("control", "none")) != "none"]
    lines = [
        f"overall passed: {payload.get('passed')}",
        f"world model passed: {payload.get('world_model_passed')}",
        f"policy enabled: {payload.get('policy_training_enabled')}",
        f"main runs: {len(main)}",
        f"control runs: {len(controls)}",
        "",
        "aggregate policy:",
        f"  random: {_fmt(payload.get('aggregate_policy_random_mean'))}",
        f"  initial: {_fmt(payload.get('aggregate_policy_initial_mean'))}",
        f"  trained: {_fmt(payload.get('aggregate_policy_trained_mean'))}",
        f"  improvement: {_fmt(payload.get('aggregate_policy_improvement'))}",
        "",
        "aggregate model:",
        f"  final JEPA: {_fmt(payload.get('aggregate_final_jepa_loss'))}",
        f"  final open-loop: {_fmt(payload.get('aggregate_final_open_loop_loss'))}",
    ]
    paired = payload.get("paired_control_differences", {})
    if paired:
        lines.extend(["", "paired control advantages:"])
        for control, metrics in sorted(paired.items()):
            policy = metrics.get("mean_policy_improvement_advantage")
            open_loop = metrics.get("mean_open_loop_advantage")
            lines.append(
                f"  {_short_label(control)}: policy {_fmt(policy)}, "
                f"open-loop {_fmt(open_loop)}",
            )
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    ax.set_title("Run Summary")


def _scatter_points(ax, xs: np.ndarray, grouped: list[list[float]]) -> None:
    for x, values in zip(xs, grouped, strict=True):
        if not values:
            continue
        offsets = np.linspace(-0.08, 0.08, num=len(values))
        ax.scatter(
            np.full(len(values), x) + offsets,
            values,
            color="#222222",
            s=24,
            alpha=0.75,
            zorder=3,
        )


def _empty(ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)


def _short_label(control: str) -> str:
    return control.replace("-world-model", "").replace("-replay", "")


def _all_positive(values: list[float | None]) -> bool:
    finite_values = [value for value in values if _finite(value)]
    return bool(finite_values) and all(float(value) > 0.0 for value in finite_values)


def _fmt(value: Any) -> str:
    if not _finite(value):
        return "n/a"
    return f"{float(value):.4g}"


if __name__ == "__main__":
    raise SystemExit(main())
