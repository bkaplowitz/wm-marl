"""Plot JEPA-vs-PPO Brax diagnostics from run summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ERROR_PATTERNS = (
    "Traceback",
    "RESOURCE_EXHAUSTED",
    "OOM",
    "out of memory",
    "Killed",
    "XlaRuntimeError",
    "cuSolver",
)


@dataclass(frozen=True)
class PpoSummary:
    env: str
    best_return: float | None
    best_steps: int | None
    last_return: float | None
    last_steps: int | None
    last_length: float | None


@dataclass(frozen=True)
class JepaSummary:
    env: str
    status: str
    passed: bool | None = None
    world_model_passed: bool | None = None
    policy_main_passed: bool | None = None
    paired_policy_ok: bool | None = None
    initial_return: float | None = None
    trained_return: float | None = None
    improvement: float | None = None
    primary_improvement: float | None = None
    online_improvement: float | None = None
    acceptance_rate: float | None = None
    open_loop_loss: float | None = None
    control_open_loop_loss: float | None = None
    recent_validation_improvement: float | None = None
    anchor_validation_degradation: float | None = None
    summary_path: str | None = None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jepa = load_jepa_summaries(args.jepa_root)
    ppo = load_ppo_summaries(args.ppo_root)
    envs = sorted(set(jepa) | set(ppo))

    rows = [build_row(env, jepa.get(env), ppo.get(env)) for env in envs]
    write_csv(args.out_dir / "summary.csv", rows)
    write_json(args.out_dir / "summary.json", rows)
    write_markdown(args.out_dir / "diagnostics.md", rows)
    write_csv(
        args.out_dir / "policy_diagnostics.csv",
        build_policy_diagnostic_rows(args.jepa_root, envs),
    )
    sample_efficiency_rows = build_sample_efficiency_rows(
        args.jepa_root,
        args.ppo_root,
        env=args.sample_env,
        step_limit=args.sample_step_limit,
    )
    write_csv(args.out_dir / "sample_efficiency.csv", sample_efficiency_rows)

    plot_returns(args.out_dir / "returns_vs_ppo.png", envs, jepa, ppo)
    plot_jepa_improvement(args.out_dir / "jepa_improvement.png", envs, jepa)
    plot_model_policy_scatter(args.out_dir / "model_vs_policy.png", envs, jepa)
    plot_ppo_curves(args.out_dir / "ppo_learning_curves.png", args.ppo_root)
    plot_jepa_metric_curves(
        args.out_dir / "jepa_model_loss_curves.png",
        args.jepa_root,
        envs,
        metric="model/total_loss",
        title="JEPA World-model Total Loss",
        ylabel="Total loss",
    )
    plot_jepa_metric_curves(
        args.out_dir / "jepa_policy_return_curves.png",
        args.jepa_root,
        envs,
        metric="policy/imagined_return",
        title="JEPA Imagined Policy Return",
        ylabel="Imagined return",
    )
    plot_policy_selection_returns(
        args.out_dir / "jepa_policy_selection_returns.png",
        args.jepa_root,
        envs,
    )
    plot_policy_training_metrics(
        args.out_dir / "jepa_policy_training_metrics.png",
        args.jepa_root,
        envs,
    )
    plot_model_head_losses(
        args.out_dir / "jepa_model_head_losses.png",
        args.jepa_root,
        envs,
    )
    plot_sample_efficiency(
        args.out_dir
        / f"{args.sample_env}_return_vs_train_steps_{args.sample_step_limit // 1000}k.png",
        sample_efficiency_rows,
        env=args.sample_env,
        step_limit=args.sample_step_limit,
    )

    print(f"Wrote diagnostics to {args.out_dir}")
    print(f"- {args.out_dir / 'summary.csv'}")
    print(f"- {args.out_dir / 'policy_diagnostics.csv'}")
    print(f"- {args.out_dir / 'sample_efficiency.csv'}")
    print(f"- {args.out_dir / 'diagnostics.md'}")
    print(f"- {args.out_dir / 'returns_vs_ppo.png'}")
    print(f"- {args.out_dir / 'jepa_improvement.png'}")
    print(f"- {args.out_dir / 'model_vs_policy.png'}")
    print(f"- {args.out_dir / 'ppo_learning_curves.png'}")
    print(f"- {args.out_dir / 'jepa_model_loss_curves.png'}")
    print(f"- {args.out_dir / 'jepa_policy_return_curves.png'}")
    print(f"- {args.out_dir / 'jepa_policy_selection_returns.png'}")
    print(f"- {args.out_dir / 'jepa_policy_training_metrics.png'}")
    print(f"- {args.out_dir / 'jepa_model_head_losses.png'}")
    print(
        "- "
        f"{args.out_dir / f'{args.sample_env}_return_vs_train_steps_{args.sample_step_limit // 1000}k.png'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jepa-root",
        type=Path,
        required=True,
        help="Root containing per-env JEPA run directories.",
    )
    parser.add_argument(
        "--ppo-root",
        type=Path,
        required=True,
        help="Root containing per-env PPO baseline summary directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for CSV, JSON, and PNG diagnostics.",
    )
    parser.add_argument(
        "--sample-env",
        default="reacher",
        help="Environment to plot for sample-efficiency diagnostics.",
    )
    parser.add_argument(
        "--sample-step-limit",
        type=int,
        default=500_000,
        help="Maximum real training-replay environment steps on the sample-efficiency plot.",
    )
    return parser.parse_args()


def load_jepa_summaries(root: Path) -> dict[str, JepaSummary]:
    envs = (
        {path.name for path in root.iterdir() if path.is_dir()}
        if root.exists()
        else set()
    )
    envs.update(nohup_env_name(path) for path in root.glob("*.nohup.log"))
    out: dict[str, JepaSummary] = {}
    for env in sorted(envs):
        paths = sorted((root / env).glob("brax_jepa_*/summary.json"))
        if paths:
            out[env] = parse_jepa_summary(env, paths[-1])
            continue

        log_path = root / f"{env}.nohup.log"
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            status = (
                "crashed"
                if any(pattern in text for pattern in ERROR_PATTERNS)
                else "running_or_incomplete"
            )
        else:
            status = "not_launched"
        out[env] = JepaSummary(env=env, status=status)
    return out


def parse_jepa_summary(env: str, path: Path) -> JepaSummary:
    payload = json.loads(path.read_text())
    return JepaSummary(
        env=env,
        status="done",
        passed=payload.get("passed"),
        world_model_passed=payload.get("world_model_passed"),
        policy_main_passed=payload.get("policy_main_passed"),
        paired_policy_ok=payload.get("paired_policy_ok"),
        initial_return=maybe_float(payload.get("aggregate_policy_initial_mean")),
        trained_return=maybe_float(payload.get("aggregate_policy_trained_mean")),
        improvement=maybe_float(payload.get("aggregate_policy_improvement")),
        primary_improvement=maybe_float(
            payload.get("aggregate_policy_primary_improvement")
        ),
        online_improvement=maybe_float(
            payload.get("aggregate_policy_online_phase_improvement")
        ),
        acceptance_rate=maybe_float(
            payload.get("aggregate_model_update_acceptance_rate")
        ),
        open_loop_loss=maybe_float(payload.get("aggregate_final_open_loop_loss")),
        control_open_loop_loss=maybe_float(
            payload.get("aggregate_control_final_open_loop_loss")
        ),
        recent_validation_improvement=maybe_float(
            payload.get("aggregate_candidate_recent_validation_improvement")
        ),
        anchor_validation_degradation=maybe_float(
            payload.get("aggregate_candidate_anchor_validation_degradation")
        ),
        summary_path=str(path),
    )


def load_ppo_summaries(root: Path) -> dict[str, PpoSummary]:
    out = {}
    for path in sorted(root.glob("*/summary.json")):
        env = path.parent.name
        payload = json.loads(path.read_text())
        history = payload.get("history", [])
        if not history:
            out[env] = PpoSummary(env, None, None, None, None, None)
            continue
        best = max(history, key=lambda row: row.get("eval/episode_reward", -math.inf))
        last = history[-1]
        out[env] = PpoSummary(
            env=env,
            best_return=maybe_float(best.get("eval/episode_reward")),
            best_steps=maybe_int(best.get("num_steps")),
            last_return=maybe_float(last.get("eval/episode_reward")),
            last_steps=maybe_int(last.get("num_steps")),
            last_length=maybe_float(last.get("eval/avg_episode_length")),
        )
    return out


def build_row(
    env: str,
    jepa: JepaSummary | None,
    ppo: PpoSummary | None,
) -> dict[str, Any]:
    trained = jepa.trained_return if jepa else None
    ppo_best = ppo.best_return if ppo else None
    return {
        "env": env,
        "jepa_status": jepa.status if jepa else "not_launched",
        "jepa_passed": jepa.passed if jepa else None,
        "jepa_world_model_passed": jepa.world_model_passed if jepa else None,
        "jepa_policy_main_passed": jepa.policy_main_passed if jepa else None,
        "jepa_initial_return": trained_or_none(jepa.initial_return if jepa else None),
        "jepa_trained_return": trained_or_none(trained),
        "jepa_improvement": trained_or_none(jepa.improvement if jepa else None),
        "jepa_online_improvement": trained_or_none(
            jepa.online_improvement if jepa else None
        ),
        "jepa_acceptance_rate": trained_or_none(jepa.acceptance_rate if jepa else None),
        "jepa_open_loop_loss": trained_or_none(jepa.open_loop_loss if jepa else None),
        "jepa_control_open_loop_loss": trained_or_none(
            jepa.control_open_loop_loss if jepa else None
        ),
        "jepa_recent_validation_improvement": trained_or_none(
            jepa.recent_validation_improvement if jepa else None
        ),
        "jepa_anchor_validation_degradation": trained_or_none(
            jepa.anchor_validation_degradation if jepa else None
        ),
        "ppo_best_return": trained_or_none(ppo_best),
        "ppo_last_return": trained_or_none(ppo.last_return if ppo else None),
        "ppo_best_steps": ppo.best_steps if ppo else None,
        "ppo_last_steps": ppo.last_steps if ppo else None,
        "ppo_last_length": trained_or_none(ppo.last_length if ppo else None),
        "jepa_minus_ppo_best": (
            trained_or_none(trained - ppo_best)
            if trained is not None and ppo_best is not None
            else None
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else ["env"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Brax JEPA Diagnostics",
        "",
        "Generated from JEPA summary files and PPO baseline summaries.",
        "",
        "| env | status | JEPA return | PPO best | delta | JEPA improvement | "
        "online improvement | accept rate | open-loop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {env} | {status} | {jepa} | {ppo} | {delta} | {improve} | "
            "{online} | {accept} | {open_loop} |".format(
                env=row["env"],
                status=row["jepa_status"],
                jepa=format_cell(row["jepa_trained_return"]),
                ppo=format_cell(row["ppo_best_return"]),
                delta=format_cell(row["jepa_minus_ppo_best"]),
                improve=format_cell(row["jepa_improvement"]),
                online=format_cell(row["jepa_online_improvement"]),
                accept=format_cell(row["jepa_acceptance_rate"]),
                open_loop=format_cell(row["jepa_open_loop_loss"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: machine-readable aggregate table.",
            "- `policy_diagnostics.csv`: best real-env policy selection "
            "checkpoints by phase.",
            "- `sample_efficiency.csv`: DreamerV3-style return checkpoints "
            "against real training replay steps for the selected environment.",
            "- `summary.json`: same aggregate table in JSON.",
            "- `returns_vs_ppo.png`: JEPA final return against PPO best and last.",
            "- `jepa_improvement.png`: offline+online and online-only JEPA gains.",
            "- `model_vs_policy.png`: final model loss against policy improvement.",
            "- `ppo_learning_curves.png`: PPO evaluation curves from baseline runs.",
            "- `jepa_model_loss_curves.png`: JEPA model loss curves from metrics logs.",
            "- `jepa_policy_return_curves.png`: imagined-return policy curves.",
            "- `jepa_policy_selection_returns.png`: real-env selection returns "
            "during policy training.",
            "- `jepa_policy_training_metrics.png`: imagined return, value loss, "
            "and action saturation during policy training.",
            "- `jepa_model_head_losses.png`: reward and control-value model losses.",
            "- `<env>_return_vs_train_steps_500k.png`: DreamerV3-style "
            "sample-efficiency curve bounded to 500k real training replay "
            "steps by default. JEPA points use actor-replay collection returns, "
            "not policy-selection or validation returns.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def plot_returns(
    path: Path,
    envs: list[str],
    jepa: dict[str, JepaSummary],
    ppo: dict[str, PpoSummary],
) -> None:
    x = list(range(len(envs)))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(9, len(envs) * 1.6), 5.5))
    ax.bar(
        [i - width for i in x],
        [
            value_or_nan(jepa.get(env).trained_return if env in jepa else None)
            for env in envs
        ],
        width,
        label="JEPA trained",
        color="#2d6cdf",
    )
    ax.bar(
        x,
        [
            value_or_nan(ppo.get(env).best_return if env in ppo else None)
            for env in envs
        ],
        width,
        label="PPO best",
        color="#4c9f70",
    )
    ax.bar(
        [i + width for i in x],
        [
            value_or_nan(ppo.get(env).last_return if env in ppo else None)
            for env in envs
        ],
        width,
        label="PPO last",
        color="#d28e2d",
    )
    annotate_status(ax, envs, jepa)
    ax.set_title("Real-environment Return")
    ax.set_ylabel("Episode return")
    ax.set_xticks(x, envs, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_jepa_improvement(
    path: Path,
    envs: list[str],
    jepa: dict[str, JepaSummary],
) -> None:
    x = list(range(len(envs)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(9, len(envs) * 1.6), 5.5))
    ax.bar(
        [i - width / 2 for i in x],
        [
            value_or_nan(jepa.get(env).improvement if env in jepa else None)
            for env in envs
        ],
        width,
        label="Total improvement",
        color="#2d6cdf",
    )
    ax.bar(
        [i + width / 2 for i in x],
        [
            value_or_nan(jepa.get(env).online_improvement if env in jepa else None)
            for env in envs
        ],
        width,
        label="Online phase improvement",
        color="#b85c9e",
    )
    annotate_status(ax, envs, jepa)
    ax.set_title("JEPA Policy Improvement")
    ax.set_ylabel("Return improvement")
    ax.set_xticks(x, envs, rotation=25, ha="right")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_policy_scatter(
    path: Path,
    envs: list[str],
    jepa: dict[str, JepaSummary],
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for env in envs:
        item = jepa.get(env)
        if item is None or item.open_loop_loss is None or item.improvement is None:
            continue
        ax.scatter(
            item.open_loop_loss,
            item.improvement,
            s=80,
            color="#2d6cdf" if item.passed else "#c4413d",
            alpha=0.9,
        )
        ax.annotate(
            env,
            (item.open_loop_loss, item.improvement),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set_title("World-model Loss vs Policy Improvement")
    ax.set_xlabel("Final open-loop latent loss")
    ax.set_ylabel("JEPA return improvement")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ppo_curves(path: Path, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text())
        history = payload.get("history", [])
        points = [
            (row.get("num_steps"), row.get("eval/episode_reward"))
            for row in history
            if row.get("num_steps") is not None
            and row.get("eval/episode_reward") is not None
        ]
        if not points:
            continue
        steps, rewards = zip(*points)
        ax.plot(
            steps,
            rewards,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=summary_path.parent.name,
        )
        plotted = True
    ax.set_title("PPO Baseline Learning Curves")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_jepa_metric_curves(
    path: Path,
    root: Path,
    envs: list[str],
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted = False
    for env in envs:
        points = load_jepa_metric_points(root, env, metric)
        if not points:
            continue
        steps, values = zip(*points)
        ax.plot(steps, values, linewidth=1.8, label=env)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Logged metric point")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_jepa_metric_points(
    root: Path,
    env: str,
    metric: str,
) -> list[tuple[int, float]]:
    run_dir = latest_jepa_run_dir(root, env)
    if run_dir is None:
        return []

    points: list[tuple[int, float]] = []
    for metrics_path in sorted(run_dir.glob("none/run_*/metrics.jsonl")):
        for line in metrics_path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = maybe_float(row.get(metric))
            if value is None:
                continue
            points.append((len(points) + 1, value))
    return points


def build_policy_diagnostic_rows(root: Path, envs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env in envs:
        metrics = load_main_metrics_rows(root, env)
        selection_rows = [
            row
            for row in metrics
            if row.get("phase") == "policy_selection"
            and row.get("policy_selection_mean_return") is not None
        ]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in selection_rows:
            control = str(row.get("control", ""))
            policy_phase = str(row.get("policy_phase", ""))
            grouped.setdefault((control, policy_phase), []).append(row)

        for (control, policy_phase), group in sorted(grouped.items()):
            best = max(
                group,
                key=lambda row: (
                    maybe_float(row.get("policy_selection_mean_return")) or -math.inf
                ),
            )
            last = group[-1]
            selected_steps = [
                maybe_int(row.get("policy_selection_step"))
                for row in group
                if row.get("policy_selection_selected")
            ]
            rows.append(
                {
                    "env": env,
                    "control": control,
                    "policy_phase": policy_phase,
                    "num_selection_points": len(group),
                    "best_selection_step": maybe_int(best.get("policy_selection_step")),
                    "best_selection_mean_return": trained_or_none(
                        maybe_float(best.get("policy_selection_mean_return"))
                    ),
                    "last_selection_step": maybe_int(last.get("policy_selection_step")),
                    "last_selection_mean_return": trained_or_none(
                        maybe_float(last.get("policy_selection_mean_return"))
                    ),
                    "selected_steps": " ".join(
                        str(step) for step in selected_steps if step is not None
                    ),
                    "logged_best_step": maybe_int(
                        last.get("policy_selection_best_step")
                    ),
                    "logged_best_mean_return": trained_or_none(
                        maybe_float(last.get("policy_selection_best_mean_return"))
                    ),
                }
            )
    return rows


def plot_policy_selection_returns(
    path: Path,
    root: Path,
    envs: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for env in envs:
        metrics = load_main_metrics_rows(root, env)
        groups: dict[str, list[tuple[int, float]]] = {}
        for row in metrics:
            if row.get("phase") != "policy_selection":
                continue
            step = maybe_int(row.get("policy_selection_step"))
            value = maybe_float(row.get("policy_selection_mean_return"))
            if step is None or value is None:
                continue
            phase = str(row.get("policy_phase", "policy"))
            groups.setdefault(phase, []).append((step, value))
        for phase, points in sorted(groups.items()):
            steps, values = zip(*points)
            label = f"{env}:{phase}"
            ax.plot(steps, values, marker="o", linewidth=1.7, markersize=3, label=label)
            plotted = True
    ax.set_title("Real-env Policy Selection Returns")
    ax.set_xlabel("Policy update within phase")
    ax.set_ylabel("Selection return")
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_policy_training_metrics(
    path: Path,
    root: Path,
    envs: list[str],
) -> None:
    metrics = [
        ("policy/imagined_return", "Imagined return"),
        ("policy/value_loss", "Value loss"),
        ("policy/action_saturation_fraction", "Action saturation fraction"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 9), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        plotted = False
        for env in envs:
            points = load_policy_training_points(root, env, metric)
            if not points:
                continue
            steps, values = zip(*points)
            ax.plot(steps, values, linewidth=1.6, label=env)
            plotted = True
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize="small")
    axes[-1].set_xlabel("Logged policy-training point")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_head_losses(
    path: Path,
    root: Path,
    envs: list[str],
) -> None:
    metrics = [
        ("model/reward_loss", "Reward loss"),
        ("model/reward_constant_mse", "Reward constant baseline"),
        ("model/control_value_loss", "Control-value loss"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 9), sharex=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        plotted = False
        for env in envs:
            points = load_jepa_metric_points(root, env, metric)
            if not points:
                continue
            steps, values = zip(*points)
            ax.plot(steps, values, linewidth=1.6, label=env)
            plotted = True
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize="small")
    axes[-1].set_xlabel("Logged model-training point")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_sample_efficiency_rows(
    jepa_root: Path,
    ppo_root: Path,
    *,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(load_ppo_sample_efficiency_rows(ppo_root, env, step_limit))
    for label, run_dir in latest_jepa_env_run_dirs(jepa_root, env):
        rows.extend(
            load_jepa_sample_efficiency_rows(
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
            str(row.get("phase", "")),
        ),
    )


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


def load_ppo_sample_efficiency_rows(
    root: Path,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    summary_path = root / env / "summary.json"
    if not summary_path.exists():
        return []

    payload = load_json_dict(summary_path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("history", []):
        step = maybe_int(item.get("num_steps"))
        value = maybe_float(item.get("eval/episode_reward"))
        if step is None or value is None or step > step_limit:
            continue
        rows.append(
            {
                "source": "ppo",
                "label": f"{env}_ppo",
                "env": env,
                "step": step,
                "return": value,
                "phase": "eval",
                "iteration": None,
                "run_dir": str(summary_path.parent),
            }
        )
    return rows


def load_jepa_sample_efficiency_rows(
    *,
    label: str,
    run_dir: Path,
    env: str,
    step_limit: int,
) -> list[dict[str, Any]]:
    run_path = run_dir / "none" / "run_000"
    outcome_path = run_path / "outcome.json"
    outcome = load_json_dict(outcome_path) if outcome_path.exists() else {}
    config = load_json_dict(run_path / "config.json")
    args = config.get("args", {})

    num_envs = maybe_int(args.get("num_envs")) or 1
    collect_steps = maybe_int(args.get("collect_steps"))
    online_collect_steps = (
        maybe_int(args.get("online_collect_steps")) or collect_steps or 0
    )
    initial_train_steps = (
        maybe_int(outcome.get("real_initial_train_replay_env_steps"))
        or maybe_int(load_json_dict(run_path / "train_replay.json").get("env_steps"))
        or ((collect_steps or 0) * num_envs)
    )

    rows: list[dict[str, Any]] = []
    online_history = outcome.get("online_history")
    if not isinstance(online_history, list):
        online_history = load_json_list(run_path / "online_history.json")
    cumulative_steps = initial_train_steps
    for index, item in enumerate(online_history, start=1):
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


def plot_sample_efficiency(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    env: str,
    step_limit: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not rows:
        ax.text(
            0.5,
            0.5,
            f"No {env} sample-efficiency rows found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["source"]), str(row["label"])), []).append(row)

    for (source, label), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: maybe_int(row.get("step")) or -1)
        steps = [maybe_int(row.get("step")) for row in group]
        returns = [maybe_float(row.get("return")) for row in group]
        points = [
            (step, value)
            for step, value in zip(steps, returns, strict=True)
            if step is not None and value is not None and step <= step_limit
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        if source == "ppo":
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.2,
                markersize=4,
                color="#333333",
                label="PPO eval",
            )
        else:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=f"JEPA {label}",
            )

    ax.set_title(f"{env.title()} DreamerV3-Style Return vs Environment Steps")
    ax.set_xlabel("Environment steps in training replay")
    ax.set_ylabel("Episode return")
    ax.set_xlim(0, step_limit)
    ax.grid(alpha=0.25)
    if rows:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_policy_training_points(
    root: Path,
    env: str,
    metric: str,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in load_main_metrics_rows(root, env):
        phase = str(row.get("phase", ""))
        if not phase.endswith("frozen_model_policy"):
            continue
        value = maybe_float(row.get(metric))
        if value is None:
            continue
        points.append((len(points) + 1, value))
    return points


def load_main_metrics_rows(root: Path, env: str) -> list[dict[str, Any]]:
    run_dir = latest_jepa_run_dir(root, env)
    if run_dir is None:
        return []

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(run_dir.glob("none/run_*/metrics.jsonl")):
        rows.extend(read_jsonl(metrics_path))
    return rows


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


def latest_jepa_run_dir(root: Path, env: str) -> Path | None:
    paths = sorted((root / env).glob("brax_jepa_*/summary.json"))
    if not paths:
        return None
    return paths[-1].parent


def nohup_env_name(path: Path) -> str:
    suffix = ".nohup.log"
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def annotate_status(
    ax: plt.Axes,
    envs: list[str],
    jepa: dict[str, JepaSummary],
) -> None:
    y0, y1 = ax.get_ylim()
    y = y0 + 0.04 * (y1 - y0)
    for index, env in enumerate(envs):
        status = jepa.get(env).status if env in jepa else "not_launched"
        if status == "done":
            continue
        ax.text(
            index,
            y,
            status.replace("_", " "),
            rotation=90,
            va="bottom",
            ha="center",
            fontsize=8,
            color="#7a2e2e",
        )


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


def trained_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def value_or_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
