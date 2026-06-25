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

    print(f"Wrote diagnostics to {args.out_dir}")
    print(f"- {args.out_dir / 'summary.csv'}")
    print(f"- {args.out_dir / 'diagnostics.md'}")
    print(f"- {args.out_dir / 'returns_vs_ppo.png'}")
    print(f"- {args.out_dir / 'jepa_improvement.png'}")
    print(f"- {args.out_dir / 'model_vs_policy.png'}")
    print(f"- {args.out_dir / 'ppo_learning_curves.png'}")
    print(f"- {args.out_dir / 'jepa_model_loss_curves.png'}")
    print(f"- {args.out_dir / 'jepa_policy_return_curves.png'}")


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
            "- `summary.json`: same aggregate table in JSON.",
            "- `returns_vs_ppo.png`: JEPA final return against PPO best and last.",
            "- `jepa_improvement.png`: offline+online and online-only JEPA gains.",
            "- `model_vs_policy.png`: final model loss against policy improvement.",
            "- `ppo_learning_curves.png`: PPO evaluation curves from baseline runs.",
            "- `jepa_model_loss_curves.png`: JEPA model loss curves from metrics logs.",
            "- `jepa_policy_return_curves.png`: imagined-return policy curves.",
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
