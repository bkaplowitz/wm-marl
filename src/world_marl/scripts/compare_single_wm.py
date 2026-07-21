"""Policy-level comparison of world-model arms on single-agent environments.

Runs one job per (env, arm) — ``jepa`` via ``world-marl-train-dmc-jepa`` and
the generative arms via ``world-marl-train-single-genwm`` — each in its own
subprocess (a crash in one arm must not take down the rest, and every job gets
a fresh JAX process). Jobs run sequentially; budgets come from the shared JEPA
presets in ``write_dmc_vector_launcher`` so all arms see the same real-env
step budget. Extra per-arm flags are appended after the preset flags, so
argparse last-wins overrides work for smoke testing.

Afterwards each job's ``summary.json`` is normalized into one row per (env,
arm) and written as ``comparison.csv`` / ``comparison.json`` /
``comparison.png``. Use ``--aggregate-only <dir>`` to rebuild the report from
an existing comparison directory without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from world_marl.scripts.write_dmc_vector_launcher import PRESETS

JEPA_ARM = "jepa"
GENWM_ARMS = ("discrete-transformer", "continuous-transformer", "llada2")
MODEL_FREE_ARM = "model-free"
DEFAULT_ARMS = (JEPA_ARM, *GENWM_ARMS, MODEL_FREE_ARM)
DEFAULT_ENVS = ("gymnax:CartPole-v1", "brax:reacher")
MAX_CYCLES = 1000

# Budget/capacity preset keys forwarded to train_single_genwm (the model-free
# arm goes through the same entry point, so it sees the identical real-step
# budget). Both scripts read collect budgets in per-env steps. Deliberately
# excludes the JEPA sequence batch_size (64 sequences x 32-step chunks); the
# genwm arms match on update counts with their own flat-transition batch.
GENWM_PRESET_KEYS = (
    "num_envs",
    "collect_steps",
    "train_steps",
    "policy_train_steps",
    "online_iterations",
    "online_collect_steps",
    "online_train_steps",
    "online_policy_train_steps",
)
GENWM_COMMON_KEYS = (
    "policy_batch_size",
    "imag_horizon",
    "model_dim",
    "num_layers",
    "num_heads",
    "mlp_ratio",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--preset", choices=tuple(PRESETS), default="jepa_500k")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/wm_comparison"))
    parser.add_argument(
        "--extra-jepa-arg",
        action="append",
        default=[],
        help="Extra flag token for jepa jobs, appended after preset flags "
        "(repeatable; last occurrence of a flag wins in argparse).",
    )
    parser.add_argument(
        "--extra-genwm-arg",
        action="append",
        default=[],
        help="Extra flag token for generative-arm jobs, appended after "
        "preset flags (repeatable).",
    )
    parser.add_argument(
        "--aggregate-only",
        type=Path,
        default=None,
        help="Rebuild comparison artifacts from this existing directory "
        "instead of running jobs.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Forward W&B logging to every job (genwm and jepa alike).",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-group",
        default=None,
        help="W&B group for this comparison's jobs (default: comparison dir name).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job commands without running them.",
    )
    args = parser.parse_args(argv)
    for arm in args.arms:
        if arm not in DEFAULT_ARMS:
            parser.error(f"unknown arm {arm!r}; choose from {DEFAULT_ARMS}")
    return args


def _flag_tokens(params: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            tokens.append(flag)
            tokens.extend(str(item) for item in value)
            continue
        tokens.extend((flag, str(value)))
    return tokens


def _add_wandb_params(params: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.wandb_project:
        return
    params["wandb_project"] = args.wandb_project
    if args.wandb_entity:
        params["wandb_entity"] = args.wandb_entity
    if args.wandb_group:
        params["wandb_group"] = args.wandb_group


def build_command(
    env: str, arm: str, job_dir: Path, args: argparse.Namespace
) -> list[str]:
    if arm == JEPA_ARM:
        params = dict(PRESETS[args.preset])
        _add_wandb_params(params, args)
        return [
            "uv",
            "run",
            "world-marl-train-dmc-jepa",
            "--env",
            env,
            "--seed",
            str(args.seed),
            *_flag_tokens(params),
            "--out-dir",
            str(job_dir),
            *args.extra_jepa_arg,
        ]
    preset = PRESETS[args.preset]
    params = {key: preset[key] for key in GENWM_PRESET_KEYS if key in preset}
    params.update({key: preset[key] for key in GENWM_COMMON_KEYS})
    params["eval_episodes"] = preset.get("final_policy_eval_episodes", 64)
    params["max_cycles"] = MAX_CYCLES
    params["allow_fail"] = True
    _add_wandb_params(params, args)
    return [
        "uv",
        "run",
        "world-marl-train-single-genwm",
        "--env",
        env,
        "--arm",
        arm,
        "--seed",
        str(args.seed),
        *_flag_tokens(params),
        "--out-dir",
        str(job_dir),
        *args.extra_genwm_arg,
    ]


def env_slug(env: str) -> str:
    return env.replace(":", "_").replace("/", "_")


def run_jobs(args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    jobs = []
    for env in args.envs:
        for arm in args.arms:
            job_dir = out_dir / env_slug(env) / arm
            command = build_command(env, arm, job_dir, args)
            jobs.append({"env": env, "arm": arm, "dir": job_dir, "command": command})

    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        return []

    records = []
    for index, job in enumerate(jobs):
        job_dir: Path = job["dir"]
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "console.log"
        print(
            f"[{index + 1}/{len(jobs)}] {job['env']} / {job['arm']} -> {log_path}",
            flush=True,
        )
        started = time.time()
        with log_path.open("wb") as log:
            log.write((shlex.join(job["command"]) + "\n\n").encode())
            log.flush()
            result = subprocess.run(
                job["command"], stdout=log, stderr=subprocess.STDOUT
            )
        elapsed = time.time() - started
        print(
            f"[{index + 1}/{len(jobs)}] exit={result.returncode} "
            f"({elapsed / 60:.1f} min)",
            flush=True,
        )
        records.append(
            {
                "env": job["env"],
                "arm": job["arm"],
                "job_dir": str(job_dir),
                "exit_code": result.returncode,
                "runtime_seconds": elapsed,
            }
        )
    return records


def _latest_summary(job_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(
        job_dir.rglob("summary.json"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def _normalize_summary(arm: str, summary: dict[str, Any]) -> dict[str, Any]:
    if arm == JEPA_ARM:
        random_mean = summary.get("aggregate_policy_random_mean")
        initial_mean = summary.get("aggregate_policy_initial_mean")
        trained_mean = summary.get("aggregate_policy_trained_mean")
    else:
        random_mean = summary.get("policy_random_mean")
        initial_mean = summary.get("policy_initial_mean")
        trained_mean = summary.get("policy_trained_mean")
    improvement = None
    if all(value is not None for value in (random_mean, initial_mean, trained_mean)):
        improvement = trained_mean - max(random_mean, initial_mean)
    return {
        "policy_random_mean": random_mean,
        "policy_initial_mean": initial_mean,
        "policy_trained_mean": trained_mean,
        "improvement_over_baseline": improvement,
        "passed": summary.get("passed"),
    }


CSV_COLUMNS = (
    "env",
    "arm",
    "policy_random_mean",
    "policy_initial_mean",
    "policy_trained_mean",
    "improvement_over_baseline",
    "passed",
    "exit_code",
    "runtime_seconds",
)


def aggregate(
    out_dir: Path,
    envs: list[str],
    arms: list[str],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_job = {(record["env"], record["arm"]): record for record in records}
    rows = []
    for env in envs:
        for arm in arms:
            job_dir = out_dir / env_slug(env) / arm
            record = by_job.get((env, arm), {})
            row: dict[str, Any] = {
                "env": env,
                "arm": arm,
                "job_dir": str(job_dir),
                "exit_code": record.get("exit_code"),
                "runtime_seconds": record.get("runtime_seconds"),
            }
            summary = _latest_summary(job_dir) if job_dir.is_dir() else None
            if summary is None:
                print(f"warning: no summary.json under {job_dir}", flush=True)
                row.update(_normalize_summary(arm, {}))
            else:
                row.update(_normalize_summary(arm, summary))
            rows.append(row)
    return rows


def write_report(out_dir: Path, rows: list[dict[str, Any]], config: dict) -> None:
    (out_dir / "comparison.json").write_text(
        json.dumps({"config": config, "rows": rows}, indent=2)
    )
    with (out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_plot(out_dir, rows)
    print(f"wrote {out_dir / 'comparison.csv'}", flush=True)
    print(f"wrote {out_dir / 'comparison.json'}", flush=True)


def write_plot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    envs = list(dict.fromkeys(row["env"] for row in rows))
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), squeeze=False)
    stages = ("policy_random_mean", "policy_initial_mean", "policy_trained_mean")
    labels = ("random", "initial", "trained")
    for axis, env in zip(axes[0], envs):
        env_rows = [row for row in rows if row["env"] == env]
        arms = [row["arm"] for row in env_rows]
        positions = np.arange(len(arms))
        width = 0.25
        for offset, (stage, label) in enumerate(zip(stages, labels)):
            values = [
                row[stage] if row[stage] is not None else np.nan for row in env_rows
            ]
            axis.bar(positions + (offset - 1) * width, values, width, label=label)
        axis.set_xticks(positions)
        axis.set_xticklabels(arms, rotation=20, ha="right")
        axis.set_title(env)
        axis.set_ylabel("mean episode return")
        axis.grid(axis="y", alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only is not None:
        out_dir = args.aggregate_only
        if not out_dir.is_dir():
            print(f"error: {out_dir} is not a directory", file=sys.stderr)
            return 2
        rows = aggregate(out_dir, args.envs, args.arms, [])
        write_report(out_dir, rows, {"aggregate_only": True, **_config(args)})
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"wm_comparison_{stamp}"
    if args.wandb_project and not args.wandb_group:
        args.wandb_group = out_dir.name
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(_config(args), indent=2))
        print(f"comparison dir: {out_dir}", flush=True)
    records = run_jobs(args, out_dir)
    if args.dry_run:
        return 0
    rows = aggregate(out_dir, args.envs, args.arms, records)
    write_report(out_dir, rows, _config(args))
    failed = [row for row in rows if row["policy_trained_mean"] is None]
    return 1 if failed else 0


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: (str(value) if isinstance(value, Path) else value)
        for name, value in vars(args).items()
    }


if __name__ == "__main__":
    sys.exit(main())
