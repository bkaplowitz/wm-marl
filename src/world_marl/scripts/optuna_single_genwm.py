"""Optuna sweep for the generative world-model arms (llada2 by default).

Wraps ``world-marl-train-single-genwm`` in one subprocess per trial, so every
trial produces the standard config/outcome/summary artifacts in a fresh JAX
process. Each trial runs ``--runs-per-trial`` seeded runs inside that one
subprocess (``--num-runs``); the objective is ``policy_trained_mean`` from the
trial's ``summary.json``, which is already the mean across runs. W&B logging
is delegated to the training script itself (``--wandb-project`` is forwarded,
one W&B run per seed, grouped per trial), keeping this wrapper a thin
controller: sqlite study, ``trials.csv``, ``best_trial.json``.

Budgets come from the shared JEPA presets (same keys ``compare_single_wm``
forwards), so tuned trials stay comparable to the benchmark arms. Needs the
``hpo`` extra: ``uv sync --extra dev --extra hpo``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from world_marl.scripts.compare_single_wm import (
    GENWM_ARMS,
    GENWM_COMMON_KEYS,
    GENWM_PRESET_KEYS,
    MAX_CYCLES,
)
from world_marl.scripts.write_dmc_vector_launcher import COMMON_PARAMS, PRESETS

FAILED_SCORE = -1_000_000.0

MODEL_DIM_CHOICES = [128, 256]
BLOCK_SIZE_CHOICES = [1, 2, 4]
STEPS_PER_BLOCK_CHOICES = [2, 4, 8]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="gymnax:CartPole-v1")
    parser.add_argument("--arm", choices=GENWM_ARMS, default="llada2")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="mainline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("runs/genwm_optuna"))
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--runs-per-trial",
        type=int,
        default=3,
        help="Seeded runs per trial; the objective averages across them.",
    )
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument(
        "--model-dims",
        type=int,
        nargs="+",
        default=MODEL_DIM_CHOICES,
        help="model_dim categorical choices for the search space.",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=BLOCK_SIZE_CHOICES,
        help="block_size categorical choices (llada2 arm only).",
    )
    parser.add_argument(
        "--steps-per-blocks",
        type=int,
        nargs="+",
        default=STEPS_PER_BLOCK_CHOICES,
        help="steps_per_block categorical choices (llada2 arm only).",
    )
    parser.add_argument(
        "--enqueue",
        action="append",
        default=[],
        metavar="JSON",
        help="JSON dict of sampled params to enqueue as an initial trial "
        "(repeatable; values must match the search-space choices).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra flag token appended to every trial command (repeatable).",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.study_name is None:
        task_slug = args.task.replace(":", "_").replace("/", "_")
        args.study_name = f"genwm_{args.arm}_{task_slug}"
    if args.storage is None:
        args.storage = f"sqlite:///{args.out_root / 'study.db'}"
    return args


def sample_params(
    trial,
    *,
    arm: str,
    model_dims: list[int] | None = None,
    block_sizes: list[int] | None = None,
    steps_per_blocks: list[int] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "wm_learning_rate": trial.suggest_float(
            "wm_learning_rate", 1e-4, 3e-3, log=True
        ),
        "model_dim": trial.suggest_categorical(
            "model_dim", model_dims or MODEL_DIM_CHOICES
        ),
        "num_layers": trial.suggest_categorical("num_layers", [2, 3, 4]),
        "obs_bins": trial.suggest_categorical("obs_bins", [16, 32, 64]),
        "imag_horizon": trial.suggest_categorical("imag_horizon", [10, 15, 25]),
        "ppo_learning_rate": trial.suggest_float(
            "ppo_learning_rate", 1e-4, 1e-3, log=True
        ),
        "ent_coef": trial.suggest_float("ent_coef", 1e-3, 3e-2, log=True),
    }
    if arm == "llada2":
        # the sampler ceil-divides the obs-token sequence into blocks, so any
        # block_size is safe on ragged lengths (reacher's 11 obs tokens incl.).
        params["block_size"] = trial.suggest_categorical(
            "block_size", block_sizes or BLOCK_SIZE_CHOICES
        )
        params["steps_per_block"] = trial.suggest_categorical(
            "steps_per_block", steps_per_blocks or STEPS_PER_BLOCK_CHOICES
        )
    return params


def base_params(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    params = {key: preset[key] for key in GENWM_PRESET_KEYS if key in preset}
    params.update({key: COMMON_PARAMS[key] for key in GENWM_COMMON_KEYS})
    params["eval_episodes"] = preset.get("policy_eval_episodes", 64)
    params["max_cycles"] = MAX_CYCLES
    params["num_runs"] = args.runs_per_trial
    params["allow_fail"] = True
    return params


def build_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    trial_dir: Path,
    seed: int,
    trial_number: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "world_marl.scripts.train_single_genwm",
        "--env",
        args.task,
        "--arm",
        args.arm,
        "--seed",
        str(seed),
    ]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    if args.wandb_project:
        command.extend(["--wandb-project", args.wandb_project])
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
        group = args.wandb_group or args.study_name
        command.extend(["--wandb-group", f"{group}_trial{trial_number:04d}"])
    command.extend(["--out-dir", str(trial_dir)])
    command.extend(args.extra_arg)
    return command


def latest_summary(trial_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(
        trial_dir.rglob("summary.json"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def run_trial(args: argparse.Namespace, trial) -> float:
    params = {
        **base_params(args),
        **sample_params(
            trial,
            arm=args.arm,
            model_dims=args.model_dims,
            block_sizes=args.block_sizes,
            steps_per_blocks=args.steps_per_blocks,
        ),
    }
    trial_dir = args.out_root / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + trial.number
    command = build_command(args, params, trial_dir, seed, trial.number)
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("command", shlex.join(command))
    if args.dry_run:
        print(f"[trial {trial.number}] dry-run: {shlex.join(command)}", flush=True)
        return 0.0
    started = time.time()
    log_path = trial_dir / "console.log"
    with log_path.open("wb") as log:
        log.write((shlex.join(command) + "\n\n").encode())
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    trial.set_user_attr("return_code", result.returncode)
    trial.set_user_attr("runtime_seconds", elapsed)
    summary = latest_summary(trial_dir)
    if summary is None or summary.get("policy_trained_mean") is None:
        print(
            f"[trial {trial.number}] no usable summary "
            f"(exit={result.returncode}) -> score {FAILED_SCORE}",
            flush=True,
        )
        return FAILED_SCORE
    score = float(summary["policy_trained_mean"])
    for key in (
        "num_runs",
        "policy_random_mean",
        "policy_initial_mean",
        "improvement_over_baseline",
    ):
        if summary.get(key) is not None:
            trial.set_user_attr(key, summary[key])
    per_run = [
        json.loads(path.read_text()).get("policy_trained_mean")
        for path in sorted(trial_dir.rglob("outcome.json"))
    ]
    trial.set_user_attr("per_run_trained", per_run)
    print(
        f"[trial {trial.number}] trained={score:.2f} per-run={per_run} "
        f"exit={result.returncode} ({elapsed / 60:.1f} min)",
        flush=True,
    )
    return score


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run with: uv sync --extra dev --extra hpo"
        ) from exc

    (args.out_root / "hpo_config.json").write_text(
        json.dumps(
            {
                "args": {
                    name: (str(value) if isinstance(value, Path) else value)
                    for name, value in vars(args).items()
                },
                "base_params": base_params(args),
            },
            indent=2,
        )
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )
    for raw in args.enqueue:
        study.enqueue_trial(json.loads(raw), skip_if_exists=True)
    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials)

    frame = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    frame.to_csv(args.out_root / "trials.csv", index=False)
    try:
        best = study.best_trial
    except ValueError:
        best = None
    (args.out_root / "best_trial.json").write_text(
        json.dumps(
            {
                "number": best.number if best else None,
                "value": best.value if best else None,
                "params": best.params if best else None,
                "user_attrs": dict(best.user_attrs) if best else {},
            },
            indent=2,
        )
    )
    if best is not None:
        print(f"best trial {best.number}: value={best.value} params={best.params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
