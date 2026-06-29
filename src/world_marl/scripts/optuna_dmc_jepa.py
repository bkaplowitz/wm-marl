"""Run Optuna sweeps for the DMC vector JEPA pipeline.

The script deliberately wraps ``train_dmc_jepa`` instead of importing its inner
training loop. Each trial therefore produces the same artifacts as a normal run,
while Optuna and W&B receive the sampled config, intermediate scores, final
summary, metrics.jsonl rows, and a run-directory artifact.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any


FAST_PRESET: dict[str, Any] = {
    "num_envs": 16,
    "env_workers": 16,
    "collect_steps": 4096,
    "validation_steps": 1024,
    "train_steps": 6000,
    "policy_train_steps": 1500,
    "online_iterations": 4,
    "online_collect_steps": 1024,
    "online_validation_steps": 512,
    "online_train_steps": 1500,
    "policy_selection_interval": 250,
    "policy_selection_episodes": 16,
    "policy_eval_episodes": 32,
    "policy_confirmation_episodes": 32,
    "final_policy_eval_episodes": 64,
}

BASE_PARAMS: dict[str, Any] = {
    "num_runs": 1,
    "critic_warmup_steps": 1000,
    "critic_horizon": 32,
    "policy_batch_size": 512,
    "policy_objective": "direct",
    "policy_return_mode": "reward-only",
    "policy_actor_baseline": "none",
    "policy_return_normalization": "none",
    "imag_horizon": 15,
    "model_grad_clip_norm": 100.0,
    "actor_grad_clip_norm": 10.0,
    "critic_grad_clip_norm": 100.0,
    "online_policy_trust_coef": 1.0,
    "online_candidate_refit": True,
    "online_candidate_eval_interval": 250,
    "online_candidate_min_recent_improvement": 0.0,
    "online_candidate_max_anchor_degradation": 0.03,
    "online_anchor_batch_fraction": 0.5,
    "online_control_value_weight": 0.1,
    "batch_size": 64,
    "chunk_length": 32,
    "open_loop_horizon": 15,
    "model_horizon": 5,
    "context_window": 4,
    "latent_dim": 128,
    "model_dim": 128,
    "num_layers": 2,
    "num_heads": 4,
    "mlp_ratio": 4,
    "dynamics_ensemble_size": 5,
    "uncertainty_penalty": 0.1,
    "regularizer": "sigreg",
    "regularizer_weight": 0.05,
    "reward_prediction_mode": "mse",
    "value_prediction_mode": "mse",
    "twohot_bins": 41,
    "twohot_min": -20.0,
    "twohot_max": 20.0,
    "clip_imagined_rewards": False,
    "imagined_reward_min": 0.0,
    "imagined_reward_max": 1.0,
    "controls": ("none",),
    "allow_fail": True,
}
WANDB_LOCK = threading.Lock()
WANDB_CONTROLLER_STEP = itertools.count()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run with: uv run --extra hpo "
            "world-marl-optuna-dmc-jepa ..."
        ) from exc

    write_json(
        args.out_root / "hpo_config.json",
        {
            "args": vars(args),
            "base_params": BASE_PARAMS,
            "fast_preset": FAST_PRESET,
        },
    )

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.pruner_startup_trials,
        n_warmup_steps=args.pruner_warmup_steps,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner if args.enable_pruning else optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    gpu_slots: queue.Queue[str] = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.jobs_per_gpu):
            gpu_slots.put(gpu)
    n_jobs = max(1, len(args.gpus) * args.jobs_per_gpu)
    controller_run = init_wandb_controller(args, n_jobs)

    def objective(trial) -> float:
        gpu = gpu_slots.get()
        try:
            return run_trial(args, trial, gpu, controller_run)
        finally:
            gpu_slots.put(gpu)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=n_jobs,
            timeout=args.timeout_seconds,
            gc_after_trial=True,
        )
        write_trials_csv(args, study)
        best = best_trial_or_none(study)
        write_json(
            args.out_root / "best_trial.json",
            {
                "number": best.number if best is not None else None,
                "value": best.value if best is not None else None,
                "params": best.params if best is not None else None,
                "user_attrs": dict(best.user_attrs) if best is not None else {},
            },
        )
        log_wandb_controller_event(
            controller_run,
            {
                "hpo/completed": True,
                "hpo/best_value": best.value if best is not None else None,
                "hpo/best_trial": best.number if best is not None else None,
            },
        )
    finally:
        finish_wandb_controller(controller_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="reacher/easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=Path("runs/dmc_jepa_optuna"))
    parser.add_argument("--study-name", default="dmc_jepa_reacher_easy_hpo")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--monitor-interval-seconds", type=int, default=60)
    parser.add_argument("--enable-pruning", action="store_true")
    parser.add_argument("--pruner-startup-trials", type=int, default=4)
    parser.add_argument("--pruner-warmup-steps", type=int, default=2)
    parser.add_argument("--selection-gap-penalty", type=float, default=0.25)
    parser.add_argument("--eval-std-penalty", type=float, default=0.05)
    parser.add_argument("--wandb-project", default="world-marl-jepa-hpo")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument(
        "--no-wandb-artifact",
        dest="wandb_artifact",
        action="store_false",
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.storage is None:
        args.storage = f"sqlite:///{args.out_root / 'study.db'}"
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be >= 1")
    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")
    if args.monitor_interval_seconds < 5:
        parser.error("--monitor-interval-seconds must be >= 5")
    return args


def run_trial(
    args: argparse.Namespace,
    trial,
    gpu: str,
    controller_run: Any | None,
) -> float:
    params = sample_params(trial)
    trial_dir = args.out_root / f"trial_{trial.number:04d}"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True)

    command = build_command(args, params, trial_dir)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": env.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
            ),
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda"),
        }
    )
    log_path = trial_dir / "trial.nohup.log"
    trial.set_user_attr("gpu", gpu)
    trial.set_user_attr("run_dir", str(trial_dir))
    trial.set_user_attr("command", " ".join(command))
    write_json(
        trial_dir / "trial_config.json",
        {
            "trial": trial.number,
            "gpu": gpu,
            "params": params,
            "command": command,
        },
    )
    log_wandb_controller_event(
        controller_run,
        {
            "hpo/trial_started": 1,
            "hpo/active_trial": trial.number,
            f"trial/{trial.number:04d}/gpu": safe_int(gpu),
            f"trial/{trial.number:04d}/model_dim": params.get("model_dim"),
            f"trial/{trial.number:04d}/num_heads": params.get("num_heads"),
            f"trial/{trial.number:04d}/learning_rate": params.get("learning_rate"),
            f"trial/{trial.number:04d}/actor_learning_rate": params.get(
                "actor_learning_rate"
            ),
        },
        step=trial.number,
    )

    if args.dry_run:
        print(f"[trial {trial.number}] gpu={gpu} dry-run: {' '.join(command)}")
        return 0.0

    wandb_stream_state: dict[str, Any] = {"step": 0, "lines": {}}
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        last_step = -1
        while process.poll() is None:
            time.sleep(args.monitor_interval_seconds)
            stream_wandb_trial_metrics(
                controller_run,
                trial_dir,
                wandb_stream_state,
                trial.number,
            )
            progress = extract_progress(trial_dir)
            if progress is None:
                continue
            step, value, metrics = progress
            if step <= last_step:
                continue
            last_step = step
            trial.report(value, step=step)
            trial.set_user_attr(f"intermediate_{step}", value)
            log_wandb_controller_event(
                controller_run,
                {
                    f"trial/{trial.number:04d}/intermediate_score": value,
                    f"trial/{trial.number:04d}/intermediate_step": step,
                    f"trial/{trial.number:04d}/gpu": safe_int(gpu),
                    **prefix_keys(
                        wandb_scalars(flatten_dict(metrics)),
                        f"trial/{trial.number:04d}/progress/",
                    ),
                },
                step=trial.number * 1000 + step,
            )
            if trial.should_prune():
                terminate_process(process)
                raise import_optuna().TrialPruned()
        return_code = process.returncode
    stream_wandb_trial_metrics(
        controller_run,
        trial_dir,
        wandb_stream_state,
        trial.number,
    )
    metrics = extract_final_metrics(trial_dir)
    metrics["return_code"] = return_code
    score = score_metrics(metrics, args)
    trial.set_user_attr("score", score)
    for key, value in metrics.items():
        if is_json_scalar(value):
            trial.set_user_attr(key, value)
    write_json(trial_dir / "trial_result.json", metrics | {"score": score})
    log_wandb_controller_event(
        controller_run,
        {
            "hpo/trial_finished": 1,
            "hpo/last_finished_trial": trial.number,
            f"trial/{trial.number:04d}/score": score,
            f"trial/{trial.number:04d}/return_code": return_code,
            **prefix_keys(
                wandb_scalars(metrics),
                f"trial/{trial.number:04d}/final/",
            ),
        },
        step=trial.number * 1000 + 999,
    )
    log_wandb_completed_trial(args, controller_run, trial_dir, metrics, score)
    if return_code != 0:
        raise RuntimeError(f"trial {trial.number} failed with return code {return_code}")
    return score


def init_wandb_controller(args: argparse.Namespace, n_jobs: int) -> Any | None:
    if args.wandb_mode == "disabled":
        return None
    with WANDB_LOCK:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "W&B is not installed. Run with: uv run --extra hpo "
                "world-marl-optuna-dmc-jepa ..."
            ) from exc
        group = args.wandb_group or args.study_name
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group,
            name=f"{args.study_name}-controller",
            mode=args.wandb_mode,
            tags=[*args.wandb_tags, "controller"],
            config={
                "task": args.task,
                "seed": args.seed,
                "n_trials": args.n_trials,
                "gpus": args.gpus,
                "jobs_per_gpu": args.jobs_per_gpu,
                "n_jobs": n_jobs,
                "study_name": args.study_name,
                "storage": args.storage,
                "out_root": str(args.out_root),
                "enable_pruning": args.enable_pruning,
                "sampler_seed": args.sampler_seed,
                "fast_preset": FAST_PRESET,
                "base_params": BASE_PARAMS,
            },
            reinit=True,
        )
        define_wandb_trial_metrics(run)
        run.log(
            {
                "hpo/started": True,
                "hpo/n_trials": args.n_trials,
                "hpo/n_jobs": n_jobs,
                "hpo/jobs_per_gpu": args.jobs_per_gpu,
            },
            step=0,
        )
        return run


def log_wandb_controller_event(
    controller_run: Any | None,
    payload: dict[str, Any],
    *,
    step: int | None = None,
) -> None:
    if controller_run is None:
        return
    scalars = wandb_scalars(payload)
    if not scalars:
        return
    with WANDB_LOCK:
        controller_run.log(scalars, step=next(WANDB_CONTROLLER_STEP))


def finish_wandb_controller(controller_run: Any | None) -> None:
    if controller_run is None:
        return
    with WANDB_LOCK:
        controller_run.finish()


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def define_wandb_trial_metrics(wandb_run) -> None:
    for metric in ("trial/*", "hpo/*"):
        try:
            wandb_run.define_metric(metric)
        except Exception:
            pass


def stream_wandb_trial_metrics(
    wandb_run,
    trial_dir: Path,
    state: dict[str, Any],
    trial_number: int,
) -> None:
    if wandb_run is None:
        return
    line_counts: dict[str, int] = state.setdefault("lines", {})
    for path in sorted((trial_dir / "run").glob("*/none/run_000/metrics.jsonl")):
        key = str(path)
        seen = line_counts.get(key, 0)
        lines = path.read_text(errors="replace").splitlines()
        if seen >= len(lines):
            continue
        for line in lines[seen:]:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = wandb_trial_payload(row, trial_number)
            if not payload:
                continue
            step = int(state.get("step", 0))
            payload[f"trial/{trial_number:04d}/stream_step"] = step
            with WANDB_LOCK:
                wandb_run.log(payload, step=step)
            state["step"] = step + 1
        line_counts[key] = len(lines)


def wandb_trial_payload(
    row: dict[str, Any],
    trial_number: int,
) -> dict[str, float | int | bool]:
    flat = wandb_scalars(flatten_dict(row))
    trial_prefix = f"trial/{trial_number:04d}/"
    payload = prefix_keys(flat, trial_prefix + "metrics/")
    env_steps = first_number(row.get("env_steps"))
    if env_steps is not None:
        payload[trial_prefix + "env_steps"] = env_steps

    phase = str(row.get("phase") or row.get("policy_phase") or "")
    aliases: dict[str, Any] = {}
    for source, target in [
        ("loss", "train/loss"),
        ("jepa", "train/jepa"),
        ("model/loss", "train/model_loss"),
        ("model/jepa_loss", "train/jepa_loss"),
        ("model/reward_loss", "train/reward_loss"),
        ("model/continue_loss", "train/continue_loss"),
        ("model/value_loss", "train/value_loss"),
        ("model/open_loop_loss", "model/open_loop_loss"),
        ("policy/loss", "policy/loss"),
        ("policy/imagined_return", "policy/imagined_return"),
        ("policy_selection_mean_return", "return/policy_selection_mean"),
        ("policy_selection_best_mean_return", "return/policy_selection_best"),
        ("mean_return", "return/mean"),
        ("std_return", "return/std"),
        ("episode_return_mean", "return/episode_mean"),
        ("episode_return_std", "return/episode_std"),
    ]:
        if source in row:
            aliases[target] = row[source]
    if phase and "mean_return" in row:
        aliases[f"return/{phase}_mean"] = row["mean_return"]
    if phase and "std_return" in row:
        aliases[f"return/{phase}_std"] = row["std_return"]
    payload.update(prefix_keys(wandb_scalars(aliases), trial_prefix))
    return payload


def log_wandb_completed_trial(
    args: argparse.Namespace,
    wandb_run,
    trial_dir: Path,
    metrics: dict[str, Any],
    score: float,
) -> None:
    if wandb_run is None:
        return
    trial_number = trial_number_from_dir(trial_dir)
    prefix = f"trial/{trial_number:04d}/final/"
    with WANDB_LOCK:
        wandb_run.log(
            {
                f"trial/{trial_number:04d}/score": score,
                **prefix_keys(wandb_scalars(metrics), prefix),
            }
        )
        wandb_run.summary[f"trial/{trial_number:04d}/score"] = score
        for key, value in metrics.items():
            if is_json_scalar(value):
                wandb_run.summary[f"trial/{trial_number:04d}/{key}"] = value
        if args.wandb_artifact:
            log_wandb_artifact(wandb_run, trial_dir, trial_number)


def trial_number_from_dir(trial_dir: Path) -> int:
    try:
        return int(trial_dir.name.split("_")[-1])
    except (IndexError, ValueError):
        return 0


def sample_params(trial) -> dict[str, Any]:
    scale = trial.suggest_categorical("scale", ["128x4", "256x8"])
    if scale == "128x4":
        latent_dim, model_dim, num_heads = 128, 128, 4
    else:
        latent_dim, model_dim, num_heads = 256, 256, 8
    actor_lr = trial.suggest_float("actor_learning_rate", 3e-5, 3e-4, log=True)
    model_lr = trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 3e-4])
    return {
        **BASE_PARAMS,
        **FAST_PRESET,
        "latent_dim": latent_dim,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "learning_rate": model_lr,
        "actor_learning_rate": actor_lr,
        "policy_train_steps": trial.suggest_categorical(
            "policy_train_steps", [1500, 3000, 4500]
        ),
        "online_policy_train_steps": trial.suggest_categorical(
            "online_policy_train_steps", [750, 1500, 2250]
        ),
        "online_policy_trust_coef": trial.suggest_categorical(
            "online_policy_trust_coef", [0.5, 1.0, 2.0]
        ),
        "uncertainty_penalty": trial.suggest_categorical(
            "uncertainty_penalty", [0.05, 0.1, 0.2]
        ),
        "actor_grad_clip_norm": trial.suggest_categorical(
            "actor_grad_clip_norm", [3.0, 10.0, 30.0]
        ),
    }


def build_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    trial_dir: Path,
) -> list[str]:
    trial_number = int(trial_dir.name.split("_")[-1])
    command = [
        sys.executable,
        "-m",
        "world_marl.scripts.train_dmc_jepa",
        "--env",
        f"dmc:{args.task}",
        "--seed",
        str(args.seed + trial_number),
    ]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            command.append(flag)
            command.extend(str(item) for item in value)
            continue
        command.extend([flag, str(value)])
    command.extend(["--out-dir", str(trial_dir / "run")])
    return command


def extract_progress(trial_dir: Path) -> tuple[int, float, dict[str, Any]] | None:
    run_dirs = sorted(
        (trial_dir / "run").glob("*/none/run_000"),
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        return None
    run_dir = run_dirs[-1]
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for path in run_dir.glob("online_*_policy_confirmation_trained_policy_evaluation.json"):
        step = iteration_from_name(path.name)
        payload = read_json(path)
        mean_return = metric_mean_return(payload)
        if step is not None and mean_return is not None:
            candidates.append(
                (
                    step,
                    mean_return,
                    {
                        "iteration": step,
                        "confirmation_return": mean_return,
                        "confirmation_std": metric_std_return(payload),
                    },
                )
            )
    for path in run_dir.glob("online_*_policy_trained_policy_evaluation.json"):
        step = iteration_from_name(path.name)
        payload = read_json(path)
        mean_return = metric_mean_return(payload)
        if step is not None and mean_return is not None:
            candidates.append(
                (
                    step,
                    mean_return,
                    {
                        "iteration": step,
                        "trained_return": mean_return,
                        "trained_std": metric_std_return(payload),
                    },
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def extract_final_metrics(trial_dir: Path) -> dict[str, Any]:
    summary_paths = sorted(
        (trial_dir / "run").glob("*/summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    metrics: dict[str, Any] = {}
    if summary_paths:
        summary = read_json(summary_paths[-1])
        metrics.update(flatten_summary(summary))
        metrics["summary_path"] = str(summary_paths[-1])
    run_dirs = sorted(
        (trial_dir / "run").glob("*/none/run_000"),
        key=lambda path: path.stat().st_mtime,
    )
    if run_dirs:
        run_metrics = extract_run_dir_metrics(run_dirs[-1])
        metrics.update({k: v for k, v in run_metrics.items() if v is not None})
    if "best_selection_return" not in metrics:
        metrics["best_selection_return"] = None
    return metrics


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "passed",
        "world_model_passed",
        "policy_main_passed",
        "online_trend_passed",
        "policy_main_beats_controls",
        "paired_policy_ok",
        "aggregate_policy_initial_mean",
        "aggregate_policy_trained_mean",
        "aggregate_policy_improvement",
        "aggregate_policy_online_phase_improvement",
        "aggregate_policy_final_champion_return",
        "aggregate_final_policy_eval_mean",
        "aggregate_final_policy_eval_std",
        "aggregate_model_update_acceptance_rate",
        "aggregate_policy_update_acceptance_rate",
        "aggregate_real_train_replay_env_steps",
        "aggregate_final_open_loop_loss",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def extract_run_dir_metrics(run_dir: Path) -> dict[str, Any]:
    best_selection = None
    for path in run_dir.glob("*policy_policy_selection_history.json"):
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else payload.get("history", [])
        for row in rows:
            value = row.get("policy_selection_mean_return")
            if value is not None:
                best_selection = (
                    float(value)
                    if best_selection is None
                    else max(best_selection, float(value))
                )
    latest_confirmation = latest_metric(
        run_dir.glob("*policy_confirmation_trained_policy_evaluation.json")
    )
    latest_trained = latest_metric(run_dir.glob("*policy_trained_policy_evaluation.json"))
    return {
        "best_selection_return": best_selection,
        "latest_confirmation_return": latest_confirmation[0],
        "latest_confirmation_std": latest_confirmation[1],
        "latest_trained_return": latest_trained[0],
        "latest_trained_std": latest_trained[1],
    }


def latest_metric(paths: Iterable[Path]) -> tuple[float | None, float | None]:
    sorted_paths = sorted(paths, key=lambda path: path.stat().st_mtime)
    if not sorted_paths:
        return None, None
    payload = read_json(sorted_paths[-1])
    return metric_mean_return(payload), metric_std_return(payload)


def score_metrics(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    champion = first_number(
        metrics.get("aggregate_policy_final_champion_return"),
        metrics.get("aggregate_policy_trained_mean"),
        metrics.get("latest_trained_return"),
    )
    final_eval = first_number(
        metrics.get("aggregate_final_policy_eval_mean"),
        metrics.get("latest_confirmation_return"),
        champion,
    )
    if final_eval is None:
        metrics["score_base_return"] = None
        metrics["score_best_selection_return"] = None
        metrics["score_eval_std"] = None
        return -1_000_000.0
    final_std = first_number(
        metrics.get("aggregate_final_policy_eval_std"),
        metrics.get("latest_confirmation_std"),
        0.0,
    )
    best_selection = first_number(metrics.get("best_selection_return"), final_eval)
    score = float(final_eval)
    score -= args.selection_gap_penalty * max(0.0, float(best_selection) - score)
    score -= args.eval_std_penalty * float(final_std)
    metrics["score_base_return"] = final_eval
    metrics["score_best_selection_return"] = best_selection
    metrics["score_eval_std"] = final_std
    return score


def log_trial_to_wandb(
    args: argparse.Namespace,
    trial,
    gpu: str,
    params: dict[str, Any],
    trial_dir: Path,
    metrics: dict[str, Any],
    score: float,
    command: list[str],
) -> None:
    if args.wandb_mode == "disabled":
        return
    group = args.wandb_group or args.study_name
    with WANDB_LOCK:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "W&B is not installed. Run with: uv run --extra hpo "
                "world-marl-optuna-dmc-jepa ..."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=group,
            name=f"{args.study_name}-trial-{trial.number:04d}",
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            config={
                "trial_number": trial.number,
                "gpu": gpu,
                "task": args.task,
                "seed": args.seed + trial.number,
                "command": " ".join(command),
                **params,
            },
            reinit=True,
        )
        try:
            wandb_run.log(prefix_keys(wandb_scalars(metrics), "final/"))
            wandb_run.summary["score"] = score
            for key, value in metrics.items():
                if is_json_scalar(value):
                    wandb_run.summary[key] = value
            log_wandb_timeseries(wandb_run, trial_dir)
            if args.wandb_artifact:
                log_wandb_artifact(wandb_run, trial_dir, trial.number)
        finally:
            wandb_run.finish()


def log_wandb_timeseries(wandb_run, trial_dir: Path) -> None:
    metric_paths = sorted((trial_dir / "run").glob("*/none/run_000/metrics.jsonl"))
    step = 0
    for path in metric_paths:
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            flat = flatten_dict(row)
            wandb_run.log(prefix_keys(wandb_scalars(flat), "metrics/"), step=step)
            step += 1
    for path in sorted((trial_dir / "run").glob("*/none/run_000/*.json")):
        payload = read_json(path)
        flat = wandb_scalars(flatten_dict(payload))
        if flat:
            wandb_run.log(prefix_keys(flat, f"artifacts/{path.stem}/"))


def log_wandb_artifact(wandb_run, trial_dir: Path, trial_number: int) -> None:
    import wandb

    artifact = wandb.Artifact(
        f"trial-{trial_number:04d}-artifacts",
        type="jepa-hpo-trial",
    )
    artifact.add_dir(str(trial_dir))
    wandb_run.log_artifact(artifact)


def build_key(prefix: str, key: str) -> str:
    return prefix + key.replace("/", "_")


def prefix_keys(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {build_key(prefix, key): value for key, value in payload.items()}


def flatten_dict(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            child_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
            result.update(flatten_dict(value, child_prefix))
        return result
    if isinstance(payload, list):
        return {}
    key = prefix[:-1] if prefix.endswith("/") else prefix
    return {key: payload}


def metric_mean_return(payload: dict[str, Any]) -> float | None:
    return first_number(
        payload.get("mean_return"),
        payload.get("episode_return_mean"),
        payload.get("policy_mean"),
    )


def metric_std_return(payload: dict[str, Any]) -> float | None:
    return first_number(
        payload.get("std_return"),
        payload.get("episode_return_std"),
        payload.get("policy_std"),
    )


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def iteration_from_name(name: str) -> int | None:
    parts = name.split("_")
    for index, part in enumerate(parts[:-1]):
        if part == "online":
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def wandb_scalars(payload: dict[str, Any]) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            result[key] = value
        elif isinstance(value, int):
            result[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[key] = value
    return result


def terminate_process(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def import_optuna():
    import optuna

    return optuna


def write_trials_csv(args: argparse.Namespace, study) -> None:
    try:
        frame = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    except Exception:
        return
    frame.to_csv(args.out_root / "trials.csv", index=False)


def best_trial_or_none(study):
    try:
        return study.best_trial
    except ValueError:
        return None


if __name__ == "__main__":
    main()
