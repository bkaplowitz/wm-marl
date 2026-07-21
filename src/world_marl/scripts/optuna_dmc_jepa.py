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
from pathlib import Path
from typing import Any

from world_marl.jepa.config import canonical_jepa_config, smoke_jepa_config

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": smoke_jepa_config(),
    "jepa_500k": canonical_jepa_config(),
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
            "resolved_preset": PRESETS[args.preset],
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
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
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
    params = sample_params(trial, base_params=PRESETS[args.preset])
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
        raise RuntimeError(
            f"trial {trial.number} failed with return code {return_code}"
        )
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
                "preset": args.preset,
                "resolved_preset": PRESETS[args.preset],
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
        next(WANDB_CONTROLLER_STEP)
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
    for path in sorted((trial_dir / "run").glob("*/run_000/metrics.jsonl")):
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
                wandb_run.log(payload, step=next(WANDB_CONTROLLER_STEP))
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
            },
            step=next(WANDB_CONTROLLER_STEP),
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


def sample_params(trial, *, base_params: dict[str, Any]) -> dict[str, Any]:
    """Sample optimizer controls while keeping the locked H8 architecture."""

    return {
        **base_params,
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [2e-5, 4e-5, 8e-5],
        ),
        "actor_learning_rate": trial.suggest_categorical(
            "actor_learning_rate",
            [2e-5, 4e-5, 8e-5],
        ),
        "actor_entropy_coef": trial.suggest_categorical(
            "actor_entropy_coef",
            [1e-3, 3e-3, 1e-2],
        ),
        "policy_actor_kl_coef": trial.suggest_categorical(
            "policy_actor_kl_coef",
            [0.0, 0.5, 1.0],
        ),
        "online_policy_actor_update_interval": trial.suggest_categorical(
            "online_policy_actor_update_interval",
            [1, 2],
        ),
        "actor_grad_clip_norm": trial.suggest_categorical(
            "actor_grad_clip_norm",
            [3.0, 10.0, 30.0],
        ),
    }


def build_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    trial_dir: Path,
) -> list[str]:
    env_name = args.task if args.task.startswith("dmc:") else f"dmc:{args.task}"
    command = [
        sys.executable,
        "-m",
        "world_marl.scripts.train_dmc_jepa_bootstrap",
        "--env",
        env_name,
        "--seed",
        str(args.seed),
    ]
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif key in {
                "actor_layer_norm",
                "critic_layer_norm",
                "deterministic_compute",
            }:
                command.append("--no-" + key.replace("_", "-"))
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            command.append(flag)
            command.extend(str(item) for item in value)
            continue
        command.extend([flag, str(value)])
    command.extend(["--out-dir", str(trial_dir / "run")])
    return command


def extract_progress(trial_dir: Path) -> tuple[int, float, dict[str, Any]] | None:
    run_dirs = sorted(
        (trial_dir / "run").glob("*/run_000"),
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        return None
    metrics_path = run_dirs[-1] / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        report = row.get("report") if isinstance(row.get("report"), dict) else {}
        value = first_number(
            row.get("paper/online_return_mean"),
            report.get("episode_return_mean"),
            row.get("episode_return_mean"),
        )
        step = first_number(row.get("budget/train_env_steps"), line_number)
        if value is not None and step is not None:
            candidates.append((int(step), value, row))
    return candidates[-1] if candidates else None


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
        (trial_dir / "run").glob("*/run_000"),
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
    outcome_path = run_dir / "outcome.json"
    if not outcome_path.is_file():
        return {}
    outcome = read_json(outcome_path)
    final_return = first_number(outcome.get("final_policy_eval_mean"))
    final_std = first_number(outcome.get("final_policy_eval_std"))
    return {
        "best_selection_return": final_return,
        "latest_confirmation_return": final_return,
        "latest_confirmation_std": final_std,
        "latest_trained_return": first_number(
            outcome.get("dreamer_style_train_return_mean"),
            final_return,
        ),
        "latest_trained_std": first_number(
            outcome.get("dreamer_style_train_return_std"),
            final_std,
        ),
    }


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
