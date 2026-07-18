"""Write reproducible launchers for the maintained DMC JEPA algorithm."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any

from world_marl.jepa.config import canonical_jepa_config, smoke_jepa_config


DEFAULT_TASKS = (
    "reacher/easy",
    "cartpole/swingup",
    "finger/spin",
    "cheetah/run",
    "walker/walk",
)

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": smoke_jepa_config(),
    "jepa_100k": canonical_jepa_config(budget_env_steps=100_000),
    "jepa_200k": canonical_jepa_config(budget_env_steps=200_000),
    "jepa_500k": canonical_jepa_config(budget_env_steps=500_000),
}

OVERRIDABLE_PARAMS = (
    "num_envs",
    "env_workers",
    "collect_steps",
    "initial_reset_interval",
    "initial_random_action_hold_steps",
    "validation_steps",
    "validation_seed",
    "online_iterations",
    "online_collect_steps",
    "train_steps",
    "online_train_steps",
    "policy_train_steps",
    "online_policy_train_steps",
    "online_policy_actor_update_interval",
    "online_policy_actor_update_interval_start_env_steps",
    "online_freeze_encoder_after_env_steps",
    "online_checkpoint_interval",
    "online_recent_replay_steps",
    "latent_dim",
    "model_dim",
    "num_layers",
    "num_heads",
    "actor_hidden_dim",
    "critic_hidden_dim",
    "imag_horizon",
    "learning_rate",
    "actor_learning_rate",
    "actor_entropy_coef",
    "value_clip",
    "value_clip_final",
    "value_clip_schedule_start_env_steps",
    "value_clip_schedule_end_env_steps",
    "policy_actor_kl_coef",
    "policy_actor_kl_target_per_dim",
    "policy_actor_kl_reference_interval",
    "policy_reset_start_fraction",
    "policy_reset_start_fraction_start_env_steps",
    "policy_reset_start_max_age",
    "online_recent_world_model_fraction",
    "online_recent_world_model_until_env_steps",
    "online_recent_replay_max_oversample",
    "dreamer_report_budget_env_steps",
    "curve_eval_interval_env_steps",
    "curve_eval_episodes",
    "curve_eval_num_envs",
    "curve_eval_seed",
    "final_policy_eval_episodes",
    "final_policy_eval_seed",
    "training_snapshot_env_steps",
    "resume_training_snapshot",
    "wandb_project",
    "wandb_entity",
    "wandb_name",
    "wandb_group",
    "wandb_tags",
    "wandb_mode",
    "wandb_videos",
)


def main() -> None:
    args = parse_args()
    params = dict(PRESETS[args.preset])
    apply_optional_overrides(args, params)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "task": task,
            "seed": int(seed),
            "short": f"{task_short_name(task)}_seed{seed}",
        }
        for task in args.tasks
        for seed in args.seeds
    ]
    manifest = {
        "algorithm": "jepa",
        "preset": args.preset,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "out_root": str(out_root),
        "params": params,
        "step_accounting": step_accounting(params),
        "jobs": jobs,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_run_one(out_root, params)
    write_launcher(
        out_root,
        jobs,
        args.gpus,
        sync=args.sync,
        tracking=bool(params.get("wandb_project")),
        repo_root=_source_checkout_root(),
    )
    write_tail(out_root)
    write_summarize(out_root)

    print(f"Wrote JEPA launcher to {out_root}")
    print(
        f"Start: nohup bash {out_root / 'launcher.sh'} > "
        f"{out_root / 'launcher.nohup.log'} 2>&1 &"
    )
    print(f"Watch: bash {out_root / 'tail.sh'}")

    if args.start:
        log = (out_root / "launcher.nohup.log").open("wb")
        process = subprocess.Popen(
            ["bash", str(out_root / "launcher.sh")],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"launcher pid: {process.pid}")


def step_accounting(params: dict[str, Any]) -> dict[str, int | float | None]:
    num_envs = _int_param(params, "num_envs")
    collect_steps = _int_param(params, "collect_steps")
    validation_steps = _int_param(params, "validation_steps")
    online_iterations = _int_param(params, "online_iterations")
    online_collect_steps = _int_param(params, "online_collect_steps")
    train_steps = _int_param(params, "train_steps")
    online_train_steps = _int_param(params, "online_train_steps")
    policy_train_steps = _int_param(params, "policy_train_steps")
    online_policy_train_steps = _int_param(params, "online_policy_train_steps")
    online_actor_update_interval = _int_param(
        params,
        "online_policy_actor_update_interval",
    )
    online_actor_interval_start_env_steps = _int_param(
        params,
        "online_policy_actor_update_interval_start_env_steps",
    )

    train_vector_steps = _sum_optional(
        collect_steps,
        _product_optional(online_iterations, online_collect_steps),
    )
    train_env_steps = _product_optional(num_envs, train_vector_steps)
    validation_env_steps = _product_optional(num_envs, validation_steps)
    world_model_updates = _phase_total(
        train_steps,
        online_iterations,
        online_train_steps,
    )
    policy_updates = _phase_total(
        policy_train_steps,
        online_iterations,
        online_policy_train_steps,
    )
    actor_updates = _scheduled_actor_update_total(
        initial_updates=policy_train_steps,
        online_iterations=online_iterations,
        online_updates=online_policy_train_steps,
        online_interval=online_actor_update_interval,
        interval_start_env_steps=online_actor_interval_start_env_steps,
        num_envs=num_envs,
        collect_steps=collect_steps,
        online_collect_steps=online_collect_steps,
    )
    sampled_transitions = _product_optional(
        world_model_updates,
        _int_param(params, "batch_size"),
        _int_param(params, "chunk_length"),
    )
    replay_ratio = (
        None
        if sampled_transitions is None or train_env_steps in (None, 0)
        else sampled_transitions / train_env_steps
    )
    return {
        "num_envs": num_envs,
        "train_replay_vector_steps": train_vector_steps,
        "train_replay_env_steps": train_env_steps,
        "validation_replay_vector_steps": validation_steps,
        "validation_replay_env_steps": validation_env_steps,
        "train_plus_validation_vector_steps": _sum_optional(
            train_vector_steps,
            validation_steps,
        ),
        "train_plus_validation_env_steps": _sum_optional(
            train_env_steps,
            validation_env_steps,
        ),
        "world_model_updates": world_model_updates,
        "policy_updates": policy_updates,
        "critic_updates": policy_updates,
        "actor_updates": actor_updates,
        "world_model_sampled_transitions": sampled_transitions,
        "world_model_replay_ratio": replay_ratio,
        "final_policy_eval_episodes": _int_param(
            params,
            "final_policy_eval_episodes",
        ),
    }


def _int_param(params: dict[str, Any], key: str) -> int | None:
    value = params.get(key)
    return None if value is None else int(value)


def _sum_optional(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _product_optional(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    result = 1
    for value in values:
        assert value is not None
        result *= value
    return result


def _phase_total(
    initial: int | None,
    iterations: int | None,
    online: int | None,
) -> int | None:
    if initial is None or iterations is None or online is None:
        return None
    return initial + iterations * online


def _scheduled_actor_update_total(
    *,
    initial_updates: int | None,
    online_iterations: int | None,
    online_updates: int | None,
    online_interval: int | None,
    interval_start_env_steps: int | None,
    num_envs: int | None,
    collect_steps: int | None,
    online_collect_steps: int | None,
) -> int | None:
    values = (
        initial_updates,
        online_iterations,
        online_updates,
        online_interval,
        interval_start_env_steps,
        num_envs,
        collect_steps,
        online_collect_steps,
    )
    if any(value is None for value in values):
        return None
    assert initial_updates is not None
    assert online_iterations is not None
    assert online_updates is not None
    assert online_interval is not None
    assert interval_start_env_steps is not None
    assert num_envs is not None
    assert collect_steps is not None
    assert online_collect_steps is not None

    actor_updates = initial_updates
    train_env_steps = num_envs * collect_steps
    for _ in range(online_iterations):
        interval = 1 if train_env_steps < interval_start_env_steps else online_interval
        actor_updates += online_updates // interval
        train_env_steps += num_envs * online_collect_steps
    return actor_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("runs/jepa"))
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="jepa_500k",
        help="Use smoke to verify a machine, then run the fixed 100k or 500k preset.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    for name in (
        "num_envs",
        "env_workers",
        "collect_steps",
        "initial_reset_interval",
        "initial_random_action_hold_steps",
        "validation_steps",
        "validation_seed",
        "online_iterations",
        "online_collect_steps",
        "train_steps",
        "online_train_steps",
        "policy_train_steps",
        "online_policy_train_steps",
        "online_policy_actor_update_interval",
        "online_policy_actor_update_interval_start_env_steps",
        "online_freeze_encoder_after_env_steps",
        "online_checkpoint_interval",
        "policy_actor_kl_reference_interval",
        "policy_reset_start_fraction_start_env_steps",
        "policy_reset_start_max_age",
        "online_recent_replay_steps",
        "online_recent_world_model_until_env_steps",
        "latent_dim",
        "model_dim",
        "num_layers",
        "num_heads",
        "actor_hidden_dim",
        "critic_hidden_dim",
        "imag_horizon",
        "final_policy_eval_episodes",
        "final_policy_eval_seed",
        "value_clip_schedule_start_env_steps",
        "value_clip_schedule_end_env_steps",
        "curve_eval_interval_env_steps",
        "curve_eval_episodes",
        "curve_eval_num_envs",
        "curve_eval_seed",
        "dreamer_report_budget_env_steps",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name in (
        "learning_rate",
        "actor_learning_rate",
        "actor_entropy_coef",
        "value_clip",
        "value_clip_final",
        "policy_actor_kl_coef",
        "policy_actor_kl_target_per_dim",
        "policy_reset_start_fraction",
        "online_recent_world_model_fraction",
        "online_recent_replay_max_oversample",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    parser.add_argument(
        "--training-snapshot-env-steps",
        nargs="*",
        type=int,
        default=None,
    )
    parser.add_argument("--resume-training-snapshot", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument(
        "--wandb-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--no-sync", dest="sync", action="store_false", default=True)
    parser.add_argument("--start", action="store_true")
    return parser.parse_args()


def apply_optional_overrides(args: argparse.Namespace, params: dict[str, Any]) -> None:
    for name in OVERRIDABLE_PARAMS:
        value = getattr(args, name)
        if value is not None:
            params[name] = value


def write_run_one(out_root: Path, params: dict[str, Any]) -> None:
    command_args = params_to_shell_args(params)
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        UV_BIN="${{UV_BIN:-$(command -v uv || true)}}"
        [[ -n "$UV_BIN" ]] || UV_BIN="/root/.local/bin/uv"
        TASK="${{TASK:?TASK is required, for example reacher/easy}}"
        SEED="${{SEED:?SEED is required}}"
        SHORT="${{SHORT:-$(echo "$TASK" | tr '/-' '__')_seed${{SEED}}}}"
        OUTROOT={shlex.quote(str(out_root))}
        OUT="$OUTROOT/$SHORT"
        mkdir -p "$OUT"

        "$UV_BIN" run world-marl-train-dmc-jepa \
          --env "dmc:$TASK" \
          --seed "$SEED" \
          {command_args} \
          --out-dir "$OUT"
        """
    )
    write_executable(out_root / "run_one.sh", body)


def params_to_shell_args(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        if value is None or value is False or value == ():
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        elif isinstance(value, (list, tuple)):
            parts.append(flag)
            parts.extend(shlex.quote(str(item)) for item in value)
        else:
            parts.extend((flag, shlex.quote(str(value))))
    separator = " " + "\\" + "\n" + "          "
    return separator.join(parts)


def write_launcher(
    out_root: Path,
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    sync: bool,
    tracking: bool,
    repo_root: Path | None = None,
) -> None:
    jobs_block = "\n".join(
        f"  {shlex.quote(job['task'] + '|' + str(job['seed']) + '|' + job['short'])}"
        for job in jobs
    )
    gpus_block = " ".join(shlex.quote(gpu) for gpu in gpus)
    tracking_extra = " --extra tracking" if tracking else ""
    sync_block = (
        f'"$UV_BIN" sync --extra dmc --extra cuda12{tracking_extra}\n'
        if sync
        else 'echo "Skipping uv sync"\n'
    )
    project_root = (repo_root or Path.cwd()).resolve()
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        DEFAULT_REPO_ROOT={shlex.quote(str(project_root))}
        REPO_ROOT="${{REPO_ROOT:-$DEFAULT_REPO_ROOT}}"
        cd "$REPO_ROOT"
        export UV_PROJECT_ENVIRONMENT="${{UV_PROJECT_ENVIRONMENT:-/tmp/wm-marl-venv}}"
        export UV_CACHE_DIR="${{UV_CACHE_DIR:-/tmp/uv-cache-wm-marl}}"
        export UV_LINK_MODE="${{UV_LINK_MODE:-copy}}"
        export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
        export JAX_PLATFORMS="${{JAX_PLATFORMS:-cuda}}"
        UV_BIN="${{UV_BIN:-$(command -v uv || true)}}"
        [[ -n "$UV_BIN" ]] || UV_BIN="/root/.local/bin/uv"
        {sync_block}
        OUTROOT={shlex.quote(str(out_root))}
        GPUS=({gpus_block})
        JOBS=(
        {jobs_block}
        )

        index=0
        while (( index < ${{#JOBS[@]}} )); do
          pids=()
          for gpu in "${{GPUS[@]}}"; do
            (( index < ${{#JOBS[@]}} )) || break
            IFS='|' read -r task seed short <<< "${{JOBS[$index]}}"
            log="$OUTROOT/$short.nohup.log"
            CUDA_VISIBLE_DEVICES="$gpu" TASK="$task" SEED="$seed" SHORT="$short" \
              UV_BIN="$UV_BIN" bash "$OUTROOT/run_one.sh" > "$log" 2>&1 &
            pids+=("$!")
            index=$((index + 1))
          done
          for pid in "${{pids[@]}}"; do wait "$pid"; done
        done
        """
    )
    write_executable(out_root / "launcher.sh", body)


def _source_checkout_root() -> Path:
    """Return the source checkout when this module is run by absolute path."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def write_tail(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        OUTROOT={shlex.quote(str(out_root))}
        pgrep -af "world-marl-train-dmc-jepa" || true
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
          --format=csv 2>/dev/null || true
        find "$OUTROOT" -maxdepth 1 -name "*.nohup.log" -print0 2>/dev/null \
          | xargs -0 -r -n1 sh -c 'echo "==== $0 ===="; tail -n 12 "$0"'
        """
    )
    write_executable(out_root / "tail.sh", body)


def write_summarize(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        python - <<'PY'
        import json
        import pathlib

        root = pathlib.Path({str(out_root)!r})
        paths = sorted(root.glob("*/*/summary.json"))
        print("job,final_mean,final_std,p10,cvar10,failure,success,train_steps,total_steps")
        for path in paths:
            row = json.loads(path.read_text())
            values = [
                path.parts[-3],
                row.get("aggregate_final_policy_eval_mean"),
                row.get("aggregate_final_policy_eval_std"),
                row.get("aggregate_final_policy_eval_return_p10"),
                row.get("aggregate_final_policy_eval_return_cvar10"),
                row.get("aggregate_final_policy_eval_failure_rate"),
                row.get("aggregate_final_policy_eval_success_rate"),
                row.get("aggregate_real_train_replay_env_steps"),
                row.get("aggregate_real_total_env_steps"),
            ]
            print(",".join("" if value is None else str(value) for value in values))
        PY
        """
    )
    write_executable(out_root / "summarize.sh", body)


def write_executable(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def task_short_name(task: str) -> str:
    return task.replace("/", "_").replace("-", "_")


if __name__ == "__main__":
    main()
