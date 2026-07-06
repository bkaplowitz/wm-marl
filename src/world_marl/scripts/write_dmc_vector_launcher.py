"""Write launcher scripts for the DMC vector JEPA benchmark track."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any


DEFAULT_TASKS = (
    "reacher/easy",
    "cartpole/swingup",
    "finger/spin",
    "cheetah/run",
    "walker/walk",
)

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "num_envs": 8,
        "env_workers": 8,
        "collect_steps": 1024,
        "validation_steps": 256,
        "train_steps": 1500,
        "policy_train_steps": 750,
        "online_iterations": 1,
        "online_collect_steps": 512,
        "online_validation_steps": 256,
        "online_train_steps": 750,
        "online_policy_train_steps": 500,
        "policy_selection_interval": 250,
        "policy_selection_episodes": 8,
        "policy_eval_episodes": 16,
        "policy_confirmation_episodes": 16,
    },
    "mainline": {
        "num_envs": 16,
        "env_workers": 16,
        "collect_steps": 8192,
        "validation_steps": 2048,
        "train_steps": 12000,
        "policy_train_steps": 3000,
        "online_iterations": 6,
        "online_collect_steps": 4096,
        "online_validation_steps": 1024,
        "online_train_steps": 3000,
        "online_policy_train_steps": 750,
        "policy_selection_interval": 250,
        "policy_selection_episodes": 32,
        "policy_eval_episodes": 64,
        "policy_confirmation_episodes": 64,
    },
    "cadence": {
        "num_envs": 16,
        "env_workers": 16,
        "collect_steps": 8192,
        "validation_steps": 2048,
        "train_steps": 12000,
        "policy_train_steps": 3000,
        "online_iterations": 12,
        "online_collect_steps": 2048,
        "online_validation_steps": 1024,
        "online_train_steps": 3000,
        "online_policy_train_steps": 750,
        "policy_selection_interval": 250,
        "policy_selection_episodes": 32,
        "policy_eval_episodes": 64,
        "policy_confirmation_episodes": 64,
        "final_policy_eval_episodes": 256,
    },
}
PRESETS["stabilized"] = {
    **PRESETS["cadence"],
    "policy_return_mode": "lambda",
    "policy_actor_baseline": "value",
    "policy_return_normalization": "batch",
    "reward_prediction_mode": "mse",
    "value_prediction_mode": "symlog-twohot",
    "clip_imagined_rewards": True,
}

COMMON_PARAMS: dict[str, Any] = {
    "num_runs": 1,
    "critic_warmup_steps": 1000,
    "critic_horizon": 32,
    "policy_batch_size": 512,
    "policy_objective": "direct",
    "policy_return_mode": "reward-only",
    "policy_actor_baseline": "none",
    "policy_return_normalization": "none",
    "imag_horizon": 15,
    "final_policy_eval_episodes": 0,
    "model_grad_clip_norm": 100.0,
    "actor_grad_clip_norm": 10.0,
    "critic_grad_clip_norm": 100.0,
    "stochastic_actor": False,
    "stochastic_collection": False,
    "actor_entropy_coef": 0.0,
    "actor_log_std_min": -5.0,
    "actor_log_std_max": 2.0,
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


def main() -> None:
    args = parse_args()
    params = {**COMMON_PARAMS, **PRESETS[args.preset]}
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
        "preset": args.preset,
        "tasks": args.tasks,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "out_root": str(out_root),
        "params": params,
        "jobs": jobs,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_run_one(out_root, params)
    write_launcher(out_root, jobs, args.gpus, sync=args.sync)
    write_tail(out_root)
    write_summarize(out_root)

    print(f"Wrote DMC vector launcher to {out_root}")
    print(f"- {out_root / 'manifest.json'}")
    print(f"- {out_root / 'run_one.sh'}")
    print(f"- {out_root / 'launcher.sh'}")
    print(f"- {out_root / 'tail.sh'}")
    print(f"- {out_root / 'summarize.sh'}")
    print()
    print(
        f"Start with: nohup bash {shlex.quote(str(out_root / 'launcher.sh'))} "
        f"> {shlex.quote(str(out_root / 'launcher.nohup.log'))} 2>&1 &"
    )
    print(f"Watch with: bash {shlex.quote(str(out_root / 'tail.sh'))}")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/dmc_jepa_vector_mainline"),
        help="Directory where launcher scripts, logs, and task runs are written.",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="mainline",
        help="Run size preset. Use smoke first to verify the pod.",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="CUDA device ids. Jobs are launched in batches, one per listed GPU.",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--env-workers", type=int, default=None)
    parser.add_argument("--collect-steps", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--online-iterations", type=int, default=None)
    parser.add_argument("--online-collect-steps", type=int, default=None)
    parser.add_argument("--online-validation-steps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--online-train-steps", type=int, default=None)
    parser.add_argument("--policy-train-steps", type=int, default=None)
    parser.add_argument("--online-policy-train-steps", type=int, default=None)
    parser.add_argument(
        "--policy-return-mode",
        choices=("reward-only", "lambda"),
        default=None,
    )
    parser.add_argument(
        "--policy-actor-baseline",
        choices=("none", "value"),
        default=None,
    )
    parser.add_argument(
        "--policy-return-normalization",
        choices=("none", "batch"),
        default=None,
    )
    parser.add_argument("--policy-selection-episodes", type=int, default=None)
    parser.add_argument("--policy-eval-episodes", type=int, default=None)
    parser.add_argument("--policy-confirmation-episodes", type=int, default=None)
    parser.add_argument("--final-policy-eval-episodes", type=int, default=None)
    parser.add_argument(
        "--reward-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default=None,
    )
    parser.add_argument(
        "--value-prediction-mode",
        choices=("mse", "symlog-twohot"),
        default=None,
    )
    parser.add_argument("--twohot-bins", type=int, default=None)
    parser.add_argument("--twohot-min", type=float, default=None)
    parser.add_argument("--twohot-max", type=float, default=None)
    parser.add_argument(
        "--clip-imagined-rewards",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--imagined-reward-min", type=float, default=None)
    parser.add_argument("--imagined-reward-max", type=float, default=None)
    parser.add_argument("--model-grad-clip-norm", type=float, default=None)
    parser.add_argument("--actor-grad-clip-norm", type=float, default=None)
    parser.add_argument("--critic-grad-clip-norm", type=float, default=None)
    parser.add_argument(
        "--stochastic-actor",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stochastic-collection",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--actor-entropy-coef", type=float, default=None)
    parser.add_argument("--actor-log-std-min", type=float, default=None)
    parser.add_argument("--actor-log-std-max", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--policy-batch-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        default=True,
        help="Do not run uv sync at launcher start.",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start launcher immediately in the background.",
    )
    return parser.parse_args()


def apply_optional_overrides(args: argparse.Namespace, params: dict[str, Any]) -> None:
    for name in (
        "num_envs",
        "env_workers",
        "collect_steps",
        "validation_steps",
        "online_iterations",
        "online_collect_steps",
        "online_validation_steps",
        "train_steps",
        "online_train_steps",
        "policy_train_steps",
        "online_policy_train_steps",
        "policy_return_mode",
        "policy_actor_baseline",
        "policy_return_normalization",
        "policy_selection_episodes",
        "policy_eval_episodes",
        "policy_confirmation_episodes",
        "final_policy_eval_episodes",
        "reward_prediction_mode",
        "value_prediction_mode",
        "twohot_bins",
        "twohot_min",
        "twohot_max",
        "clip_imagined_rewards",
        "imagined_reward_min",
        "imagined_reward_max",
        "model_grad_clip_norm",
        "actor_grad_clip_norm",
        "critic_grad_clip_norm",
        "stochastic_actor",
        "stochastic_collection",
        "actor_entropy_coef",
        "actor_log_std_min",
        "actor_log_std_max",
        "batch_size",
        "policy_batch_size",
        "latent_dim",
        "model_dim",
        "num_layers",
        "num_heads",
    ):
        value = getattr(args, name)
        if value is not None:
            params[name] = value


def write_run_one(out_root: Path, params: dict[str, Any]) -> None:
    command_args = params_to_shell_args(params)
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        UV_BIN="${{UV_BIN:-uv}}"
        TASK="${{TASK:?TASK is required, for example reacher/easy}}"
        SEED="${{SEED:?SEED is required}}"
        SHORT="${{SHORT:-$(echo "$TASK" | tr '/-' '__')_seed${{SEED}}}}"
        OUTROOT="{out_root}"
        OUT="$OUTROOT/$SHORT"
        mkdir -p "$OUT"

        echo "==== starting dmc:$TASK seed=$SEED on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}} ===="
        echo "out=$OUT"

        "$UV_BIN" run world-marl-train-dmc-jepa \\
          --env "dmc:$TASK" \\
          --seed "$SEED" \\
          {command_args} \\
          --out-dir "$OUT"

        echo "==== finished dmc:$TASK seed=$SEED ===="
        """
    )
    write_executable(out_root / "run_one.sh", body)


def params_to_shell_args(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            parts.append(flag)
            parts.extend(shlex.quote(str(item)) for item in value)
            continue
        parts.extend((flag, shlex.quote(str(value))))
    return " \\\n          ".join(parts)


def write_launcher(
    out_root: Path,
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    sync: bool,
) -> None:
    jobs_block = "\n".join(
        f"  {shlex.quote(job['task'] + '|' + str(job['seed']) + '|' + job['short'])}"
        for job in jobs
    )
    gpus_block = " ".join(shlex.quote(gpu) for gpu in gpus)
    sync_block = (
        '"$UV_BIN" sync --extra dmc --extra cuda12\n'
        if sync
        else 'echo "Skipping uv sync because launcher was generated with --no-sync"\n'
    )
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

        export UV_PROJECT_ENVIRONMENT="${{UV_PROJECT_ENVIRONMENT:-/tmp/wm-marl-venv}}"
        export UV_CACHE_DIR="${{UV_CACHE_DIR:-/tmp/uv-cache-wm-marl}}"
        export UV_LINK_MODE="${{UV_LINK_MODE:-copy}}"
        export XLA_PYTHON_CLIENT_PREALLOCATE="${{XLA_PYTHON_CLIENT_PREALLOCATE:-false}}"
        export JAX_PLATFORMS="${{JAX_PLATFORMS:-cuda}}"

        UV_BIN="${{UV_BIN:-uv}}"
        {sync_block}
        OUTROOT="{out_root}"
        GPUS=({gpus_block})
        JOBS=(
        {jobs_block}
        )

        index=0
        total="${{#JOBS[@]}}"
        while (( index < total )); do
          pids=()
          for gpu in "${{GPUS[@]}}"; do
            if (( index >= total )); then
              break
            fi
            IFS='|' read -r task seed short <<< "${{JOBS[$index]}}"
            log="$OUTROOT/$short.nohup.log"
            echo "launching dmc:$task seed=$seed on GPU $gpu -> $log"
            CUDA_VISIBLE_DEVICES="$gpu" TASK="$task" SEED="$seed" SHORT="$short" \\
              UV_BIN="$UV_BIN" bash "$OUTROOT/run_one.sh" > "$log" 2>&1 &
            pids+=("$!")
            index=$((index + 1))
          done
          for pid in "${{pids[@]}}"; do
            wait "$pid"
          done
        done

        echo "all DMC vector jobs finished"
        """
    )
    write_executable(out_root / "launcher.sh", body)


def write_tail(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
        OUTROOT="{out_root}"

        echo "== processes =="
        pgrep -af "world-marl-train-dmc-jepa|dmc_jepa_vector" || true
        echo
        echo "== gpu =="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv 2>/dev/null || true
        echo
        echo "== summaries =="
        find "$OUTROOT" -path "*/summary.json" -print | sort || true
        echo
        echo "== latest logs =="
        find "$OUTROOT" -maxdepth 1 -name "*.nohup.log" -printf "%T@ %p\\n" 2>/dev/null \\
          | sort -n | tail -4 | cut -d' ' -f2- | while read -r log; do
              echo
              echo "==== $log ===="
              tail -n 12 "$log" | tr '\\r' '\\n' | tail -n 12
            done
        """
    )
    write_executable(out_root / "tail.sh", body)


def write_summarize(out_root: Path) -> None:
    body = dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
        python - <<'PY'
        import json
        import pathlib

        root = pathlib.Path({str(out_root)!r})
        paths = sorted(root.glob("*/*/summary.json"))
        if not paths:
            print("no summaries yet")
            raise SystemExit(0)

        print("job,passed,world,policy,initial,trained,champion,final_eval,final_eval_std,improve,online,model_accept,policy_accept,train_replay_steps,strict_total_steps,open_loop")
        for path in paths:
            job = path.parts[-3]
            summary = json.loads(path.read_text())
            values = [
                job,
                summary.get("passed"),
                summary.get("world_model_passed"),
                summary.get("policy_main_passed"),
                summary.get("aggregate_policy_initial_mean"),
                summary.get("aggregate_policy_trained_mean"),
                summary.get("aggregate_policy_final_champion_return"),
                summary.get("aggregate_final_policy_eval_mean"),
                summary.get("aggregate_final_policy_eval_std"),
                summary.get("aggregate_policy_improvement"),
                summary.get("aggregate_policy_online_phase_improvement"),
                summary.get("aggregate_model_update_acceptance_rate"),
                summary.get("aggregate_policy_update_acceptance_rate"),
                summary.get("aggregate_real_train_replay_env_steps"),
                summary.get("aggregate_real_total_env_steps"),
                summary.get("aggregate_final_open_loop_loss"),
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
