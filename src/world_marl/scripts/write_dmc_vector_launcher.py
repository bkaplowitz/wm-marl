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
}

COMMON_PARAMS: dict[str, Any] = {
    "num_runs": 1,
    "critic_warmup_steps": 1000,
    "critic_horizon": 32,
    "policy_batch_size": 512,
    "policy_objective": "direct",
    "policy_return_mode": "reward-only",
    "imag_horizon": 15,
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
    print(f"Start with: nohup bash {shlex.quote(str(out_root / 'launcher.sh'))} "
          f"> {shlex.quote(str(out_root / 'launcher.nohup.log'))} 2>&1 &")
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
    parser.add_argument("--online-iterations", type=int, default=None)
    parser.add_argument("--online-collect-steps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--online-train-steps", type=int, default=None)
    parser.add_argument("--policy-train-steps", type=int, default=None)
    parser.add_argument("--online-policy-train-steps", type=int, default=None)
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
        "online_iterations",
        "online_collect_steps",
        "train_steps",
        "online_train_steps",
        "policy_train_steps",
        "online_policy_train_steps",
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

        "$UV_BIN" run world-marl-validate-single-agent-world-model \\
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
        pgrep -af "world-marl-validate-single-agent-world-model|dmc_jepa_vector" || true
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

        print("job,passed,world,policy,initial,trained,improve,online,model_accept,policy_accept,train_replay_steps,open_loop")
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
                summary.get("aggregate_policy_improvement"),
                summary.get("aggregate_policy_online_phase_improvement"),
                summary.get("aggregate_model_update_acceptance_rate"),
                summary.get("aggregate_policy_update_acceptance_rate"),
                summary.get("aggregate_real_train_replay_env_steps"),
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
