"""Train a Brax PPO baseline and write plotter-friendly diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from brax import envs
from brax.training.agents.ppo import train as ppo_train

from world_marl.logging import to_jsonable


def main() -> None:
    args = parse_args()
    env_name = normalize_env_name(args.env)
    out_dir = args.out_dir / env_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    history_path = out_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    started = time.time()

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        row = {
            "num_steps": int(num_steps),
            "walltime_seconds": time.time() - started,
            **to_jsonable(metrics),
        }
        if row["walltime_seconds"] > 0:
            row["training/sps_since_start"] = row["num_steps"] / row["walltime_seconds"]
        history.append(row)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    environment = make_env(env_name, args)
    num_evals = max(2, math.ceil(args.total_timesteps / args.eval_interval) + 1)
    _, _, final_metrics = ppo_train.train(
        environment=environment,
        num_timesteps=args.total_timesteps,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        num_envs=args.num_envs,
        num_eval_envs=args.eval_episodes,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        seed=args.seed,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        num_evals=num_evals,
        normalize_observations=args.normalize_observations,
        reward_scaling=args.reward_scaling,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        deterministic_eval=args.deterministic_eval,
        normalize_advantage=args.normalize_advantage,
        progress_fn=progress,
    )

    summary = {
        "env": env_name,
        "args": to_jsonable(vars(args)),
        "history": history,
        "final_metrics": to_jsonable(final_metrics),
        "best": best_history_row(history),
        "last": history[-1] if history else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote PPO summary to {out_dir / 'summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="brax:reacher")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/brax_ppo"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.95)
    parser.add_argument("--unroll-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-minibatches", type=int, default=16)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--reward-scaling", type=float, default=10.0)
    parser.add_argument("--clipping-epsilon", type=float, default=0.3)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deterministic-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def normalize_env_name(env: str) -> str:
    return env.removeprefix("brax:")


def make_env(env_name: str, args: argparse.Namespace):
    kwargs: dict[str, Any] = {
        "episode_length": args.episode_length,
        "action_repeat": args.action_repeat,
    }
    if args.backend:
        kwargs["backend"] = args.backend
    return envs.create(env_name, **kwargs)


def best_history_row(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        item
        for item in history
        if isinstance(item.get("eval/episode_reward"), (int, float))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["eval/episode_reward"])


if __name__ == "__main__":
    main()
