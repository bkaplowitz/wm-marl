"""Evaluate saved JEPA policies on fixed DeepMind Control episodes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from world_marl.checkpointing import load_metadata, load_params
from world_marl.envs.dmc_adapter import DMCVectorAdapter, dmc_env_name
from world_marl.jepa.models import JepaConfig
from world_marl.jepa.training import create_jepa_train_state, select_continuous_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--env",
        default=None,
        help="DMC env, e.g. dmc:reacher/easy. Defaults to checkpoint metadata.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env-workers", type=int, default=16)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9_000_000)
    parser.add_argument(
        "--stochastic-actions",
        action="store_true",
        help=(
            "Sample from the saved actor distribution instead of using its "
            "deterministic mean. Run deterministic and stochastic evaluations "
            "with the same --seed to pair initial states."
        ),
    )
    parser.add_argument(
        "--action-seed",
        type=int,
        default=None,
        help="Action-sampling seed. Defaults to --seed.",
    )
    parser.add_argument("--failure-return-threshold", type=float, default=100.0)
    parser.add_argument("--success-return-threshold", type=float, default=900.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.num_envs < 1:
        parser.error("--num-envs must be >= 1")
    if args.env_workers < 1:
        parser.error("--env-workers must be >= 1")
    if args.max_cycles < 1:
        parser.error("--max-cycles must be >= 1")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.action_seed is not None and args.action_seed < 0:
        parser.error("--action-seed must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    evaluations = [evaluate_checkpoint(path, args) for path in args.checkpoint]
    result = {
        "protocol": {
            "episodes": args.episodes,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "stochastic_actions": args.stochastic_actions,
            "action_seed": (
                args.seed if args.action_seed is None else args.action_seed
            ),
            "checkpoint_selection": False,
        },
        "evaluations": evaluations,
        "paired_comparisons": [
            compare_evaluations(evaluations[index - 1], evaluations[index])
            for index in range(1, len(evaluations))
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def evaluate_checkpoint(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = load_metadata(path)
    env = args.env or metadata.get("env")
    if not isinstance(env, str) or not env.startswith("dmc:"):
        raise ValueError(
            "--env is required unless checkpoint metadata contains a DMC env"
        )
    config, ignored_keys = jepa_config_from_metadata(metadata)
    state = create_jepa_train_state(jax.random.PRNGKey(0), config)
    state = state.replace(
        params=load_params(path / "checkpoint.msgpack", state.params)
    )

    adapter = DMCVectorAdapter(
        dmc_env_name(env),
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed,
        num_workers=min(args.env_workers, args.num_envs),
    )
    try:
        observations = adapter.reset()
        action_low = jnp.asarray(adapter.action_low, dtype=jnp.float32)
        action_high = jnp.asarray(adapter.action_high, dtype=jnp.float32)
        returns: list[float] = []
        lengths: list[int] = []
        step_calls = 0
        action_seed = args.seed if args.action_seed is None else args.action_seed
        action_key = jax.random.PRNGKey(action_seed)
        with tqdm(
            total=args.episodes,
            desc=f"evaluate {path.name}",
            unit="episode",
            disable=args.quiet,
        ) as progress:
            while len(returns) < args.episodes:
                before = len(returns)
                action_key, step_action_key = jax.random.split(action_key)
                actions = np.asarray(
                    select_continuous_actions(
                        state,
                        jnp.asarray(observations[:, 0], dtype=jnp.float32),
                        config,
                        action_low,
                        action_high,
                        key=step_action_key,
                        stochastic=args.stochastic_actions,
                    )
                )
                step = adapter.step(actions[:, None, :])
                step_calls += 1
                returns.extend(float(item[0]) for item in step.completed_returns)
                lengths.extend(int(item) for item in step.completed_lengths)
                observations = step.observations
                progress.update(
                    max(
                        0,
                        min(len(returns), args.episodes) - min(before, args.episodes),
                    )
                )
    finally:
        adapter.close()

    returns = returns[: args.episodes]
    lengths = lengths[: args.episodes]
    return {
        "checkpoint": str(path),
        "checkpoint_kind": metadata.get("checkpoint_kind"),
        "checkpoint_online_iteration": metadata.get("online_iteration"),
        "checkpoint_train_env_steps": metadata.get("train_replay_env_steps"),
        "env": env,
        "model_seed": metadata.get("seed"),
        "ignored_legacy_jepa_config_keys": ignored_keys,
        "episodes": len(returns),
        "num_envs": args.num_envs,
        "evaluation_seed": args.seed,
        "stochastic_actions": args.stochastic_actions,
        "action_seed": action_seed,
        "env_steps": step_calls * args.num_envs,
        "completed_episode_steps": int(sum(lengths)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
        "lengths": lengths,
        **return_tail_metrics(
            returns,
            failure_threshold=args.failure_return_threshold,
            success_threshold=args.success_return_threshold,
        ),
    }


def jepa_config_from_metadata(
    metadata: dict[str, Any],
) -> tuple[JepaConfig, list[str]]:
    payload = metadata.get("jepa_config")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata is missing jepa_config")
    field_names = {field.name for field in dataclasses.fields(JepaConfig)}
    ignored = sorted(set(payload) - field_names)
    return JepaConfig(
        **{key: value for key, value in payload.items() if key in field_names}
    ), ignored


def return_tail_metrics(
    returns: list[float],
    *,
    failure_threshold: float,
    success_threshold: float,
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float32)
    failures = values < float(failure_threshold)
    successes = values >= float(success_threshold)
    tail_count = max(1, int(math.ceil(0.10 * values.size)))
    nonfailures = values[~failures]
    return {
        "failure_return_threshold": float(failure_threshold),
        "success_return_threshold": float(success_threshold),
        "failure_count": int(np.sum(failures)),
        "failure_rate": float(np.mean(failures)),
        "success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "return_min": float(np.min(values)),
        "return_max": float(np.max(values)),
        "return_p05": float(np.percentile(values, 5)),
        "return_p10": float(np.percentile(values, 10)),
        "return_p25": float(np.percentile(values, 25)),
        "return_cvar10": float(np.mean(np.sort(values)[:tail_count])),
        "nonfailure_mean_return": (
            float(np.mean(nonfailures)) if nonfailures.size else None
        ),
    }


def compare_evaluations(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    before_returns = np.asarray(before["returns"], dtype=np.float32)
    after_returns = np.asarray(after["returns"], dtype=np.float32)
    if before_returns.shape != after_returns.shape:
        raise ValueError("paired checkpoint evaluations must have equal episode counts")
    before_failure = before_returns < float(before["failure_return_threshold"])
    after_failure = after_returns < float(after["failure_return_threshold"])
    deltas = after_returns - before_returns
    return {
        "before_checkpoint": before["checkpoint"],
        "after_checkpoint": after["checkpoint"],
        "mean_return_delta": float(np.mean(deltas)),
        "episode_return_deltas": [float(value) for value in deltas],
        "regressed_episode_indices": [
            int(index) for index in np.flatnonzero(deltas < 0.0)
        ],
        "new_failure_indices": [
            int(index) for index in np.flatnonzero(~before_failure & after_failure)
        ],
        "recovered_failure_indices": [
            int(index) for index in np.flatnonzero(before_failure & ~after_failure)
        ],
    }


if __name__ == "__main__":
    main()
