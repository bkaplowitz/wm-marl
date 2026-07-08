from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.imagination import open_loop_diagnostic
from world_marl.dreamer_v3_baseline.training import train_dreamer_world_model
from world_marl.dreamer_v3_baseline.validation import (
    finite_metric_check,
    loss_decreased,
)
from world_marl.world_model_foundation.collect import (
    synthetic_sequence_collector,
    write_json_artifact,
    write_jsonl_metrics,
)
from world_marl.world_model_foundation.sources import world_model_sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="synthetic:image-grid")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/dreamer_v3_baseline")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--policy-train-steps", type=int, default=10)
    parser.add_argument("--time-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args(argv)


def _config_payload(
    args: argparse.Namespace, config: DreamerV3Config
) -> dict[str, Any]:
    return {
        "env": args.env,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "policy_train_steps": args.policy_train_steps,
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "action_dim": args.action_dim,
        "observation_shape": config.observation_shape,
        "rssm": {
            "deterministic_size": config.rssm.deterministic_size,
            "stochastic_size": config.rssm.stochastic_size,
            "discrete_classes": config.rssm.discrete_classes,
        },
    }


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, np.clip(image, 0.0, 1.0))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env != "synthetic:image-grid":
        raise SystemExit("v1 smoke supports --env synthetic:image-grid")

    out_dir = args.out_dir
    config = DreamerV3Config(
        action_dim=args.action_dim,
        observation_shape=(args.image_size, args.image_size, 3),
    )
    batch = synthetic_sequence_collector(
        env_name=args.env,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        observation_shape=config.observation_shape,
        action_dim=config.action_dim,
    )
    state, world_metrics = train_dreamer_world_model(
        batch=batch,
        config=config,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    finite_metric_check(world_metrics[-1])
    outputs = state.apply_fn(
        state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        jnp.asarray(batch.actions, dtype=jnp.int32),
    )
    diagnostic = open_loop_diagnostic(outputs["features"], args.policy_train_steps)
    actor_critic_metrics = [
        {
            "step": step,
            "imagined_reward": float(jnp.mean(diagnostic.rewards)),
            "imagined_value": float(jnp.mean(diagnostic.values)),
        }
        for step in range(args.policy_train_steps)
    ]
    gate_passed = loss_decreased(world_metrics)
    status = "ok" if gate_passed else "learning_gate_failed"

    write_json_artifact(out_dir / "config.json", _config_payload(args, config))
    write_json_artifact(out_dir / "sources.json", world_model_sources())
    write_jsonl_metrics(out_dir / "world_model_metrics.jsonl", world_metrics)
    write_jsonl_metrics(out_dir / "actor_critic_metrics.jsonl", actor_critic_metrics)
    first_obs = np.asarray(batch.observations[0, 0])
    first_reconstruction = np.asarray(outputs["reconstructions"][0, 0])
    _write_png(
        out_dir / "open_loop_reconstruction.png",
        np.concatenate([first_obs, first_reconstruction], axis=1),
    )
    rollout = np.asarray(
        batch.observations[: min(args.policy_train_steps, args.time_steps), 0]
    )
    _write_png(out_dir / "imagined_rollout.png", np.concatenate(list(rollout), axis=1))
    write_json_artifact(
        out_dir / "outcome.json",
        {
            "status": status,
            "final_loss": world_metrics[-1]["loss"],
            "initial_loss": world_metrics[0]["loss"],
        },
    )
    write_json_artifact(
        out_dir / "summary.json",
        {
            "status": status,
            "model": "dreamer_v3_baseline",
            "final_loss": world_metrics[-1]["loss"],
            "learning_gate_passed": gate_passed,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
