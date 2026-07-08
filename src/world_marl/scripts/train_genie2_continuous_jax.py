from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from world_marl.genie2_continuous_jax.action_bridge import fit_linear_action_bridge
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.training import train_genie2_world_model
from world_marl.genie2_continuous_jax.validation import (
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
        "--out-dir", type=Path, default=Path("runs/genie2_continuous_jax")
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
    args: argparse.Namespace, config: Genie2ContinuousConfig
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
        "representation": config.representation,
        "latent_dim": config.autoencoder.latent_dim,
        "latent_action_dim": config.lam.latent_action_dim,
        "dynamics_objective": config.dynamics.objective,
    }


def _split_metrics(metrics: list[dict[str, float]], key: str) -> list[dict[str, float]]:
    return [{"step": row["step"], key: row[key]} for row in metrics]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.env != "synthetic:image-grid":
        raise SystemExit("v1 smoke supports --env synthetic:image-grid")

    config = Genie2ContinuousConfig()
    observation_shape = (args.image_size, args.image_size, 3)
    batch = synthetic_sequence_collector(
        env_name=args.env,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        observation_shape=observation_shape,
        action_dim=args.action_dim,
    )
    _, metrics = train_genie2_world_model(
        batch=batch,
        observation_shape=observation_shape,
        config=config,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    finite_metric_check(metrics[-1])
    latent_actions = np.eye(config.lam.latent_action_dim, dtype=np.float32)[
        : args.action_dim
    ]
    real_actions = np.arange(args.action_dim, dtype=np.float32)[:, None]
    bridge = fit_linear_action_bridge(latent_actions, real_actions)
    gate_passed = loss_decreased(metrics)
    status = "ok" if gate_passed else "learning_gate_failed"

    out_dir = args.out_dir
    write_json_artifact(out_dir / "config.json", _config_payload(args, config))
    write_json_artifact(out_dir / "sources.json", world_model_sources())
    write_jsonl_metrics(
        out_dir / "autoencoder_metrics.jsonl",
        _split_metrics(metrics, "reconstruction_loss"),
    )
    write_jsonl_metrics(
        out_dir / "lam_metrics.jsonl", _split_metrics(metrics, "lam_kl_loss")
    )
    write_jsonl_metrics(
        out_dir / "dynamics_metrics.jsonl",
        _split_metrics(metrics, "dynamics_loss"),
    )
    write_jsonl_metrics(
        out_dir / "reward_continue_metrics.jsonl",
        [
            {
                "step": row["step"],
                "reward_loss": row["reward_loss"],
                "continue_loss": row["continue_loss"],
            }
            for row in metrics
        ],
    )
    write_json_artifact(
        out_dir / "latent_action_usage.json",
        {"latent_action_dim": config.lam.latent_action_dim, "usage": "synthetic_smoke"},
    )
    write_json_artifact(
        out_dir / "latent_action_bridge.json",
        {
            "latent_action_dim": bridge.latent_action_dim,
            "real_action_dim": bridge.real_action_dim,
        },
    )
    write_json_artifact(
        out_dir / "outcome.json",
        {
            "status": status,
            "initial_loss": metrics[0]["loss"],
            "final_loss": metrics[-1]["loss"],
        },
    )
    write_json_artifact(
        out_dir / "summary.json",
        {
            "status": status,
            "model": "genie2_continuous_jax",
            "final_loss": metrics[-1]["loss"],
            "learning_gate_passed": gate_passed,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
