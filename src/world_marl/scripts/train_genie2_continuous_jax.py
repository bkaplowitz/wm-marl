from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np

from world_marl.genie2_continuous_jax.action_bridge import fit_linear_action_bridge
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.training import train_genie2_world_model
from world_marl.genie2_continuous_jax.validation import (
    finite_metric_check,
    loss_decreased,
)
from world_marl.world_model_foundation.collect import (
    collect_world_model_sequence,
    make_single_agent_adapter,
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
    parser.add_argument("--collect-steps", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--brax-backend", default=None)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--allow-fail", action="store_true")
    return parser.parse_args(argv)


def _config_payload(
    args: argparse.Namespace,
    config: Genie2ContinuousConfig,
    *,
    observation_shape: tuple[int, ...],
    action_mode: str,
    action_dim: int,
) -> dict[str, Any]:
    return {
        "env": args.env,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "policy_train_steps": args.policy_train_steps,
        "collect_steps": _collect_steps(args),
        "batch_size": args.batch_size,
        "num_envs": args.num_envs,
        "max_cycles": args.max_cycles,
        "image_size": args.image_size,
        "action_dim": action_dim,
        "action_mode": action_mode,
        "observation_shape": observation_shape,
        "representation": config.representation,
        "latent_dim": config.autoencoder.latent_dim,
        "latent_action_dim": config.lam.latent_action_dim,
        "dynamics_objective": config.dynamics.objective,
    }


def _split_metrics(metrics: list[dict[str, float]], key: str) -> list[dict[str, float]]:
    return [{"step": row["step"], key: row[key]} for row in metrics]


def _collect_steps(args: argparse.Namespace) -> int:
    return args.time_steps if args.collect_steps is None else args.collect_steps


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, np.clip(image, 0.0, 1.0))


def _to_rgb_panel(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :, None]
    elif values.ndim == 2:
        values = values[..., None]
    elif values.ndim == 3 and values.shape[-1] not in {1, 3, 4}:
        values = values.reshape((1, -1, 1))
    elif values.ndim != 3:
        values = values.reshape((1, -1, 1))

    values = values[..., :3]
    if values.shape[-1] == 1:
        values = np.repeat(values, 3, axis=-1)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value < 0.0 or max_value > 1.0:
        scale = max(max_value - min_value, 1e-6)
        values = (values - min_value) / scale
    return np.clip(values, 0.0, 1.0)


def _make_batch(args: argparse.Namespace):
    collect_steps = _collect_steps(args)
    if args.env.startswith("synthetic:"):
        return collect_world_model_sequence(
            env_name=args.env,
            time_steps=collect_steps,
            batch_size=args.batch_size,
            observation_shape=(args.image_size, args.image_size, 3),
            action_dim=args.action_dim,
            seed=args.seed,
        )
    return collect_world_model_sequence(
        env_name=args.env,
        time_steps=collect_steps,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
    )


def _infer_latent_actions(state: Any, batch: Any) -> np.ndarray:
    outputs = state.apply_fn(
        state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        jnp.asarray(batch.rewards, dtype=jnp.float32),
        jnp.asarray(batch.continues, dtype=jnp.float32),
    )
    return np.asarray(outputs["latent_actions"], dtype=np.float32)


def _bridge_real_actions(
    batch_actions: np.ndarray,
    *,
    action_mode: str,
    action_dim: int,
) -> np.ndarray:
    actions = np.asarray(batch_actions[:-1])
    if action_mode == "discrete":
        return actions.reshape((-1, 1)).astype(np.float32)
    return actions.reshape((-1, action_dim)).astype(np.float32)


def _bridged_policy_action(
    bridge,
    latent_actions: np.ndarray,
    *,
    action_mode: str,
    action_dim: int,
    num_envs: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> np.ndarray:
    latent_action = np.mean(latent_actions, axis=0, keepdims=True)
    real_action = bridge.predict(latent_action)[0]
    if action_mode == "discrete":
        action = int(np.clip(np.rint(real_action[0]), 0, action_dim - 1))
        return np.full((num_envs, 1), action, dtype=np.int32)

    if action_low is not None and action_high is not None:
        real_action = np.clip(real_action, action_low, action_high)
    return np.broadcast_to(real_action, (num_envs, action_dim)).astype(np.float32)


def _evaluate_bridged_real_env(
    args: argparse.Namespace,
    bridge,
    latent_actions: np.ndarray,
    *,
    action_mode: str,
    action_dim: int,
    batch: Any,
) -> list[dict[str, float]]:
    if args.env.startswith("synthetic:"):
        return [
            {
                "episode": 0,
                "return": float(np.mean(np.sum(batch.rewards, axis=0))),
                "length": float(batch.time_steps),
            }
        ]

    adapter = make_single_agent_adapter(
        args.env,
        num_envs=args.num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed + 20_000,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
    )
    try:
        rows: list[dict[str, float]] = []
        action_low = getattr(adapter, "action_low", None)
        action_high = getattr(adapter, "action_high", None)
        if action_low is not None:
            action_low = np.asarray(action_low, dtype=np.float32).reshape((action_dim,))
        if action_high is not None:
            action_high = np.asarray(action_high, dtype=np.float32).reshape(
                (action_dim,)
            )
        policy_action = _bridged_policy_action(
            bridge,
            latent_actions,
            action_mode=action_mode,
            action_dim=action_dim,
            num_envs=adapter.num_envs,
            action_low=action_low,
            action_high=action_high,
        )
        for episode in range(max(args.eval_episodes, 1)):
            adapter.reset()
            episode_return = np.zeros((adapter.num_envs,), dtype=np.float32)
            for _ in range(args.max_cycles):
                step = adapter.step(policy_action)
                episode_return += np.asarray(step.rewards, dtype=np.float32).reshape(
                    (adapter.num_envs,)
                )
            rows.append(
                {
                    "episode": episode,
                    "return": float(np.mean(episode_return)),
                    "length": float(args.max_cycles),
                }
            )
        return rows
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = Genie2ContinuousConfig()
    batch = _make_batch(args)
    observation_shape = batch.observation_shape
    action_mode = str(batch.metadata.get("action_mode", "discrete"))
    action_dim = int(batch.metadata.get("action_dim", args.action_dim))
    state, metrics = train_genie2_world_model(
        batch=batch,
        observation_shape=observation_shape,
        config=config,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    finite_metric_check(metrics[-1])
    latent_actions = _infer_latent_actions(state, batch)
    real_actions = _bridge_real_actions(
        batch.actions,
        action_mode=action_mode,
        action_dim=action_dim,
    )
    bridge = fit_linear_action_bridge(latent_actions, real_actions)
    real_env_metrics = _evaluate_bridged_real_env(
        args,
        bridge,
        latent_actions,
        action_mode=action_mode,
        action_dim=action_dim,
        batch=batch,
    )
    real_env_return = float(np.mean([row["return"] for row in real_env_metrics]))
    gate_passed = loss_decreased(metrics)
    status = "ok" if gate_passed else "learning_gate_failed"

    out_dir = args.out_dir
    write_json_artifact(
        out_dir / "config.json",
        _config_payload(
            args,
            config,
            observation_shape=observation_shape,
            action_mode=action_mode,
            action_dim=action_dim,
        ),
    )
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
        {
            "latent_action_dim": config.lam.latent_action_dim,
            "num_samples": int(latent_actions.shape[0]),
            "mean_abs": float(np.mean(np.abs(latent_actions))),
            "std": float(np.std(latent_actions)),
        },
    )
    write_json_artifact(
        out_dir / "latent_action_bridge.json",
        {
            "latent_action_dim": bridge.latent_action_dim,
            "real_action_dim": bridge.real_action_dim,
            "action_mode": action_mode,
            "source": "lam_replay_actions",
        },
    )
    write_jsonl_metrics(out_dir / "real_env_metrics.jsonl", real_env_metrics)
    rollout = np.asarray(
        batch.observations[: min(args.policy_train_steps, batch.time_steps), 0]
    )
    _write_png(
        out_dir / "open_loop_rollout.png",
        np.concatenate([_to_rgb_panel(item) for item in rollout], axis=1),
    )
    _write_png(out_dir / "latent_action_grid.png", _to_rgb_panel(latent_actions[:32]))
    write_json_artifact(
        out_dir / "outcome.json",
        {
            "status": status,
            "initial_loss": metrics[0]["loss"],
            "final_loss": metrics[-1]["loss"],
            "real_env_bridged_return": real_env_return,
        },
    )
    write_json_artifact(
        out_dir / "summary.json",
        {
            "status": status,
            "model": "genie2_continuous_jax",
            "env": args.env,
            "action_mode": action_mode,
            "observation_shape": observation_shape,
            "final_loss": metrics[-1]["loss"],
            "real_env_bridged_return": real_env_return,
            "learning_gate_passed": gate_passed,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
