from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.imagination import (
    dreamer_policy_action,
    train_dreamer_actor_critic,
)
from world_marl.dreamer_v3_baseline.rssm import (
    flatten_rssm_state,
    initial_rssm_state,
    reset_rssm_state,
)
from world_marl.dreamer_v3_baseline.training import (
    DreamerWorldModel,
    dreamer_action_features,
    train_dreamer_world_model,
)
from world_marl.dreamer_v3_baseline.validation import (
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
        "--out-dir", type=Path, default=Path("runs/dreamer_v3_baseline")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--policy-train-steps", type=int, default=10)
    parser.add_argument("--imagination-horizon", type=int, default=15)
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
    parser.add_argument("--dmc-camera-id", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
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
        "imagination_horizon": args.imagination_horizon,
        "collect_steps": _collect_steps(args),
        "batch_size": args.batch_size,
        "num_envs": args.num_envs,
        "max_cycles": args.max_cycles,
        "image_size": args.image_size,
        "dmc_camera_id": args.dmc_camera_id,
        "action_dim": config.action_dim,
        "action_mode": config.action_mode,
        "observation_shape": config.observation_shape,
        "rssm": {
            "deterministic_size": config.rssm.deterministic_size,
            "stochastic_size": config.rssm.stochastic_size,
            "discrete_classes": config.rssm.discrete_classes,
        },
        "actor_critic": {
            "value_bins": config.actor_critic.value_bins,
            "discount_lambda": config.actor_critic.discount_lambda,
            "entropy_scale": config.actor_critic.entropy_scale,
        },
    }


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


def _squeeze_single_agent_axis(array: Any, *, num_envs: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim >= 2 and values.shape[:2] == (num_envs, 1):
        return values[:, 0, ...]
    return values


def _evaluate_real_env(
    args: argparse.Namespace,
    batch: Any,
    config: DreamerV3Config,
    world_model_state: TrainState,
    actor_state: TrainState,
) -> list[dict[str, float | str]]:
    if args.env.startswith("synthetic:"):
        return [
            {
                "episode": 0,
                "return": float(np.mean(np.sum(batch.rewards, axis=0))),
                "length": float(batch.time_steps),
                "policy_source": "imagined_actor",
            }
        ]

    evaluation_num_envs = min(args.num_envs, max(args.eval_episodes, 1))
    adapter = make_single_agent_adapter(
        args.env,
        num_envs=evaluation_num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed + 10_000,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
    )
    try:
        rows: list[dict[str, float | str]] = []
        model = DreamerWorldModel(config)
        action_low = getattr(adapter, "action_low", None)
        action_high = getattr(adapter, "action_high", None)
        if action_low is not None:
            action_low = np.asarray(action_low, dtype=np.float32).reshape(
                (config.action_dim,)
            )
        if action_high is not None:
            action_high = np.asarray(action_high, dtype=np.float32).reshape(
                (config.action_dim,)
            )
        observations = _squeeze_single_agent_axis(
            adapter.reset(),
            num_envs=adapter.num_envs,
        ).reshape((adapter.num_envs, *config.observation_shape))
        latent_state = initial_rssm_state(
            batch_size=adapter.num_envs,
            config=config.rssm,
        )
        if config.action_mode == "discrete":
            previous_action = jnp.zeros((adapter.num_envs,), dtype=jnp.int32)
        else:
            previous_action = jnp.zeros(
                (adapter.num_envs, config.action_dim), dtype=jnp.float32
            )
        target_episodes = max(args.eval_episodes, 1)
        while len(rows) < target_episodes:
            action_features = dreamer_action_features(previous_action, config)
            _, latent_state, _ = model.apply(
                world_model_state.params,
                latent_state,
                action_features,
                jnp.asarray(observations, dtype=jnp.float32),
                method=model.observe_step,
            )
            actions = dreamer_policy_action(
                actor_state,
                flatten_rssm_state(latent_state),
                config,
            )
            if config.action_mode == "discrete":
                policy_action = np.asarray(actions, dtype=np.int32).reshape(
                    (adapter.num_envs, 1)
                )
                previous_action = actions
            else:
                policy_action = np.asarray(actions, dtype=np.float32).reshape(
                    (adapter.num_envs, config.action_dim)
                )
                if action_low is not None and action_high is not None:
                    policy_action = np.clip(policy_action, action_low, action_high)
                previous_action = jnp.asarray(policy_action, dtype=jnp.float32)
            step = adapter.step(policy_action)
            for episode_return, episode_length in zip(
                step.completed_returns,
                step.completed_lengths,
                strict=True,
            ):
                if len(rows) >= target_episodes:
                    break
                rows.append(
                    {
                        "episode": len(rows),
                        "return": float(episode_return[0]),
                        "length": float(episode_length),
                        "policy_source": "imagined_actor",
                    }
                )
            terminals = _squeeze_single_agent_axis(
                step.dones,
                num_envs=adapter.num_envs,
            ).reshape((adapter.num_envs,))
            latent_state = reset_rssm_state(
                latent_state,
                jnp.asarray(terminals > 0.5),
                config=config.rssm,
            )
            if config.action_mode == "discrete":
                previous_action = jnp.where(
                    jnp.asarray(terminals > 0.5),
                    jnp.zeros_like(previous_action),
                    previous_action,
                )
            else:
                previous_action = jnp.where(
                    jnp.asarray(terminals > 0.5)[:, None],
                    jnp.zeros_like(previous_action),
                    previous_action,
                )
            observations = _squeeze_single_agent_axis(
                step.observations,
                num_envs=adapter.num_envs,
            ).reshape((adapter.num_envs, *config.observation_shape))
        return rows
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


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
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
    )


def _run_accounting(
    args: argparse.Namespace,
    batch: Any,
    real_env_metrics: list[dict[str, float | str]],
) -> dict[str, Any]:
    environment_backend = str(batch.metadata.get("environment_backend", "unknown"))
    evaluation_transitions = 0
    if environment_backend not in {"synthetic", "unknown"}:
        evaluation_transitions = int(
            sum(float(row["length"]) for row in real_env_metrics)
        )
    return {
        "seed": args.seed,
        "evaluation_seed": args.seed + 10_000,
        "environment_backend": environment_backend,
        "observation_mode": str(batch.metadata.get("observation_mode", "unknown")),
        "real_env_transitions": int(batch.metadata.get("real_env_transitions", 0)),
        "evaluation_env_transitions": evaluation_transitions,
        "evaluation_episodes": len(real_env_metrics),
        "model_updates": args.train_steps,
        "policy_updates": args.policy_train_steps,
        "imagined_transitions": (
            args.policy_train_steps * args.imagination_horizon * batch.batch_size
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir
    batch = _make_batch(args)
    action_mode = str(batch.metadata.get("action_mode", "discrete"))
    action_dim = int(batch.metadata.get("action_dim", args.action_dim))
    config = DreamerV3Config(
        action_dim=action_dim,
        action_mode=action_mode,
        observation_shape=batch.observation_shape,
    )
    state, world_metrics = train_dreamer_world_model(
        batch=batch,
        config=config,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    finite_metric_check(world_metrics[-1])
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    outputs = state.apply_fn(
        state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        jnp.asarray(batch.actions, dtype=action_dtype),
        jnp.asarray(batch.is_first, dtype=bool),
    )
    actor_state, critic_state, actor_critic_metrics, imagined_rollout = (
        train_dreamer_actor_critic(
            world_model_state=state,
            batch=batch,
            config=config,
            train_steps=args.policy_train_steps,
            learning_rate=args.learning_rate,
            imagination_horizon=args.imagination_horizon,
            seed=args.seed + 1,
        )
    )
    finite_metric_check(actor_critic_metrics[-1])
    gate_passed = loss_decreased(world_metrics)
    status = "ok" if gate_passed else "learning_gate_failed"
    real_env_metrics = _evaluate_real_env(
        args,
        batch,
        config,
        state,
        actor_state,
    )
    real_env_return = float(np.mean([row["return"] for row in real_env_metrics]))
    accounting = _run_accounting(args, batch, real_env_metrics)

    write_json_artifact(out_dir / "config.json", _config_payload(args, config))
    write_json_artifact(out_dir / "sources.json", world_model_sources())
    write_jsonl_metrics(out_dir / "world_model_metrics.jsonl", world_metrics)
    write_jsonl_metrics(out_dir / "actor_critic_metrics.jsonl", actor_critic_metrics)
    write_jsonl_metrics(out_dir / "real_env_metrics.jsonl", real_env_metrics)
    first_obs = np.asarray(batch.observations[0, 0])
    first_reconstruction = np.asarray(outputs["reconstructions"][0, 0])
    _write_png(
        out_dir / "open_loop_reconstruction.png",
        np.concatenate(
            [_to_rgb_panel(first_obs), _to_rgb_panel(first_reconstruction)], axis=1
        ),
    )
    model = DreamerWorldModel(config)
    imagined_predictions = model.apply(
        state.params,
        imagined_rollout.features[:, 0],
        method=model.predict,
    )
    rollout = np.asarray(imagined_predictions["reconstructions"])
    _write_png(
        out_dir / "imagined_rollout.png",
        np.concatenate([_to_rgb_panel(item) for item in rollout], axis=1),
    )
    write_json_artifact(
        out_dir / "outcome.json",
        {
            "status": status,
            "final_loss": world_metrics[-1]["loss"],
            "initial_loss": world_metrics[0]["loss"],
            "real_env_return": real_env_return,
            "policy_source": "imagined_actor",
            **accounting,
        },
    )
    write_json_artifact(
        out_dir / "summary.json",
        {
            "status": status,
            "model": "dreamer_v3_baseline",
            "env": args.env,
            "action_mode": config.action_mode,
            "observation_shape": config.observation_shape,
            "final_loss": world_metrics[-1]["loss"],
            "real_env_return": real_env_return,
            "policy_source": "imagined_actor",
            "learning_gate_passed": gate_passed,
            **accounting,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
