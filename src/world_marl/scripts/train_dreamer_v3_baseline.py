from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.image as mpimg
import numpy as np
from flax.training.train_state import TrainState

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.imagination import (
    dreamer_policy_action,
)
from world_marl.dreamer_v3_baseline.rssm import (
    flatten_rssm_state,
    initial_rssm_state,
    reset_rssm_state,
)
from world_marl.dreamer_v3_baseline.training import (
    DreamerWorldModel,
    dreamer_agent_views,
    dreamer_action_features,
    observe_dreamer_sequence,
    train_dreamer_agent,
    train_dreamer_agent_online,
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
from world_marl.world_model_foundation.metrics import scanned_episode_metrics
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
    parser.add_argument("--time-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--model-size", choices=("12m", "debug"), default="12m")
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
        "train_batch_size": args.train_batch_size,
        "train_ratio": (
            config.replay.train_ratio if args.train_ratio is None else args.train_ratio
        ),
        "sequence_length": args.sequence_length,
        "num_envs": args.num_envs,
        "max_cycles": args.max_cycles,
        "image_size": args.image_size,
        "model_size": args.model_size,
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
        model = DreamerWorldModel(config)
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
        scan_recurrent_rollout = getattr(adapter, "scan_recurrent_rollout", None)
        if scan_recurrent_rollout is None:
            raise RuntimeError(
                f"adapter for {args.env!r} must provide scan_recurrent_rollout"
            )

        def policy_step(policy_state, carry, obs_flat, is_first):
            world_params, policy_train_state = policy_state
            current_state, previous_action, policy_key = carry
            policy_key, sample_key, action_key = jax.random.split(policy_key, 3)
            current_state = reset_rssm_state(
                current_state,
                is_first,
                config=config.rssm,
            )
            if config.action_mode == "discrete":
                previous_action = jnp.where(
                    is_first,
                    jnp.zeros_like(previous_action),
                    previous_action,
                )
            else:
                previous_action = jnp.where(
                    is_first[:, None],
                    jnp.zeros_like(previous_action),
                    previous_action,
                )
            _, current_state, _ = model.apply(
                world_params,
                current_state,
                dreamer_action_features(previous_action, config),
                obs_flat.reshape((adapter.num_envs, *config.observation_shape)),
                sample_key,
                method=model.observe_step,
            )
            actions = dreamer_policy_action(
                policy_train_state,
                flatten_rssm_state(current_state),
                config,
                action_key,
            )
            return (current_state, actions, policy_key), actions

        evaluation_steps = math.ceil(target_episodes / adapter.num_envs) * (
            args.max_cycles + 1
        )
        ys, _, _ = scan_recurrent_rollout(
            policy_step,
            (world_model_state.params, actor_state),
            (latent_state, previous_action, jax.random.PRNGKey(args.seed + 20_000)),
            evaluation_steps,
            observations=observations,
        )
        _, _, rewards, _, dones = ys
        return scanned_episode_metrics(
            rewards,
            dones,
            target_episodes=target_episodes,
            policy_source="imagined_actor",
            arrival_aligned=True,
        )
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


def _minimum_online_environment_steps(
    *,
    requested_steps: int,
    num_envs: int,
    sequence_length: int,
    batch_size: int,
    train_ratio: float,
    train_steps: int,
) -> int:
    if train_ratio <= 0:
        raise ValueError("online Dreamer training requires a positive train ratio")
    ready_step = max(
        sequence_length + 1,
        math.ceil(batch_size * sequence_length / num_envs),
    )
    update_ratio = num_envs * train_ratio / (batch_size * sequence_length)
    additional_steps = (
        0 if train_steps <= 1 else math.ceil((train_steps - 1) / update_ratio)
    )
    return max(requested_steps, ready_step + additional_steps)


def _run_accounting(
    args: argparse.Namespace,
    batch: Any,
    real_env_metrics: list[dict[str, float | str]],
    *,
    model_updates: int,
    sequence_length: int,
    train_batch_size: int,
    training_execution: str,
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
        "physics_backend": str(batch.metadata.get("physics_backend", "unknown")),
        "observation_mode": str(batch.metadata.get("observation_mode", "unknown")),
        "collection_execution": str(
            batch.metadata.get("collection_execution", "unknown")
        ),
        "collection_policy": str(batch.metadata.get("collection_policy", "unknown")),
        "training_execution": training_execution,
        "requested_collection_steps": _collect_steps(args),
        "online_environment_steps": batch.time_steps,
        "real_env_transitions": int(batch.metadata.get("real_env_transitions", 0)),
        "evaluation_env_transitions": evaluation_transitions,
        "evaluation_episodes": len(real_env_metrics),
        "evaluation_execution": str(
            real_env_metrics[0].get("evaluation_execution", "synthetic")
        ),
        "requested_model_updates": args.train_steps,
        "model_updates": model_updates,
        "policy_updates": model_updates,
        "imagined_transitions": (
            model_updates
            * args.imagination_horizon
            * sequence_length
            * train_batch_size
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.policy_train_steps != args.train_steps:
        raise ValueError(
            "paper-faithful DreamerV3 updates world model, actor, and critic "
            "together; --policy-train-steps must equal --train-steps"
        )
    if args.env.startswith("dmc-pixels:"):
        raise RuntimeError(
            "host-loop collection is not supported by the scanned Dreamer trainer"
        )
    out_dir = args.out_dir
    config_factory = (
        DreamerV3Config.debug if args.model_size == "debug" else DreamerV3Config
    )
    if args.env.startswith("synthetic:"):
        batch = _make_batch(args)
        action_mode = str(batch.metadata.get("action_mode", "discrete"))
        action_dim = int(batch.metadata.get("action_dim", args.action_dim))
        config = config_factory(
            action_dim=action_dim,
            action_mode=action_mode,
            observation_shape=batch.observation_shape,
        )
        resolved_sequence_length = min(
            batch.time_steps - 1,
            config.replay.batch_length,
            args.sequence_length,
        )
        resolved_train_batch_size = (
            min(batch.batch_size, config.replay.batch_size)
            if args.train_batch_size is None
            else args.train_batch_size
        )
        agent_state, agent_metrics, imagined_rollout = train_dreamer_agent(
            batch=batch,
            config=config,
            train_steps=args.train_steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            sequence_length=resolved_sequence_length,
            batch_size=resolved_train_batch_size,
            imagination_horizon=args.imagination_horizon,
        )
        training_execution = "jax_scan_synthetic_fixture"
    else:
        adapter = make_single_agent_adapter(
            args.env,
            num_envs=args.num_envs,
            max_cycles=args.max_cycles,
            seed=args.seed,
            brax_backend=args.brax_backend,
            dmc_workers=args.dmc_workers,
            image_size=args.image_size,
            dmc_camera_id=args.dmc_camera_id,
        )
        try:
            observations = _squeeze_single_agent_axis(
                adapter.reset(),
                num_envs=adapter.num_envs,
            )
            action_mode = (
                "discrete"
                if tuple(getattr(adapter, "action_shape", ())) == ()
                else "continuous"
            )
            config = config_factory(
                action_dim=int(adapter.action_dim),
                action_mode=action_mode,
                observation_shape=tuple(adapter.observation_shape),
            )
            requested_environment_steps = _collect_steps(args)
            resolved_sequence_length = min(
                requested_environment_steps - 1,
                config.replay.batch_length,
                args.sequence_length,
            )
            if resolved_sequence_length <= 0:
                raise ValueError(
                    "collect steps must provide a nonempty replay sequence"
                )
            resolved_train_batch_size = (
                config.replay.batch_size
                if args.train_batch_size is None
                else args.train_batch_size
            )
            resolved_train_ratio = (
                config.replay.train_ratio
                if args.train_ratio is None
                else args.train_ratio
            )
            environment_steps = _minimum_online_environment_steps(
                requested_steps=requested_environment_steps,
                num_envs=adapter.num_envs,
                sequence_length=resolved_sequence_length,
                batch_size=resolved_train_batch_size,
                train_ratio=resolved_train_ratio,
                train_steps=args.train_steps,
            )
            agent_state, agent_metrics, imagined_rollout, batch = (
                train_dreamer_agent_online(
                    adapter=adapter,
                    observations=observations,
                    config=config,
                    environment_steps=environment_steps,
                    max_train_steps=args.train_steps,
                    seed=args.seed,
                    train_ratio=resolved_train_ratio,
                    learning_rate=args.learning_rate,
                    sequence_length=resolved_sequence_length,
                    batch_size=resolved_train_batch_size,
                    imagination_horizon=args.imagination_horizon,
                )
            )
            batch.metadata["requested_collection_steps"] = requested_environment_steps
        finally:
            close = getattr(adapter, "close", None)
            if close is not None:
                close()
        training_execution = "nested_jax_scan"
    world_model_state, actor_state, _ = dreamer_agent_views(agent_state, config)
    world_metric_keys = {
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "kl_loss",
        "dynamics_kl_loss",
        "representation_kl_loss",
    }
    world_metrics = [
        {
            "step": row["step"],
            "loss": row["world_model_loss"],
            **{key: row[key] for key in world_metric_keys},
        }
        for row in agent_metrics
    ]
    actor_metric_keys = {
        "actor_loss",
        "critic_loss",
        "replay_critic_loss",
        "return_scale",
        "imagined_reward",
        "imagined_value",
        "actor_entropy",
    }
    actor_critic_metrics = [
        {
            "step": row["step"],
            **{key: row[key] for key in actor_metric_keys},
        }
        for row in agent_metrics
    ]
    finite_metric_check(world_metrics[-1])
    action_dtype = jnp.int32 if config.action_mode == "discrete" else jnp.float32
    actions = jnp.asarray(batch.actions, dtype=action_dtype)
    previous_actions = jnp.concatenate(
        [jnp.zeros_like(actions[:1]), actions[:-1]],
        axis=0,
    )
    outputs = observe_dreamer_sequence(
        world_model_state.params,
        jnp.asarray(batch.observations, dtype=jnp.float32),
        previous_actions,
        jnp.asarray(batch.is_first, dtype=bool),
        config,
        jax.random.PRNGKey(args.seed + 2),
    )
    finite_metric_check(actor_critic_metrics[-1])
    gate_passed = loss_decreased(world_metrics)
    status = "ok" if gate_passed else "learning_gate_failed"
    real_env_metrics = _evaluate_real_env(
        args,
        batch,
        config,
        world_model_state,
        actor_state,
    )
    real_env_return = float(np.mean([row["return"] for row in real_env_metrics]))
    accounting = _run_accounting(
        args,
        batch,
        real_env_metrics,
        model_updates=len(agent_metrics),
        sequence_length=resolved_sequence_length,
        train_batch_size=resolved_train_batch_size,
        training_execution=training_execution,
    )

    write_json_artifact(out_dir / "config.json", _config_payload(args, config))
    write_json_artifact(out_dir / "sources.json", world_model_sources())
    write_jsonl_metrics(out_dir / "world_model_metrics.jsonl", world_metrics)
    write_jsonl_metrics(out_dir / "actor_critic_metrics.jsonl", actor_critic_metrics)
    write_jsonl_metrics(out_dir / "agent_metrics.jsonl", agent_metrics)
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
        world_model_state.params,
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
