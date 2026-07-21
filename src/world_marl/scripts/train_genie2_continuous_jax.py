"""Train and evaluate the public-source Genie2 latent-diffusion alternative."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.policy import (
    latent_policy_action,
    train_genie2_latent_policy,
    update_observation_history,
)
from world_marl.genie2_continuous_jax.training import (
    create_genie2_train_state,
    decode_genie2_latents,
    encode_genie2_observations,
    metrics_to_host,
    scan_genie2_training_phases,
)
from world_marl.world_model_foundation.collect import (
    collect_world_model_sequence,
    make_single_agent_adapter,
    write_json_artifact,
    write_jsonl_metrics,
)
from world_marl.world_model_foundation.metrics import scanned_episode_metrics
from world_marl.world_model_foundation.replay import sequence_batch_to_jax
from world_marl.world_model_foundation.sources import world_model_sources


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="synthetic:image-grid")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/genie2_continuous_jax")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", choices=("jasmine", "debug"), default="jasmine")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--tokenizer-steps", type=int, default=10)
    parser.add_argument("--dynamics-steps", type=int, default=10)
    parser.add_argument("--reward-continue-steps", type=int, default=10)
    parser.add_argument("--policy-train-steps", type=int, default=10)
    parser.add_argument(
        "--policy-objective",
        choices=("reinforce", "candidate-distill"),
        default="reinforce",
    )
    parser.add_argument("--num-policy-candidates", type=int, default=64)
    parser.add_argument("--candidate-min-gap", type=float, default=0.0)
    parser.add_argument("--imagination-horizon", type=int, default=5)
    parser.add_argument("--collect-steps", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--brax-backend", default=None)
    parser.add_argument("--dmc-workers", type=int, default=1)
    parser.add_argument("--dmc-camera-id", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)
    if args.train_steps is not None:
        args.tokenizer_steps = args.train_steps
        args.dynamics_steps = args.train_steps
        args.reward_continue_steps = args.train_steps
    for name in (
        "tokenizer_steps",
        "dynamics_steps",
        "reward_continue_steps",
        "policy_train_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_policy_candidates < 2:
        parser.error("--num-policy-candidates must be at least 2")
    if args.candidate_min_gap < 0.0:
        parser.error("--candidate-min-gap must be non-negative")
    return args


def _collect_steps(args: argparse.Namespace) -> int:
    requested = (
        args.collect_steps if args.collect_steps is not None else args.time_steps
    )
    return max(int(requested), int(args.sequence_length))


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


def _to_rgb_panel(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] in {1, 3, 4}:
        values = values[..., :3]
        if values.shape[-1] == 1:
            values = np.repeat(values, 3, axis=-1)
    else:
        values = values.reshape(1, -1)
        values = np.repeat(values[..., None], 3, axis=-1)
    minimum, maximum = float(values.min()), float(values.max())
    if maximum > minimum:
        values = (values - minimum) / (maximum - minimum)
    return np.asarray(np.clip(values * 255.0, 0.0, 255.0), dtype=np.uint8)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)


def _evaluate_real_env(
    args: argparse.Namespace,
    batch,
    world_state,
    actor_state,
    config: Genie2ContinuousConfig,
) -> list[dict[str, float | str]]:
    if args.env.startswith("synthetic:"):
        return [
            {
                "episode": 0,
                "return": float(np.mean(np.sum(batch.rewards, axis=0))),
                "length": float(batch.time_steps),
                "policy_source": "direct_action_policy",
                "evaluation_execution": "synthetic",
            }
        ]
    target_episodes = max(args.eval_episodes, 1)
    evaluation_num_envs = math.gcd(args.num_envs, target_episodes)
    adapter = make_single_agent_adapter(
        args.env,
        num_envs=evaluation_num_envs,
        max_cycles=args.max_cycles,
        seed=args.seed + 20_000,
        brax_backend=args.brax_backend,
        dmc_workers=args.dmc_workers,
        image_size=args.image_size,
        dmc_camera_id=args.dmc_camera_id,
    )
    try:
        scan_rollout = getattr(adapter, "scan_recurrent_rollout", None)
        if scan_rollout is None:
            raise RuntimeError(
                f"{args.env} must expose scan_recurrent_rollout for Genie2 evaluation"
            )
        observations = np.asarray(adapter.reset(), dtype=np.float32).reshape(
            (adapter.num_envs, *config.observation_shape)
        )

        evaluation_context = args.sequence_length if config.is_image_observation else 1
        initial_history = jnp.broadcast_to(
            jnp.asarray(observations)[None],
            (evaluation_context, *observations.shape),
        )

        def policy_step(policy_state, carry, flat_observations, is_first):
            model_state, policy_train_state = policy_state
            observation_history, policy_key = carry
            policy_key, encode_key = jax.random.split(policy_key)
            current_observations = flat_observations.reshape(
                (adapter.num_envs, *config.observation_shape)
            )
            observation_history = update_observation_history(
                observation_history,
                current_observations,
                is_first,
            )
            latents = encode_genie2_observations(
                model_state,
                observation_history,
                config,
                encode_key,
            )
            pooled = jnp.mean(latents[:, -1], axis=1)
            actions = latent_policy_action(policy_train_state, pooled, config)
            return (observation_history, policy_key), actions

        evaluation_steps = math.ceil(target_episodes / adapter.num_envs) * (
            args.max_cycles + 1
        )
        ys, _, _ = scan_rollout(
            policy_step,
            (world_state, actor_state),
            (initial_history, jax.random.PRNGKey(args.seed + 30_000)),
            evaluation_steps,
            observations=observations,
        )
        _, _, rewards, _, dones = ys
        return scanned_episode_metrics(
            rewards,
            dones,
            target_episodes=target_episodes,
            policy_source="direct_action_policy",
            arrival_aligned=True,
        )
    finally:
        close = getattr(adapter, "close", None)
        if close is not None:
            close()


def _accounting(
    args: argparse.Namespace,
    batch,
    config: Genie2ContinuousConfig,
    real_env_metrics: list[dict[str, float | str]],
) -> dict[str, Any]:
    backend = str(batch.metadata.get("environment_backend", "unknown"))
    evaluation_transitions = 0
    if backend not in {"synthetic", "unknown"}:
        evaluation_transitions = int(
            sum(float(row["length"]) for row in real_env_metrics)
        )
    candidate_action_evaluations = 0
    imagined_transitions = 0
    if args.policy_objective == "candidate-distill":
        candidate_action_evaluations = (
            args.policy_train_steps
            * config.latent_policy.batch_size
            * args.num_policy_candidates
        )
        imagined_transitions = candidate_action_evaluations * args.imagination_horizon
    else:
        imagined_transitions = (
            args.policy_train_steps
            * args.imagination_horizon
            * config.latent_policy.batch_size
        )
    return {
        "seed": args.seed,
        "evaluation_seed": args.seed + 20_000,
        "environment_backend": backend,
        "physics_backend": str(batch.metadata.get("physics_backend", "unknown")),
        "observation_mode": str(batch.metadata.get("observation_mode", "unknown")),
        "collection_execution": str(
            batch.metadata.get("collection_execution", "unknown")
        ),
        "training_execution": "nested_jax_scan",
        "real_env_transitions": int(batch.metadata.get("real_env_transitions", 0)),
        "evaluation_env_transitions": evaluation_transitions,
        "evaluation_episodes": len(real_env_metrics),
        "evaluation_execution": str(real_env_metrics[0]["evaluation_execution"]),
        "tokenizer_updates": args.tokenizer_steps,
        "dynamics_updates": args.dynamics_steps,
        "reward_continue_updates": args.reward_continue_steps,
        "model_updates": (
            args.tokenizer_steps + args.dynamics_steps + args.reward_continue_steps
        ),
        "policy_updates": args.policy_train_steps,
        "policy_imagination_batch_size": config.latent_policy.batch_size,
        "policy_imagination_horizon": args.imagination_horizon,
        "imagined_transitions": imagined_transitions,
        "candidate_action_evaluations": candidate_action_evaluations,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batch = _make_batch(args)
    action_mode = str(batch.metadata.get("action_mode", "discrete"))
    action_dim = int(batch.metadata.get("action_dim", args.action_dim))
    config_factory = (
        Genie2ContinuousConfig.debug
        if args.model_size == "debug"
        else Genie2ContinuousConfig
    )
    config = config_factory(
        action_dim=action_dim,
        action_mode=action_mode,
        action_low=(
            None
            if batch.metadata.get("action_low") is None
            else tuple(float(value) for value in batch.metadata["action_low"])
        ),
        action_high=(
            None
            if batch.metadata.get("action_high") is None
            else tuple(float(value) for value in batch.metadata["action_high"])
        ),
        observation_shape=batch.observation_shape,
    )
    replay = sequence_batch_to_jax(batch)
    train_batch_size = args.train_batch_size or min(batch.batch_size, 4)
    sequence_length = min(
        args.sequence_length, batch.time_steps, config.dynamics.max_context
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(args.seed),
        config=config,
        learning_rate=args.learning_rate,
    )
    state, metric_arrays, validation_arrays = jax.jit(
        scan_genie2_training_phases,
        static_argnames=(
            "config",
            "tokenizer_steps",
            "dynamics_steps",
            "reward_continue_steps",
            "sequence_length",
            "batch_size",
        ),
    )(
        state,
        replay,
        jax.random.PRNGKey(args.seed + 1),
        config=config,
        tokenizer_steps=args.tokenizer_steps,
        dynamics_steps=args.dynamics_steps,
        reward_continue_steps=args.reward_continue_steps,
        sequence_length=sequence_length,
        batch_size=train_batch_size,
    )
    metric_arrays, validation_arrays = jax.device_get(
        (metric_arrays, validation_arrays)
    )
    phase_metrics = metrics_to_host(metric_arrays)
    validation_metrics = {
        stage: {name: float(value) for name, value in values.items()}
        for stage, values in validation_arrays.items()
    }
    actor_state, _, policy_metrics, imagined_rollout = train_genie2_latent_policy(
        world_model_state=state,
        batch=batch,
        observation_shape=batch.observation_shape,
        config=config,
        train_steps=args.policy_train_steps,
        learning_rate=args.learning_rate or 1e-4,
        imagination_horizon=args.imagination_horizon,
        seed=args.seed + 2,
        objective=args.policy_objective,
        num_candidates=args.num_policy_candidates,
        candidate_min_gap=args.candidate_min_gap,
    )
    real_env_metrics = _evaluate_real_env(args, batch, state, actor_state, config)
    real_env_return = float(np.mean([row["return"] for row in real_env_metrics]))
    gate_passed = all(
        values["final_loss"] < values["initial_loss"]
        for values in validation_metrics.values()
    )
    status = "ok" if gate_passed else "learning_gate_failed"
    accounting = _accounting(args, batch, config, real_env_metrics)

    out_dir = args.out_dir
    config_payload = {
        "model": "genie2_continuous_jax",
        "model_size": args.model_size,
        "conditioning_mode": config.conditioning_mode,
        "representation": config.representation,
        "action_mode": config.action_mode,
        "action_dim": config.action_dim,
        "observation_shape": config.observation_shape,
        "sequence_length": sequence_length,
        "train_batch_size": train_batch_size,
        "policy_objective": args.policy_objective,
        "num_policy_candidates": args.num_policy_candidates,
        "candidate_min_gap": args.candidate_min_gap,
    }
    write_json_artifact(out_dir / "config.json", config_payload)
    write_json_artifact(out_dir / "sources.json", world_model_sources())
    write_jsonl_metrics(out_dir / "tokenizer_metrics.jsonl", phase_metrics["tokenizer"])
    write_jsonl_metrics(
        out_dir / "autoencoder_metrics.jsonl", phase_metrics["tokenizer"]
    )
    write_jsonl_metrics(out_dir / "dynamics_metrics.jsonl", phase_metrics["dynamics"])
    write_jsonl_metrics(
        out_dir / "reward_continue_metrics.jsonl",
        phase_metrics["reward_continue"],
    )
    write_jsonl_metrics(out_dir / "policy_metrics.jsonl", policy_metrics)
    write_jsonl_metrics(out_dir / "real_env_metrics.jsonl", real_env_metrics)
    write_json_artifact(out_dir / "validation_metrics.json", validation_metrics)
    write_json_artifact(
        out_dir / "conditioning.json",
        {
            "mode": "real_action",
            "source": "public_genie2_disclosure",
            "lam_enabled": False,
            "bridge_required": False,
        },
    )
    decoded_rollout = decode_genie2_latents(
        state,
        jnp.swapaxes(imagined_rollout.latents, 0, 1),
        config,
    )
    rollout_panels = np.asarray(decoded_rollout[0])
    _write_png(
        out_dir / "open_loop_rollout.png",
        np.concatenate([_to_rgb_panel(frame) for frame in rollout_panels], axis=1),
    )
    action_panel = np.asarray(imagined_rollout.model_actions).reshape(
        (imagined_rollout.model_actions.shape[0], -1)
    )
    _write_png(out_dir / "action_grid.png", _to_rgb_panel(action_panel))
    outcome = {
        "status": status,
        "final_tokenizer_loss": phase_metrics["tokenizer"][-1]["tokenizer_loss"],
        "final_dynamics_loss": phase_metrics["dynamics"][-1]["dynamics_loss"],
        "validation_losses": validation_metrics,
        "real_env_return": real_env_return,
        "policy_source": "direct_action_policy",
        "policy_objective": args.policy_objective,
        "conditioning_mode": config.conditioning_mode,
        "representation": config.representation,
        "action_mode": config.action_mode,
        "action_dim": config.action_dim,
        "observation_shape": config.observation_shape,
        **accounting,
    }
    write_json_artifact(out_dir / "outcome.json", outcome)
    write_json_artifact(
        out_dir / "summary.json",
        {
            **outcome,
            "model": "genie2_continuous_jax",
            "env": args.env,
            "learning_gate_passed": gate_passed,
        },
    )
    return 0 if gate_passed or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
