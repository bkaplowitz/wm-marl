"""Training-loop helpers shared by CLIs and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.algs.gae import compute_gae
from world_marl.algs.ippo import RolloutBatch
from world_marl.algs.mappo import MAPPORolloutBatch
from world_marl.envs.meltingpot_adapter import (
    flatten_agent_batch,
    unflatten_agent_actions,
)


ObservationMode = Literal["image", "vector"]


@dataclass(frozen=True)
class RolloutResult:
    batch: RolloutBatch | MAPPORolloutBatch
    next_observations: np.ndarray
    last_values: jnp.ndarray
    metrics: dict[str, Any]


def collect_rollout(
    adapter: Any,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutResult:
    """Collect a rollout by stepping a vectorized multi-agent adapter."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    obs_rows = []
    action_rows = []
    log_prob_rows = []
    reward_rows = []
    done_rows = []
    value_rows = []
    entropy_rows = []
    step_infos: list[dict[str, Any]] = []
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    infer_fn = jax.jit(_ippo_infer_with_entropy)
    value_fn = jax.jit(
        lambda state, flat_obs: state.apply_fn(
            {"params": state.params},
            flat_obs,
        )[1]
    )

    current_observations = observations
    for _ in range(rollout_steps):
        flat_observations = flatten_agent_batch(current_observations)
        rng, action_rng = jax.random.split(rng)
        actions, log_probs, values, entropies = infer_fn(
            train_state,
            action_rng,
            jnp.asarray(flat_observations),
        )
        env_actions = unflatten_agent_actions(
            np.asarray(actions),
            num_envs=adapter.num_envs,
            num_agents=adapter.num_agents,
        )
        step = adapter.step(env_actions)

        obs_rows.append(flat_observations)
        action_rows.append(np.asarray(actions, dtype=np.int32))
        log_prob_rows.append(np.asarray(log_probs, dtype=np.float32))
        reward_rows.append(step.rewards.reshape((-1,)))
        done_rows.append(step.dones.reshape((-1,)))
        value_rows.append(np.asarray(values, dtype=np.float32))
        entropy_rows.append(np.asarray(entropies, dtype=np.float32))
        step_infos.extend(step.step_infos)
        completed_returns.extend(step.completed_returns)
        completed_lengths.extend(step.completed_lengths)
        current_observations = step.observations

    last_flat_observations = flatten_agent_batch(current_observations)
    last_values = value_fn(
        train_state,
        jnp.asarray(last_flat_observations),
    )

    batch = RolloutBatch(
        observations=jnp.asarray(np.stack(obs_rows, axis=0), dtype=jnp.float32),
        actions=jnp.asarray(np.stack(action_rows, axis=0), dtype=jnp.int32),
        log_probs=jnp.asarray(np.stack(log_prob_rows, axis=0), dtype=jnp.float32),
        rewards=jnp.asarray(np.stack(reward_rows, axis=0), dtype=jnp.float32),
        dones=jnp.asarray(np.stack(done_rows, axis=0), dtype=jnp.float32),
        values=jnp.asarray(np.stack(value_rows, axis=0), dtype=jnp.float32),
    )

    completed_array = (
        np.asarray(completed_returns, dtype=np.float32)
        if completed_returns
        else np.asarray([], dtype=np.float32)
    )
    metrics = {
        "rollout_mean_reward": float(batch.rewards.mean()),
        "completed_episodes": len(completed_returns),
        "episode_return_mean": (
            float(completed_array.mean()) if completed_returns else None
        ),
        "episode_length_mean": (
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
    }
    metrics.update(
        _rollout_diagnostics(
            batch=batch,
            last_values=last_values,
            entropies=np.stack(entropy_rows, axis=0),
            completed_returns=completed_returns,
            step_infos=step_infos,
            action_dim=adapter.action_dim,
            num_envs=adapter.num_envs,
            num_agents=adapter.num_agents,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    )
    return RolloutResult(
        batch=batch,
        next_observations=current_observations,
        last_values=last_values,
        metrics=metrics,
    )


def collect_mappo_rollout(
    adapter: Any,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    observation_mode: ObservationMode = "image",
) -> RolloutResult:
    """Collect a rollout for MAPPO with centralized critic observations."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    obs_rows = []
    central_obs_rows = []
    action_rows = []
    log_prob_rows = []
    reward_rows = []
    done_rows = []
    value_rows = []
    entropy_rows = []
    step_infos: list[dict[str, Any]] = []
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    infer_fn = jax.jit(_mappo_infer_with_entropy)
    value_fn = jax.jit(
        lambda state, flat_obs, flat_central_obs: state.apply_fn(
            {"params": state.params},
            flat_obs,
            flat_central_obs,
        )[1]
    )

    current_observations = observations
    for _ in range(rollout_steps):
        central_observations = build_central_observations(
            current_observations,
            observation_mode=observation_mode,
        )
        flat_observations = flatten_agent_batch(current_observations)
        flat_central_observations = flatten_agent_batch(central_observations)
        rng, action_rng = jax.random.split(rng)
        actions, log_probs, values, entropies = infer_fn(
            train_state,
            action_rng,
            jnp.asarray(flat_observations),
            jnp.asarray(flat_central_observations),
        )
        env_actions = unflatten_agent_actions(
            np.asarray(actions),
            num_envs=adapter.num_envs,
            num_agents=adapter.num_agents,
        )
        step = adapter.step(env_actions)

        obs_rows.append(flat_observations)
        central_obs_rows.append(flat_central_observations)
        action_rows.append(np.asarray(actions, dtype=np.int32))
        log_prob_rows.append(np.asarray(log_probs, dtype=np.float32))
        reward_rows.append(step.rewards.reshape((-1,)))
        done_rows.append(step.dones.reshape((-1,)))
        value_rows.append(np.asarray(values, dtype=np.float32))
        entropy_rows.append(np.asarray(entropies, dtype=np.float32))
        step_infos.extend(step.step_infos)
        completed_returns.extend(step.completed_returns)
        completed_lengths.extend(step.completed_lengths)
        current_observations = step.observations

    last_central_observations = build_central_observations(
        current_observations,
        observation_mode=observation_mode,
    )
    last_flat_observations = flatten_agent_batch(current_observations)
    last_flat_central_observations = flatten_agent_batch(last_central_observations)
    last_values = value_fn(
        train_state,
        jnp.asarray(last_flat_observations),
        jnp.asarray(last_flat_central_observations),
    )

    batch = MAPPORolloutBatch(
        observations=jnp.asarray(np.stack(obs_rows, axis=0), dtype=jnp.float32),
        central_observations=jnp.asarray(
            np.stack(central_obs_rows, axis=0),
            dtype=jnp.float32,
        ),
        actions=jnp.asarray(np.stack(action_rows, axis=0), dtype=jnp.int32),
        log_probs=jnp.asarray(np.stack(log_prob_rows, axis=0), dtype=jnp.float32),
        rewards=jnp.asarray(np.stack(reward_rows, axis=0), dtype=jnp.float32),
        dones=jnp.asarray(np.stack(done_rows, axis=0), dtype=jnp.float32),
        values=jnp.asarray(np.stack(value_rows, axis=0), dtype=jnp.float32),
    )

    completed_array = (
        np.asarray(completed_returns, dtype=np.float32)
        if completed_returns
        else np.asarray([], dtype=np.float32)
    )
    metrics = {
        "rollout_mean_reward": float(batch.rewards.mean()),
        "completed_episodes": len(completed_returns),
        "episode_return_mean": (
            float(completed_array.mean()) if completed_returns else None
        ),
        "episode_length_mean": (
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
    }
    metrics.update(
        _rollout_diagnostics(
            batch=batch,
            last_values=last_values,
            entropies=np.stack(entropy_rows, axis=0),
            completed_returns=completed_returns,
            step_infos=step_infos,
            action_dim=adapter.action_dim,
            num_envs=adapter.num_envs,
            num_agents=adapter.num_agents,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    )
    return RolloutResult(
        batch=batch,
        next_observations=current_observations,
        last_values=last_values,
        metrics=metrics,
    )


def central_observation_shape(
    observation_shape: tuple[int, ...],
    num_agents: int,
    *,
    observation_mode: ObservationMode = "image",
) -> tuple[int, ...]:
    """Shape for centralized critic observations built by this module."""
    if observation_mode == "vector":
        flat_dim = int(np.prod(observation_shape))
        return (flat_dim * num_agents + num_agents,)
    if observation_mode != "image":
        raise ValueError(f"unsupported observation_mode {observation_mode!r}")
    height, width, channels = observation_shape
    return (height, width, channels * num_agents + num_agents)


def build_vector_central(observations, xp):
    """Array-module-agnostic vector-mode centralized-observation builder.

    ``xp`` is either ``numpy`` or ``jax.numpy``; the same flatten/repeat/one-hot
    logic serves both the numpy data pipeline and the jnp model rollouts.
    """
    num_envs, num_agents = observations.shape[:2]
    flat = observations.reshape((num_envs, num_agents, -1))
    central = xp.repeat(
        flat.reshape((num_envs, num_agents * flat.shape[-1]))[:, None, :],
        num_agents,
        axis=1,
    )
    target_ids = xp.broadcast_to(
        xp.eye(num_agents, dtype=xp.float32)[None],
        (num_envs, num_agents, num_agents),
    )
    return xp.concatenate([central, target_ids], axis=-1)


def build_central_observations(
    observations: np.ndarray,
    *,
    observation_mode: ObservationMode = "image",
) -> np.ndarray:
    """Build centralized critic observations shaped [env, agent, H, W, C].

    The centralized input for each target agent contains all agents' local
    observations concatenated along channels, plus one-hot target-agent channels
    so the critic can estimate individual values.
    """
    observations = np.asarray(observations, dtype=np.float32)
    if observation_mode == "vector":
        if observations.ndim < 3:
            raise ValueError("expected observations shaped [env, agent, ...]")
        return build_vector_central(observations, np)
    if observation_mode != "image":
        raise ValueError(f"unsupported observation_mode {observation_mode!r}")
    if observations.ndim != 5:
        raise ValueError("expected observations shaped [env, agent, H, W, C]")
    num_envs, num_agents, height, width, channels = observations.shape
    central = observations.transpose(0, 2, 3, 1, 4).reshape(
        num_envs,
        height,
        width,
        num_agents * channels,
    )
    central = np.repeat(central[:, None, :, :, :], repeats=num_agents, axis=1)
    target_ids = np.eye(num_agents, dtype=np.float32)
    target_ids = target_ids.reshape(1, num_agents, 1, 1, num_agents)
    target_ids = np.broadcast_to(
        target_ids,
        (num_envs, num_agents, height, width, num_agents),
    )
    return np.concatenate([central, target_ids], axis=-1)


def _ippo_infer_with_entropy(state, key, flat_obs):
    policy, values = state.apply_fn({"params": state.params}, flat_obs)
    actions = policy.sample(seed=key)
    return (
        actions.astype(jnp.int32),
        policy.log_prob(actions),
        values,
        policy.entropy(),
    )


def _mappo_infer_with_entropy(state, key, flat_obs, flat_central_obs):
    policy, values = state.apply_fn(
        {"params": state.params},
        flat_obs,
        flat_central_obs,
    )
    actions = policy.sample(seed=key)
    return (
        actions.astype(jnp.int32),
        policy.log_prob(actions),
        values,
        policy.entropy(),
    )


def _rollout_diagnostics(
    *,
    batch: RolloutBatch | MAPPORolloutBatch,
    last_values: jnp.ndarray,
    entropies: np.ndarray,
    completed_returns: list[tuple[float, ...]],
    step_infos: list[dict[str, Any]],
    action_dim: int,
    num_envs: int,
    num_agents: int,
    gamma: float,
    gae_lambda: float,
) -> dict[str, Any]:
    actions = np.asarray(batch.actions)
    rewards = np.asarray(batch.rewards, dtype=np.float32)
    values = np.asarray(batch.values, dtype=np.float32)
    targets = np.asarray(
        compute_gae(
            batch.rewards,
            batch.values,
            batch.dones,
            last_values,
            gamma,
            gae_lambda,
        )[1],
        dtype=np.float32,
    )

    action_counts = np.bincount(actions.reshape(-1), minlength=action_dim)
    action_total = max(1, int(action_counts.sum()))
    action_counts_by_agent = []
    action_freq_by_agent = []
    actions_by_agent = actions.reshape((actions.shape[0], num_envs, num_agents))
    for agent_index in range(num_agents):
        counts = np.bincount(
            actions_by_agent[:, :, agent_index].reshape(-1),
            minlength=action_dim,
        )
        total = max(1, int(counts.sum()))
        action_counts_by_agent.append(counts.astype(int).tolist())
        action_freq_by_agent.append((counts / total).astype(float).tolist())

    rewards_by_agent = rewards.reshape((rewards.shape[0], num_envs, num_agents))
    entropies_by_agent = entropies.reshape((entropies.shape[0], num_envs, num_agents))
    completed_array = (
        np.asarray(completed_returns, dtype=np.float32)
        if completed_returns
        else np.zeros((0, num_agents), dtype=np.float32)
    )

    metrics: dict[str, Any] = {
        "action_counts": action_counts.astype(int).tolist(),
        "action_frequencies": (action_counts / action_total).astype(float).tolist(),
        "action_counts_by_agent": action_counts_by_agent,
        "action_frequencies_by_agent": action_freq_by_agent,
        "policy_entropy_mean": float(entropies.mean()),
        "policy_entropy_by_agent": entropies_by_agent.mean(axis=(0, 1))
        .astype(float)
        .tolist(),
        "policy_entropy_min": float(entropies.min()),
        "policy_entropy_max": float(entropies.max()),
        "rollout_reward_mean_by_agent": rewards_by_agent.mean(axis=(0, 1))
        .astype(float)
        .tolist(),
        "rollout_reward_sum_by_agent": rewards_by_agent.sum(axis=(0, 1))
        .astype(float)
        .tolist(),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "value_target_mean": float(targets.mean()),
        "value_target_std": float(targets.std()),
        "value_explained_variance": _explained_variance(values, targets),
    }
    if completed_returns:
        metrics["episode_return_mean_by_agent"] = (
            completed_array.mean(axis=0).astype(float).tolist()
        )
    else:
        metrics["episode_return_mean_by_agent"] = None
    metrics.update(_info_diagnostics(step_infos))
    return metrics


def _explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    target_variance = float(np.var(targets))
    if target_variance < 1e-8:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_variance)


def _info_diagnostics(step_infos: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "info_items_seen": 0,
        "coin_related_info_items": 0,
        "coin_consumed_events": 0,
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                counters["info_items_seen"] += 1
                key_text = str(key).lower()
                if "coin" in key_text:
                    counters["coin_related_info_items"] += 1
                if "coin_consumed" in key_text:
                    counters["coin_consumed_events"] += 1
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            text = value.lower()
            if "coin" in text:
                counters["coin_related_info_items"] += 1
            if "coin_consumed" in text:
                counters["coin_consumed_events"] += 1

    for info in step_infos:
        visit(info)
    return counters


def training_window_means(
    rows: list[dict[str, Any]],
    *,
    fraction: float = 1 / 3,
) -> tuple[float, float]:
    """Return early/final means from available training metrics."""
    if not rows:
        return 0.0, 0.0
    values = [
        row["episode_return_mean"]
        if row.get("episode_return_mean") is not None
        else row["rollout_mean_reward"]
        for row in rows
    ]
    window = max(1, int(len(values) * fraction))
    return float(np.mean(values[:window])), float(np.mean(values[-window:]))
