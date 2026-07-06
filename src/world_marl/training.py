"""Training-loop helpers shared by CLIs and tests."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.algs.gae import compute_gae
from world_marl.algs.ippo import RolloutBatch, ppo_update
from world_marl.algs.mappo import MAPPORolloutBatch, mappo_update
from world_marl.envs.meltingpot_adapter import (
    MeltingPotVectorAdapter,
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
    adapter: MeltingPotVectorAdapter,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> RolloutResult:
    """Collect a rollout by stepping the Python-side Melting Pot adapter."""
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
    get_action_and_value = jax.jit(_ippo_get_action_and_value)
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
        actions, log_probs, values, entropies = get_action_and_value(
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


class TrainOnEnvCarry(NamedTuple):
    """State threaded through ``train_on_real_env``'s update-loop ``lax.scan``."""

    train_state: TrainState
    rng: jax.Array
    env_state: Any
    env_keys: jax.Array
    obs_flat: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array


def train_on_real_env(
    adapter: Any,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    num_updates: int,
    config: Any,
    rollout_steps: int,
    algorithm: str,
    freeze_policy: bool = False,
):
    """Fold the whole model-free PPO update loop into a single ``lax.scan``.

    Reproduces the host loop (``collect_rollout``/``collect_mappo_rollout`` + a
    standalone jitted update, called ``num_updates`` times) but keeps every
    intermediate on device: the env ``state``/``keys``/flat-obs and the per-env
    episode return/length accumulators ride in the scan carry, so nothing syncs
    to the host until the stacked metrics are read afterwards. Coins is lockstep
    and fixed-horizon, so per-agent dones equal ``done["__all__"]`` and
    completions are timer-driven. Per update the carried key is split three ways
    (``rng, rollout_key, update_key``) exactly like the host loop. MAPPO's
    centralized-critic observations are a pure function of the joint obs
    (``build_vector_central``), so they are rebuilt inside the scan rather than
    threaded through the adapter. The adapter's mutable carry is written back at
    the end so a chained caller (e.g. warmup then main) resumes correctly.
    Returns ``(final_train_state, final_observations, final_rng,
    stacked_metrics)`` where ``stacked_metrics`` maps each metric name to a
    ``[num_updates]`` device array (``nan`` marks an update with no completed
    episode; the host maps that to ``None``).
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if num_updates < 1:
        raise ValueError("num_updates must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    is_mappo = algorithm == "mappo"

    num_envs = adapter.num_envs
    num_agents = adapter.num_agents

    get_action_and_value = (
        make_mappo_get_action_and_value(num_envs, num_agents)
        if is_mappo
        else _ippo_get_action_and_value
    )
    rollout_fn = adapter._build_rollout_fn(get_action_and_value, rollout_steps)
    update_fn = mappo_update if is_mappo else ppo_update

    def central_flat(obs_flat):
        grouped = obs_flat.reshape((-1, num_agents, obs_flat.shape[-1]))
        return build_vector_central(grouped, jnp).reshape((obs_flat.shape[0], -1))

    def value_fn(state, flat_obs):
        if is_mappo:
            return state.apply_fn(
                {"params": state.params}, flat_obs, central_flat(flat_obs)
            )[1]
        return state.apply_fn({"params": state.params}, flat_obs)[1]

    def accumulate_episode_stats(episode_return, episode_length, rewards_ea, dones_ea):
        def accumulate_step(carry, step_data):
            episode_return, episode_length, return_sum, length_sum, num_completed = (
                carry
            )
            reward, done = step_data
            episode_return = episode_return + reward
            episode_length = episode_length + 1
            done_env = done.all(axis=1)
            return_sum = return_sum + jnp.sum(
                jnp.where(done_env[:, None], episode_return, 0.0)
            )
            length_sum = length_sum + jnp.sum(jnp.where(done_env, episode_length, 0))
            num_completed = num_completed + jnp.sum(done_env.astype(jnp.int32))
            episode_return = jnp.where(done_env[:, None], 0.0, episode_return)
            episode_length = jnp.where(done_env, 0, episode_length)
            return (
                episode_return,
                episode_length,
                return_sum,
                length_sum,
                num_completed,
            ), None

        init = (
            episode_return,
            episode_length,
            jnp.float32(0.0),
            jnp.int32(0),
            jnp.int32(0),
        )
        (
            (
                final_episode_return,
                final_episode_length,
                return_sum,
                length_sum,
                num_completed,
            ),
            _,
        ) = jax.lax.scan(accumulate_step, init, (rewards_ea, dones_ea))
        return (
            final_episode_return,
            final_episode_length,
            return_sum,
            length_sum,
            num_completed,
        )

    def update_step(carry, _):
        key, rollout_key, update_key = jax.random.split(carry.rng, 3)
        trajectory, last_obs_flat, final_env_state, final_env_keys = rollout_fn(
            carry.train_state,
            carry.env_state,
            carry.env_keys,
            rollout_key,
            carry.obs_flat,
        )
        common = {
            "observations": jnp.asarray(trajectory.observations, dtype=jnp.float32),
            "actions": jnp.asarray(trajectory.actions, dtype=jnp.int32),
            "log_probs": jnp.asarray(trajectory.log_probs, dtype=jnp.float32),
            "rewards": jnp.asarray(trajectory.rewards, dtype=jnp.float32),
            "dones": jnp.asarray(trajectory.dones, dtype=jnp.float32),
            "values": jnp.asarray(trajectory.values, dtype=jnp.float32),
        }
        if is_mappo:
            obs_2d = common["observations"].reshape(
                (-1, trajectory.observations.shape[-1])
            )
            central_seq = central_flat(obs_2d).reshape(
                (rollout_steps, num_envs * num_agents, -1)
            )
            batch = MAPPORolloutBatch(central_observations=central_seq, **common)
        else:
            batch = RolloutBatch(**common)
        last_values = value_fn(carry.train_state, last_obs_flat)

        if freeze_policy:
            next_train_state = carry.train_state
            ppo_metrics: dict[str, Any] = {}
        else:
            next_train_state, update_metrics = update_fn(
                carry.train_state, batch, last_values, update_key, config
            )
            ppo_metrics = {
                f"ppo/{metric_name}": val for metric_name, val in update_metrics.items()
            }

        rewards_ea = trajectory.rewards.reshape((rollout_steps, num_envs, num_agents))
        dones_ea = trajectory.dones.reshape((rollout_steps, num_envs, num_agents))
        (
            next_episode_return,
            next_episode_length,
            return_sum,
            length_sum,
            num_completed,
        ) = accumulate_episode_stats(
            carry.episode_return, carry.episode_length, rewards_ea, dones_ea
        )
        num_completed_f = num_completed.astype(jnp.float32)
        episode_return_mean = jnp.where(
            num_completed > 0, return_sum / (num_completed_f * num_agents), jnp.nan
        )
        episode_length_mean = jnp.where(
            num_completed > 0, length_sum.astype(jnp.float32) / num_completed_f, jnp.nan
        )
        out = {
            "rollout_mean_reward": jnp.mean(batch.rewards),
            "completed_episodes": num_completed,
            "episode_return_mean": episode_return_mean,
            "episode_length_mean": episode_length_mean,
        }
        out.update(ppo_metrics)
        next_carry = TrainOnEnvCarry(
            train_state=next_train_state,
            rng=key,
            env_state=final_env_state,
            env_keys=final_env_keys,
            obs_flat=last_obs_flat,
            episode_return=next_episode_return,
            episode_length=next_episode_length,
        )
        return next_carry, out

    obs_flat0 = jnp.asarray(observations, dtype=jnp.float32).reshape(
        (num_envs * num_agents, -1)
    )
    init_carry = TrainOnEnvCarry(
        train_state=train_state,
        rng=rng,
        env_state=adapter._state,
        env_keys=adapter._keys,
        obs_flat=obs_flat0,
        episode_return=jnp.asarray(adapter._episode_returns, dtype=jnp.float32),
        episode_length=jnp.asarray(adapter._episode_lengths, dtype=jnp.int32),
    )
    final_carry, stacked = jax.lax.scan(
        update_step, init_carry, None, length=num_updates
    )

    adapter._state = final_carry.env_state
    adapter._keys = final_carry.env_keys
    adapter._episode_returns = np.asarray(final_carry.episode_return, dtype=np.float32)
    adapter._episode_lengths = np.asarray(
        final_carry.episode_length, dtype=adapter._episode_lengths.dtype
    )

    final_observations = np.asarray(final_carry.obs_flat, dtype=np.float32).reshape(
        (num_envs, num_agents, -1)
    )
    return final_carry.train_state, final_observations, final_carry.rng, stacked


def collect_mappo_rollout(
    adapter: MeltingPotVectorAdapter,
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
    get_action_and_value = jax.jit(_mappo_get_action_and_value)
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
        actions, log_probs, values, entropies = get_action_and_value(
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


def _ippo_get_action_and_value(state, key, flat_obs):
    policy, values = state.apply_fn({"params": state.params}, flat_obs)
    actions = policy.sample(seed=key)
    return (
        actions.astype(jnp.int32),
        policy.log_prob(actions),
        values,
        policy.entropy(),
    )


def _mappo_get_action_and_value(state, key, flat_obs, flat_central_obs):
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


@functools.lru_cache(maxsize=None)
def make_mappo_get_action_and_value(num_envs: int, num_agents: int):
    """MAPPO ``get_action_and_value`` in ``adapter.rollout``'s 3-arg shape.

    Rebuilds centralized-critic obs from the joint obs. Cached per env geometry so the returned closure keeps a stable identity —
    ``adapter.rollout`` caches compiled programs by ``id(get_action_and_value)``.
    """

    def get_action_and_value(state, key, flat_obs):
        central = build_vector_central(
            flat_obs.reshape((num_envs, num_agents, -1)), jnp
        ).reshape((num_envs * num_agents, -1))
        return _mappo_get_action_and_value(state, key, flat_obs, central)

    return get_action_and_value


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


# TODO: Fit rollout and return summary statistics of fit.
