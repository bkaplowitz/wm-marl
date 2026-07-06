"""Helpers for collecting vector-state batches for prefit world models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.training import (
    _ippo_get_action_and_value,
    _mappo_get_action_and_value,
    build_central_observations,
)
from world_marl.world_model import (
    VectorTransitionBatch,
    _apply_vector_policy,
    train_world_model_step,
)

TrainingAdapter = MeltingPotVectorAdapter | JaxMARLCoinGameVectorAdapter


@dataclass(frozen=True)
class TransitionCollectionStats:
    real_env_steps: int
    completed_episodes: int
    episode_return_mean: float | None
    episode_length_mean: float | None


def flatten_state_observations(observations: np.ndarray) -> np.ndarray:
    """Flatten local observations while preserving env and agent axes."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim < 3:
        raise ValueError("expected observations shaped [env, agent, ...]")
    return observations.reshape((observations.shape[0], observations.shape[1], -1))


def collect_random_transition_batch(
    adapter: TrainingAdapter,
    observations: np.ndarray,
    rng: np.random.Generator,
    *,
    rollout_steps: int,
) -> tuple[
    VectorTransitionBatch,
    np.ndarray,
    jnp.ndarray,
    TransitionCollectionStats,
]:
    """Collect vector-state transitions using adapter-sampled random actions."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")

    current_observations = observations
    rows = _TransitionRows()
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    for _ in range(rollout_steps):
        states = flatten_state_observations(current_observations)
        actions = adapter.sample_actions(rng)
        step = adapter.step(actions)
        rows.append(
            states=states,
            actions=actions,
            next_states=flatten_state_observations(step.observations),
            rewards=step.rewards,
            dones=step.dones,
        )
        completed_returns.extend(step.completed_returns)
        completed_lengths.extend(step.completed_lengths)
        current_observations = step.observations

    batch = rows.to_batch()
    return (
        batch,
        current_observations,
        batch.states,
        _collection_stats(
            real_env_steps=rollout_steps * adapter.num_envs,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
        ),
    )


def collect_policy_transition_batch(
    adapter: TrainingAdapter,
    train_state: TrainState,
    observations: np.ndarray,
    rng: jax.Array,
    *,
    rollout_steps: int,
    algorithm: str,
) -> tuple[
    VectorTransitionBatch,
    np.ndarray,
    jax.Array,
    jnp.ndarray,
    TransitionCollectionStats,
]:
    """Collect vector-state transitions using the current IPPO/MAPPO policy."""
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")

    current_observations = observations
    rows = _TransitionRows()
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    for _ in range(rollout_steps):
        states = flatten_state_observations(current_observations)
        flat_states = states.reshape((adapter.num_envs * adapter.num_agents, -1))
        rng, action_key = jax.random.split(rng)
        central_states = (
            jnp.asarray(
                build_central_observations(
                    states,
                    observation_mode="vector",
                ).reshape((adapter.num_envs * adapter.num_agents, -1))
            )
            if algorithm == "mappo"
            else None
        )
        policy, _ = _apply_vector_policy(
            train_state, jnp.asarray(flat_states), central_states
        )
        actions = np.asarray(policy.sample(seed=action_key), dtype=np.int32).reshape(
            (adapter.num_envs, adapter.num_agents)
        )
        step = adapter.step(actions)
        rows.append(
            states=states,
            actions=actions,
            next_states=flatten_state_observations(step.observations),
            rewards=step.rewards,
            dones=step.dones,
        )
        completed_returns.extend(step.completed_returns)
        completed_lengths.extend(step.completed_lengths)
        current_observations = step.observations

    batch = rows.to_batch()
    return (
        batch,
        current_observations,
        rng,
        batch.states,
        _collection_stats(
            real_env_steps=rollout_steps * adapter.num_envs,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
        ),
    )


def _collection_stats(
    *,
    real_env_steps: int,
    completed_returns: Sequence[tuple[float, ...]],
    completed_lengths: Sequence[int],
) -> TransitionCollectionStats:
    completed_array = (
        np.asarray(completed_returns, dtype=np.float32)
        if completed_returns
        else np.asarray([], dtype=np.float32)
    )
    return TransitionCollectionStats(
        real_env_steps=real_env_steps,
        completed_episodes=len(completed_returns),
        episode_return_mean=(
            float(completed_array.mean()) if completed_returns else None
        ),
        episode_length_mean=(
            float(np.mean(completed_lengths)) if completed_lengths else None
        ),
    )


def concatenate_transition_batches(
    batches: Sequence[VectorTransitionBatch],
) -> VectorTransitionBatch:
    """Concatenate non-empty transition batches along the batch dimension."""
    if not batches:
        raise ValueError("expected at least one transition batch")
    return VectorTransitionBatch(
        states=jnp.concatenate([batch.states for batch in batches], axis=0),
        actions=jnp.concatenate([batch.actions for batch in batches], axis=0),
        next_states=jnp.concatenate([batch.next_states for batch in batches], axis=0),
        rewards=jnp.concatenate([batch.rewards for batch in batches], axis=0),
        dones=jnp.concatenate([batch.dones for batch in batches], axis=0),
    )


def _transition_batch_from_scan(
    ys,
    last_obs_flat: jax.Array,
    *,
    num_envs: int,
    num_agents: int,
    rollout_steps: int,
) -> VectorTransitionBatch:
    """Repackage ``adapter.scan_rollout`` outputs into a ``VectorTransitionBatch``.

    ``scan_rollout`` stacks per-step arrays as ``[T, E*A, ...]`` and returns the
    post-rollout ``last_obs_flat[E*A, d]``. The host collectors store one
    ``[E, A, d]`` row per step and concatenate over ``T`` (env-major within each
    step), so the folded layout is ``[T*E, A, ...]``; ``next_states[t]`` is
    ``obs[t+1]`` with ``last_obs_flat`` closing the final step.
    """
    obs_seq, action_seq, _log_probs, _values, _entropies, reward_seq, done_seq = ys
    state_dim = obs_seq.shape[-1]

    def fold(array: jax.Array) -> jax.Array:
        reshaped = array.reshape(
            (rollout_steps, num_envs, num_agents) + array.shape[2:]
        )
        return reshaped.reshape(
            (rollout_steps * num_envs, num_agents) + array.shape[2:]
        )

    states = jnp.asarray(obs_seq, dtype=jnp.float32).reshape(
        (rollout_steps, num_envs, num_agents, state_dim)
    )
    last = jnp.asarray(last_obs_flat, dtype=jnp.float32).reshape(
        (1, num_envs, num_agents, state_dim)
    )
    next_states = jnp.concatenate([states[1:], last], axis=0)
    return VectorTransitionBatch(
        states=states.reshape((rollout_steps * num_envs, num_agents, state_dim)),
        actions=fold(jnp.asarray(action_seq, dtype=jnp.int32)),
        next_states=next_states.reshape(
            (rollout_steps * num_envs, num_agents, state_dim)
        ),
        rewards=fold(jnp.asarray(reward_seq, dtype=jnp.float32)),
        dones=fold(jnp.asarray(done_seq, dtype=jnp.float32)),
    )


def _replay_scan_episode_bookkeeping(
    adapter,
    ys,
    rollout_steps: int,
) -> tuple[list[tuple[float, ...]], list[int]]:
    """Advance the adapter's episode accumulators over a scan batch (host-side).

    ``scan_rollout`` advances ``adapter._state``/``_keys`` but not the episode
    return/length accumulators, so -- exactly as ``train_real_scan`` does --
    replay them over the collected ``(reward, done)`` sequences and write the
    partial-episode state back, so a chained collector (random -> policy) and the
    following training loop resume from the right boundary.
    """
    _obs, _actions, _log_probs, _values, _entropies, reward_seq, done_seq = ys
    num_envs = adapter.num_envs
    num_agents = adapter.num_agents
    rewards_ea = np.asarray(reward_seq, dtype=np.float32).reshape(
        (rollout_steps, num_envs, num_agents)
    )
    dones_ea = np.asarray(done_seq).reshape((rollout_steps, num_envs, num_agents))
    ep_returns = adapter._episode_returns.copy()
    ep_lengths = adapter._episode_lengths.copy()
    completed_returns: list[tuple[float, ...]] = []
    completed_lengths: list[int] = []
    for t in range(rollout_steps):
        ep_returns += rewards_ea[t]
        ep_lengths += 1
        for env_index in np.flatnonzero(dones_ea[t].all(axis=1)):
            completed_returns.append(tuple(float(x) for x in ep_returns[env_index]))
            completed_lengths.append(int(ep_lengths[env_index]))
            ep_returns[env_index] = 0.0
            ep_lengths[env_index] = 0
    adapter._episode_returns = ep_returns
    adapter._episode_lengths = ep_lengths
    return completed_returns, completed_lengths


def collect_random_transition_batch_scan(
    adapter,
    observations: np.ndarray,
    key: jax.Array,
    *,
    rollout_steps: int,
) -> tuple[VectorTransitionBatch, np.ndarray, jnp.ndarray, TransitionCollectionStats]:
    """On-device twin of ``collect_random_transition_batch`` via ``lax.scan``.

    Reuses the adapter's proven ``scan_rollout`` with a uniform on-device action
    sampler, so the whole random-action rollout runs on the accelerator instead of
    a host Python loop. Uniform sampling matches ``sample_actions`` (uniform over
    the action set), but the PRNG source differs (jax vs numpy ``Generator``), so
    this is distribution-equivalent, not bit-for-bit, with the host version.
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    action_dim = adapter.action_dim

    def random_infer(_state, action_key, flat_obs):
        num_rows = flat_obs.shape[0]
        actions = jax.random.randint(action_key, (num_rows,), 0, action_dim)
        zeros = jnp.zeros((num_rows,), dtype=jnp.float32)
        return actions.astype(jnp.int32), zeros, zeros, zeros

    ys, last_obs_flat = adapter.scan_rollout(
        random_infer,
        None,
        rollout_steps,
        policy_key=key,
        observations=observations,
    )
    batch = _transition_batch_from_scan(
        ys,
        last_obs_flat,
        num_envs=adapter.num_envs,
        num_agents=adapter.num_agents,
        rollout_steps=rollout_steps,
    )
    completed_returns, completed_lengths = _replay_scan_episode_bookkeeping(
        adapter, ys, rollout_steps
    )
    last_observations = np.asarray(last_obs_flat, dtype=np.float32).reshape(
        (adapter.num_envs, adapter.num_agents, -1)
    )
    return (
        batch,
        last_observations,
        batch.states,
        _collection_stats(
            real_env_steps=rollout_steps * adapter.num_envs,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
        ),
    )


def collect_policy_transition_batch_scan(
    adapter,
    train_state: TrainState,
    observations: np.ndarray,
    key: jax.Array,
    *,
    rollout_steps: int,
    algorithm: str,
) -> tuple[
    VectorTransitionBatch,
    np.ndarray,
    jax.Array,
    jnp.ndarray,
    TransitionCollectionStats,
]:
    """On-device twin of ``collect_policy_transition_batch`` via ``lax.scan``.

    Reuses ``scan_rollout`` with the same inference the host loop applies --
    ``_ippo_get_action_and_value``, or its MAPPO wrapper that rebuilds central
    observations from the joint obs -- and mirrors its
    ``policy-key-then-env-key`` split order, so the collected batch matches the
    host loop bit-for-bit (integer actions exact, continuous tensors to float
    tolerance).
    """
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be >= 1")
    if algorithm not in {"ippo", "mappo"}:
        raise ValueError(f"unsupported algorithm {algorithm!r}")
    get_action_and_value = (
        _mappo_get_action_and_value(adapter.num_envs, adapter.num_agents)
        if algorithm == "mappo"
        else _ippo_get_action_and_value
    )

    ys, last_obs_flat = adapter.scan_rollout(
        get_action_and_value,
        train_state,
        rollout_steps,
        policy_key=key,
        observations=observations,
    )
    batch = _transition_batch_from_scan(
        ys,
        last_obs_flat,
        num_envs=adapter.num_envs,
        num_agents=adapter.num_agents,
        rollout_steps=rollout_steps,
    )
    completed_returns, completed_lengths = _replay_scan_episode_bookkeeping(
        adapter, ys, rollout_steps
    )
    last_observations = np.asarray(last_obs_flat, dtype=np.float32).reshape(
        (adapter.num_envs, adapter.num_agents, -1)
    )
    return (
        batch,
        last_observations,
        jax.random.fold_in(key, rollout_steps),
        batch.states,
        _collection_stats(
            real_env_steps=rollout_steps * adapter.num_envs,
            completed_returns=completed_returns,
            completed_lengths=completed_lengths,
        ),
    )


def fit_world_model_steps(
    model_state: TrainState,
    rng: jax.Array,
    batch: VectorTransitionBatch,
    config,
    *,
    steps: int,
) -> tuple[TrainState, jax.Array, jnp.ndarray, jnp.ndarray]:
    """Run full-batch world-model fitting steps.

    Returns the updated state, the advanced rng, the final step loss, and the
    per-step loss history (length ``steps``) for plotting fit convergence.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    model_state, rng, loss_history = _fit_world_model_updates(
        model_state, rng, batch, config, steps=steps
    )
    return model_state, rng, loss_history[-1], loss_history


@partial(jax.jit, static_argnames=("config", "steps"))
def _fit_world_model_updates(
    model_state: TrainState,
    rng: jax.Array,
    batch: VectorTransitionBatch,
    config,
    *,
    steps: int,
) -> tuple[TrainState, jax.Array, jnp.ndarray]:
    """Fused full-batch fitting: one ``lax.scan`` step per gradient update.

    The carry is ``(model_state, rng)`` and ``scan`` stacks each step's loss into
    the returned history.
    """

    def update(carry, _):
        state, rng = carry
        rng, fit_key = jax.random.split(rng)
        state, loss = train_world_model_step(state, fit_key, batch, config)
        return (state, rng), loss

    (model_state, rng), loss_history = jax.lax.scan(
        update, (model_state, rng), xs=None, length=steps
    )
    return model_state, rng, loss_history


def sample_initial_states(
    states: jnp.ndarray,
    rng: jax.Array,
    *,
    num_envs: int,
) -> jnp.ndarray:
    """Sample model-rollout initial states from a collected state pool."""
    if states.shape[0] < 1:
        raise ValueError("expected at least one collected state")
    indices = jax.random.randint(rng, (num_envs,), minval=0, maxval=states.shape[0])
    return states[indices]


class _TransitionRows:
    def __init__(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.next_states: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []

    def append(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.states.append(states)
        self.actions.append(np.asarray(actions, dtype=np.int32))
        self.next_states.append(next_states)
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))

    def to_batch(self) -> VectorTransitionBatch:
        states = np.concatenate(self.states, axis=0)
        return VectorTransitionBatch(
            states=jnp.asarray(states, dtype=jnp.float32),
            actions=jnp.asarray(np.concatenate(self.actions, axis=0), dtype=jnp.int32),
            next_states=jnp.asarray(
                np.concatenate(self.next_states, axis=0),
                dtype=jnp.float32,
            ),
            rewards=jnp.asarray(
                np.concatenate(self.rewards, axis=0),
                dtype=jnp.float32,
            ),
            dones=jnp.asarray(np.concatenate(self.dones, axis=0), dtype=jnp.float32),
        )
