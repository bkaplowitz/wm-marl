"""Helpers for collecting vector-state batches for prefit world models."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.training import build_central_observations
from world_marl.world_model import (
    VectorTransitionBatch,
    _apply_vector_policy,
    train_world_model_step,
)


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
    adapter: MeltingPotVectorAdapter,
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
    adapter: MeltingPotVectorAdapter,
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
