"""Replay layout conversion for latent-action world-model backends.

This repository-specific module converts the shared time-major replay contract
to the batch-major HWC RGB layout used by both source-derived arms. Transition
targets deliberately use the next observation, reward, and continuation, and
episode-boundary transitions are excluded.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from world_marl.world_model_foundation.replay import JaxSequenceBatch


class BackendSequenceBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    continues: jax.Array
    is_first: jax.Array
    is_terminal: jax.Array
    is_last: jax.Array
    valid_transitions: jax.Array


class TransitionBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    rewards: jax.Array
    continues: jax.Array


def _batch_major(values: jax.Array) -> jax.Array:
    return jnp.swapaxes(values, 0, 1)


def to_backend_sequence(batch: JaxSequenceBatch) -> BackendSequenceBatch:
    observations = jnp.asarray(batch.observations)
    if observations.ndim != 5 or observations.shape[-1] != 3:
        raise ValueError(
            "latent-action world models require time-major HWC RGB observations"
        )
    if observations.shape[0] < 2:
        raise ValueError("at least two time steps are required")

    observations = _batch_major(observations)
    actions = _batch_major(jnp.asarray(batch.actions))
    rewards = _batch_major(jnp.asarray(batch.rewards, dtype=jnp.float32))
    continues = _batch_major(jnp.asarray(batch.continues, dtype=jnp.float32))
    is_first = _batch_major(jnp.asarray(batch.is_first, dtype=bool))
    is_terminal = _batch_major(jnp.asarray(batch.is_terminal, dtype=bool))
    is_last = _batch_major(jnp.asarray(batch.is_last, dtype=bool))
    return BackendSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=is_terminal,
        is_last=is_last,
        valid_transitions=~is_first[:, 1:],
    )


def pair_valid_transitions(batch: BackendSequenceBatch) -> TransitionBatch:
    valid = batch.valid_transitions.reshape(-1)
    current_observations = batch.observations[:, :-1].reshape(
        -1, *batch.observations.shape[2:]
    )
    next_observations = batch.observations[:, 1:].reshape(
        -1, *batch.observations.shape[2:]
    )
    actions = batch.actions[:, :-1].reshape(-1, *batch.actions.shape[2:])
    rewards = batch.rewards[:, 1:].reshape(-1, *batch.rewards.shape[2:])
    continues = batch.continues[:, 1:].reshape(-1, *batch.continues.shape[2:])
    return TransitionBatch(
        observations=current_observations[valid],
        actions=actions[valid],
        next_observations=next_observations[valid],
        rewards=rewards[valid],
        continues=continues[valid],
    )
