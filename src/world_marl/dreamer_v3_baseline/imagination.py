from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class DreamerImaginedRollout:
    features: jnp.ndarray
    rewards: jnp.ndarray
    continues: jnp.ndarray
    values: jnp.ndarray


def open_loop_diagnostic(features: jnp.ndarray, horizon: int) -> DreamerImaginedRollout:
    clipped = features[:horizon]
    rewards = jnp.mean(clipped, axis=-1)
    continues = jnp.ones_like(rewards)
    values = jnp.cumsum(rewards[::-1], axis=0)[::-1]
    return DreamerImaginedRollout(
        features=clipped,
        rewards=rewards,
        continues=continues,
        values=values,
    )
