"""Helper utils for envs."""
from collections.abc import Iterable
from typing import Any
import jax

import jax.numpy as jnp
Agent = Any

def batchify(
    x: jax.Array, agents: dict[Agent, jax.Array], num_actors: int
) -> jax.Array:  # dict -> stacked array
    return jnp.stack([x[a] for a in agents]).reshape((num_actors, -1))


def unbatchify(
    x, agents: Iterable[Agent], num_envs: int, num_actors: int
) -> dict[Agent, jax.Array]:  # array -> dict
    return {a: x.reshape((num_actors, num_envs, -1))[i] for i, a in enumerate(agents)}
