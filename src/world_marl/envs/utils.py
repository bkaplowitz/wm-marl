"""Helper utils for envs."""

import jax.numpy as jnp


def batchify(x, agents, num_actors):  # dict -> stacked array
    return jnp.stack([x[a] for a in agents]).reshape((num_actors, -1))


def unbatchify(x, agents, num_envs, num_actors):  # array -> dict
    return {a: x.reshape((num_actors, num_envs, -1))[i] for i, a in enumerate(agents)}
