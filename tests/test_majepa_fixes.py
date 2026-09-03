"""Focused invariants for independently switchable MA-JEPA fixes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import embodied.jax.outs as outs

from majepa.models.heads import apply_support_preserving_availability
from majepa.training.ctde import (
    alive_weighted_team_logits,
    alive_weighted_team_signal,
)


def test_dead_world_head_rows_cannot_change_team_signal() -> None:
    alive = jnp.asarray([[[True, False, True], [False, False, False]]])
    value = jnp.asarray([[[1.0, 100.0, 3.0], [7.0, 8.0, 9.0]]])
    changed = value.at[0, 0, 1].set(-1e6)

    expected = jnp.asarray([[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]])
    np.testing.assert_allclose(alive_weighted_team_signal(value, alive), expected)
    np.testing.assert_allclose(alive_weighted_team_signal(changed, alive), expected)


def test_dead_critic_rows_cannot_change_team_distribution() -> None:
    alive = jnp.asarray([[[True, False, True]]])
    logits = jnp.asarray([[[[1.0, 3.0], [100.0, -100.0], [3.0, 1.0]]]])
    changed = logits.at[0, 0, 1].set(jnp.asarray([-1e6, 1e6]))
    expected = jnp.asarray([[[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]])

    np.testing.assert_allclose(alive_weighted_team_logits(logits, alive), expected)
    np.testing.assert_allclose(alive_weighted_team_logits(changed, alive), expected)


def test_support_preserving_availability_keeps_every_live_action() -> None:
    policy = {"action": outs.Categorical(jnp.zeros((2, 4), jnp.float32))}
    availability = jnp.asarray(
        [[-100.0, -5.0, 0.0, 5.0], [-100.0, 100.0, 100.0, 100.0]]
    )
    selected = apply_support_preserving_availability(
        policy,
        availability,
        jnp.asarray([True, False]),
        "action",
        probability_floor=0.05,
    )["action"]

    assert bool((selected.logits[0] > -1e20).all())
    np.testing.assert_allclose(
        np.asarray(jnp.exp(selected.logits[0, 0])), 0.05, atol=1e-6
    )
    np.testing.assert_allclose(
        jnp.asarray(jax_softmax(selected.logits[1])), [1.0, 0.0, 0.0, 0.0]
    )


def jax_softmax(logits):
    shifted = logits - logits.max()
    probability = jnp.exp(shifted)
    return probability / probability.sum()
