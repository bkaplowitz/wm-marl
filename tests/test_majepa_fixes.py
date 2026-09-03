"""Focused invariants for independently switchable MA-JEPA fixes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import embodied.jax.outs as outs

from majepa.models.heads import apply_support_preserving_availability
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
