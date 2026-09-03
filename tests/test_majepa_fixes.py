"""Focused invariants for independently switchable MA-JEPA fixes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import embodied.jax.outs as outs

from majepa.models.heads import apply_support_preserving_availability
from majepa.training.learner import actor_entropy_coefficient
from majepa.training.objectives import normalized_policy_entropy


def test_actor_entropy_cosine_schedule_uses_environment_steps() -> None:
    coefficient = actor_entropy_coefficient(
        jnp.asarray([0, 20_000, 40_000, 50_000]),
        initial=1e-3,
        final=3e-4,
        decay_steps=40_000,
        schedule="cosine",
    )
    np.testing.assert_allclose(
        np.asarray(coefficient), [1e-3, 6.5e-4, 3e-4, 3e-4], rtol=1e-6
    )


def test_actor_entropy_normalizes_by_live_action_support() -> None:
    distribution = outs.Categorical(
        jnp.asarray([[0.0, 0.0, -1e30], [0.0, 0.0, 0.0]], jnp.float32)
    )
    normalized = normalized_policy_entropy(distribution, distribution.entropy())
    np.testing.assert_allclose(np.asarray(normalized), [1.0, 1.0], rtol=1e-6)

    sequence = outs.Categorical(jnp.zeros((2, 6, 4), jnp.float32))
    normalized_sequence = normalized_policy_entropy(sequence, sequence.entropy())[
        :, :-1
    ]
    assert normalized_sequence.shape == (2, 5)
    np.testing.assert_allclose(np.asarray(normalized_sequence), 1.0, rtol=1e-6)


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
