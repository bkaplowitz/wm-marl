from __future__ import annotations

import jax
import jax.numpy as jnp

from dreamarl.training.churn import categorical_forward_kl, relative_churn_scale


class Categorical:
    def __init__(self, logits):
        self.logits = logits


def test_policy_churn_stops_the_reference_and_preserves_current_gradient() -> None:
    def loss(reference_logits, current_logits):
        reference = {"action": Categorical(reference_logits)}
        current = {"action": Categorical(current_logits)}
        return categorical_forward_kl(reference, current, "action").mean()

    reference = jnp.array([[1.0, -0.5, 0.25]], jnp.float32)
    current = jnp.array([[-0.25, 0.5, 0.0]], jnp.float32)
    reference_grad, current_grad = jax.grad(loss, argnums=(0, 1))(
        reference,
        current,
    )

    assert jnp.array_equal(reference_grad, jnp.zeros_like(reference_grad))
    assert jnp.linalg.norm(current_grad) > 0


def test_relative_churn_scale_has_the_requested_loss_fraction() -> None:
    policy_magnitude = jnp.asarray(2.0, jnp.float32)
    churn = jnp.asarray(0.25, jnp.float32)
    scale = relative_churn_scale(
        policy_magnitude,
        churn,
        beta=0.02,
        maximum=1.0,
        epsilon=0.0,
    )

    assert jnp.isclose(scale * churn, 0.02 * policy_magnitude)
