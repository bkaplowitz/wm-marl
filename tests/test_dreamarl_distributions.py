from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.dreamarl.config import DistributionConfig
from world_marl.dreamarl.distributions import (
    symexp,
    symlog,
    two_hot,
    two_hot_loss,
    two_hot_mean,
)


def test_symlog_roundtrip_and_two_hot_normalization() -> None:
    values = jnp.array([-1e4, -10.0, 0.0, 2.0, 1e4])
    np.testing.assert_allclose(symexp(symlog(values)), values, rtol=1e-5)
    labels = two_hot(values, DistributionConfig())
    np.testing.assert_allclose(jnp.sum(labels, axis=-1), 1.0, atol=1e-6)
    assert bool(jnp.all(labels >= 0.0))


def test_distributional_loss_and_mean_have_finite_gradients() -> None:
    config = DistributionConfig(bins=31, low=-10.0, high=10.0)
    logits = jnp.zeros((5, config.bins))
    targets = jnp.array([-100.0, -1.0, 0.0, 2.0, 100.0])
    gradients = jax.grad(
        lambda current: jnp.mean(two_hot_loss(current, targets, config))
    )(logits)
    assert bool(jnp.all(jnp.isfinite(gradients)))
    assert two_hot_mean(logits, config).shape == targets.shape
