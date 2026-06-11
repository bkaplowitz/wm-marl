from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from world_marl.algs.gae import compute_gae


def test_gae_shape_and_values():
  advantages, targets = compute_gae(
    rewards=jnp.asarray([[1.0], [1.0]]),
    values=jnp.asarray([[0.0], [0.0]]),
    dones=jnp.asarray([[0.0], [1.0]]),
    last_values=jnp.asarray([0.0]),
    gamma=1.0,
    gae_lambda=1.0,
  )
  assert advantages.shape == (2, 1)
  np.testing.assert_allclose(advantages, [[2.0], [1.0]], rtol=1e-6)
  np.testing.assert_allclose(targets, [[2.0], [1.0]], rtol=1e-6)
