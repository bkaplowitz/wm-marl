"""Tests for the LeWM-style CEM-MPC planner (world_marl.genwm.cem)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.genwm import CEMConfig, cem_solve, sample_candidates


def test_sample_candidates_forces_mean_first():
    key = jax.random.PRNGKey(0)
    mean = jnp.arange(2 * 3 * 2, dtype=jnp.float32).reshape(2, 3, 2)
    std = jnp.ones_like(mean)
    candidates = sample_candidates(key, mean, std, num_samples=5)
    assert candidates.shape == (5, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(candidates[0]), np.asarray(mean))
    assert not np.allclose(np.asarray(candidates[1]), np.asarray(mean))


def test_cem_solve_converges_on_quadratic_cost():
    config = CEMConfig(num_samples=64, num_elites=8, num_iters=20, horizon=3)
    target = jnp.asarray(np.linspace(-0.5, 0.5, 2 * 3 * 2), dtype=jnp.float32).reshape(
        2, 3, 2
    )

    def cost_fn(candidates, key):
        del key
        return jnp.sum((candidates - target[None]) ** 2, axis=(2, 3))

    mean_init = jnp.zeros((2, 3, 2), dtype=jnp.float32)
    mean, std, elite_cost = cem_solve(cost_fn, jax.random.PRNGKey(1), mean_init, config)
    assert mean.shape == (2, 3, 2)
    assert std.shape == (2, 3, 2)
    assert elite_cost.shape == (2,)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(target), atol=0.05)
    assert float(elite_cost.mean()) < 0.01


def test_cem_solve_batches_envs_independently():
    config = CEMConfig(num_samples=64, num_elites=8, num_iters=20, horizon=1)
    targets = jnp.asarray([[[0.7]], [[-0.7]]], dtype=jnp.float32)  # (2, 1, 1)

    def cost_fn(candidates, key):
        del key
        return jnp.sum((candidates - targets[None]) ** 2, axis=(2, 3))

    mean, _, _ = cem_solve(cost_fn, jax.random.PRNGKey(2), jnp.zeros((2, 1, 1)), config)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(targets), atol=0.05)
