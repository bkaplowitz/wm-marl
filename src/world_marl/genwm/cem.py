"""LeWM-style CEM-MPC planning for the generative world-model arms.

The solver mirrors ``stable-worldmodel``'s ``CEMSolver`` (the planner LeWM
evaluates with, arXiv 2603.19312): diagonal-Gaussian action sequences, the
first candidate forced to the current mean, top-k selection, and a
plain top-k mean/std refit with no momentum or std floor. LeWM's benchmarks
are goal-conditioned so its cost is terminal goal-latent MSE; this harness is
reward-based, so the shooting cost is negative predicted return through the
same reward/continue head the PPO arms imagine with.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CEMConfig:
    num_samples: int = 300
    topk: int = 30
    num_iters: int = 30
    horizon: int = 5
    receding_horizon: int = 5
    init_std: float = 1.0
    action_low: float = -1.0
    action_high: float = 1.0


def sample_candidates(
    key: jax.Array, mean: jax.Array, std: jax.Array, num_samples: int
) -> jax.Array:
    """Sample ``(num_samples, *mean.shape)`` candidates, candidate 0 = mean."""
    noise = jax.random.normal(key, (num_samples, *mean.shape), dtype=mean.dtype)
    return (mean[None] + std[None] * noise).at[0].set(mean)


def cem_solve(cost_fn, key: jax.Array, mean_init: jax.Array, config: CEMConfig):
    """Run the reference CEM loop; returns final ``(mean, std, topk_cost)``.

    ``cost_fn(candidates (S, N, H, A), key) -> costs (S, N)``, lower is
    better. The key argument lets stochastic world models resample per
    iteration.
    """
    std_init = jnp.full_like(mean_init, config.init_std)

    def iteration(carry, it_key):
        mean, std = carry
        sample_key, cost_key = jax.random.split(it_key)
        candidates = sample_candidates(sample_key, mean, std, config.num_samples)
        costs = cost_fn(candidates, cost_key)  # (S, N)
        neg_costs = -costs.T  # (N, S); top_k keeps largest
        _, topk_idx = jax.lax.top_k(neg_costs, config.topk)  # (N, K)
        per_env = candidates.transpose(1, 0, 2, 3)  # (N, S, H, A)
        topk_candidates = jnp.take_along_axis(
            per_env, topk_idx[:, :, None, None], axis=1
        )  # (N, K, H, A)
        topk_cost = jnp.take_along_axis(costs.T, topk_idx, axis=1).mean(axis=1)
        new_mean = topk_candidates.mean(axis=1)
        new_std = topk_candidates.std(axis=1, ddof=1)
        return (new_mean, new_std), topk_cost

    keys = jax.random.split(key, config.num_iters)
    (mean, std), topk_costs = jax.lax.scan(iteration, (mean_init, std_init), keys)
    return mean, std, topk_costs[-1]
