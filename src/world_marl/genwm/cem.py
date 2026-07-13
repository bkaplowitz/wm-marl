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
from flax.training.train_state import TrainState

from world_marl.genwm.tokenizer import QuantileTokenizer
from world_marl.genwm.world_model import (
    GenWMConfig,
    action_features,
    genwm_predict_next,
)


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


def discounted_return(
    rewards: jax.Array,
    continue_probs: jax.Array,
    gamma: float,
) -> jax.Array:
    """Compute discounted return with survival weighting.

    Args:
        rewards: shape ``(S, N, H)`` — samples × envs × horizon.
        continue_probs: same shape; ``c_t = P(episode continues past step t)``.
        gamma: discount factor.

    Returns:
        Discounted return ``(S, N)``:
        ``sum_t  gamma^t * (prod_{s<t} c_s) * r_t``
        where the product at ``t=0`` is the empty product (= 1).
    """
    H = rewards.shape[2]
    time_indices = jnp.arange(H, dtype=rewards.dtype)
    gamma_t = gamma**time_indices  # (H,)

    cum = jnp.cumprod(continue_probs, axis=2)  # (S, N, H): [c0, c0*c1, ...]
    ones = jnp.ones_like(cum[..., :1])
    shifted = jnp.concatenate([ones, cum[..., :-1]], axis=2)  # empty product at t=0

    return jnp.sum(rewards * shifted * gamma_t, axis=2)  # (S, N)


def make_genwm_plan_fn(
    wm_state: TrainState,
    head_state: TrainState,
    start_observations: jax.Array,
    obs_tokenizer: QuantileTokenizer,
    action_tokenizer: QuantileTokenizer | None,
    config: GenWMConfig,
    cem_config: CEMConfig,
    gamma: float,
):
    """Return a jitted shooting cost function for CEM-MPC.

    The returned ``cost_fn(candidates, key) -> costs`` rolls ``candidates``
    through the world model starting from ``start_observations``, accumulates
    predicted rewards and continue probabilities from the fitted head, and
    returns negative discounted return (shape ``(S, N)``; lower = better for
    the minimising CEM solver).

    ``start_observations`` is captured in the closure; callers recreate this
    closure each time the planning context changes (e.g. each call to
    ``cem_solve``). JAX reuses the compiled kernel as long as shapes are the
    same.
    """

    action_low = cem_config.action_low
    action_high = cem_config.action_high

    @jax.jit
    def _shoot(candidates: jax.Array, key: jax.Array) -> jax.Array:
        # candidates: (S, N, H, A)
        S, N, H, _ = candidates.shape
        clipped = jnp.clip(candidates, action_low, action_high)
        # Permute to (H, S, N, A) for scan over horizon
        actions_by_t = clipped.transpose(2, 0, 1, 3)

        def step(carry, t_actions):
            observations, step_key = carry
            # observations: (S, N, obs_dim); t_actions: (S, N, A)
            step_key, model_key = jax.random.split(step_key)
            flat_obs = observations.reshape(S * N, -1)
            flat_act = t_actions.reshape(S * N, -1)
            next_flat_obs = genwm_predict_next(
                wm_state,
                model_key,
                flat_obs,
                flat_act,
                obs_tokenizer,
                action_tokenizer,
                config,
            )
            act_feats = action_features(flat_act, config)
            reward_flat, continue_logit_flat = head_state.apply_fn(
                {"params": head_state.params}, flat_obs, act_feats
            )
            reward = reward_flat.reshape(S, N)
            continue_prob = jax.nn.sigmoid(continue_logit_flat).reshape(S, N)
            next_obs = next_flat_obs.reshape(S, N, -1)
            return (next_obs, step_key), (reward, continue_prob)

        init_obs = jnp.broadcast_to(
            start_observations[None, :, :], (S, N, start_observations.shape[-1])
        )
        (_, _), (rewards, continue_probs) = jax.lax.scan(
            step, (init_obs, key), actions_by_t
        )
        # rewards: (H, S, N) → (S, N, H)
        rewards = rewards.transpose(1, 2, 0)
        continue_probs = continue_probs.transpose(1, 2, 0)
        return -discounted_return(rewards, continue_probs, gamma)

    return _shoot
