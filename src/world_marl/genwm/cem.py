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

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
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
    rewards: jax.Array, continue_probs: jax.Array, gamma: float
) -> jax.Array:
    """``sum_t gamma^t * prod_{s<t} c_s * r_t`` over the trailing axis."""
    ones = jnp.ones_like(continue_probs[..., :1])
    survival = jnp.cumprod(
        jnp.concatenate([ones, continue_probs[..., :-1]], axis=-1), axis=-1
    )
    discounts = gamma ** jnp.arange(rewards.shape[-1], dtype=rewards.dtype)
    return jnp.sum(discounts * survival * rewards, axis=-1)


def make_genwm_plan_fn(config: GenWMConfig, cem_config: CEMConfig, gamma: float):
    """Build a jitted CEM plan function over a fitted genwm world model.

    The cost is the negative continue-weighted discounted predicted return —
    the reward-based analogue of LeWM's terminal goal cost (this harness has
    rewards, not goal observations). Candidate actions are clipped to the CEM
    bounds before entering the model, matching ``imagined_rollout``.

    Built once per run; ``wm_state``/``head_state``/tokenizers/observations/
    ``mean_init``/``key`` are traced arguments so the compiled kernel is
    reused across replans instead of retraced per call.
    """

    def plan_fn(
        wm_state: TrainState,
        head_state: TrainState,
        obs_tokenizer,
        action_tokenizer: QuantileTokenizer | None,
        observations: jax.Array,
        mean_init: jax.Array,
        key: jax.Array,
    ):
        num_envs = observations.shape[0]

        def cost_fn(candidates: jax.Array, cost_key: jax.Array) -> jax.Array:
            num_samples = candidates.shape[0]
            batch = num_samples * num_envs
            obs0 = jnp.broadcast_to(
                observations[None], (num_samples, num_envs, observations.shape[-1])
            ).reshape(batch, -1)
            actions = jnp.clip(
                candidates, cem_config.action_low, cem_config.action_high
            ).reshape(batch, cem_config.horizon, -1)
            action_seq = actions.transpose(1, 0, 2)  # (H, B, A)

            def step(carry, act_t):
                obs_t, step_key = carry
                step_key, model_key = jax.random.split(step_key)
                reward, continue_logit = head_state.apply_fn(
                    {"params": head_state.params},
                    obs_t,
                    action_features(act_t, config),
                )
                next_obs = genwm_predict_next(
                    wm_state,
                    model_key,
                    obs_t,
                    act_t,
                    obs_tokenizer,
                    action_tokenizer,
                    config,
                )
                return (next_obs, step_key), (reward, jax.nn.sigmoid(continue_logit))

            (_, _), (rewards, continues) = jax.lax.scan(
                step, (obs0, cost_key), action_seq
            )
            returns = discounted_return(rewards.T, continues.T, gamma)  # (B,)
            return -returns.reshape(num_samples, num_envs)

        mean, _, topk_cost = cem_solve(cost_fn, key, mean_init, cem_config)
        return mean, topk_cost

    return jax.jit(plan_fn)


class CEMPlanner:
    """Receding-horizon CEM actor: one jitted solve per replan, buffered acts.

    Fixed-cadence replanning like LeWM's MPC loop: plan ``horizon`` actions,
    execute ``receding_horizon`` of them, replan. When ``receding_horizon <
    horizon`` the next solve warm-starts from the unexecuted tail of the
    previous plan (zero-padded), mirroring the reference's init_action path.
    """

    def __init__(
        self,
        plan_fn,
        *,
        wm_state,
        head_state,
        obs_tokenizer,
        action_tokenizer,
        cem_config: CEMConfig,
        num_envs: int,
        action_dim: int,
        key: jax.Array,
    ) -> None:
        self._plan_fn = plan_fn
        self._wm_state = wm_state
        self._head_state = head_state
        self._obs_tokenizer = obs_tokenizer
        self._action_tokenizer = action_tokenizer
        self._cem_config = cem_config
        self._key = key
        self._mean_shape = (num_envs, cem_config.horizon, action_dim)
        self.solve_seconds: list[float] = []
        self.topk_costs: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._plan: np.ndarray | None = None
        self._offset = 0
        self._next_mean = np.zeros(self._mean_shape, dtype=np.float32)

    def act(self, flat_obs: np.ndarray) -> np.ndarray:
        cem_config = self._cem_config
        if self._plan is None or self._offset >= cem_config.receding_horizon:
            self._key, solve_key = jax.random.split(self._key)
            started = time.perf_counter()
            mean, topk_cost = self._plan_fn(
                self._wm_state,
                self._head_state,
                self._obs_tokenizer,
                self._action_tokenizer,
                jnp.asarray(flat_obs, dtype=jnp.float32),
                jnp.asarray(self._next_mean),
                solve_key,
            )
            mean = np.asarray(jax.block_until_ready(mean), dtype=np.float32)
            self.solve_seconds.append(time.perf_counter() - started)
            self.topk_costs.append(float(np.mean(np.asarray(topk_cost))))
            self._plan = np.clip(mean, cem_config.action_low, cem_config.action_high)
            self._offset = 0
            shifted = np.zeros(self._mean_shape, dtype=np.float32)
            remaining = cem_config.horizon - cem_config.receding_horizon
            if remaining > 0:
                shifted[:, :remaining] = mean[:, cem_config.receding_horizon :]
            self._next_mean = shifted
        actions = self._plan[:, self._offset]
        self._offset += 1
        return actions
