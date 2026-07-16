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


class CEMPlanner:
    """Stateful receding-horizon MPC actor backed by the CEM solver.

    ``make_plan_fn`` must be a callable with the keyword signature::

        make_plan_fn(
            wm_state, head_state, start_observations,
            obs_tokenizer, action_tokenizer, gamma,
        ) -> cost_fn

    where the returned ``cost_fn(candidates, key) -> costs`` matches the
    interface expected by ``cem_solve``.  The caller is responsible for
    partially-binding ``config`` and ``cem_config`` before passing the
    callable here (e.g. via ``functools.partial`` or a local closure).
    """

    def __init__(
        self, make_plan_fn, key: jax.Array, cem_config: CEMConfig, action_dim: int = 1
    ):
        self._make_plan_fn = make_plan_fn
        self._key = key
        self._cem_config = cem_config
        self._plan_horizon = cem_config.horizon
        self._receding = cem_config.receding_horizon
        self._action_dim = action_dim

        # Buffers are allocated lazily on first act() call, once N is known.
        self._action_buffer: jax.Array | None = None
        self._N: int | None = None

        self._step_index: int = 0
        self._needs_replan: bool = True
        self._topk_costs: jax.Array | None = None
        self._solve_seconds: float = 0.0

    def _ensure_buffers(self, N: int) -> None:
        if self._N != N:
            self._N = N
            self._action_buffer = jnp.zeros(
                (N, self._plan_horizon, self._action_dim), dtype=jnp.float32
            )
            self._topk_costs = jnp.full((N,), jnp.inf, dtype=jnp.float32)

    def reset(self) -> None:
        """Reset episode state; next act() call will trigger a replan."""
        self._step_index = 0
        self._needs_replan = True
        if self._action_buffer is not None:
            self._action_buffer = jnp.zeros_like(self._action_buffer)
        if self._topk_costs is not None:
            self._topk_costs = jnp.full_like(self._topk_costs, jnp.inf)

    def _warm_start_shift(self) -> None:
        """Drop the first action from the buffer and append a zero column.

        After executing ``receding_horizon`` steps the old plan is stale; the
        next replan will sample fresh candidates from the CEM prior.  The
        zeroed tail action lets the solver's init_std exploration dominate the
        extended horizon.
        """
        assert self._action_buffer is not None
        shifted = self._action_buffer[:, 1:, :]  # (N, H-1, A)
        zeros = jnp.zeros(
            (self._N, 1, self._action_dim), dtype=self._action_buffer.dtype
        )
        self._action_buffer = jnp.concatenate([shifted, zeros], axis=1)  # (N, H, A)

    def act(
        self,
        flat_obs,
        key: jax.Array,
        wm_state: TrainState,
        head_state: TrainState,
        start_obs_for_horizon: jax.Array,
        current_obs_tokenizer,
        current_action_tokenizer,
        gamma: float,
    ) -> jax.Array:
        """Execute one environment step, replanning when needed.

        Args:
            flat_obs: current flat observation (unused in the CEM path;
                included for cross-actor API consistency).
            key: JAX RNG key (consumed; caller should split before passing).
            wm_state: current world-model TrainState.
            head_state: current reward/continue head TrainState.
            start_obs_for_horizon: shape ``(N, obs_dim)`` — initial
                observations from which to unroll the plan.
            current_obs_tokenizer: QuantileTokenizer for observations.
            current_action_tokenizer: QuantileTokenizer (or None) for actions.
            gamma: discount factor.

        Returns:
            Planned actions for this step, shape ``(N, action_dim)``.
        """
        del flat_obs  # not used in the CEM path

        N = int(start_obs_for_horizon.shape[0])
        self._ensure_buffers(N)

        if self._needs_replan:
            cost_fn = self._make_plan_fn(
                wm_state=wm_state,
                head_state=head_state,
                start_observations=start_obs_for_horizon,
                obs_tokenizer=current_obs_tokenizer,
                action_tokenizer=current_action_tokenizer,
                gamma=gamma,
            )

            self._key, solve_key = jax.random.split(self._key)
            mean_init = jnp.zeros(
                (N, self._plan_horizon, self._action_dim), dtype=jnp.float32
            )

            t0 = time.time()
            final_mean, _std, topk_cost = cem_solve(
                cost_fn, solve_key, mean_init, self._cem_config
            )
            self._solve_seconds += time.time() - t0

            self._action_buffer = jnp.clip(
                final_mean,
                self._cem_config.action_low,
                self._cem_config.action_high,
            )
            self._topk_costs = topk_cost
            self._step_index = 0
            self._needs_replan = False

        actions = self._action_buffer[:, self._step_index, :]  # (N, A)
        self._step_index += 1

        if self._step_index >= self._receding:
            self._warm_start_shift()
            self._needs_replan = True

        return actions
