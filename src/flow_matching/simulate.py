"""Simulation helpers for learned and analytic vector fields."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from flow_matching.distributions import sample_standard_normal
from flow_matching.paths import factorized_jump_rates


def euler_integrate(
    drift_fn: Callable[[jax.Array, jax.Array], jax.Array],
    x0: jax.Array,
    ts: jax.Array,
) -> jax.Array:
    """Integrate an ODE with Euler steps and record every timestep."""

    def _euler_step(
        xt: jax.Array, dt_t: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:  # (carry, x) -> (carry, y)
        """Takes one euler update step."""
        dt, t = dt_t
        next_xt = xt + drift_fn(xt, t) * dt
        return next_xt, next_xt

    dts = ts[1:] - ts[:-1]  # Δt
    _, xs = jax.lax.scan(_euler_step, init=x0, xs=(dts, ts[:-1]))
    return jnp.concatenate((x0[None, ...], xs))  # prepend the initial x0


def sample_marginal_flow_model(
    apply_fn: Any,
    params: Any,
    key: jax.Array,
    cond_vars: jax.Array,
    *,
    dim: int,
    steps: int,
) -> jax.Array:
    """Euler-integrate the marginal vector field from N(0, I) to t=1.

    Returns the terminal sample ``x1`` of shape ``(cond_vars.shape[0], dim)``.
    The conditioning is passed straight to the model at every drift evaluation.
    """
    x0 = sample_standard_normal(key, cond_vars.shape[0], dim)
    ts = jnp.linspace(0.0, 1.0, steps + 1)

    def drift(xt: jax.Array, t: jax.Array) -> jax.Array:
        tt = jnp.full((xt.shape[0], 1), t)
        return apply_fn({"params": params}, xt, tt, cond_vars)

    return euler_integrate(drift, x0, ts)[-1]


def sample_marginal_discrete_flow_model(
    apply_fn: Any,
    params: Any,
    key: jax.Array,
    cond_vars: jax.Array,
    *,
    num_factors: int,
    num_categories: int,
    steps: int,
) -> jax.Array:
    """Sample tokens from the simulated CTMC, where the model is used to predict the data.

    The discrete twin of :func:`sample_marginal_flow_model`: start from the uniform
    source ``X_0 ~ Uniform(V)^d`` and take ``steps`` Euler/tau-leaping updates of
    the per-factor jump process. Each step feeds the current integer tokens to the
    tokenized denoiser, reads per-factor logits, estimates Q_t(y|x), and samples
    the next tokens (Algorithm 7: the rate network is evaluated every step).
    """
    batch = cond_vars.shape[0]
    h = 1.0 / steps
    key, init_key = jax.random.split(key)
    x0 = jax.random.randint(init_key, (batch, num_factors), 0, num_categories)
    ts = jnp.arange(steps) / steps  # left endpoints: 0, 1/n, ..., (n-1)/n

    def ctmc_step(
        carry: tuple[jax.Array, jax.Array], t: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        xt, step_key = carry
        step_key, sample_key = jax.random.split(step_key)
        tt = jnp.full((batch, 1), t)
        logits = apply_fn({"params": params}, xt, tt, cond_vars)  # (B, d, V)
        # Predicted probability of x1, aka drawing from data.
        p_x1 = jax.nn.softmax(logits, axis=-1)  # p_{1|t}(.|x_t), (B, d, V)
        rates = factorized_jump_rates(p_x1, t)  # q_j(v), (B, d, V)
        # This is δ_{x=y} needed in algorithm.
        current = jax.nn.one_hot(xt, num_categories)  # (B, d, V)
        # h * q_j(y) for y ≠ x ( 1- δ_{y=x})
        off_diag = h * rates * (1.0 - current)
        # 1 - h ∑_{y ≠x} (q_j (y)) for y = x ( δ_{y=x})
        self_prob = 1.0 - jnp.sum(off_diag, axis=-1, keepdims=True)
        # p(x_{t+h}|z,t) = 1_(y=x) (1-h sum_{y' \in S \{x}}q_j(y'))
        probs = current * self_prob + off_diag
        # Sample x_{t+h} ∼ Categorical(p_{x_{t+h}|z,t})
        next_x = jax.random.categorical(sample_key, jnp.log(probs), axis=-1)
        return (next_x, step_key), next_x

    (x1, _), _ = jax.lax.scan(ctmc_step, (x0, key), ts)
    return x1
