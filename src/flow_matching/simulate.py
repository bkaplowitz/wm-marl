"""Simulation helpers for learned and analytic vector fields."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from flow_matching.distributions import sample_standard_normal
from flow_matching.paths import mixture_path_rates


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


def sample_conditioned_flow(
    apply_fn: Any,
    params: Any,
    key: jax.Array,
    cond_vars: jax.Array,
    *,
    dim: int,
    steps: int,
) -> jax.Array:
    """Euler-integrate the conditioned vector field from N(0, I) to t=1.

    Returns the terminal sample ``x1`` of shape ``(cond_vars.shape[0], dim)``.
    The conditioning is passed straight to the model at every drift evaluation.
    """
    x0 = sample_standard_normal(key, cond_vars.shape[0], dim)
    ts = jnp.linspace(0.0, 1.0, steps + 1)

    def drift(xt: jax.Array, t: jax.Array) -> jax.Array:
        tt = jnp.full((xt.shape[0], 1), t)
        return apply_fn({"params": params}, xt, tt, cond_vars)

    return euler_integrate(drift, x0, ts)[-1]


def sample_conditioned_discrete_flow(
    apply_fn: Any,
    params: Any,
    key: jax.Array,
    cond_vars: jax.Array,
    *,
    num_factors: int,
    num_categories: int,
    steps: int,
) -> jax.Array:
    """Sample tokens from the factorized CTMC (discrete.md Alg 7, Euler/tau-leaping).

    The discrete twin of :func:`sample_conditioned_flow`: start from the uniform
    source ``X_0 ~ Uniform(V)^d`` and take ``steps`` Euler/tau-leaping updates of
    the per-factor jump process. Each step one-hots the current tokens into the
    ``(B, d*V)`` model input, reads per-factor logits, forms the denoising
    posterior, converts it to jump rates, and samples the next tokens from the
    Euler transition probabilities.

    Rates are evaluated only at the LEFT endpoints ``t_i = i/steps`` (i = 0..n-1),
    mirroring :func:`euler_integrate`. There ``h * q = posterior / (steps - i) <=
    1`` so the self-probability ``1 - sum_{v != x} h*q`` is automatically valid
    (>= posterior(x) >= 0) with no clamping, and the final step (``i = n-1``)
    degenerates to sampling straight from the posterior. Returns integer tokens
    ``X_1`` of shape ``(B, d)``; the caller owns any one-hot/layout conversion.
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
        xt_onehot = jax.nn.one_hot(xt, num_categories).reshape(batch, -1)
        tt = jnp.full((batch, 1), t)
        logits = apply_fn({"params": params}, xt_onehot, tt, cond_vars)
        logits = logits.reshape(batch, num_factors, num_categories)
        posterior = jax.nn.softmax(logits, axis=-1)  # p_{1|t}(.|x_t), (B, d, V)
        rates = mixture_path_rates(posterior, t)  # q_j(v), (B, d, V)
        current = jax.nn.one_hot(xt, num_categories)  # (B, d, V)
        off_diag = h * rates * (1.0 - current)  # h*q for v != x, 0 at v = x
        self_prob = 1.0 - jnp.sum(off_diag, axis=-1, keepdims=True)
        probs = off_diag + current * self_prob  # Euler transition probs (B, d, V)
        next_x = jax.random.categorical(sample_key, jnp.log(probs), axis=-1)
        return (next_x, step_key), next_x

    (x1, _), _ = jax.lax.scan(ctmc_step, (x0, key), ts)
    return x1
