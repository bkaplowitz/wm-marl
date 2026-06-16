"""Simulation helpers for learned and analytic vector fields."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from flow_matching.distributions import sample_standard_normal


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
