"""Simulation helpers for learned and analytic vector fields."""

from collections.abc import Callable

import jax
import jax.numpy as jnp


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
