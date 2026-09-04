"""Real-transition value targets, separate from the imagined PPO actor batch."""

import jax
import jax.numpy as jnp


def replay_lambda_return(
    reward, is_first, is_last, is_terminal, present, bootstrap, *, discount, lam
):
    """State-aligned replay lambda returns with explicit episode boundaries.

    The transition leaving t receives reward[t+1]. Terminal arrivals retain
    that reward and stop bootstrapping. Truncations bootstrap once, but never
    continue the trace into a reset episode. Individual death does not end a
    shared-reward team's return; ``present`` denotes membership, not liveness.
    """

    values = (reward, is_first, is_last, is_terminal, present, bootstrap)
    shapes = {tuple(jnp.shape(value)) for value in values}
    if len(shapes) != 1 or reward.ndim != 2 or reward.shape[1] < 2:
        raise ValueError("replay value inputs must have matching [B,T>=2] shapes")
    if not 0.0 <= float(discount) <= 1.0 or not 0.0 <= float(lam) <= 1.0:
        raise ValueError("replay value discount and lambda must be in [0,1]")
    reward, bootstrap = [
        jnp.asarray(value, jnp.float32) for value in (reward, bootstrap)
    ]
    is_first, is_last, is_terminal, present = [
        jnp.asarray(value, bool) for value in (is_first, is_last, is_terminal, present)
    ]
    valid = present[:, :-1] & present[:, 1:] & ~is_last[:, :-1] & ~is_first[:, 1:]
    live = float(discount) * (~is_terminal[:, 1:]).astype(jnp.float32)
    trace = float(lam) * (~is_last[:, 1:]).astype(jnp.float32)

    def step(future, inputs):
        next_reward, next_bootstrap, gamma, mix, selected, current_bootstrap = inputs
        target = next_reward + gamma * ((1.0 - mix) * next_bootstrap + mix * future)
        # An invalid transition cannot bridge two episodes or absent slots.
        target = jnp.where(selected, target, current_bootstrap)
        return target, target

    _, returns = jax.lax.scan(
        step,
        bootstrap[:, -1],
        tuple(
            value.T
            for value in (
                reward[:, 1:],
                bootstrap[:, 1:],
                live,
                trace,
                valid,
                bootstrap[:, :-1],
            )
        ),
        reverse=True,
    )
    return jax.lax.stop_gradient(returns.T), jax.lax.stop_gradient(valid)
