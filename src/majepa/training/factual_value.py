"""Bounded off-policy value targets on recorded team transitions."""

import jax
import jax.numpy as jnp


def joint_action_logratio(current, behavior, controllable, team):
    """Recorded action a_t probabilities, folded as [B*A,T].

    All acting agents contribute to the probability of a joint action. Dead or
    absent slots contribute an identity ratio, not an extra policy factor.
    """
    if current.shape != behavior.shape or current.shape != controllable.shape:
        raise ValueError("action log probabilities and acting mask must match")
    difference = jnp.where(controllable, current - behavior, 0.0)
    joint = team.unfold_sequence(difference).sum(-1)
    return jax.lax.stop_gradient(team.broadcast_sequence(joint))


def factual_vtrace_return(
    reward,
    first,
    last,
    terminal,
    present,
    value,
    logratio,
    *,
    discount,
    lam,
    rho_clip,
    c_clip,
):
    """V-trace with a lambda trace, real rewards, and factual state values.

    Action/logratio[t] leaves state t and receives reward[t+1]. Truncations
    bootstrap from their final observation exactly once. A terminal arrival
    retains its reward. Neither resets nor missing roster slots bridge traces.
    This is a bounded replay correction, not a claim of unbiased returns with
    a changing latent state representation.
    """
    arrays = (reward, first, last, terminal, present, value, logratio)
    if len({x.shape for x in arrays}) != 1 or reward.ndim != 2 or reward.shape[1] < 2:
        raise ValueError("factual value inputs must share [B,T>=2] shape")
    if not 0 <= discount <= 1 or not 0 <= lam <= 1:
        raise ValueError("discount and lambda must be in [0,1]")
    if not 0 < c_clip <= rho_clip:
        raise ValueError("require 0 < c_clip <= rho_clip")
    reward, value, logratio = [
        jnp.asarray(x, jnp.float32) for x in (reward, value, logratio)
    ]
    first, last, terminal, present = [
        jnp.asarray(x, bool) for x in (first, last, terminal, present)
    ]
    valid = present[:, :-1] & present[:, 1:] & ~last[:, :-1] & ~first[:, 1:]
    # Clip in log space so a large joint ratio cannot overflow before clipping.
    rho = jnp.exp(jnp.minimum(logratio[:, :-1], jnp.log(rho_clip)))
    c = jnp.exp(jnp.minimum(logratio[:, :-1], jnp.log(c_clip)))
    gamma = discount * (~terminal[:, 1:]).astype(jnp.float32)
    delta = rho * (reward[:, 1:] + gamma * value[:, 1:] - value[:, :-1])
    trace = gamma * lam * c * (~last[:, 1:])

    def step(correction, inputs):
        td, coefficient, selected = inputs
        correction = jnp.where(selected, td + coefficient * correction, 0.0)
        return correction, correction

    _, correction = jax.lax.scan(
        step,
        jnp.zeros_like(value[:, -1]),
        (delta.T, trace.T, valid.T),
        reverse=True,
    )
    targets = value[:, :-1] + correction.T
    count = jnp.maximum(valid.sum(), 1)
    weights = jnp.where(valid, rho, 0.0)
    metrics = {
        "rho_mean": weights.sum() / count,
        "rho_clipped_fraction": (valid & (logratio[:, :-1] > jnp.log(rho_clip))).sum()
        / count,
        "relative_ess": weights.sum() ** 2
        / jnp.maximum(count * (weights**2).sum(), 1e-8),
        "target_mean": jnp.where(valid, targets, 0.0).sum() / count,
    }
    return jax.tree.map(jax.lax.stop_gradient, (targets, valid, metrics))
