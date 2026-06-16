"""Generalized advantage estimation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_values: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and value targets.

    Args:
      rewards: Array shaped [T, N].
      values: Value predictions shaped [T, N].
      dones: Float/bool terminal mask shaped [T, N], where 1 means terminal.
      last_values: Bootstrap values shaped [N].
      gamma: Discount.
      gae_lambda: GAE trace parameter.

    Returns:
      ``(advantages, targets)``, each shaped [T, N].
    """
    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values)
    dones = jnp.asarray(dones)
    last_values = jnp.asarray(last_values)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must share shape [T, N]")
    if last_values.shape != rewards.shape[1:]:
        raise ValueError("last_values must have shape [N]")

    def step(carry, transition):
        gae, next_value = carry
        reward_t, value_t, done_t = transition
        nonterminal = 1.0 - done_t.astype(jnp.float32)
        delta = reward_t + gamma * next_value * nonterminal - value_t
        gae = delta + gamma * gae_lambda * nonterminal * gae
        return (gae, value_t), gae

    (_, _), advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_values), last_values),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values
