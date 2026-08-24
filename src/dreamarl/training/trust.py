"""Replay-supported trust regions for the executable actor."""

import math

import jax
import jax.numpy as jnp
import ninjax as nj


f32 = jnp.float32
sg = jax.lax.stop_gradient


def categorical_forward_kl(reference_logits, current_logits):
    """Return ``KL(stop_gradient(reference) || current)`` per state."""

    reference_logits = sg(reference_logits).astype(f32)
    current_logits = current_logits.astype(f32)
    reference_logprob = jax.nn.log_softmax(reference_logits, axis=-1)
    current_logprob = jax.nn.log_softmax(current_logits, axis=-1)
    reference_prob = jnp.exp(reference_logprob)
    divergence = jnp.sum(
        reference_prob * (reference_logprob - current_logprob), axis=-1
    )
    return jnp.maximum(divergence, 0.0)


def masked_average(value, mask):
    """Average over decision states with at least two legal actions."""

    weight = mask.astype(f32)
    return (value * weight).sum() / jnp.maximum(weight.sum(), 1.0)


class AdaptiveKLCoefficient(nj.Module):
    """Checkpointed dual variable for a target behavioral KL."""

    target: float = 0.005
    rate: float = 0.01
    initial: float = 1.0
    minimum: float = 1e-4
    maximum: float = 1e3
    ema_rate: float = 0.05

    def __init__(self, **kwargs):
        del kwargs
        if self.target <= 0:
            raise ValueError("actor trust target must be positive")
        if not 0 < self.rate <= 1:
            raise ValueError("actor trust dual rate must be in (0, 1]")
        if not 0 < self.ema_rate <= 1:
            raise ValueError("actor trust EMA rate must be in (0, 1]")
        if not 0 < self.minimum <= self.initial <= self.maximum:
            raise ValueError("actor trust coefficient bounds are inconsistent")
        self.log_value = nj.Variable(
            jnp.array,
            math.log(self.initial),
            f32,
            name="log_value",
        )
        self.kl_ema = nj.Variable(
            jnp.array,
            self.target,
            f32,
            name="kl_ema",
        )

    def value(self):
        return jnp.exp(self.log_value.read())

    def average(self):
        return self.kl_ema.read()

    def update(self, divergence):
        divergence = sg(divergence.astype(f32))
        average = (
            (1.0 - self.ema_rate) * self.kl_ema.read()
            + self.ema_rate * divergence
        )
        self.kl_ema.write(average)
        violation = jnp.clip(average / self.target - 1.0, -1.0, 1.0)
        lower = math.log(self.minimum)
        upper = math.log(self.maximum)
        self.log_value.write(
            jnp.clip(self.log_value.read() + self.rate * violation, lower, upper)
        )


__all__ = [
    "AdaptiveKLCoefficient",
    "categorical_forward_kl",
    "masked_average",
]
