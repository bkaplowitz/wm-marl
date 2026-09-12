"""Critic EMA supporting the full convex update-rate interval."""

import math

import embodied.jax


class CriticTarget(embodied.jax.SlowModel):
    def __init__(self, model, *, source, rate=1.0, every=1):
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Critic target rate must lie in [0, 1]")
        # The external helper restricts its constructor to rate<.5 or rate=1;
        # its actual convex averaging supports every rate in [0, 1]. Preserve
        # its parameter paths, counter and update implementation unchanged.
        super().__init__(model, source=source, rate=1.0, every=every)
        self.rate = rate
