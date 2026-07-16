"""Jafar MaskGIT and autoregressive sampling.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, paths ``genie.py`` and
``sample.py``. Integration changes: sampler schedules are public pure helpers;
runtime refinement and frame loops are implemented with ``jax.lax.scan``.
"""

import jax
import jax.numpy as jnp


def unmasked_ratio(step: jax.Array, steps: int) -> jax.Array:
    return jnp.cos(jnp.pi * (step + 1) / (steps * 2))
