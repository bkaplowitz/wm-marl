"""Jasmine diffusion sampling helpers.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, paths
``jasmine/models/genie.py`` (``GenieDiffusion.sample``) and
``jasmine/baselines/diffusion/sample_diffusion.py``. Integration changes: the
context-level calculation is a public pure helper and runtime loops are Linen
compatible ``jax.lax.scan`` operations.
"""

import jax.numpy as jnp


def snapped_context_signal_level(
    diffusion_steps: int,
    context_corruption: float,
) -> jnp.ndarray:
    target_signal = 1.0 - context_corruption
    step = jnp.argmin(
        jnp.abs(jnp.arange(diffusion_steps) / diffusion_steps - target_signal)
    )
    return step / diffusion_steps
