"""DreaMARL output heads with effective categorical uniform mixing."""

from typing import Callable

import embodied.jax.heads as upstream
import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import numpy as np


class Head(upstream.Head):
    """Pinned DreamerV3 head with categorical ``unimix`` wired through."""

    minstd: float = 1.0
    maxstd: float = 1.0
    unimix: float = 0.0
    bins: int = 255
    outscale: float = 1.0

    def categorical(self, x):
        assert self.space.discrete
        classes = np.asarray(self.space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        shape = (*self.space.shape, classes[0].item())
        logits = self.sub("logits", upstream.nets.Linear, shape, **self.kw)(x)
        output = outs.Categorical(logits, self.unimix)
        output.minent = 0
        output.maxent = np.log(logits.shape[-1])
        return output


class DictHead(upstream.DictHead):
    """Use the corrected categorical head for each dictionary output."""

    def __call__(self, x):
        outputs = {}
        for key, impl in self.outputs.items():
            outputs[key] = self.sub(
                key,
                Head,
                self.spaces[key],
                impl,
                **self.kw,
            )(x)
        return outputs


class MLPHead(upstream.MLPHead):
    """Parameter-name-compatible MLP head using the corrected dictionary head."""

    units: int = 1024
    layers: int = 5
    act: str = "silu"
    norm: str = "rms"
    bias: bool = True
    winit: str | Callable = upstream.nets.Initializer("trunc_normal")
    binit: str | Callable = upstream.nets.Initializer("zeros")

    def __init__(self, space, output, **hkw):
        shared = dict(bias=self.bias, winit=self.winit, binit=self.binit)
        mkw = dict(**shared, act=self.act, norm=self.norm)
        hkw = dict(**shared, **hkw)
        self.mlp = upstream.nets.MLP(
            self.layers,
            self.units,
            **mkw,
            name="mlp",
        )
        if isinstance(space, dict):
            self.head = DictHead(space, output, **hkw, name="head")
        else:
            self.head = Head(space, output, **hkw, name="head")


def apply_action_mask(distributions, mask, action_key):
    """Condition one categorical policy on a nonempty availability mask."""

    if mask is None:
        return distributions
    distribution = distributions[action_key]
    if not isinstance(distribution, outs.Categorical):
        raise TypeError("action masks require a categorical policy output")
    mask = jnp.asarray(mask, bool)
    if mask.shape != distribution.logits.shape:
        raise ValueError(
            f"action mask shape {mask.shape} does not match logits "
            f"{distribution.logits.shape}"
        )
    fallback = jax.nn.one_hot(
        jnp.argmax(distribution.logits, axis=-1),
        distribution.logits.shape[-1],
        dtype=bool,
    )
    mask = jnp.where(mask.any(axis=-1, keepdims=True), mask, fallback)
    masked = outs.Categorical(jnp.where(mask, distribution.logits, -1e30))
    for name in ("minent", "maxent"):
        if hasattr(distribution, name):
            setattr(masked, name, getattr(distribution, name))
    return dict(distributions, **{action_key: masked})


def apply_predicted_action_mask(
    distributions,
    availability_logits,
    action_key,
    *,
    probability_floor=1e-6,
):
    """Use uncertain future availability without creating hard support changes.

    Environment-provided masks can safely remove actions because collection and
    evaluation observe the same mask. During imagination, availability is itself
    predicted. A hard threshold can therefore make an action sampled during the
    rollout have zero probability when the policy is evaluated again for the
    actor loss. Weighting by a floored availability probability keeps that loss
    finite while still discouraging actions predicted to be unavailable.
    """

    distribution = distributions[action_key]
    if not isinstance(distribution, outs.Categorical):
        raise TypeError("action masks require a categorical policy output")
    availability_logits = jnp.asarray(availability_logits)
    if availability_logits.shape != distribution.logits.shape:
        raise ValueError(
            f"action availability shape {availability_logits.shape} does not "
            f"match logits {distribution.logits.shape}"
        )
    availability = jax.nn.sigmoid(availability_logits)
    availability = jnp.clip(availability, probability_floor, 1.0)
    masked = outs.Categorical(distribution.logits + jnp.log(availability))
    for name in ("minent", "maxent"):
        if hasattr(distribution, name):
            setattr(masked, name, getattr(distribution, name))
    return dict(distributions, **{action_key: masked})


__all__ = ["MLPHead", "apply_action_mask", "apply_predicted_action_mask"]
