"""Small array helpers shared by MA-JEPA runtime components."""

import jax
import jax.numpy as jnp
import ninjax as nj

f32 = jnp.float32
i32 = jnp.int32


def sg(xs, skip=False):
    return xs if skip else jax.lax.stop_gradient(xs)


def sample(xs):
    return jax.tree.map(lambda x: x.sample(nj.seed()), xs)


def predict(xs):
    return jax.tree.map(lambda x: x.pred(), xs)


def prefix(xs, name):
    return {f"{name}/{key}": value for key, value in xs.items()}


def concat(xs, axis):
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis), *xs)


def masked_mean(value, valid, *, alignment="tail"):
    """Average a per-transition loss over valid local agent transitions."""

    if alignment == "tail":
        weight = valid[:, -value.shape[1] :]
    elif alignment == "replay_value":
        start = valid.shape[1] - value.shape[1] - 1
        weight = valid[:, start : start + value.shape[1]]
    elif alignment == "exact":
        if valid.shape != value.shape:
            raise ValueError(
                f"validity {valid.shape} does not match loss {value.shape}"
            )
        weight = valid
    else:
        raise ValueError(f"unknown validity alignment: {alignment!r}")
    weight = weight.astype(jnp.float32)
    return (value * weight).mean() / jnp.maximum(weight.mean(), 1e-8)
