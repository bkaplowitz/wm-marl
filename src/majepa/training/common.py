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
