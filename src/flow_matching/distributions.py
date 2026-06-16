"""Small distribution helpers for the 2D flow-matching exercise."""

import jax
import jax.numpy as jnp
from flax import struct


class GaussianMixture2D(struct.PyTreeNode):
    """A uniform or weighted mixture of isotropic 2D Gaussian components."""

    means: jax.Array
    std: float
    weights: jax.Array

    @property
    def dim(self) -> int:
        """Return the data dimension."""
        return int(self.means.shape[1])

    @property
    def nmodes(self) -> int:
        """Return the number of mixture components."""
        return int(self.means.shape[0])


def make_symmetric_gmm_2d(
    nmodes: int, std: float, scale: float = 10.0
) -> GaussianMixture2D:
    """Place `nmodes` Gaussian means evenly on a radius-`scale` circle."""
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, nmodes + 1)[:nmodes]
    means = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * scale
    weights = jnp.ones((nmodes,)) / nmodes
    return GaussianMixture2D(means=means, std=std, weights=weights)


def sample_standard_normal(key: jax.Array, n: int, dim: int = 2) -> jax.Array:
    """Sample from the source distribution p0 = N(0, I)."""
    return jax.random.normal(key, shape=(n, dim))


def sample_gmm(key: jax.Array, gmm: GaussianMixture2D, n: int) -> jax.Array:
    """Sample `n` points from a 2D isotropic Gaussian mixture."""
    key_labels, key_noise = jax.random.split(key)
    component_label = jax.random.choice(
        key_labels, gmm.nmodes, shape=(n,), p=gmm.weights
    )
    means = gmm.means[component_label, :]  # (bs, 2)
    return means + gmm.std * jax.random.normal(key_noise, shape=(n, gmm.dim))
