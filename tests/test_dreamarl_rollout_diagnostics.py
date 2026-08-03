import jax.numpy as jnp

from world_marl.dreamarl.agent import _pearson_correlation


def test_pearson_correlation_tracks_direction_and_constant_inputs():
    values = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.isclose(_pearson_correlation(values, values), 1.0)
    assert jnp.isclose(_pearson_correlation(values, -values), -1.0)
    assert jnp.isclose(_pearson_correlation(values, jnp.ones_like(values)), 0.0)
