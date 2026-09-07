import jax
import jax.numpy as jnp
import numpy as np

from majepa.training.ctde import imagined_action_mask


def test_mask_sampling_frequency_independence_and_boundary_rules():
    probability = jnp.broadcast_to(jnp.array([0., 1., .49, .9]), (20000, 4))
    alive = jnp.ones(20000, bool)
    threshold = imagined_action_mask(probability, alive)
    assert not np.asarray(threshold[:, 2]).any()
    sampled = imagined_action_mask(probability, alive, jax.random.key(73))
    frequency = np.asarray(sampled).mean(0)
    np.testing.assert_allclose(frequency, [0., 1., .49, .9], atol=.015)
    assert abs(float(np.corrcoef(np.asarray(sampled)[:, 2:4].T)[0, 1])) < .03
    np.testing.assert_array_equal(
        imagined_action_mask(jnp.zeros((2, 4)), jnp.array([True, False]), jax.random.key(2)),
        [[True, False, False, False]] * 2,
    )
    np.testing.assert_array_equal(
        imagined_action_mask(jnp.ones((1, 4)), jnp.array([False]), jax.random.key(2)),
        [[True, False, False, False]],
    )
