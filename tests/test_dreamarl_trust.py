import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.training.trust import (
    AdaptiveKLCoefficient,
    categorical_forward_kl,
    masked_average,
)


def test_forward_kl_stops_reference_and_excludes_forced_actions() -> None:
    reference = jnp.asarray([[2.0, 0.0], [0.0, -1e30]])
    current = jnp.asarray([[0.0, 2.0], [0.0, -1e30]])
    decision = jnp.asarray([True, False])

    def objective(old, new):
        return masked_average(categorical_forward_kl(old, new), decision)

    old_grad, new_grad = jax.grad(objective, (0, 1))(reference, current)
    np.testing.assert_array_equal(old_grad, jnp.zeros_like(old_grad))
    assert float(jnp.linalg.norm(new_grad[0])) > 0.0
    np.testing.assert_array_equal(new_grad[1], jnp.zeros_like(new_grad[1]))


def test_adaptive_coefficient_tracks_target_and_is_checkpointed() -> None:
    controller = AdaptiveKLCoefficient(
        target=0.005,
        rate=0.1,
        initial=1.0,
        minimum=1e-4,
        maximum=1e3,
        ema_rate=1.0,
        name="trust",
    )

    def update(divergence):
        before = controller.value()
        controller.update(divergence)
        return before, controller.value(), controller.average()

    state = nj.init(lambda: update(jnp.asarray(0.005)))({}, seed=1)
    state, at_target = nj.pure(lambda: update(jnp.asarray(0.005)))(state, seed=2)
    state, above = nj.pure(lambda: update(jnp.asarray(0.02)))(state, seed=3)
    state, below = nj.pure(lambda: update(jnp.asarray(0.0)))(state, seed=4)

    np.testing.assert_allclose(at_target[0], at_target[1])
    assert float(above[1]) > float(above[0])
    assert float(below[1]) < float(below[0])
    assert "trust/log_value/value" in state
    assert "trust/kl_ema/value" in state
