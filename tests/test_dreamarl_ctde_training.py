"""Decision-critical checks for bounded two-step CTDE replay training."""

import jax
import jax.numpy as jnp
import numpy as np

from dreamarl.training.ctde import (
    detach_self_feed,
    gather_anchors,
    predicted_controllable_alive,
    sample_two_step_anchors,
    two_step_anchor_mask,
    two_step_objective,
)


def test_two_step_anchors_respect_resets_without_dropping_death_targets() -> None:
    first = jnp.zeros((1, 7), bool).at[0, 4].set(True)
    alive = jnp.ones((1, 7, 2), bool)
    alive = alive.at[0, 1:, 0].set(False)
    valid = two_step_anchor_mask(first, alive)
    np.testing.assert_array_equal(valid, [[True, True, False, False, True]])

    terminal_death = jnp.zeros((1, 7, 2), bool).at[:, 0].set(True)
    death_valid = two_step_anchor_mask(jnp.zeros_like(first), terminal_death)
    np.testing.assert_array_equal(
        death_valid, [[True, False, False, False, False]]
    )

    anchors = sample_two_step_anchors(jax.random.key(1), valid, count=5)
    assert int(anchors.valid.sum()) == 3
    selected = {
        int(time)
        for time, selected_valid in zip(anchors.time, anchors.valid)
        if bool(selected_valid)
    }
    assert selected == {0, 1, 4}

    replay = jnp.arange(7)[None]
    np.testing.assert_array_equal(
        np.asarray(gather_anchors(replay, anchors, offset=2)[anchors.valid]),
        np.asarray(anchors.time[anchors.valid] + 2),
    )


def test_two_step_self_feed_has_last_step_gradient_only() -> None:
    def objective(first_input, first_keys, last_weight):
        first_prediction = {
            "embedding": 2.0 * first_input,
            "joint_carry": {"keys": 3.0 * first_keys},
        }
        detached = detach_self_feed(first_prediction)
        return jnp.sum(
            (
                detached["embedding"]
                + detached["joint_carry"]["keys"]
            )
            * last_weight
        )

    first = jnp.asarray([1.0, -2.0])
    keys = jnp.asarray([0.5, 0.25])
    weight = jnp.asarray([3.0, 4.0])
    first_grad, key_grad, last_grad = jax.grad(objective, (0, 1, 2))(
        first, keys, weight
    )
    np.testing.assert_array_equal(first_grad, jnp.zeros_like(first))
    np.testing.assert_array_equal(key_grad, jnp.zeros_like(keys))
    np.testing.assert_array_equal(last_grad, 2.0 * first + 3.0 * keys)

    next_alive = predicted_controllable_alive(
        jnp.asarray([[True, True, False]]),
        jnp.asarray([[True, False, True]]),
        jnp.asarray([[0.9, 0.9, 0.9]]),
    )
    np.testing.assert_array_equal(next_alive, [[True, False, False]])


def test_two_step_objective_matches_sample_mean_and_stops_ema_target() -> None:
    anchor_valid = jnp.asarray([[True, True, False]])
    anchors = sample_two_step_anchors(jax.random.key(2), anchor_valid, count=2)
    # Losses are scattered at their source indices on a full replay grid. This
    # is the exact alignment consumed by the learner's full-length validity.
    destination_valid = jnp.ones((1, 5, 2), bool)
    supervision_valid = jnp.asarray([[True, False], [True, True]])
    prediction = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ]
    )
    target = jnp.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    reward_loss = jnp.asarray([[2.0, 100.0], [4.0, 6.0]])
    alive_loss = jnp.asarray([[1.0, 3.0], [5.0, 7.0]])

    def losses(predicted, ema_target):
        losses, _ = two_step_objective(
            predicted,
            ema_target,
            {"reward": reward_loss, "alive": alive_loss},
            anchors,
            supervision_valid,
            destination_valid,
            auxiliary_valid={"alive": jnp.ones_like(supervision_valid)},
        )
        weight = destination_valid.astype(jnp.float32)
        return {
            name: (value * weight).sum() / weight.sum()
            for name, value in losses.items()
        }

    expected = (
        reward_loss * supervision_valid.astype(jnp.float32)
    ).sum() / supervision_valid.sum()
    np.testing.assert_allclose(losses(prediction, target)["reward"], expected)
    np.testing.assert_allclose(
        losses(prediction, target)["alive"], alive_loss.mean()
    )

    def embedding_loss(predicted, ema_target):
        return losses(predicted, ema_target)["embedding"]

    prediction_grad, target_grad = jax.grad(embedding_loss, (0, 1))(
        prediction, target
    )
    assert float(jnp.linalg.norm(prediction_grad)) > 0.0
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target))
