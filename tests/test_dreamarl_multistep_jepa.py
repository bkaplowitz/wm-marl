"""Decision-critical tests for direct action-conditioned multi-step JEPA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.models.multistep_jepa import ActionConditionedMultiStepJEPA
from dreamarl.training.multistep_jepa import (
    action_window_validity,
    aligned_action_windows,
    direct_multistep_objective,
    same_focal_legal_action_interventions,
)


def test_action_windows_preserve_agent_identity_and_transition_alignment() -> None:
    batch, length, agents, classes = 1, 10, 2, 4
    transition = jnp.arange(length - 1, dtype=jnp.int32)[None, :, None]
    identity = jnp.arange(agents, dtype=jnp.int32)[None, None, :]
    actions = (transition + identity) % classes
    action_mask = jnp.ones((batch, length - 1, agents, classes), bool)
    present = jnp.ones((batch, length, agents), bool)
    alive = jnp.ones_like(present)
    first = jnp.zeros((batch, length), bool)

    windows, valid = aligned_action_windows(
        actions,
        action_mask,
        present,
        alive,
        first,
        action_low=0,
        max_horizon=8,
    )

    assert windows.shape == (1, 2, 2, 8)
    np.testing.assert_array_equal(windows[0, 0, 0], np.arange(8) % classes)
    np.testing.assert_array_equal(windows[0, 0, 1], (np.arange(8) + 1) % classes)
    np.testing.assert_array_equal(windows[0, 1, 0], np.arange(1, 9) % classes)
    for horizon in range(1, 9):
        np.testing.assert_array_equal(valid[horizon], np.ones((1, 2, 2), bool))


def test_action_window_masks_crossed_resets_illegal_actions_and_source_death() -> None:
    length, agents, classes = 10, 3, 4
    actions = jnp.ones((1, length - 1, agents), jnp.int32)
    action_mask = jnp.ones((1, length - 1, agents, classes), bool)
    # Root 0 / agent 0 becomes invalid at H2 because its second action is illegal.
    action_mask = action_mask.at[0, 1, 0, 1].set(False)
    present = jnp.ones((1, length, agents), bool)
    alive = jnp.ones_like(present)
    # Death exactly at the H1 target remains supervised, but makes H2 invalid.
    alive = alive.at[0, 1, 1].set(False)
    # A reset at observation 4 invalidates every prefix crossing that boundary.
    first = jnp.zeros((1, length), bool).at[0, 4].set(True)

    _, valid = aligned_action_windows(
        actions,
        action_mask,
        present,
        alive,
        first,
        action_low=0,
        max_horizon=8,
    )

    assert bool(valid[1][0, 0, 0])
    assert not bool(valid[2][0, 0, 0])
    assert bool(valid[1][0, 0, 1])
    assert not bool(valid[2][0, 0, 1])
    assert bool(valid[2][0, 1, 2])
    assert not bool(valid[4][0, 0, 2])


def test_same_focal_intervention_changes_only_last_legal_action() -> None:
    batch, roots, agents, horizon, classes = 1, 2, 3, 8, 5
    windows = jnp.zeros((batch, roots, agents, horizon), jnp.int32)
    masks = jnp.zeros((batch, roots + horizon - 1, agents, classes), bool)
    masks = masks.at[..., 0].set(True)
    masks = masks.at[..., 3].set(True)
    changed, distinct = same_focal_legal_action_interventions(
        windows,
        masks,
        action_low=0,
        horizons=(1, 2, 4, 8),
    )

    np.testing.assert_array_equal(changed[1], windows)
    assert not bool(distinct[1].any())
    for selected in (2, 4, 8):
        expected = windows.at[..., selected - 1].set(3)
        np.testing.assert_array_equal(changed[selected], expected)
        assert bool(distinct[selected].all())


def test_same_focal_intervention_requires_an_alternative_and_revalidates() -> None:
    length, agents, classes = 9, 2, 4
    actions = jnp.ones((1, length - 1, agents), jnp.int32)
    action_mask = jnp.zeros((1, length - 1, agents, classes), bool)
    action_mask = action_mask.at[..., 1].set(True)
    action_mask = action_mask.at[:, 1:, 1, 2].set(True)
    present = jnp.ones((1, length, agents), bool)
    alive = jnp.ones_like(present)
    first = jnp.zeros((1, length), bool)
    windows, factual_valid = aligned_action_windows(
        actions,
        action_mask,
        present,
        alive,
        first,
        action_low=0,
        max_horizon=8,
    )
    changed, distinct = same_focal_legal_action_interventions(
        windows,
        action_mask,
        action_low=0,
        horizons=(1, 2, 4, 8),
    )
    changed_valid = action_window_validity(
        changed[2],
        action_mask,
        present,
        alive,
        first,
        action_low=0,
    )

    assert bool(factual_valid[2][0, 0].all())
    assert not bool(distinct[2][0, 0, 0])
    assert bool(distinct[2][0, 0, 1])
    assert int(changed[2][0, 0, 0, 1]) == 1
    assert int(changed[2][0, 0, 1, 1]) == 2
    assert bool(changed_valid[2][0, 0].all())


def test_horizon_heads_cannot_see_actions_after_their_prefix() -> None:
    model = ActionConditionedMultiStepJEPA(
        4,
        0,
        5,
        (1, 2, 4, 8),
        8,
        width=8,
        layers=1,
        units=8,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(1), (1, 2, 3, 6))
    actions = jnp.zeros((1, 2, 3, 8), jnp.int32)
    changed = actions.at[..., 1:].set(3)

    def predict(window):
        return model(hidden, window)

    params = nj.init(lambda: predict(actions))({}, seed=5)
    original = nj.pure(lambda: predict(actions))(params, seed=6)[1]
    intervened = nj.pure(lambda: predict(changed))(params, seed=6)[1]

    np.testing.assert_array_equal(original[1], intervened[1])
    assert not np.allclose(
        np.asarray(original[8], np.float32), np.asarray(intervened[8], np.float32)
    )


def test_direct_objective_stops_targets_and_trains_both_action_queries() -> None:
    horizons = (1, 2, 4, 8)
    shape = (1, 2, 3, 4)
    base = jax.random.normal(jax.random.key(11), shape)
    target = jax.random.normal(jax.random.key(12), shape)
    valid = {horizon: jnp.ones(shape[:-1], bool) for horizon in horizons}

    shuffled_valid = {horizon: jnp.ones(shape[:-1], bool) for horizon in horizons}
    distinct = {
        horizon: jnp.full(shape[:-1], horizon > 1, bool) for horizon in horizons
    }

    def objective(prediction, ema_target, shuffled):
        losses, metrics = direct_multistep_objective(
            {horizon: prediction + horizon / 100 for horizon in horizons},
            {horizon: ema_target for horizon in horizons},
            valid,
            {horizon: shuffled for horizon in horizons},
            shuffled_valid,
            distinct,
            horizons=horizons,
            decay=0.75,
            action_margin=0.1,
        )
        return losses["cosine"].mean() + losses["action"].mean(), metrics

    (prediction_grad, target_grad, shuffle_grad), metrics = jax.grad(
        objective,
        (0, 1, 2),
        has_aux=True,
    )(base, target, -base)

    assert float(jnp.linalg.norm(prediction_grad)) > 0.0
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target_grad))
    assert float(jnp.linalg.norm(shuffle_grad)) > 0.0
    for horizon in horizons:
        assert float(metrics[f"h{horizon}_valid_count"]) == 6.0
        assert np.isfinite(float(metrics[f"h{horizon}_cosine"]))
        assert 1.0 <= float(metrics[f"h{horizon}_within_team_mean_rank"]) <= 3.0
    assert float(metrics["h1_action_distinct_legal_count"]) == 0.0


def test_predictor_has_a_nonzero_gradient_into_shared_joint_hidden() -> None:
    model = ActionConditionedMultiStepJEPA(
        3,
        0,
        4,
        (1, 2, 4, 8),
        8,
        width=4,
        layers=1,
        units=4,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(20), (1, 1, 2, 5))
    actions = jnp.zeros((1, 1, 2, 8), jnp.int32)

    def scalar(value):
        return sum(output.sum() for output in model(value, actions).values())

    params = nj.init(lambda: scalar(hidden))({}, seed=21)

    def pure_scalar(value):
        return nj.pure(lambda: scalar(value))(params, seed=22)[1]

    gradient = jax.grad(pure_scalar)(hidden)
    assert float(jnp.linalg.norm(gradient)) > 0.0
