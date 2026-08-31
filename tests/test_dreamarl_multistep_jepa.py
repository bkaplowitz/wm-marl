"""Decision-critical tests for direct action-conditioned multi-step JEPA."""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamarl.models.multistep_jepa import ActionConditionedMultiStepJEPA
from dreamarl.training.multistep_jepa import (
    _cosine,
    action_window_validity,
    aligned_action_windows,
    all_legal_same_focal_action_interventions,
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


def test_all_legal_intervention_enumerates_exact_legal_alternatives() -> None:
    batch, roots, agents, max_horizon, classes = 1, 2, 2, 4, 5
    action_low = 3
    windows = jnp.full((batch, roots, agents, max_horizon), action_low + 1, jnp.int32)
    masks = jnp.zeros((batch, roots + max_horizon - 1, agents, classes), bool)
    masks = masks.at[..., 1].set(True)
    masks = masks.at[:, 1 : 1 + roots, :, 0].set(True)
    masks = masks.at[:, 1 : 1 + roots, 1, 4].set(True)
    masks = masks.at[:, 3 : 3 + roots, :, 2].set(True)

    candidates, alternative = all_legal_same_focal_action_interventions(
        windows,
        masks,
        action_low=action_low,
        horizons=(1, 2, 4),
    )

    assert candidates[2].shape == (batch, roots, agents, classes, max_horizon)
    assert alternative[2].shape == (batch, roots, agents, classes)
    assert not bool(alternative[1].any())
    np.testing.assert_array_equal(alternative[2][..., 0], True)
    np.testing.assert_array_equal(alternative[2][..., 1], False)
    np.testing.assert_array_equal(alternative[2][..., 2:4], False)
    np.testing.assert_array_equal(alternative[2][:, :, 0, 4], False)
    np.testing.assert_array_equal(alternative[2][:, :, 1, 4], True)
    np.testing.assert_array_equal(candidates[2][..., 0, 1], action_low)
    np.testing.assert_array_equal(candidates[2][..., 4, 1], action_low + 4)
    np.testing.assert_array_equal(candidates[2][..., 0, 0], windows[..., 0])
    np.testing.assert_array_equal(candidates[4][..., 2, 3], action_low + 2)


def test_all_legal_intervention_is_action_label_permutation_equivariant() -> None:
    classes = 4
    permutation = jnp.asarray([2, 0, 3, 1], jnp.int32)
    windows = jnp.asarray(
        [[[[0, 1, 2, 3], [3, 2, 1, 0]], [[1, 0, 3, 2], [2, 3, 0, 1]]]],
        jnp.int32,
    )
    masks = jnp.asarray(
        jax.random.bernoulli(jax.random.key(44), 0.7, (1, 5, 2, classes)), bool
    )
    masks = masks.at[..., windows[:, :1, :, 0]].set(True)
    original, original_mask = all_legal_same_focal_action_interventions(
        windows, masks, action_low=0, horizons=(1, 2, 4)
    )
    permuted_windows = permutation[windows]
    permuted_masks = jnp.zeros_like(masks).at[..., permutation].set(masks)
    permuted, permuted_mask = all_legal_same_focal_action_interventions(
        permuted_windows, permuted_masks, action_low=0, horizons=(1, 2, 4)
    )

    for horizon in (1, 2, 4):
        for old_action, new_action in enumerate(np.asarray(permutation)):
            np.testing.assert_array_equal(
                permuted_mask[horizon][..., new_action],
                original_mask[horizon][..., old_action],
            )
            np.testing.assert_array_equal(
                permuted[horizon][..., new_action, :],
                permutation[original[horizon][..., old_action, :]],
            )


def test_all_legal_objective_is_candidate_normalized_and_has_finite_grads() -> None:
    horizons = (1, 2)
    shape = (1, 2, 2, 3)
    factual = jax.random.normal(jax.random.key(51), shape)
    target = jax.random.normal(jax.random.key(52), shape)
    valid = {horizon: jnp.ones(shape[:-1], bool) for horizon in horizons}

    def objective(prediction, ema_target, counterfactual, candidates):
        counterfactuals = {
            horizon: jnp.broadcast_to(
                counterfactual[..., None, :], (*shape[:-1], candidates, shape[-1])
            )
            for horizon in horizons
        }
        candidate_valid = {
            1: jnp.zeros((*shape[:-1], candidates), bool),
            2: jnp.ones((*shape[:-1], candidates), bool),
        }
        losses, metrics = direct_multistep_objective(
            {horizon: prediction for horizon in horizons},
            {horizon: ema_target for horizon in horizons},
            valid,
            counterfactuals,
            candidate_valid,
            candidate_valid,
            horizons=horizons,
            decay=0.75,
            action_margin=0.1,
        )
        return losses["cosine"].mean() + losses["action"].mean(), metrics

    (prediction_grad, target_grad, counterfactual_grad), metrics = jax.grad(
        lambda prediction, ema_target, counterfactual: objective(
            prediction, ema_target, counterfactual, 3
        ),
        (0, 1, 2),
        has_aux=True,
    )(factual, target, -factual)
    for gradient in (prediction_grad, target_grad, counterfactual_grad):
        assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.linalg.norm(prediction_grad)) > 0.0
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target_grad))
    assert float(jnp.linalg.norm(counterfactual_grad)) > 0.0
    loss_two, _ = objective(factual, target, -factual, 2)
    loss_five, _ = objective(factual, target, -factual, 5)
    np.testing.assert_allclose(loss_two, loss_five, rtol=0.0, atol=1e-6)
    assert float(metrics["h2_action_counterfactual_candidates_per_root"]) == 3.0
    assert float(metrics["h2_action_counterfactual_all_legal"]) == 1.0


def test_cosine_zero_and_masked_rows_have_finite_vjps() -> None:
    horizons = (1, 2)
    shape = (1, 2, 2, 3)
    prediction = (
        jnp.zeros(shape, jnp.float32)
        .at[:, 1]
        .set(jax.random.normal(jax.random.key(55), (1, 2, 3)))
    )
    target = (
        jnp.zeros(shape, jnp.float32)
        .at[:, 1]
        .set(jax.random.normal(jax.random.key(56), (1, 2, 3)))
    )
    counterfactual = (
        jnp.zeros((*shape[:-1], 3, shape[-1]), jnp.float32)
        .at[:, 1]
        .set(jax.random.normal(jax.random.key(57), (1, 2, 3, 3)))
    )
    valid = jnp.asarray([[[False, False], [True, True]]])
    candidate_valid = jnp.broadcast_to(valid[..., None], (*valid.shape, 3))

    def objective(value, alternative, ema_target):
        losses, _ = direct_multistep_objective(
            {horizon: value for horizon in horizons},
            {horizon: ema_target for horizon in horizons},
            {horizon: valid for horizon in horizons},
            {horizon: alternative for horizon in horizons},
            {
                1: jnp.zeros_like(candidate_valid),
                2: candidate_valid,
            },
            {
                1: jnp.zeros_like(candidate_valid),
                2: candidate_valid,
            },
            horizons=horizons,
            decay=0.75,
            action_margin=0.1,
        )
        return losses["cosine"].mean() + losses["action"].mean()

    value, gradients = jax.value_and_grad(objective, (0, 1, 2))(
        prediction, counterfactual, target
    )
    assert bool(jnp.isfinite(value))
    for gradient in gradients:
        assert bool(jnp.isfinite(gradient).all())
    np.testing.assert_array_equal(gradients[2], jnp.zeros_like(target))


def test_safe_cosine_matches_canonical_cosine_on_nondegenerate_vectors() -> None:
    left = jax.random.normal(jax.random.key(58), (2, 3, 7)) + 0.25
    right = jax.random.normal(jax.random.key(59), (2, 3, 7)) - 0.4

    def canonical_cosine(value, target):
        value = value / jnp.maximum(
            jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8
        )
        target = target / jnp.maximum(
            jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8
        )
        return jnp.sum(value * target, axis=-1).sum()

    safe_value, safe_gradients = jax.value_and_grad(
        lambda value, target: _cosine(value, target).sum(), (0, 1)
    )(left, right)
    canonical_value, canonical_gradients = jax.value_and_grad(canonical_cosine, (0, 1))(
        left, right
    )
    np.testing.assert_allclose(safe_value, canonical_value, rtol=1e-6, atol=1e-7)
    for safe, canonical in zip(safe_gradients, canonical_gradients, strict=True):
        np.testing.assert_allclose(safe, canonical, rtol=1e-6, atol=1e-7)


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


def test_all_legal_candidate_batch_preserves_candidate_and_agent_axes() -> None:
    batch, roots, agents, classes, max_horizon = 1, 2, 3, 5, 4
    model = ActionConditionedMultiStepJEPA(
        classes,
        0,
        4,
        (1, 2, 4),
        max_horizon,
        width=8,
        layers=1,
        units=8,
        name="multistep",
    )
    hidden = jax.random.normal(jax.random.key(61), (batch, roots, agents, 6))
    windows = jnp.zeros((batch, roots, agents, max_horizon), jnp.int32)
    masks = jnp.ones((batch, roots + max_horizon - 1, agents, classes), bool)
    candidates, _ = all_legal_same_focal_action_interventions(
        windows, masks, action_low=0, horizons=(1, 2, 4)
    )

    def predict(value, candidate_windows):
        model(value, windows)
        expanded = jnp.broadcast_to(
            value[:, None], (batch, classes, *value.shape[1:])
        ).reshape((batch * classes, *value.shape[1:]))
        flat_windows = jnp.transpose(candidate_windows, (0, 3, 1, 2, 4)).reshape(
            (batch * classes, roots, agents, max_horizon)
        )
        flat = model(expanded, flat_windows, selected_horizon=4)[4]
        return jnp.transpose(
            flat.reshape((batch, classes, roots, agents, 4)),
            (0, 2, 3, 1, 4),
        )

    params = nj.init(lambda: predict(hidden, candidates[4]))({}, seed=62)
    output = nj.pure(lambda: predict(hidden, candidates[4]))(params, seed=63)[1]
    assert output.shape == (batch, roots, agents, classes, 4)
    permutation = jnp.asarray([3, 1, 4, 0, 2])
    permuted = nj.pure(lambda: predict(hidden, candidates[4][..., permutation, :]))(
        params, seed=63
    )[1]
    np.testing.assert_array_equal(permuted, output[..., permutation, :])

    def scalar(value):
        return nj.pure(lambda: predict(value, candidates[4]))(params, seed=63)[1].sum()

    gradient = jax.grad(scalar)(hidden)
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_all_legal_predictor_has_finite_parameter_gradient_leaves() -> None:
    batch, roots, agents, classes, target_dim, max_horizon = 1, 2, 3, 5, 4, 4
    horizons = (1, 2, 4)
    model = ActionConditionedMultiStepJEPA(
        classes,
        0,
        target_dim,
        horizons,
        max_horizon,
        plan_aggregation="mean",
        width=8,
        layers=1,
        units=8,
        name="alllegal_param_grad",
    )
    root = 0.1 * jax.random.normal(
        jax.random.key(71), (batch, roots, agents, target_dim)
    )
    windows = (
        jnp.arange(batch * roots * agents * max_horizon).reshape(
            (batch, roots, agents, max_horizon)
        )
        % classes
    )
    masks = jnp.ones((batch, roots + max_horizon - 1, agents, classes), bool)
    plan = 0.1 * jax.random.normal(
        jax.random.key(72),
        (batch, roots, agents, max_horizon - 1, agents - 1, classes),
    )
    target = jax.random.normal(jax.random.key(73), (batch, roots, agents, target_dim))
    candidates, candidate_mask = all_legal_same_focal_action_interventions(
        windows, masks, action_low=0, horizons=horizons
    )
    valid = {horizon: jnp.ones((batch, roots, agents), bool) for horizon in horizons}

    def objective(value):
        predictions = model(value, windows, plan)
        expanded_root = jnp.broadcast_to(
            value[:, None], (batch, classes, *value.shape[1:])
        ).reshape((batch * classes, *value.shape[1:]))
        expanded_plan = jnp.broadcast_to(
            plan[:, None], (batch, classes, *plan.shape[1:])
        ).reshape((batch * classes, *plan.shape[1:]))
        counterfactuals = {}
        for horizon in horizons:
            if horizon == 1:
                counterfactuals[horizon] = jnp.broadcast_to(
                    jax.lax.stop_gradient(predictions[horizon])[..., None, :],
                    (*predictions[horizon].shape[:-1], classes, target_dim),
                )
                continue
            candidate_windows = candidates[horizon]
            flat_windows = jnp.transpose(candidate_windows, (0, 3, 1, 2, 4)).reshape(
                (batch * classes, roots, agents, max_horizon)
            )
            flat = model(
                expanded_root,
                flat_windows,
                expanded_plan,
                selected_horizon=horizon,
            )[horizon]
            counterfactuals[horizon] = jnp.transpose(
                flat.reshape((batch, classes, roots, agents, target_dim)),
                (0, 2, 3, 1, 4),
            )
        losses, _ = direct_multistep_objective(
            predictions,
            {horizon: target for horizon in horizons},
            valid,
            counterfactuals,
            candidate_mask,
            candidate_mask,
            horizons=horizons,
            decay=0.75,
            action_margin=0.1,
        )
        return losses["cosine"].mean() + 0.25 * losses["action"].mean()

    params = nj.init(lambda: objective(root))({}, seed=74)

    def parameter_loss(parameters, value):
        return nj.pure(lambda: objective(value))(parameters, seed=75)[1]

    parameter_gradients, root_gradient = jax.grad(parameter_loss, argnums=(0, 1))(
        params, root
    )
    leaves = jax.tree_util.tree_leaves(parameter_gradients)
    assert leaves
    for leaf in leaves:
        assert bool(jnp.isfinite(leaf).all())
    total_norm = jnp.sqrt(
        sum(jnp.square(leaf.astype(jnp.float32)).sum() for leaf in leaves)
    )
    assert float(total_norm) > 0.0
    assert bool(jnp.isfinite(root_gradient).all())
    assert float(jnp.linalg.norm(root_gradient)) > 0.0


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


def test_legacy_cyclic_objective_step_hash_is_unchanged() -> None:
    """Guard the exact e02498e cyclic path used by c4."""

    horizons = (1, 2, 4, 8)
    shape = (1, 2, 3, 4)
    base = (jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) - 7.0) / 11.0
    target = jnp.cos(jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 7.0)
    valid = {horizon: jnp.ones(shape[:-1], bool) for horizon in horizons}
    distinct = {
        horizon: jnp.full(shape[:-1], horizon > 1, bool) for horizon in horizons
    }

    def objective(prediction, counterfactual):
        losses, metrics = direct_multistep_objective(
            {horizon: prediction + horizon / 100 for horizon in horizons},
            {horizon: target for horizon in horizons},
            valid,
            {horizon: counterfactual - horizon / 200 for horizon in horizons},
            valid,
            distinct,
            horizons=horizons,
            decay=0.75,
            action_margin=0.1,
        )
        scalar = losses["cosine"].sum() + losses["action"].sum()
        return scalar, (losses, metrics)

    ((value, (losses, metrics)), gradients) = jax.value_and_grad(
        objective, (0, 1), has_aux=True
    )(base, -base)
    windows = jnp.arange(48, dtype=jnp.int32).reshape((1, 2, 3, 8)) % 5
    masks = jnp.ones((1, 9, 3, 5), bool).at[:, 1:3, 0, 2].set(False)
    changed, changed_mask = same_focal_legal_action_interventions(
        windows, masks, action_low=0, horizons=horizons
    )
    leaves = [np.asarray(value), *(np.asarray(x) for x in gradients)]
    leaves.extend(np.asarray(losses[name]) for name in sorted(losses))
    leaves.extend(np.asarray(metrics[name]) for name in sorted(metrics))
    for horizon in horizons:
        leaves.extend([np.asarray(changed[horizon]), np.asarray(changed_mask[horizon])])
    digest = hashlib.sha256()
    for value in leaves:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())

    assert len(metrics) == 74
    assert digest.hexdigest() == (
        "f922c5e451f0a726c6ce3d9b67a224987731b96536ddcebbc45c7e321b642c44"
    )


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
