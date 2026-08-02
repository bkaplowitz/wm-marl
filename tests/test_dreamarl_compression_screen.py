from __future__ import annotations

import jax
import numpy as np

from world_marl.dreamarl.compression_screen import (
    ScreenConfig,
    ScreenInput,
    _signed_projection,
    init_screen_predictor,
    parameter_count,
    screen_predictor,
)


def test_signed_projection_is_deterministic_and_not_order_averaging():
    values = np.arange(2 * 3 * 2 * 8, dtype=np.float32).reshape(2, 3, 2, 8)
    first = _signed_projection(values, 4, seed=7)
    second = _signed_projection(values, 4, seed=7)
    different = _signed_projection(values, 4, seed=8)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 3, 2, 4)
    assert not np.array_equal(first, different)


def test_all_screen_inputs_have_identical_parameters():
    config = ScreenConfig(feature_width=8, hidden=16, heads=4, temporal_layers=1)
    expected = parameter_count(
        init_screen_predictor(jax.random.key(0), config, action_dim=3, output_dim=5)
    )
    for variant in ScreenInput:
        del variant
        params = init_screen_predictor(
            jax.random.key(0), config, action_dim=3, output_dim=5
        )
        assert parameter_count(params) == expected


def test_local_screen_cannot_see_other_agent_but_joint_screen_can():
    config = ScreenConfig(feature_width=8, hidden=16, heads=4, temporal_layers=1)
    params = init_screen_predictor(
        jax.random.key(0), config, action_dim=3, output_dim=5
    )
    state = np.ones((1, 3, 2, 8), np.float32)
    changed = state.copy()
    changed[:, :, 1] = 20
    action = np.zeros((1, 3, 2), np.int32)
    valid = np.ones((1, 3, 2), bool)
    reset = np.zeros((1, 3, 2), bool)
    reset[:, 0] = True
    local = screen_predictor(
        params,
        state,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_LOCAL,
        heads=4,
    )
    local_changed = screen_predictor(
        params,
        changed,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_LOCAL,
        heads=4,
    )
    joint = screen_predictor(
        params,
        state,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_JOINT,
        heads=4,
    )
    joint_changed = screen_predictor(
        params,
        changed,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_JOINT,
        heads=4,
    )
    np.testing.assert_allclose(local[:, :, 0], local_changed[:, :, 0], atol=1e-5)
    assert not np.allclose(joint[:, :, 0], joint_changed[:, :, 0])


def test_temporal_predictor_is_causal():
    config = ScreenConfig(feature_width=8, hidden=16, heads=4, temporal_layers=1)
    params = init_screen_predictor(
        jax.random.key(0), config, action_dim=3, output_dim=5
    )
    state = np.ones((1, 4, 2, 8), np.float32)
    future_changed = state.copy()
    future_changed[:, 2:] = 100
    action = np.zeros((1, 4, 2), np.int32)
    valid = np.ones((1, 4, 2), bool)
    reset = np.zeros((1, 4, 2), bool)
    reset[:, 0] = True
    baseline = screen_predictor(
        params,
        state,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_JOINT,
        heads=4,
    )
    changed = screen_predictor(
        params,
        future_changed,
        action,
        valid,
        reset,
        ScreenInput.OBSERVATION_JOINT,
        heads=4,
    )
    np.testing.assert_allclose(baseline[:, :2], changed[:, :2], atol=1e-5)
