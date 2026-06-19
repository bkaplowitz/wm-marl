from __future__ import annotations

import numpy as np

from baselines.softmax_model import (
    NUM_CELLS,
    SoftmaxBaselineData,
    SoftmaxBaselinePredictions,
    encode_coin_positions,
    expected_deterministic_next_positions,
)
from world_marl.visualize import (
    build_next_state_comparison,
    plot_next_state_comparison,
)


def test_perfect_predictions_on_deterministic_transitions():
    positions = np.asarray(
        [
            [[0, 8, 4, 5], [8, 0, 5, 4]],
            [[1, 7, 3, 6], [7, 1, 6, 3]],
        ],
        dtype=np.int32,
    )
    actions = np.asarray([[4, 4], [4, 4]], dtype=np.int32)
    next_positions = expected_deterministic_next_positions(positions, actions)
    data = _softmax_data(positions, next_positions, actions)

    softmax_predictions = SoftmaxBaselinePredictions(
        next_position_logits=_one_hot_logits(next_positions)
    )
    flow_cell_samples = np.broadcast_to(
        next_positions, (3, *next_positions.shape)
    ).copy()

    comparison = build_next_state_comparison(
        data, softmax_predictions, flow_cell_samples
    )

    assert comparison.det_transition_count == 2
    assert comparison.respawn_event_count == 0
    assert comparison.det_exact_softmax == 1.0
    assert comparison.det_exact_flow == 1.0
    # Every populated deterministic cell is perfectly predicted.
    populated = np.isfinite(comparison.det_softmax_accuracy)
    np.testing.assert_allclose(comparison.det_softmax_accuracy[populated], 1.0)
    np.testing.assert_allclose(comparison.det_flow_accuracy[populated], 1.0)
    # No respawn events -> distributions fall back to uniform, zero divergence.
    np.testing.assert_allclose(comparison.respawn_softmax, 1.0 / NUM_CELLS)
    np.testing.assert_allclose(comparison.respawn_flow, 1.0 / NUM_CELLS)
    assert comparison.respawn_tv_softmax == 0.0
    assert comparison.respawn_kl_flow == 0.0
    _assert_distribution(comparison.respawn_empirical)
    assert comparison.num_flow_samples == 3


def test_uniform_softmax_is_calibrated_on_respawns():
    positions = np.asarray(
        [
            [[0, 4, 0, 8], [4, 0, 8, 0]],
            [[1, 4, 1, 8], [4, 1, 8, 1]],
            [[2, 4, 2, 8], [4, 2, 8, 2]],
        ],
        dtype=np.int32,
    )
    actions = np.asarray([[4, 4], [4, 4], [4, 4]], dtype=np.int32)
    next_positions = np.asarray(
        [
            [[0, 4, 5, 8], [4, 0, 8, 5]],
            [[1, 4, 6, 8], [4, 1, 8, 6]],
            [[2, 4, 7, 8], [4, 2, 8, 7]],
        ],
        dtype=np.int32,
    )
    data = _softmax_data(positions, next_positions, actions)

    # All-zero logits -> uniform softmax distribution over the nine cells.
    softmax_predictions = SoftmaxBaselinePredictions(
        next_position_logits=np.zeros((3, 2, 4, NUM_CELLS), dtype=np.float32)
    )
    flow_cell_samples = np.broadcast_to(
        next_positions, (4, *next_positions.shape)
    ).copy()

    comparison = build_next_state_comparison(
        data, softmax_predictions, flow_cell_samples
    )

    assert comparison.det_transition_count == 0
    assert comparison.respawn_event_count == 3
    _assert_distribution(comparison.respawn_softmax)
    _assert_distribution(comparison.respawn_flow)
    _assert_distribution(comparison.respawn_empirical)
    np.testing.assert_allclose(comparison.respawn_softmax, 1.0 / NUM_CELLS)
    assert comparison.respawn_tv_softmax == 0.0
    np.testing.assert_allclose(comparison.respawn_kl_softmax, 0.0, atol=1e-9)
    # Flow sampled only cells {5, 6, 7}; empirical matches the true respawn cells.
    np.testing.assert_array_equal(
        np.flatnonzero(comparison.respawn_flow.reshape(-1) > 0.0),
        np.asarray([5, 6, 7]),
    )


def test_plot_writes_png(tmp_path):
    positions = np.asarray([[[0, 8, 4, 5], [8, 0, 5, 4]]], dtype=np.int32)
    actions = np.asarray([[4, 4]], dtype=np.int32)
    next_positions = expected_deterministic_next_positions(positions, actions)
    data = _softmax_data(positions, next_positions, actions)
    softmax_predictions = SoftmaxBaselinePredictions(
        next_position_logits=_one_hot_logits(next_positions)
    )
    comparison = build_next_state_comparison(
        data, softmax_predictions, next_positions[None, ...]
    )

    output = tmp_path / "next_state_comparison.png"
    plot_next_state_comparison(comparison, output)

    assert output.exists()
    assert output.stat().st_size > 0


def _softmax_data(
    positions: np.ndarray,
    next_positions: np.ndarray,
    actions: np.ndarray,
) -> SoftmaxBaselineData:
    count = positions.shape[0]
    return SoftmaxBaselineData(
        states=encode_coin_positions(positions),
        positions=positions,
        actions=actions,
        next_states=encode_coin_positions(next_positions),
        next_positions=next_positions,
        rewards=np.zeros((count, 2), dtype=np.float32),
        dones=np.zeros((count, 2), dtype=np.float32),
        action_dim=5,
        num_agents=2,
    )


def _one_hot_logits(next_positions: np.ndarray) -> np.ndarray:
    one_hot = np.eye(NUM_CELLS, dtype=np.float32)[next_positions]
    return (one_hot * 20.0 - 10.0).astype(np.float32)


def _assert_distribution(grid: np.ndarray) -> None:
    assert np.all(grid >= 0.0)
    np.testing.assert_allclose(grid.sum(), 1.0, atol=1e-6)
