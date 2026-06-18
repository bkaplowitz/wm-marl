from __future__ import annotations

import jax
import numpy as np

from baselines.softmax_model import (
    NUM_CELLS,
    SoftmaxBaselineConfig,
    SoftmaxBaselineData,
    SoftmaxBaselinePredictions,
    create_softmax_train_state,
    decode_coin_positions,
    encode_coin_positions,
    evaluate_softmax_baseline,
    expected_deterministic_next_positions,
    prepare_softmax_data,
    softmax_target_distributions,
    split_softmax_data,
    train_softmax_baseline,
)
from world_marl.checkpoint.train_state import load_params, save_checkpoint
from world_marl.world_model import VectorTransitionBatch


def test_coin_position_codec_round_trips():
    positions = np.asarray(
        [
            [[0, 4, 2, 8], [4, 0, 8, 2]],
            [[3, 5, 1, 7], [5, 3, 7, 1]],
        ],
        dtype=np.int32,
    )

    observations = encode_coin_positions(positions)
    decoded = decode_coin_positions(observations)

    assert observations.shape == (2, 2, 36)
    np.testing.assert_array_equal(decoded, positions)


def test_prepare_softmax_data_consumes_vector_transition_batch():
    positions = np.asarray([[[0, 4, 2, 8], [4, 0, 8, 2]]], dtype=np.int32)
    actions = np.asarray([[0, 4]], dtype=np.int32)
    next_positions = expected_deterministic_next_positions(positions, actions)
    states = encode_coin_positions(positions)
    next_states = encode_coin_positions(next_positions)
    batch = VectorTransitionBatch(
        states=jax.numpy.asarray(states),
        actions=jax.numpy.asarray(actions),
        next_states=jax.numpy.asarray(next_states),
        rewards=jax.numpy.asarray([[0.0, 0.0]], dtype=jax.numpy.float32),
        dones=jax.numpy.asarray([[0.0, 0.0]], dtype=jax.numpy.float32),
    )

    data = prepare_softmax_data(batch)

    assert data.num_transitions == 1
    np.testing.assert_array_equal(data.states, states)
    np.testing.assert_array_equal(data.positions, positions)
    np.testing.assert_array_equal(data.actions, actions)
    np.testing.assert_array_equal(data.next_positions, next_positions)


def test_softmax_targets_use_uniform_for_respawn_and_reset():
    positions = np.asarray(
        [
            [[0, 4, 0, 8], [4, 0, 8, 0]],  # red coin collected
            [[4, 0, 8, 0], [0, 4, 0, 8]],  # blue coin collected
            [[0, 4, 8, 2], [4, 0, 2, 8]],  # terminal reset
        ],
        dtype=np.int32,
    )
    actions = np.asarray([[4, 4], [4, 4], [4, 4]], dtype=np.int32)
    next_positions = np.asarray(
        [
            [[0, 4, 5, 8], [4, 0, 8, 5]],
            [[4, 0, 8, 6], [0, 4, 6, 8]],
            [[1, 2, 3, 4], [2, 1, 4, 3]],
        ],
        dtype=np.int32,
    )
    data = _softmax_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=np.zeros((3, 2), dtype=np.float32),
        dones=np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    target_bundle = softmax_target_distributions(
        data,
        stochastic_target_weight=3.0,
    )
    targets = target_bundle.distributions
    weights = target_bundle.weights
    uniform = np.full((NUM_CELLS,), 1.0 / NUM_CELLS, dtype=np.float32)

    np.testing.assert_allclose(targets[0, 0, 2], uniform)
    np.testing.assert_allclose(targets[0, 1, 3], uniform)
    np.testing.assert_allclose(targets[1, 0, 3], uniform)
    np.testing.assert_allclose(targets[1, 1, 2], uniform)
    np.testing.assert_allclose(targets[2], np.broadcast_to(uniform, targets[2].shape))
    assert weights[0, 0, 2] == 3.0
    assert weights[0, 1, 3] == 3.0
    assert weights[1, 0, 3] == 3.0
    assert weights[1, 1, 2] == 3.0
    np.testing.assert_allclose(weights[2], np.full_like(weights[2], 3.0))

    assert targets[0, 0, 0, next_positions[0, 0, 0]] == 1.0
    assert targets[0, 0, 0].sum() == 1.0
    assert weights[0, 0, 0] == 1.0


def test_softmax_training_metrics_and_reload(tmp_path):
    positions = []
    actions = []
    for red in [0, 3, 6]:
        for blue in [2, 5, 8]:
            for red_action in [0, 4]:
                for blue_action in [1, 4]:
                    positions.append([[red, blue, 4, 7], [blue, red, 7, 4]])
                    actions.append([red_action, blue_action])
    positions = np.asarray(positions, dtype=np.int32)
    actions = np.asarray(actions, dtype=np.int32)
    next_positions = expected_deterministic_next_positions(positions, actions)
    data = _softmax_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=np.zeros((positions.shape[0], 2), dtype=np.float32),
        dones=np.zeros((positions.shape[0], 2), dtype=np.float32),
    )
    train_data, validation_data = split_softmax_data(
        data,
        validation_fraction=0.25,
        seed=0,
    )
    config = SoftmaxBaselineConfig(
        hidden_dims=(32,),
        learning_rate=1e-2,
        batch_size=16,
        train_steps=80,
    )

    state, rows = train_softmax_baseline(
        jax.random.PRNGKey(0),
        train_data,
        config=config,
    )
    predictions = SoftmaxBaselinePredictions(
        next_position_logits=np.asarray(
            state.apply_fn(
                {"params": state.params},
                jax.numpy.asarray(validation_data.positions, dtype=jax.numpy.int32),
                jax.numpy.asarray(validation_data.actions, dtype=jax.numpy.int32),
            ),
            dtype=np.float32,
        )
    )
    metrics = evaluate_softmax_baseline(train_data, validation_data, predictions)

    assert rows[-1]["loss"] < rows[0]["loss"]
    assert metrics["deterministic_full_state_exact_accuracy"] is not None
    assert (
        metrics["full_state_exact_accuracy"]
        > metrics["marginal_full_state_exact_accuracy"]
    )

    save_checkpoint(tmp_path / "checkpoint", state, metadata={"kind": "test"})
    reload_state = create_softmax_train_state(
        jax.random.PRNGKey(1),
        config=config,
    )
    reload_state = reload_state.replace(
        params=load_params(
            tmp_path / "checkpoint" / "checkpoint.msgpack",
            reload_state.params,
        )
    )
    reload_logits = np.asarray(
        reload_state.apply_fn(
            {"params": reload_state.params},
            jax.numpy.asarray(validation_data.positions, dtype=jax.numpy.int32),
            jax.numpy.asarray(validation_data.actions, dtype=jax.numpy.int32),
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        reload_logits,
        predictions.next_position_logits,
        atol=0.0,
    )


def test_train_simple_categorical_policy_diagnostic_smoke(tmp_path, monkeypatch):
    from world_marl.scripts.diagnostics import train_simple_categorical_policy

    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m world_marl.scripts.diagnostics.train_simple_categorical_policy",
            "--num-envs",
            "2",
            "--collect-steps",
            "4",
            "--train-steps",
            "2",
            "--batch-size",
            "4",
            "--max-cycles",
            "10",
            "--out-dir",
            str(tmp_path),
            "--quiet",
        ],
    )

    train_simple_categorical_policy.main()

    run_dirs = list(tmp_path.glob("coin_softmax_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "prediction_metrics.json").exists()
    assert (run_dir / "training_summary.json").exists()
    assert (run_dir / "checkpoint" / "metadata.json").exists()
    assert (run_dir / "reload_evaluation.json").exists()
    assert (run_dir / "outcome.json").exists()


def _softmax_data(
    *,
    positions: np.ndarray,
    next_positions: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
) -> SoftmaxBaselineData:
    return SoftmaxBaselineData(
        states=encode_coin_positions(positions),
        positions=positions,
        actions=actions,
        next_states=encode_coin_positions(next_positions),
        next_positions=next_positions,
        rewards=rewards,
        dones=dones,
        action_dim=5,
        num_agents=2,
    )
