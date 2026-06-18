from __future__ import annotations

import jax
import numpy as np

from world_marl.checkpointing import load_params, save_checkpoint
from world_marl.coingame_dynamics import (
    CoinDynamicsConfig,
    CoinDynamicsData,
    CoinDynamicsPredictions,
    NUM_CELLS,
    collect_coin_dynamics_dataset,
    coin_collected,
    coin_dynamics_target_distributions,
    collected_coin_masks,
    create_coin_dynamics_train_state,
    decode_coin_positions,
    derive_coin_rewards,
    deterministic_transition_mask,
    distributional_cross_entropy_positions,
    encode_coin_positions,
    evaluate_coin_dynamics,
    expected_deterministic_next_positions,
    predict_coin_dynamics,
    prepare_coin_dynamics_data,
    reward_prediction_metrics,
    split_coin_dynamics_data,
    stochastic_respawn_metrics,
    summarize_coin_dynamics_outcome,
    train_coin_dynamics_model,
)
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter


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


def test_collect_and_split_coin_dynamics_dataset():
    adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=10, seed=0)
    try:
        dataset = collect_coin_dynamics_dataset(
            adapter,
            np.random.default_rng(0),
            rollout_steps=4,
        )
    finally:
        adapter.close()

    assert dataset.observations.shape == (8, 2, 36)
    assert dataset.next_observations.shape == (8, 2, 36)
    assert dataset.actions.shape == (8, 2)
    assert dataset.rewards.shape == (8, 2)

    data = prepare_coin_dynamics_data(dataset)
    derived_rewards = derive_coin_rewards(data.positions, data.actions)
    reward_metrics = reward_prediction_metrics(data, derived_rewards)
    assert reward_metrics["nonterminal_transition_exact_accuracy"] == 1.0

    train_data, validation_data = split_coin_dynamics_data(
        data,
        validation_fraction=0.25,
        seed=0,
    )
    assert train_data.num_transitions == 6
    assert validation_data.num_transitions == 2
    assert train_data.positions.shape[1:] == (2, 4)


def test_expected_deterministic_next_positions_and_mask():
    positions = np.asarray(
        [
            [[0, 8, 4, 5], [8, 0, 5, 4]],  # no pickup
            [[0, 8, 1, 5], [8, 0, 5, 1]],  # red moves onto red coin
        ],
        dtype=np.int32,
    )
    actions = np.asarray([[0, 4], [0, 4]], dtype=np.int32)
    next_positions = expected_deterministic_next_positions(positions, actions)
    data = _coin_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=np.zeros((2, 2), dtype=np.float32),
        dones=np.zeros((2, 2), dtype=np.float32),
    )

    np.testing.assert_array_equal(coin_collected(positions, actions), [False, True])
    red_collected, blue_collected = collected_coin_masks(positions, actions)
    np.testing.assert_array_equal(red_collected, [False, True])
    np.testing.assert_array_equal(blue_collected, [False, False])
    np.testing.assert_array_equal(deterministic_transition_mask(data), [True, False])
    np.testing.assert_array_equal(next_positions[0, 0], np.asarray([1, 8, 4, 5]))
    np.testing.assert_array_equal(next_positions[0, 1], np.asarray([8, 1, 5, 4]))


def test_derived_coin_rewards_cover_payoff_branches():
    stay = 4
    actions = np.asarray([[stay, stay]], dtype=np.int32)
    cases = {
        "red_takes_red": (
            np.asarray([[[0, 4, 0, 8], [4, 0, 8, 0]]], dtype=np.int32),
            [[1.0, 0.0]],
        ),
        "red_takes_blue": (
            np.asarray([[[0, 4, 8, 0], [4, 0, 0, 8]]], dtype=np.int32),
            [[1.0, -2.0]],
        ),
        "blue_takes_red": (
            np.asarray([[[4, 0, 0, 8], [0, 4, 8, 0]]], dtype=np.int32),
            [[-2.0, 1.0]],
        ),
        "blue_takes_blue": (
            np.asarray([[[4, 0, 8, 0], [0, 4, 0, 8]]], dtype=np.int32),
            [[0.0, 1.0]],
        ),
        "neither": (
            np.asarray([[[0, 4, 8, 2], [4, 0, 2, 8]]], dtype=np.int32),
            [[0.0, 0.0]],
        ),
    }
    for name, (positions, expected) in cases.items():
        np.testing.assert_allclose(
            derive_coin_rewards(positions, actions),
            np.asarray(expected, dtype=np.float32),
            err_msg=name,
        )


def test_stochastic_respawn_metrics_use_distribution_not_argmax():
    positions = np.asarray(
        [
            [[0, 4, 0, 8], [4, 0, 8, 0]],  # red coin collected
            [[4, 0, 8, 0], [0, 4, 0, 8]],  # blue coin collected
        ],
        dtype=np.int32,
    )
    actions = np.asarray([[4, 4], [4, 4]], dtype=np.int32)
    next_positions = np.asarray(
        [
            [[0, 4, 5, 8], [4, 0, 8, 5]],
            [[4, 0, 8, 6], [0, 4, 6, 8]],
        ],
        dtype=np.int32,
    )
    data = _coin_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=derive_coin_rewards(positions, actions),
        dones=np.zeros((2, 2), dtype=np.float32),
    )
    predictions = CoinDynamicsPredictions(
        next_position_logits=np.zeros((2, 2, 4, NUM_CELLS), dtype=np.float32)
    )

    metrics = stochastic_respawn_metrics(data, predictions)

    assert metrics["num_respawn_targets"] == 2
    assert metrics["red_coin_respawn_count"] == 1
    assert metrics["blue_coin_respawn_count"] == 1
    np.testing.assert_allclose(metrics["cross_entropy"], np.log(NUM_CELLS))
    np.testing.assert_allclose(metrics["mean_entropy"], np.log(NUM_CELLS))
    np.testing.assert_allclose(
        metrics["uniform_target_cross_entropy"], np.log(NUM_CELLS)
    )
    np.testing.assert_allclose(metrics["uniform_target_kl"], 0.0)
    assert metrics["aggregate_distribution_tv_to_uniform"] == 0.0
    assert metrics["mean_target_probability"] == 1.0 / NUM_CELLS


def test_coin_dynamics_targets_use_uniform_for_respawn_and_reset():
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
    data = _coin_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=derive_coin_rewards(positions, actions),
        dones=np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )

    target_bundle = coin_dynamics_target_distributions(
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
    assert targets[0, 0, 1, next_positions[0, 0, 1]] == 1.0

    zero_logits = np.zeros_like(targets)
    np.testing.assert_allclose(
        distributional_cross_entropy_positions(zero_logits, targets),
        np.log(NUM_CELLS),
    )


def test_discrete_dynamics_training_metrics_and_reload(tmp_path):
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
    rewards = derive_coin_rewards(positions, actions)
    data = _coin_data(
        positions=positions,
        next_positions=next_positions,
        actions=actions,
        rewards=rewards,
        dones=np.zeros((positions.shape[0], 2), dtype=np.float32),
    )
    train_data, validation_data = split_coin_dynamics_data(
        data,
        validation_fraction=0.25,
        seed=0,
    )
    config = CoinDynamicsConfig(
        hidden_dims=(32,),
        learning_rate=1e-2,
        batch_size=16,
        train_steps=80,
    )

    state, rows = train_coin_dynamics_model(
        jax.random.PRNGKey(0),
        train_data,
        config=config,
    )
    predictions = predict_coin_dynamics(state, validation_data)
    metrics = evaluate_coin_dynamics(train_data, validation_data, predictions)

    assert rows[-1]["loss"] < rows[0]["loss"]
    assert metrics["deterministic_full_state_exact_accuracy"] is not None
    assert (
        metrics["full_state_exact_accuracy"]
        > metrics["marginal_full_state_exact_accuracy"]
    )
    passed, criteria = summarize_coin_dynamics_outcome(
        metrics,
        finite_losses=True,
        reload_passed=True,
        min_deterministic_exact=0.0,
        min_reward_exact=0.0,
    )
    assert isinstance(passed, bool)
    assert criteria["valid_predictions"]

    save_checkpoint(tmp_path / "checkpoint", state, metadata={"kind": "test"})
    reload_state = create_coin_dynamics_train_state(
        jax.random.PRNGKey(1),
        config=config,
    )
    reload_state = reload_state.replace(
        params=load_params(
            tmp_path / "checkpoint" / "checkpoint.msgpack",
            reload_state.params,
        )
    )
    reload_predictions = predict_coin_dynamics(reload_state, validation_data)
    np.testing.assert_allclose(
        reload_predictions.next_position_logits,
        predictions.next_position_logits,
        atol=0.0,
    )


def test_train_coin_dynamics_cli_smoke(tmp_path, monkeypatch):
    from world_marl.scripts import train_coin_dynamics

    monkeypatch.setattr(
        "sys.argv",
        [
            "world-marl-train-coin-dynamics",
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

    train_coin_dynamics.main()

    run_dirs = list(tmp_path.glob("coin_dynamics_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "config.json").exists()
    assert (run_dir / "prediction_metrics.json").exists()
    assert (run_dir / "checkpoint" / "metadata.json").exists()
    assert (run_dir / "reload_evaluation.json").exists()
    assert (run_dir / "outcome.json").exists()


def _coin_data(
    *,
    positions: np.ndarray,
    next_positions: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
) -> CoinDynamicsData:
    return CoinDynamicsData(
        positions=positions,
        actions=actions,
        next_positions=next_positions,
        rewards=rewards,
        dones=dones,
        action_dim=5,
        num_agents=2,
    )
