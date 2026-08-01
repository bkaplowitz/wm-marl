from __future__ import annotations

import jax
import numpy as np

from world_marl.dreamarl.information_ladder import (
    LadderConfig,
    PreparedDataset,
    Rung,
    cosine_error,
    evaluate_predictor,
    init_residual,
    init_predictor,
    residual_predictor,
    train_predictor,
)


def _synthetic_dataset(seed=0):
    generator = np.random.default_rng(seed)
    trajectories, time, agents = 24, 5, 2
    action = generator.integers(0, 2, (trajectories, time, agents), dtype=np.int32)
    belief = generator.normal(size=(trajectories, time, agents, 8)).astype(np.float32)
    target = np.zeros((trajectories, time, agents, 2), np.float32)
    target[..., 0, :] = np.eye(2, dtype=np.float32)[action[..., 1]]
    target[..., 1, :] = np.eye(2, dtype=np.float32)[action[..., 0]]
    valid = np.ones((trajectories, time, agents), bool)
    return PreparedDataset(
        belief=belief,
        observation=None,
        oracle=None,
        action=action,
        target=target,
        valid=valid,
        reward_event=np.zeros_like(valid),
        trajectory_id=np.arange(trajectories),
        action_dim=2,
    )


def test_all_rungs_have_identical_trainable_parameter_count():
    params = init_predictor(jax.random.key(0), 8, 3, 16, 5)
    expected = sum(value.size for value in jax.tree.leaves(params))
    for rung in (Rung.X0_LOCAL, Rung.X1_JOINT_ACTION, Rung.X2_JOINT_BELIEF):
        del rung
        current = init_predictor(jax.random.key(0), 8, 3, 16, 5)
        assert sum(value.size for value in jax.tree.leaves(current)) == expected


def test_zero_initialized_residual_is_exact_local_baseline():
    dataset = _synthetic_dataset()
    base = init_predictor(jax.random.key(0), 8, 2, 16, 2)
    residual = init_residual(jax.random.key(1), 16, 2)
    baseline = evaluate_predictor(
        base, dataset, np.arange(3), Rung.X0_LOCAL
    )["error"]
    prediction = residual_predictor(
        base,
        residual,
        dataset.belief[:3],
        dataset.belief[:3],
        dataset.action[:3],
        dataset.valid[:3],
        Rung.X1_JOINT_ACTION,
    )
    residual_error = np.asarray(cosine_error(prediction, dataset.target[:3]))
    np.testing.assert_allclose(residual_error, baseline, atol=0, rtol=0)


def test_joint_action_rung_recovers_other_action_signal():
    dataset = _synthetic_dataset()
    train = np.arange(18)
    test = np.arange(18, 24)
    config = LadderConfig(
        feature_width=8,
        hidden=32,
        learning_rate=1e-3,
        steps=500,
        batch_trajectories=8,
        seed=0,
        bootstrap_samples=100,
    )
    local, _ = train_predictor(dataset, train, Rung.X0_LOCAL, config)
    joint, _ = train_predictor(dataset, train, Rung.X1_JOINT_ACTION, config)
    local_error = evaluate_predictor(local, dataset, test, Rung.X0_LOCAL)["error"].mean()
    joint_error = evaluate_predictor(
        joint, dataset, test, Rung.X1_JOINT_ACTION
    )["error"].mean()
    assert joint_error < 0.2 * local_error
