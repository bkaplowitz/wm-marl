"""Correctness invariants for JEPA-native imagined PPO."""

import embodied.jax.nets as nets
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax

from majepa.training.optimization import GroupedOptimizer
from majepa.training.ppo import (
    clipped_policy_objective,
    generalized_advantage_estimate,
    normalize_advantage,
    scheduled_entropy_coefficient,
    value_objective,
)
from majepa.marl.axes import TeamAxis
from majepa.marl.core import MARLCore


def test_gae_counts_death_transition_and_cuts_bootstrap() -> None:
    reward = jnp.asarray([[0.0, 1.0, 2.0, 99.0]])
    continuation = jnp.asarray([[1.0, 0.9, 0.9, 0.9]])
    value = jnp.asarray([[10.0, 20.0, 30.0, 40.0]])
    state_valid = jnp.asarray([[True, True, False, False]])

    returns, advantage, valid, weight = generalized_advantage_estimate(
        reward,
        continuation,
        value,
        state_valid,
        lam=0.95,
    )

    # The second action receives its death-transition reward but cannot
    # bootstrap from the dead next state. Earlier GAE can still include it.
    np.testing.assert_allclose(advantage, [[-6.39, -18.0, 0.0]], atol=1e-5)
    np.testing.assert_allclose(returns, [[3.61, 2.0, 30.0]], atol=1e-5)
    np.testing.assert_array_equal(valid, [[True, True, False]])
    np.testing.assert_allclose(weight, [[1.0, 0.9, 0.0]])


def test_ctde_state_validity_is_agent_specific_and_state_aligned() -> None:
    core = object.__new__(MARLCore)
    core.ctde_enabled = True
    core.team = TeamAxis(2)
    auxiliary = {
        "present": jnp.ones((1, 3, 2), bool),
        "controllable_alive": jnp.asarray(
            [[[True, True], [True, False], [False, False]]]
        ),
    }

    validity = core.imagination_state_validity({}, 2, auxiliary)

    np.testing.assert_array_equal(
        validity,
        [[True, True, False], [True, False, False]],
    )


def test_advantage_normalization_uses_effective_trajectory_weight() -> None:
    advantage = jnp.asarray([[0.0, 2.0, 100.0], [4.0, -100.0, 6.0]])
    valid = jnp.asarray([[True, True, False], [True, True, True]])
    weight = jnp.asarray([[1.0, 0.5, 1.0], [0.25, 0.0, 0.125]])
    normalized = normalize_advantage(advantage, valid, weight)

    selected = valid.astype(jnp.float32) * weight
    count = selected.sum()
    np.testing.assert_allclose((normalized * selected).sum() / count, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        (jnp.square(normalized) * selected).sum() / count,
        1.0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(normalized[weight == 0.0], jnp.zeros((1,)))


def test_policy_objective_clips_ratio_and_reports_divergence() -> None:
    old_logits = jnp.zeros((2, 1, 2))
    new_logits = jnp.asarray([[[8.0, -8.0]], [[8.0, -8.0]]])
    action = jnp.zeros((2, 1), jnp.int32)
    advantage = jnp.asarray([[1.0], [-1.0]])
    valid = jnp.ones((2, 1), bool)
    weight = jnp.ones((2, 1))

    loss, metrics = clipped_policy_objective(
        new_logits,
        old_logits,
        action,
        advantage,
        valid,
        weight,
        clip_epsilon=0.2,
        entropy_coefficient=0.0,
    )

    np.testing.assert_allclose(loss, 0.4, atol=1e-3)
    assert float(metrics["exact_kl"]) > 0.0
    assert float(metrics["approx_kl"]) > 0.0
    np.testing.assert_allclose(metrics["clip_fraction"], 1.0)
    np.testing.assert_allclose(metrics["ratio"], 2.0, atol=1e-3)


def test_entropy_schedule_reaches_endpoints_and_supports_linear() -> None:
    cosine = scheduled_entropy_coefficient(
        jnp.asarray([0, 20_000, 40_000, 80_000]),
        initial=1e-3,
        final=3e-4,
        decay_steps=40_000,
        schedule="cosine",
    )
    linear = scheduled_entropy_coefficient(
        jnp.asarray([0, 20_000, 40_000]),
        initial=1e-3,
        final=3e-4,
        decay_steps=40_000,
        schedule="linear",
    )
    np.testing.assert_allclose(cosine, [1e-3, 6.5e-4, 3e-4, 3e-4])
    np.testing.assert_allclose(linear, [1e-3, 6.5e-4, 3e-4])


def test_normalized_entropy_is_invariant_to_legal_action_count() -> None:
    logits = jnp.asarray([[[0.0, 0.0, -1e30, -1e30]], [[0.0, 0.0, 0.0, 0.0]]])
    action = jnp.zeros((2, 1), jnp.int32)
    advantage = jnp.zeros((2, 1))
    valid = jnp.ones((2, 1), bool)
    weight = jnp.ones((2, 1))
    coefficient = jnp.asarray(0.25)

    loss, metrics = clipped_policy_objective(
        logits,
        logits,
        action,
        advantage,
        valid,
        weight,
        entropy_coefficient=coefficient,
        normalize_entropy=True,
    )

    np.testing.assert_allclose(metrics["normalized_entropy"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["entropy"], 1.0, atol=1e-6)
    np.testing.assert_allclose(metrics["entropy_coefficient"], coefficient)
    np.testing.assert_allclose(loss, -coefficient, atol=1e-6)


def test_policy_targets_are_frozen_and_invalid_entries_have_no_gradient() -> None:
    old = jnp.asarray([[[0.0, 0.0]], [[0.0, 0.0]]])
    new = jnp.asarray([[[0.2, -0.2]], [[4.0, -4.0]]])
    action = jnp.zeros((2, 1), jnp.int32)
    advantage = jnp.ones((2, 1))
    valid = jnp.asarray([[True], [False]])
    weight = jnp.ones((2, 1))

    def objective(candidate, reference, target):
        return clipped_policy_objective(
            candidate,
            reference,
            action,
            target,
            valid,
            weight,
            entropy_coefficient=0.0,
        )[0]

    new_grad, old_grad, advantage_grad = jax.grad(objective, (0, 1, 2))(
        new, old, advantage
    )
    assert float(jnp.linalg.norm(new_grad[0])) > 0.0
    np.testing.assert_array_equal(new_grad[1], jnp.zeros_like(new_grad[1]))
    np.testing.assert_array_equal(old_grad, jnp.zeros_like(old_grad))
    np.testing.assert_array_equal(advantage_grad, jnp.zeros_like(advantage_grad))


class _SquaredValueOutput:
    def __init__(self, prediction):
        self.prediction = prediction

    def pred(self):
        return self.prediction

    def loss(self, target):
        return jnp.square(self.prediction - target)


def test_value_targets_are_frozen_and_weighted() -> None:
    prediction = jnp.asarray([[1.0, 2.0], [8.0, 4.0]])
    target = jnp.asarray([[3.0, 0.0], [100.0, 6.0]])
    valid = jnp.asarray([[True, True], [False, True]])
    weight = jnp.asarray([[1.0, 0.5], [1.0, 0.25]])

    def objective(candidate, frozen_target):
        return value_objective(
            _SquaredValueOutput(candidate),
            frozen_target,
            valid,
            weight,
        )[0]

    prediction_grad, target_grad = jax.grad(objective, (0, 1))(prediction, target)
    assert float(jnp.linalg.norm(prediction_grad)) > 0.0
    np.testing.assert_array_equal(prediction_grad[1, 0], 0.0)
    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target_grad))


def test_ppo_group_step_cannot_modify_world_or_peer_optimizer_state() -> None:
    local_world = nets.Linear(1, name="local_world")
    joint_world = nets.Linear(1, name="joint_world")
    actor = nets.Linear(1, name="actor")
    critic = nets.Linear(1, name="critic")
    optimizer = GroupedOptimizer(
        {
            "local_world": ((local_world,), optax.sgd(0.05)),
            "joint_world": ((joint_world,), optax.sgd(0.05)),
            "actor": ((actor,), optax.sgd(0.05)),
            "critic": ((critic,), optax.sgd(0.05)),
        },
        name="optimizer",
    )
    inputs = nets.cast(jnp.ones((3, 2)))

    def world_loss():
        outputs = [
            module(inputs).astype(jnp.float32)
            for module in (local_world, joint_world, actor, critic)
        ]
        return sum(jnp.square(output).mean() for output in outputs)

    def actor_loss():
        prediction = actor(inputs).astype(jnp.float32)
        return jnp.square(prediction - 1.0).mean()

    def critic_loss():
        prediction = critic(inputs).astype(jnp.float32)
        return jnp.square(prediction + 1.0).mean()

    def initialize():
        optimizer(world_loss, skip_groups=("actor", "critic"))
        optimizer.step_group("actor", actor_loss, active=False)
        optimizer.step_group("critic", critic_loss, active=False)

    def world_step():
        return optimizer(world_loss, skip_groups=("actor", "critic"))

    def actor_step():
        return optimizer.step_group("actor", actor_loss)

    state = nj.init(initialize)({}, seed=41)
    state, metrics = nj.pure(world_step)(state, seed=42)
    np.testing.assert_allclose(metrics["optimizer/actor/skipped"], 1.0)
    np.testing.assert_allclose(metrics["optimizer/critic/skipped"], 1.0)
    before = {key: np.asarray(value) for key, value in state.items()}
    after, metrics = nj.pure(actor_step)(state, seed=43)

    actor_changed = False
    for key, old_value in before.items():
        changed = not np.array_equal(old_value, np.asarray(after[key]))
        if key.startswith("actor/") or key.startswith("optimizer/actor_"):
            actor_changed |= changed
        elif key.startswith(
            ("local_world/", "joint_world/", "critic/", "optimizer/critic_")
        ):
            assert not changed, f"actor-only PPO step modified {key}"
    assert actor_changed
    np.testing.assert_allclose(metrics["optimizer/actor/skipped"], 0.0)
