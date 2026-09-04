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


def test_gae_counts_last_transition_and_cuts_absent_state_bootstrap() -> None:
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

    # The second action receives its transition reward but cannot bootstrap
    # from an absent/padded next state. Earlier GAE can still include it.
    np.testing.assert_allclose(advantage, [[-6.39, -18.0, 0.0]], atol=1e-5)
    np.testing.assert_allclose(returns, [[3.61, 2.0, 30.0]], atol=1e-5)
    np.testing.assert_array_equal(valid, [[True, True, False]])
    np.testing.assert_allclose(weight, [[1.0, 0.9, 0.0]])


def test_team_gae_preserves_reward_after_focal_death_and_masks_actor() -> None:
    # Focal agent dies after the first action, then surviving teammates win.
    reward = jnp.asarray([[0.0, 1.0, 2.0, 10.0]])
    continuation = jnp.asarray([[1.0, 0.9, 0.9, 0.0]])
    value = jnp.zeros_like(reward)
    present = jnp.ones_like(reward, bool)
    alive = jnp.asarray([[True, False, False, False]])

    returns, advantage, actor_valid, weight = generalized_advantage_estimate(
        reward,
        continuation,
        value,
        present,
        decision_state_valid=alive,
        lam=1.0,
    )

    np.testing.assert_allclose(returns, [[10.9, 11.0, 10.0]], atol=1e-6)
    np.testing.assert_allclose(advantage, [[10.9, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_array_equal(actor_valid, [[True, False, False]])
    np.testing.assert_allclose(weight, [[1.0, 0.9, 0.81]], atol=1e-6)

    # The critic must learn those post-death values to supply the bootstrap
    # when a shorter imagination ends before the surviving teammates win.
    critic_grad = jax.grad(
        lambda prediction: value_objective(
            _SquaredValueOutput(prediction), returns, present[:, :-1], weight
        )[0]
    )(jnp.zeros_like(returns))
    assert bool((critic_grad < 0.0).all())


def test_team_gae_prefers_cooperative_sacrifice_over_survival_only_reward() -> None:
    # The sacrifice has delayed team payoff 10; surviving yields only 1.
    # Treating focal death as termination reverses this action preference.
    reward = jnp.asarray([[0.0, 0.0, 10.0], [0.0, 1.0, 0.0]])
    continuation = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    value = jnp.zeros_like(reward)
    alive = jnp.asarray([[True, False, False], [True, True, False]])
    present = jnp.ones_like(alive)

    returns, advantage, valid, weight = generalized_advantage_estimate(
        reward,
        continuation,
        value,
        present,
        decision_state_valid=alive,
        lam=1.0,
    )
    old_returns = generalized_advantage_estimate(
        reward, continuation, value, alive, lam=1.0
    )[0]
    assert returns[0, 0] > returns[1, 0]
    assert old_returns[0, 0] < old_returns[1, 0]

    root_advantage = normalize_advantage(advantage[:, :1], valid[:, :1], weight[:, :1])
    old_logits = jnp.zeros((2, 1, 2))
    actions = jnp.asarray([[0], [1]])

    def actor_loss(shared_logits):
        return clipped_policy_objective(
            jnp.broadcast_to(shared_logits, old_logits.shape),
            old_logits,
            actions,
            root_advantage,
            valid[:, :1],
            weight[:, :1],
            entropy_coefficient=0.0,
        )[0]

    gradient = jax.grad(actor_loss)(jnp.zeros((2,)))
    assert gradient[0] < 0.0  # Gradient descent increases sacrifice probability.
    assert gradient[1] > 0.0


def test_team_gae_cannot_cross_episode_terminal_after_focal_death() -> None:
    reward = jnp.asarray([[0.0, 1.0, 1000.0]])
    continuation = jnp.asarray([[1.0, 0.0, 1.0]])
    value = jnp.asarray([[0.0, 500.0, 500.0]])
    present = jnp.ones_like(reward, bool)
    alive = jnp.asarray([[True, False, False]])

    returns, advantage, _, weight = generalized_advantage_estimate(
        reward,
        continuation,
        value,
        present,
        decision_state_valid=alive,
        lam=1.0,
    )

    np.testing.assert_allclose(returns[:, :1], [[1.0]])
    np.testing.assert_allclose(advantage, [[1.0, 0.0]])
    np.testing.assert_allclose(weight, [[1.0, 0.0]])


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


def test_nonfinite_world_gradient_skips_complete_optimizer_transaction() -> None:
    local_world = nets.Linear(1, name="local_world")
    joint_world = nets.Linear(1, name="joint_world")
    optimizer = GroupedOptimizer(
        {
            "local_world": ((local_world,), optax.adam(0.01)),
            "joint_world": ((joint_world,), optax.adam(0.01)),
        },
        name="optimizer",
    )
    inputs = nets.cast(jnp.ones((3, 2)))

    def good_loss():
        local = local_world(inputs).astype(jnp.float32)
        joint = joint_world(inputs).astype(jnp.float32)
        return jnp.square(local - 1.0).mean() + jnp.square(joint + 1.0).mean()

    def bad_loss():
        local = local_world(inputs).astype(jnp.float32)
        joint = joint_world(inputs).astype(jnp.float32)
        # Finite objective, undefined local derivative; the joint derivative
        # remains valid. Neither half of the coupled world update can commit.
        zero = local - jax.lax.stop_gradient(local)
        return jnp.sqrt(jnp.square(zero).sum()) + jnp.square(joint + 1.0).mean()

    def good_step():
        return optimizer(good_loss)

    def bad_step():
        return optimizer(bad_loss)

    state = nj.init(good_step)({}, seed=810)
    state, _ = jax.jit(nj.pure(good_step))(state, seed=811)
    before = {key: np.asarray(value) for key, value in state.items()}
    failed_state, metrics = jax.jit(nj.pure(bad_step))(state, seed=812)

    assert float(metrics["optimizer/finite"]) == 0.0
    assert float(metrics["optimizer/local_world/active"]) == 0.0
    assert float(metrics["optimizer/joint_world/active"]) == 0.0
    assert not bool(jnp.isfinite(metrics["optimizer/local_world/grad_norm"]))
    assert bool(jnp.isfinite(metrics["optimizer/joint_world/grad_norm"]))
    for key, value in before.items():
        np.testing.assert_array_equal(value, failed_state[key], err_msg=key)

    recovered_state, recovered_metrics = jax.jit(nj.pure(good_step))(
        failed_state, seed=813
    )
    assert float(recovered_metrics["optimizer/finite"]) == 1.0
    for key, value in recovered_state.items():
        assert np.isfinite(np.asarray(value).astype(np.float32)).all(), key
    assert any(
        not np.array_equal(value, recovered_state[key])
        for key, value in before.items()
        if key.startswith(("local_world/", "joint_world/"))
    )
