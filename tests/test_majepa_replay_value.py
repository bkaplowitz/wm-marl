"""Real rewards, episode boundaries, and team ordering in critic grounding."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from majepa.marl.axes import TeamAxis
from majepa.training.learner import LearnerMixin
from majepa.training.replay_value import replay_lambda_return


def test_replay_returns_keep_terminal_reward_and_do_not_cross_reset():
    reward = jnp.array([[0.0, 1.0, 7.0, 0.0, 100.0]])
    first = jnp.array([[True, False, False, True, False]])
    last = jnp.array([[False, False, True, False, True]])
    bootstrap = jnp.full_like(reward, 999.0)
    returns, valid = replay_lambda_return(
        reward,
        first,
        last,
        last,
        jnp.ones_like(first),
        bootstrap,
        discount=0.9,
        lam=1.0,
    )
    np.testing.assert_allclose(returns, [[7.3, 7.0, 999.0, 100.0]])
    np.testing.assert_array_equal(valid, [[True, True, False, True]])


def test_replay_truncation_bootstraps_once_without_reset_leakage():
    reward = jnp.array([[0.0, 2.0, 0.0, 100.0]])
    first = jnp.array([[True, False, True, False]])
    last = jnp.array([[False, True, False, False]])
    terminal = jnp.zeros_like(first)
    bootstrap = jnp.array([[3.0, 10.0, 200.0, 300.0]])
    returns, valid = replay_lambda_return(
        reward,
        first,
        last,
        terminal,
        jnp.ones_like(first),
        bootstrap,
        discount=0.9,
        lam=0.95,
    )
    np.testing.assert_allclose(returns[0, 0], 11.0)
    np.testing.assert_array_equal(valid, [[True, False, True]])


def test_replay_bootstrap_is_frozen_and_absence_excludes_transition():
    reward = jnp.array([[0.0, 1.0, 2.0]])
    flags = jnp.zeros_like(reward, bool)
    present = jnp.array([[True, True, False]])

    def objective(bootstrap, rewards):
        return replay_lambda_return(
            rewards,
            flags,
            flags,
            flags,
            present,
            bootstrap,
            discount=0.9,
            lam=0.5,
        )[0].sum()

    gradients = jax.grad(objective, (0, 1))(reward, reward)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(reward))
    _, valid = replay_lambda_return(
        reward,
        flags,
        flags,
        flags,
        present,
        reward,
        discount=0.9,
        lam=0.5,
    )
    np.testing.assert_array_equal(valid, [[True, False]])


def test_replay_value_restores_team_agent_time_order_and_keeps_dead_slots():
    learner = object.__new__(LearnerMixin)
    learner.team = TeamAxis(2)
    learner.config = SimpleNamespace(
        horizon=10, ppo=SimpleNamespace(replay_value_lam=0.0)
    )
    learner._controllable = lambda obs: obs["controllable_alive"]
    learner.critic = lambda features, *args, **kwargs: SimpleNamespace(
        pred=lambda: jnp.full(features["deter"].shape[:2], 20.0)
    )
    # Unique values for [team, time, agent], deliberately unequal across agents.
    grouped = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
    folded = learner.team.fold_sequence(grouped)
    flags = jnp.zeros_like(folded, bool)
    obs = {
        "reward": jnp.zeros_like(folded),
        "is_first": flags,
        "is_last": flags,
        "is_terminal": flags,
        "agent_present": jnp.ones_like(flags),
        "controllable_alive": flags.at[0, :].set(True),
    }
    batch = learner._prepare_replay_value_batch(
        {"deter": folded[..., None]}, obs, grouped.reshape(-1), 3
    )
    np.testing.assert_allclose(batch["target_return"], 0.9 * folded[:, 1:])
    np.testing.assert_array_equal(batch["features"]["deter"], folded[:, :-1, None])
    np.testing.assert_array_equal(batch["valid"], jnp.ones((4, 2), bool))
    np.testing.assert_array_equal(
        batch["context"]["controllable_alive"],
        learner.team.unfold_sequence(obs["controllable_alive"])[:, :-1],
    )
    # A truncated imagination root has an absent roster and its root_return
    # cannot be used as a factual final-state bootstrap.
    truncated = dict(obs, is_last=flags.at[:, 1].set(True))
    batch = learner._prepare_replay_value_batch(
        {"deter": folded[..., None]}, truncated, grouped.reshape(-1), 3
    )
    np.testing.assert_allclose(batch["target_return"][:, 0], 18.0)
    np.testing.assert_array_equal(batch["valid"][:, 1], False)
