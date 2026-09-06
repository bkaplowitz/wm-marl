"""Recorded-action alignment, bounded traces, and representation gradients."""

from types import SimpleNamespace

import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from majepa.marl.axes import TeamAxis
from majepa.models.heads import apply_action_mask, apply_legal_unimix
from majepa.training.factual_value import factual_vtrace_return, joint_action_logratio
from majepa.training.learner import LearnerMixin
from majepa.training.policy import PolicyMixin
from majepa.training.replay import ReplayMixin
from majepa.training.replay_value import replay_lambda_return


def _targets(
    reward,
    value,
    ratio,
    *,
    first=None,
    last=None,
    terminal=None,
    present=None,
    lam=0.95,
):
    flags = jnp.zeros_like(reward, bool)
    return factual_vtrace_return(
        reward,
        flags if first is None else first,
        flags if last is None else last,
        flags if terminal is None else terminal,
        ~flags if present is None else present,
        value,
        ratio,
        discount=0.9,
        lam=lam,
        rho_clip=1.0,
        c_clip=1.0,
    )


def test_on_policy_trace_matches_real_lambda_with_terminal_reset_and_truncation():
    reward = jnp.array([[0.0, 1.0, 7.0, 0.0, 2.0, 0.0, 100.0]])
    first = jnp.array([[True, False, False, True, False, True, False]])
    last = jnp.array([[False, False, True, False, True, False, False]])
    terminal = last.at[:, 4].set(False)
    value = jnp.array([[3.0, 4.0, 50.0, 6.0, 10.0, 200.0, 300.0]])
    expected, mask = replay_lambda_return(
        reward,
        first,
        last,
        terminal,
        ~jnp.zeros_like(first),
        value,
        discount=0.9,
        lam=0.95,
    )
    actual, valid, _ = _targets(
        reward, value, jnp.zeros_like(value), first=first, last=last, terminal=terminal
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_array_equal(valid, mask)
    assert float(actual[0, 1]) == 7.0
    assert float(actual[0, 3]) == 11.0


def test_vtrace_uses_source_action_ratio_and_bounds_extreme_ratios():
    # V=[2,4,8], rewards=[1,3], rho=[.5,1]. Last logratio is unused.
    # delta1=3+.9*8-4=6.2; delta0=.5*(1+.9*4-2)=1.3.
    # v0=2+1.3+.9*.5*6.2=6.09, v1=4+6.2=10.2.
    reward = jnp.array([[0.0, 1.0, 3.0]])
    value = jnp.array([[2.0, 4.0, 8.0]])
    ratio = jnp.array([[np.log(0.5), 1000.0, -1000.0]])
    actual, _, metrics = _targets(reward, value, ratio, lam=1.0)
    np.testing.assert_allclose(actual, [[6.09, 10.2]], rtol=1e-5)
    assert np.isfinite(np.asarray(actual)).all()
    assert float(metrics["rho_clipped_fraction"]) == 0.5
    for grad in jax.grad(lambda r, v, w: _targets(r, v, w)[0].sum(), (0, 1, 2))(
        reward, value, ratio
    ):
        np.testing.assert_array_equal(grad, jnp.zeros_like(grad))


def test_joint_ratios_use_all_acting_agents_and_keep_dead_team_returns():
    team = TeamAxis(2)
    current = team.fold_sequence(
        jnp.log(jnp.array([[[0.5, 0.2], [0.4, 0.1], [0.2, 0.3]]]))
    )
    behavior = team.fold_sequence(
        jnp.log(jnp.array([[[0.25, 0.4], [0.8, 0.9], [0.6, 0.6]]]))
    )
    alive = team.fold_sequence(
        jnp.array([[[True, True], [True, False], [False, False]]])
    )
    ratio = joint_action_logratio(current, behavior, alive, team)
    np.testing.assert_allclose(
        jnp.exp(ratio), [[1.0, 0.5, 1.0], [1.0, 0.5, 1.0]], rtol=1e-5
    )
    reward = jnp.array([[0.0, 1.0, 10.0], [0.0, 1.0, 10.0]])
    result, valid, _ = _targets(
        reward, jnp.zeros_like(reward), jnp.zeros_like(reward), lam=1.0
    )
    np.testing.assert_allclose(result, [[10.0, 10.0], [10.0, 10.0]])
    assert bool(valid.all())  # death alone is not an episode boundary


def test_absence_breaks_trace_without_leaking_next_episode():
    reward = jnp.array([[0.0, 1.0, 1000.0, 3000.0]])
    value = jnp.array([[2.0, 4.0, 6.0, 8.0]])
    present = jnp.array([[True, True, False, True]])
    target, valid, _ = _targets(
        reward, value, jnp.zeros_like(value), present=present, lam=1.0
    )
    np.testing.assert_allclose(target, [[4.6, 4.0, 6.0]], rtol=1e-5)
    np.testing.assert_array_equal(valid, [[True, False, False]])


def test_policy_records_actual_masked_collection_mixture_probability():
    actor = object.__new__(PolicyMixin)
    actor.factual_value_enabled = True
    actor.action_mask_key = "action"
    actor.config = SimpleNamespace(collection_unimix=0.2, replay_context=0)
    actor.enc = lambda carry, obs, reset, **kw: (carry, {}, obs["observation"])
    actor.observe_dynamics = lambda carry, tokens, prev, reset, obs, **kw: (
        carry,
        {},
        {"x": tokens},
        {},
    )
    actor.feat2tensor = lambda feat: feat["x"]
    actor.dec = None
    logits = jnp.log(jnp.array([[0.9, 0.09, 0.01]]))
    actor.policy_distribution = lambda tensor, bdims, action_mask: apply_action_mask(
        {"action": outs.Categorical(logits)}, action_mask, "action"
    )
    obs = {
        "observation": jnp.ones((1, 2)),
        "is_first": jnp.array([True]),
        "action_mask": jnp.array([[True, True, False]]),
    }
    # sample() uses Ninjax RNG; exercise the actual policy through its context.
    import ninjax as nj

    _, (_, act, output) = nj.pure(actor.policy)(
        {}, ({}, {}, {}, {"action": jnp.zeros(1, jnp.int32)}), obs, seed=12
    )
    expected = apply_legal_unimix(
        actor.policy_distribution(None, 1, obs["action_mask"]), "action", 0.2
    )["action"]
    np.testing.assert_allclose(
        output["behavior_logprob"], expected.logp(act["action"]), atol=1e-6
    )
    assert int(act["action"][0]) != 2


def test_factual_target_ignores_imagined_root_returns_and_auxiliary_reaches_features():
    learner = object.__new__(LearnerMixin)
    learner.factual_value_enabled = True
    learner.team = TeamAxis(2)
    learner.action_mask_key = "action"
    learner.config = SimpleNamespace(
        horizon=10,
        ppo=SimpleNamespace(
            replay_value_lam=0.0,
            factual_value=SimpleNamespace(rho_clip=1.0, c_clip=1.0),
        ),
    )
    learner._controllable = lambda obs: obs["controllable_alive"]
    learner.feat2tensor = lambda f: f["deter"]
    learner.critic = lambda features, *args, **kwargs: outs.MSE(
        features["deter"][..., 0] * 2
    )
    learner.real_value = lambda x, bdims: outs.MSE(x[..., 0])
    learner.policy_distribution = lambda tensor, bdims, action_mask: {
        "action": outs.Categorical(jnp.zeros((*tensor.shape[:2], 2)))
    }
    x = jnp.arange(1.0, 7.0).reshape(2, 3, 1)
    flags = jnp.zeros((2, 3), bool)
    obs = {
        "reward": jnp.ones((2, 3)),
        "is_first": flags,
        "is_last": flags,
        "is_terminal": flags,
        "agent_present": ~flags,
        "controllable_alive": ~flags,
        "action_mask": jnp.ones((2, 3, 2), bool),
        "behavior_logprob": jnp.full((2, 3), -jnp.log(2.0)),
        "_replay_action": jnp.zeros((2, 3), jnp.int32),
    }
    a = learner._prepare_replay_value_batch({"deter": x}, obs, jnp.zeros(6), 3)
    b = learner._prepare_replay_value_batch({"deter": x}, obs, jnp.full(6, 9999.0), 3)
    np.testing.assert_array_equal(a["target_return"], b["target_return"])
    np.testing.assert_allclose(a["target_return"], 1 + 0.9 * 2 * x[:, 1:, 0])
    targetgrad = jax.grad(
        lambda f: learner._prepare_factual_value_batch({"deter": f}, obs)[
            "target_return"
        ].sum()
    )(x)
    np.testing.assert_array_equal(targetgrad, jnp.zeros_like(x))
    grad = jax.grad(
        lambda f: learner._factual_representation_loss({"deter": f}, obs)[0]
    )(x)
    assert bool((jnp.abs(grad[:, :-1]) > 0).all())
    np.testing.assert_array_equal(grad[:, -1], jnp.zeros_like(grad[:, -1]))


def test_replay_metadata_is_not_an_encoder_observation():
    replay = object.__new__(ReplayMixin)
    replay.obs_space = {"observation": None}
    replay.action_mask_key = "action"
    replay.factual_value_enabled = True
    data = {
        "observation": jnp.ones((1, 3, 2)),
        "behavior_logprob": jnp.full((1, 3), -0.5),
        "action": jnp.array([[1, 0, 1]]),
    }
    obs = replay._replay_observations(data)
    assert "behavior_logprob" not in replay.obs_space
    np.testing.assert_array_equal(obs["_replay_action"], data["action"])
    with pytest.raises(KeyError):
        replay._replay_observations(
            {"observation": data["observation"], "action": data["action"]}
        )
