"""Temporal credit, frozen local weights and current-history reconstruction."""

import elements
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from majepa.marl.core import MARLCore
from majepa.models.target import CriticTarget
from majepa.training import self_fed as module
from majepa.training.ctde import TwoStepAnchors
from test_majepa_self_fed_training import make_tiny, toy_case
from test_majepa_learner_smoke import _synthetic_replay, _assert_finite


@pytest.mark.parametrize("horizon,expected", [(2, 18.0), (4, 140.0), (5, 150.0)])
def test_two_step_credit_stops_at_chunk_boundary_and_freezes_local(
    monkeypatch, horizon, expected
):
    monkeypatch.setattr(
        module,
        "sample_two_step_anchors",
        lambda *_: TwoStepAnchors(jnp.array([0]), jnp.array([0]), jnp.ones(1, bool)),
    )
    agent, tokens, features, entries, obs, actions = toy_case(
        length=horizon + 1, horizons=(horizon,)
    )
    agent.config = elements.Config(
        {**agent.config.flat, "marl.ctde.self_fed.bptt_steps": 2}
    )

    def loss():
        losses, _ = module.self_fed_losses(
            agent, tokens, features, entries, tokens, obs, actions
        )
        return losses["ctde_self_fed_reward"].mean()

    state = nj.init(loss)({}, seed=21)
    pure = nj.pure(loss)
    gradient = jax.grad(lambda params: pure(params, seed=22)[1])(state)
    assert float(gradient["joint/weight"]) == expected
    assert float(gradient["dyn/gain"]) == 0


@pytest.mark.parametrize(
    "fresh,steps,rate", [(True, 1, 1.0), (False, 2, 1.0), (False, 1, 0.5)]
)
def test_extensions_execute_full_active_ppo(fresh, steps, rate):
    original, observations, actions = make_tiny(enabled=True, length=6, anchors=8)
    config = elements.Config(
        {
            **original.config.flat,
            "marl.ctde.self_fed.fresh_history": fresh,
            "marl.ctde.self_fed.bptt_steps": steps,
            "slowvalue.rate": rate,
        }
    )
    learner = object.__new__(MARLCore)
    MARLCore.__init__(learner, observations, actions, config)
    data = _synthetic_replay(learner, observations, actions)
    data = {
        key: jnp.concatenate([value, jnp.repeat(value[:, -1:], 2, axis=1)], axis=1)
        for key, value in data.items()
    }
    for key in data:
        if key.endswith(("is_last", "is_terminal")):
            data[key] = data[key].at[:, 7:].set(False).at[:, 9].set(True)
    data = dict(data, _environment_step=jnp.full((2, 10), 10, jnp.int32))
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=981)
    updated, (_, _, metrics) = jax.jit(nj.pure(learner.train))(
        state, carry, data, seed=982
    )
    _assert_finite((updated, metrics))
    assert float(metrics["ppo/active"]) == 1
    assert float(metrics["ctde/self_fed_train_h5/team_count"]) > 0


def test_fresh_roots_ignore_stored_latents_but_use_raw_observations():
    learner, observations, actions = make_tiny(enabled=True)
    learner.config = elements.Config(
        {**learner.config.flat, "marl.ctde.self_fed.fresh_history": True}
    )
    data = _synthetic_replay(learner, observations, actions)
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=93)
    primary = {k: v for k, v in data.items() if not k.startswith("_behavior_replay/")}
    raw = learner.team.local_sequence_data(primary)

    def roots(data):
        return module.recurrent_training_inputs(
            learner, None, None, {"_self_fed_raw": data}
        )

    pure = nj.pure(roots)
    baseline = pure(state, raw, seed=94, create=False)[1]
    changed_codes = dict(
        raw,
        **{
            key: value + 7
            for key, value in raw.items()
            if key in ("dyn/stoch", "dyn/pair")
        },
    )
    changed = pure(state, changed_codes, seed=94, create=False)[1]
    for x, y in zip(jax.tree.leaves(baseline), jax.tree.leaves(changed)):
        np.testing.assert_array_equal(x, y)
    changed_obs = dict(raw, observation=raw["observation"].at[:, :4].add(7))
    observed = pure(state, changed_obs, seed=94, create=False)[1]
    assert any(
        not np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(baseline), jax.tree.leaves(observed))
    )


class Weight(nj.Module):
    def __call__(self):
        return self.value("weight", lambda: jnp.float32(2))


@pytest.mark.parametrize("rate", [0.02, 0.5, 1.0])
def test_critic_target_convex_update_and_default_parity(rate):
    source, target = Weight(name="source"), Weight(name="target")
    slow = CriticTarget(target, source=source, rate=rate)

    def initialize():
        source()
        slow.count.read()
        return slow()

    state = nj.init(initialize)({}, seed=1)
    state["source/weight"] = jnp.float32(10)

    def update():
        slow.update()
        return slow()

    _, result = nj.pure(update)(state, seed=2)
    assert float(result) == pytest.approx((1 - rate) * 2 + rate * 10)
    if rate != 0.5:
        original = embodied.jax.SlowModel(target, source=source, rate=rate)

        def baseline():
            original.update()
            return original()

        expected, _ = nj.pure(baseline)(state, seed=2)
        actual, _ = nj.pure(update)(state, seed=2)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
