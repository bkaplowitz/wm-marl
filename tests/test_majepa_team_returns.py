"""Shared SMAC outcomes must survive individual death in the joint model."""

from types import SimpleNamespace

import embodied.jax.outs as outs
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from majepa.marl.axes import TeamAxis
from majepa.marl.core import MARLCore
from majepa.training.ctde import broadcast_team_mean, shared_team_outcomes


def test_shared_team_signal_includes_dead_roster_and_ignores_absent_padding():
    present = jnp.asarray([[True, True, False], [False, False, False]])
    values = jnp.asarray([[2.0, 6.0, 1000.0], [10.0, 20.0, 30.0]])

    actual = broadcast_team_mean(values, present)

    np.testing.assert_allclose(actual, [[4.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    grad = jax.grad(lambda x: broadcast_team_mean(x, present)[0, 0])(values)
    np.testing.assert_allclose(grad, [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])


def test_ctde_value_roster_survives_death_while_actor_decisions_stop():
    core = object.__new__(MARLCore)
    core.ctde_enabled = True
    core.team = TeamAxis(2)
    auxiliary = {
        "present": jnp.ones((1, 3, 2), bool),
        "controllable_alive": jnp.asarray(
            [[[True, True], [True, False], [False, False]]]
        ),
    }

    decision = core.imagination_state_validity({}, 2, auxiliary)
    bootstrap = core.imagination_bootstrap_validity({}, 2, auxiliary)

    np.testing.assert_array_equal(decision, [[True, True, False], [True, False, False]])
    np.testing.assert_array_equal(bootstrap, np.ones((2, 3), bool))


def test_shared_outcomes_keep_final_reward_and_stop_absorbing_tail():
    reward, continuation = shared_team_outcomes(
        jnp.asarray([[7.0, 9.0], [99.0, 99.0], [3.0, 5.0]]),
        jnp.ones((3, 2)),
        jnp.ones((3, 2), bool),
        jnp.asarray([[True, False], [False, False], [True, True]]),
        jnp.asarray([[False, False], [False, False], [True, False]]),
    )

    np.testing.assert_allclose(reward, [[8.0, 8.0], [0.0, 0.0], [4.0, 4.0]])
    np.testing.assert_allclose(continuation, [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])


def _replay_losses(reward_prediction):
    """Exercise the actual loss alignment with deterministic, tiny heads."""

    core = object.__new__(MARLCore)
    core.team = TeamAxis(2)
    core.ctde_action_key = "action"
    core.ctde_soft_liveness = False
    core.ctde_teammate_belief_enabled = False
    core.ctde_multistep_jepa_enabled = False
    core.ctde_mask_calibration = False
    core.ctde_rollout_steps = 1
    core.action_mask_reduction = "mean"
    core.config = SimpleNamespace(contdisc=False)
    core.feat2tensor = lambda features: features["deter"]
    core.dyn = SimpleNamespace(
        posterior=lambda embedding, deter: jnp.zeros((*embedding.shape[:2], 1, 2))
    )

    def joint_sequence(cache, states, *args):
        del args
        embedding = jnp.zeros_like(states)
        return cache, {"hidden": embedding, "embedding": embedding}, {}

    core.ctde_joint = SimpleNamespace(sequence=joint_sequence)
    core.ctde_rew = lambda hidden, bdims: SimpleNamespace(
        loss=lambda target: jnp.square(reward_prediction - target)
    )
    core.ctde_con = lambda hidden, bdims: outs.Binary(jnp.full(hidden.shape[:-1], 2.0))
    core.ctde_mask = lambda hidden, bdims: outs.Binary(
        jnp.zeros((*hidden.shape[:-1], 2))
    )
    core.ctde_alive = lambda hidden, bdims: outs.Binary(jnp.zeros(hidden.shape[:-1]))

    # Unit 1 dies on the first transition. The team receives its final reward
    # on the second transition. The third transition crosses an episode reset.
    fold = core.team.fold_sequence
    first = jnp.asarray([[[True, True], [False, False], [False, False], [True, True]]])
    obs = {
        "is_first": fold(first),
        "is_terminal": fold(
            jnp.asarray(
                [[[False, False], [False, False], [True, True], [False, False]]]
            )
        ),
        "agent_present": jnp.ones((2, 4), bool),
        "controllable_alive": fold(
            jnp.asarray([[[True, True], [True, False], [False, False], [True, True]]])
        ),
        "reward": fold(
            jnp.asarray([[[0.0, 0.0], [2.0, 2.0], [7.0, 7.0], [99.0, 99.0]]])
        ),
        "action_mask": jnp.ones((2, 4, 2), bool),
    }
    tokens = jnp.ones((2, 4, 2))
    return core._ctde_replay_losses(
        tokens,
        {"deter": tokens},
        {"ctde_joint_carry": {}},
        tokens,
        obs,
        {"action": jnp.zeros((2, 4), jnp.int32)},
        training=True,
    )


def test_ctde_shared_outcomes_train_dead_slots_and_keep_terminal_reward():
    losses, metrics = _replay_losses(jnp.zeros((1, 3, 2)))

    # Averaging the normalized replay loss gives the mean over both factual
    # transitions and both roster slots: (2**2 + 7**2) / 2.
    np.testing.assert_allclose(losses["ctde_reward"].mean(), 26.5)
    np.testing.assert_allclose(metrics["ctde/reward_loss"], 26.5)
    assert float(losses["ctde_reward"][1, 1]) > 0.0
    assert float(losses["ctde_continuation"][1, 1]) > 0.0
    np.testing.assert_array_equal(losses["ctde_reward"][:, 2:], 0.0)
    np.testing.assert_array_equal(losses["ctde_continuation"][:, 2:], 0.0)
    # Local embedding supervision still ends once the source unit is dead.
    assert float(losses["ctde_embedding"][1, 1]) == 0.0

    grad = jax.grad(
        lambda prediction: _replay_losses(prediction)[0]["ctde_reward"].mean()
    )(jnp.zeros((1, 3, 2)))
    np.testing.assert_allclose(grad[0, 1], [-3.5, -3.5])
    np.testing.assert_array_equal(grad[0, 2], 0.0)


def test_self_fed_report_uses_predicted_states_and_excludes_reset_crossings():
    core = object.__new__(MARLCore)
    core.team = TeamAxis(2)
    core.ctde_action_key = "action"
    core.ctde_soft_liveness = False
    core.ctde_mask_calibration = False
    core.config = SimpleNamespace(contdisc=False)
    core.feat2tensor = lambda feature: feature["deter"]
    core.ctde_joint = SimpleNamespace(
        step=lambda cache, state, *args, **kwargs: (
            cache,
            {"embedding": state + 1.0, "hidden": state},
        )
    )

    def logits(tokens):
        return jnp.zeros((*tokens.shape[:-1], 1, 2))

    def complete(cache, deter, tokens, sample):
        del cache, deter, sample
        features = {"deter": tokens, "stoch": logits(tokens).astype(tokens.dtype)}
        return features, dict(features, logit=logits(tokens))

    core.dyn = SimpleNamespace(
        advance=lambda carry, *args, **kwargs: ({}, carry["deter"]),
        complete_from_observation=complete,
        posterior=lambda tokens, deter: logits(tokens),
    )
    core.ctde_rew = lambda hidden, bdims: SimpleNamespace(pred=lambda: hidden[..., 0])
    core.ctde_con = lambda hidden, bdims: SimpleNamespace(
        prob=lambda event: jnp.ones(hidden.shape[:-1])
    )
    core.ctde_alive = core.ctde_con
    core.ctde_mask = lambda hidden, bdims: outs.Binary(
        jnp.full((*hidden.shape[:-1], 2), 10.0)
    )
    deter = jnp.full((2, 6, 1), 100.0).at[:, 0].set(0.0)
    entries = {
        "deter": deter,
        "stoch": jnp.zeros((2, 6, 1, 2)),
        "ctde_joint_carry": {"dummy": jnp.zeros((2,))},
    }
    obs = {
        "agent_present": jnp.ones((2, 6), bool),
        "controllable_alive": jnp.ones((2, 6), bool),
        "is_first": jnp.zeros((2, 6), bool).at[:, 0].set(True).at[:, 3].set(True),
        "is_terminal": jnp.zeros((2, 6), bool).at[:, 2].set(True),
        "reward": jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 1.0, 100.0, 100.0, 100.0]), (2, 6)
        ),
        "action_mask": jnp.ones((2, 6, 2), bool),
    }
    tokens = jnp.ones((2, 6, 1))

    def report():
        return core._ctde_self_fed_report(
            {"deter": deter},
            entries,
            {"dummy": jnp.zeros((2, 5))},
            tokens,
            tokens,
            obs,
            {"action": jnp.zeros((2, 6), jnp.int32)},
        )

    params = nj.init(report)({}, seed=0)
    _, metrics = jax.jit(nj.pure(report))(params, seed=1)

    # Second reward is generated from the first predicted state (1), not the
    # factual posterior (100). The terminal transition itself stays valid.
    np.testing.assert_allclose(metrics["ctde/self_fed_h2/reward_rmse"], 0.0)
    np.testing.assert_allclose(metrics["ctde/self_fed_h2/valid_count"], 2.0)
    np.testing.assert_allclose(metrics["ctde/self_fed_h4/valid_count"], 0.0)
    np.testing.assert_allclose(metrics["ctde/self_fed_h5/valid_count"], 0.0)
