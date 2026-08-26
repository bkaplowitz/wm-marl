"""Experimental gate contract for the candidate CTDE generalist profiles."""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import ninjax as nj
import elements

from dreamarl.main import _load_configs, _resolve_config_profiles, make_replay
from dreamarl.marl.core import MARLCore
from dreamarl.marl.spaces import add_agent_axis
from dreamarl.replay import DualViewReplay, ExponentialRecency
from dreamarl.train import _with_prefixed_batch


def _resolved(*profiles: str):
    return _resolve_config_profiles(_load_configs(), ["smac_vector", "ctde", *profiles])


def test_generalist_profile_selects_the_candidate_algorithm() -> None:
    config = _resolved("ctde_generalist")

    assert config.agent.marl.stage == "ctde"
    assert config.agent.marl.execution == "strict_decentralized"
    assert config.agent.marl.ctde.rollout_steps == 1
    assert config.agent.marl.ctde.multistep.anchors == 0
    assert config.agent.marl.ctde.joint.width == 256
    assert config.agent.marl.ctde.joint.temporal_layers == 12
    assert config.agent.imag_length == 15
    assert config.replay_context == 192

    assert config.agent.opt.lr == pytest.approx(4e-5)
    assert config.agent.marl.ctde.opt.lr == pytest.approx(4e-5)
    assert config.agent.marl.ctde.actor_lr == pytest.approx(1e-5)
    assert config.agent.marl.ctde.actor_update_every == 1
    assert config.agent.marl.ctde.critic.width == 256

    # Disabled calibration selects the original central hard-thresholded
    # availability and irreversible-liveness imagination path.
    assert not config.agent.marl.ctde.mask_calibration.enabled
    assert not config.agent.marl.ctde.mask_calibration.soft_liveness
    assert config.agent.action_mask_reduction == "sum"

    assert config.replay.size == 250_000
    assert not config.replay.online
    assert config.replay.sampling == "recent_world_uniform_behavior"
    assert config.replay.recency_decay == pytest.approx(0.9998)
    assert config.run.train_ratio == 256


def test_generalist_is_original_jepa_reinforce_without_experimental_paths() -> None:
    config = _resolved("ctde_generalist")

    assert config.agent.loss_scales.posterior_jepa == pytest.approx(2.0)
    assert config.agent.loss_scales.dynamics_jepa == pytest.approx(2.0)
    assert config.agent.loss_scales.ctde_embedding == pytest.approx(2.0)
    assert config.agent.actor_trust.mode == "none"

    prohibited = ("ppo", "pge", "consumer")
    assert not {
        key
        for key in config.flat
        if any(fragment in key.lower() for fragment in prohibited)
    }
    assert config.agent.marl.ctde.rollout_steps == 1
    assert config.agent.marl.ctde.multistep.anchors == 0
    assert not config.agent.marl.ctde.mask_calibration.enabled
    assert not config.agent.marl.ctde.mask_calibration.soft_liveness


def test_generalist_is_a_narrow_override_of_maintained_ctde() -> None:
    maintained = _resolved()
    generalist = _resolved("ctde_generalist")
    keys = set(maintained.flat) | set(generalist.flat)
    differences = {
        key: (maintained.flat.get(key), generalist.flat.get(key))
        for key in keys
        if maintained.flat.get(key) != generalist.flat.get(key)
    }

    assert differences == {
        "agent.marl.ctde.actor_lr": (4e-5, 1e-5),
        "agent.marl.ctde.joint.temporal_layers": (4, 12),
        "replay_context": (128, 192),
        "replay.online": (True, False),
        "replay.sampling": ("uniform", "recent_world_uniform_behavior"),
        "replay.size": (5e6, 250_000),
    }


def test_mask_mean_treatment_changes_only_per_action_bce_reduction() -> None:
    control = _resolved("ctde_generalist")
    treatment = _resolved("ctde_generalist", "ctde_generalist_mask_mean")
    keys = set(control.flat) | set(treatment.flat)
    differences = {
        key: (control.flat.get(key), treatment.flat.get(key))
        for key in keys
        if control.flat.get(key) != treatment.flat.get(key)
    }

    assert differences == {"agent.action_mask_reduction": ("sum", "mean")}


def test_generalist_replay_factory_selects_the_dual_view_transport(tmp_path) -> None:
    config = _resolved("ctde_generalist").update(logdir=str(tmp_path / "run"))
    replay = make_replay(config, "replay")
    try:
        assert isinstance(replay, DualViewReplay)
        assert isinstance(replay.sampler, ExponentialRecency)
        assert replay.dual_view
        assert replay.capacity == 250_000
        assert not replay.online
        assert replay.length == config.batch_length + config.replay_context
        assert replay.optimized_length == config.batch_length
    finally:
        replay.workers.shutdown(wait=True)


def test_dual_view_replay_routes_world_and_behavior_without_merging_axes() -> None:
    replay = DualViewReplay(
        length=3,
        capacity=32,
        online=False,
        recency_decay=0.9998,
        seed=17,
    )
    for step in range(12):
        replay.add(
            {
                "is_first": np.asarray(step == 0),
                "value": np.asarray(step, np.int32),
            }
        )

    world = replay.sample(2, "train_world")
    behavior = replay.sample(2, "train_behavior")
    paired = next(
        _with_prefixed_batch(
            iter([world]),
            iter([behavior]),
            "_behavior_replay/",
        )
    )

    assert world["value"].shape == (2, 3)
    assert behavior["value"].shape == (2, 3)
    assert paired["value"].shape == (2, 3)
    assert paired["_behavior_replay/value"].shape == (2, 3)
    np.testing.assert_array_equal(paired["value"], world["value"])
    np.testing.assert_array_equal(paired["_behavior_replay/value"], behavior["value"])
    assert not any("role" in key for key in paired)

    stats = replay.stats()
    assert stats["world_samples"] == 2
    assert stats["behavior_samples"] == 2
    assert stats["world_replay_ratio"] == pytest.approx(stats["behavior_replay_ratio"])
    assert stats["world_read_ratio"] == pytest.approx(stats["behavior_read_ratio"])
    assert stats["optimized_replay_ratio"] == pytest.approx(
        stats["world_replay_ratio"] + stats["behavior_replay_ratio"]
    )
    for view in ("world", "behavior"):
        assert f"{view}_sample_age_mean" in stats
        assert f"{view}_sample_age_p50" in stats
        assert f"{view}_sample_age_p95" in stats


def test_prefixed_behavior_transport_rejects_axis_mismatch() -> None:
    world = iter([{"is_first": np.zeros((2, 3), bool)}])
    behavior = iter([{"is_first": np.zeros((3, 3), bool)}])

    with pytest.raises(ValueError, match="identical BxT shapes"):
        next(_with_prefixed_batch(world, behavior, "_behavior_replay/"))


def test_dual_learner_keeps_world_and_behavior_updates_isolated() -> None:
    """Changing one replay view cannot update the other view's modules."""

    team_size = 2
    local_observations = {
        "vector": elements.Space(np.float32, (3,)),
        "reward": elements.Space(np.float32, ()),
        "agent_present": elements.Space(bool, ()),
        "agent_alive": elements.Space(bool, ()),
        "controllable_alive": elements.Space(bool, ()),
        "action_mask": elements.Space(bool, (4,)),
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
    }
    global_fields = {"is_first", "is_last", "is_terminal"}
    observations = {
        key: space if key in global_fields else add_agent_axis(space, team_size)
        for key, space in local_observations.items()
    }
    actions = {"action": add_agent_axis(elements.Space(np.int32, (), 0, 4), team_size)}
    resolved = _resolved("ctde_generalist", "debug").update(
        {
            "replay_context": 2,
            "agent.num_agents": team_size,
            "agent.imag_length": 1,
            "agent.imag_last": 2,
            "agent.enc.simple.units": 2,
            "agent.dyn.parallel_transformer.deter": 2,
            "agent.dyn.parallel_transformer.hidden": 2,
            "agent.dyn.parallel_transformer.stoch": 1,
            "agent.dyn.parallel_transformer.classes": 2,
            "agent.dyn.parallel_transformer.model": 2,
            "agent.dyn.parallel_transformer.heads": 1,
            "agent.dyn.parallel_transformer.context": 2,
            "agent.sigreg.knots": 3,
            "agent.sigreg.num_proj": 1,
            "agent.rewhead.units": 2,
            "agent.conhead.units": 2,
            "agent.maskhead.units": 2,
            "agent.policy.units": 2,
            "agent.value.units": 2,
            "agent.marl.ctde.joint.width": 2,
            "agent.marl.ctde.joint.heads": 1,
            "agent.marl.ctde.joint.context": 2,
            "agent.marl.ctde.head.units": 2,
            "agent.marl.ctde.critic.width": 2,
            "agent.marl.ctde.critic.heads": 1,
            "agent.marl.ctde.critic.value_units": 2,
            "agent.opt.warmup": 0,
            "agent.marl.ctde.opt.warmup": 0,
        }
    )
    config = elements.Config(
        **resolved.agent,
        logdir="/tmp/dreamarl-dual-learner-test",
        seed=0,
        jax=resolved.jax,
        batch_size=1,
        batch_length=2,
        replay_context=2,
        replay_sampling="recent_world_uniform_behavior",
        report_length=1,
        replica=0,
        replicas=1,
    )
    agent = object.__new__(MARLCore)
    MARLCore.__init__(agent, observations, actions, config)
    carry = agent.init_train(1)

    def replay_view(seed, reward_shift=0.0):
        length = 4
        data = {
            "vector": jax.random.normal(
                jax.random.key(seed), (1, length, team_size, 3)
            ),
            "reward": jax.random.normal(
                jax.random.key(seed + 1), (1, length, team_size)
            )
            + reward_shift,
            "agent_present": jnp.ones((1, length, team_size), bool),
            "agent_alive": jnp.ones((1, length, team_size), bool),
            "controllable_alive": jnp.ones((1, length, team_size), bool),
            "action_mask": jnp.ones((1, length, team_size, 4), bool),
            "is_first": jnp.zeros((1, length), bool).at[:, 0].set(True),
            "is_last": jnp.zeros((1, length), bool),
            "is_terminal": jnp.zeros((1, length), bool),
            "action": jax.random.randint(
                jax.random.key(seed + 2),
                (1, length, team_size),
                0,
                4,
            ),
            "stepid": jnp.zeros((1, length, 20), jnp.uint8),
            "consec": jnp.zeros((1, length), jnp.int32),
        }
        for key, space in agent.ext_space.items():
            if key.startswith("_behavior_replay/") or key in data:
                continue
            shape = (1, length, *space.shape)
            data[key] = (
                jnp.ones(shape, space.dtype)
                if space.dtype == np.bool_
                else jnp.zeros(shape, space.dtype)
            )
        data["dyn/reset"] = jnp.broadcast_to(
            data["is_first"][:, :, None], (1, length, team_size)
        )
        data["dyn/position"] = jnp.broadcast_to(
            jnp.arange(length, dtype=jnp.int32)[None, :, None],
            (1, length, team_size),
        )
        data["dyn/active"] = jnp.ones((1, length, team_size), bool)
        return data

    def paired(world, behavior):
        return {
            **world,
            **{f"_behavior_replay/{key}": value for key, value in behavior.items()},
        }

    world_a = replay_view(1)
    world_b = replay_view(2, reward_shift=2.0)
    behavior_a = replay_view(3)
    behavior_b = replay_view(4, reward_shift=3.0)

    def step(batch):
        return agent.train(carry, batch)

    initial = nj.init(lambda: step(paired(world_a, behavior_a)))({}, seed=10)
    state_aa, result_aa = nj.pure(lambda: step(paired(world_a, behavior_a)))(
        initial, seed=11
    )
    state_ab, result_ab = nj.pure(lambda: step(paired(world_a, behavior_b)))(
        initial, seed=11
    )
    state_ba, _ = nj.pure(lambda: step(paired(world_b, behavior_a)))(initial, seed=11)

    def assert_parameters_equal(left, right, prefixes):
        keys = [key for key in left if key.startswith(prefixes)]
        assert keys
        for key in keys:
            np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]))

    assert_parameters_equal(
        state_aa,
        state_ab,
        (
            "enc/",
            "dyn/",
            "rew/",
            "con/",
            "actmask/",
            "ctde_joint/",
            "ctde_rew/",
            "ctde_con/",
            "ctde_mask/",
            "ctde_alive/",
        ),
    )
    assert_parameters_equal(state_aa, state_ba, ("pol/", "ctde_val/"))

    carry_aa = result_aa[0]
    carry_ab = result_ab[0]
    assert jax.tree.structure(carry_aa) == jax.tree.structure(carry_ab)
    for left, right in zip(jax.tree.leaves(carry_aa), jax.tree.leaves(carry_ab)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    replay_aa = result_aa[1]["replay"]
    replay_ab = result_ab[1]["replay"]
    assert replay_aa
    assert not any(key.startswith("_behavior_replay/") for key in replay_aa)
    for key in replay_aa:
        np.testing.assert_array_equal(
            np.asarray(replay_aa[key]), np.asarray(replay_ab[key])
        )

    metrics = result_aa[2]
    for key in (
        "replay_views/world_loss",
        "replay_views/behavior_loss",
        "loss/ctde_embedding",
        "loss/policy",
        "loss/value",
        "loss/repval",
        "opt/local_world/grad_norm",
        "opt/joint_world/grad_norm",
        "opt/actor/grad_norm",
        "opt/critic/grad_norm",
    ):
        assert np.isfinite(float(metrics[key])), key


def test_dual_view_selectors_have_distinct_age_distributions() -> None:
    replay = DualViewReplay(
        length=3,
        capacity=128,
        online=False,
        recency_decay=0.9,
        seed=23,
    )
    for step in range(180):
        replay.add(
            {
                "is_first": np.asarray(step % 20 == 0),
                "value": np.asarray(step, np.int32),
            }
        )
    for _ in range(128):
        replay.sample(1, "train_world")
        replay.sample(1, "train_behavior")

    stats = replay.stats()
    assert len(replay.items) == len(replay.sampler) == len(replay.behavior_sampler)
    assert stats["world_sample_age_mean"] < 20
    assert stats["behavior_sample_age_mean"] > 45
    assert stats["world_sample_age_mean"] < stats["behavior_sample_age_mean"]
