"""CPU integration gate for full JEPA world, replay, and imagined PPO training."""

from __future__ import annotations

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from majepa.main import _load_configs, _resolve_config_profiles
from majepa.marl.axes import BEHAVIOR_REPLAY_PREFIX, split_prefixed_data
from majepa.marl.core import MARLCore


def _tiny_learner():
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa", "debug")
    ).update(
        {
            "batch_size": 2,
            "batch_length": 4,
            "replay_context": 4,
            "agent.num_agents": 2,
            "agent.imag_length": 3,
            "agent.ppo.epochs": 2,
            "agent.opt.warmup": 0,
            "agent.marl.ctde.opt.warmup": 0,
            "agent.dyn.parallel_transformer.deter": 8,
            "agent.dyn.parallel_transformer.hidden": 8,
            "agent.dyn.parallel_transformer.model": 8,
            "agent.dyn.parallel_transformer.layers": 1,
            "agent.dyn.parallel_transformer.context": 4,
            "agent.marl.ctde.joint.temporal_layers": 1,
            "agent.marl.ctde.joint.context": 4,
            "agent.marl.ctde.multistep_jepa.horizons": [1, 2],
            "agent.marl.ctde.multistep_jepa.max_horizon": 2,
            "agent.sigreg.num_proj": 8,
        }
    )
    config = elements.Config(
        **config.agent,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        replay_sampling=config.replay.sampling,
        ppo_start_step=10,
    )
    agents, actions = 2, 8
    obs_space = {
        "observation": elements.Space(np.float32, (agents, 6)),
        "reward": elements.Space(np.float32, (agents,)),
        "agent_present": elements.Space(bool, (agents,)),
        "agent_alive": elements.Space(bool, (agents,)),
        "controllable_alive": elements.Space(bool, (agents,)),
        "action_mask": elements.Space(bool, (agents, actions)),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {"action": elements.Space(np.int32, (agents,), 0, actions)}
    # Test the full model directly inside Ninjax/JIT, without the outer runtime
    # wrapper starting device meshes, checkpointing, or environment workers.
    learner = object.__new__(MARLCore)
    MARLCore.__init__(learner, obs_space, act_space, config)
    return learner, obs_space, act_space


def _synthetic_replay(learner, obs_space, act_space):
    batch, length, agents = 2, 8, learner.team.size
    spaces = {**obs_space, **act_space, **learner.ext_space}
    data = {
        name: jnp.zeros((batch, length, *space.shape), space.dtype)
        for name, space in spaces.items()
        if not name.startswith(BEHAVIOR_REPLAY_PREFIX)
    }
    data["observation"] = jax.random.normal(
        jax.random.key(701), (batch, length, agents, 6)
    )
    present = jnp.ones((batch, length, agents), bool)
    alive = present.at[0, 5:, 0].set(False)
    data["agent_present"] = present
    data["agent_alive"] = present
    data["controllable_alive"] = alive
    data["observation"] = jnp.where(alive[..., None], data["observation"], 0.0)
    mask = jnp.ones((batch, length, agents, 8), bool).at[..., 0].set(False)
    noop = jnp.zeros_like(mask).at[..., 0].set(True)
    data["action_mask"] = jnp.where(alive[..., None], mask, noop)
    data["action"] = jnp.where(alive, 1, 0).astype(jnp.int32)
    data["is_first"] = data["is_first"].at[:, 0].set(True).at[1, 5].set(True)
    data["is_last"] = data["is_last"].at[0, 7].set(True).at[1, 4].set(True)
    data["is_terminal"] = data["is_last"]
    reward = jnp.asarray(
        [
            [0, 0.2, 0.1, 0.3, 0.1, 0.2, 1.0, 2.0],
            [0, 0.1, 0.2, 0.1, 1.0, 0.0, 0.2, 0.4],
        ],
        jnp.float32,
    )
    data["reward"] = jnp.broadcast_to(reward[..., None], present.shape)
    data["dyn/reset"] = jnp.broadcast_to(data["is_first"][..., None], present.shape)
    position = jnp.asarray([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 0, 1, 2]])
    data["dyn/position"] = jnp.broadcast_to(position[..., None], present.shape)
    data["dyn/active"] = present
    data["dyn/stoch"] = data["dyn/stoch"].at[..., 0].set(1.0)
    data["dyn/pair"] = data["dyn/pair"].at[..., -7].set(1.0)
    data.update(
        {
            f"{BEHAVIOR_REPLAY_PREFIX}{key}": value
            for key, value in tuple(data.items())
            if key != "_environment_step"
        }
    )
    return data


def _assert_finite(tree):
    for value in jax.tree.leaves(tree):
        array = np.asarray(value)
        assert np.isfinite(array.astype(np.float32)).all()


def test_full_learner_jit_warmup_and_ppo_with_death_and_episode_resets():
    learner, obs_space, act_space = _tiny_learner()
    data = _synthetic_replay(learner, obs_space, act_space)
    carry = learner.init_train(2)
    state = nj.init(learner.train)({}, carry, data, seed=702)
    train = jax.jit(nj.pure(learner.train))

    frozen_state, (carry, _, frozen_metrics) = train(state, carry, data, seed=703)
    _assert_finite((frozen_state, carry, frozen_metrics))
    assert float(frozen_metrics["ppo/active"]) == 0.0
    for key, value in state.items():
        if key.startswith(("pol/", "ctde_teammate_actor/", "ctde_val/")):
            np.testing.assert_array_equal(value, frozen_state[key])
    assert float(frozen_metrics["opt/actor/updates"]) == 0.0
    assert float(frozen_metrics["opt/critic/updates"]) == 0.0

    data = dict(data, _environment_step=jnp.full((2, 8), 10, jnp.int32))
    updated_state, (carry, _, metrics) = train(frozen_state, carry, data, seed=704)
    _assert_finite((updated_state, carry, metrics))
    assert float(metrics["ppo/active"]) == 1.0
    assert float(metrics["opt/actor/updates"]) > 0.0
    assert float(metrics["opt/critic/updates"]) == learner.config.ppo.epochs
    assert float(metrics["ppo/batch_illegal_action_fraction"]) == 0.0
    for prefix in ("pol/", "ctde_val/"):
        assert any(
            not np.array_equal(value, updated_state[key])
            for key, value in frozen_state.items()
            if key.startswith(prefix)
        ), f"PPO did not update {prefix}"

    _, behavior = split_prefixed_data(data)
    behavior = learner.team.local_sequence_data(behavior)

    def inspect_batch():
        batch, batch_metrics = learner._prepare_ppo_batch(behavior, jnp.float32(0.01))
        _, actor_metrics = learner._ppo_actor_loss(batch)
        _, critic_metrics = learner._ppo_critic_loss(batch)
        return batch, batch_metrics, actor_metrics, critic_metrics

    _, (batch, batch_metrics, actor_metrics, critic_metrics) = jax.jit(
        nj.pure(inspect_batch)
    )(updated_state, seed=705)
    _assert_finite((batch, batch_metrics, actor_metrics, critic_metrics))
    # Re-evaluation before any optimizer step must reproduce the exact policy
    # distribution that sampled the frozen imagination, including its masks.
    np.testing.assert_allclose(actor_metrics["ratio"], 1.0, atol=1e-6)
    np.testing.assert_allclose(actor_metrics["exact_kl"], 0.0, atol=1e-6)
    sampled_legal = jnp.take_along_axis(
        batch["action_mask"], batch["action"][..., None], axis=-1
    )[..., 0]
    assert bool(sampled_legal.all())
    assert bool((batch["critic_valid"] & ~batch["valid"]).any())

    _, (report_carry, report_metrics) = jax.jit(nj.pure(learner.report))(
        updated_state, carry, data, seed=706
    )
    _assert_finite((report_carry, report_metrics))
    assert "ctde/self_fed_h1/reward_rmse" in report_metrics
    assert "ctde/self_fed_h2/reward_rmse" in report_metrics
