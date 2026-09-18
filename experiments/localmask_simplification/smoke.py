"""CPU integration gate for full JEPA world, replay, and imagined PPO training."""

from __future__ import annotations

from unittest.mock import patch

import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from majepa.main import _load_configs, _resolve_config_profiles
from majepa.marl.axes import BEHAVIOR_REPLAY_PREFIX, split_prefixed_data
from majepa.marl.core import MARLCore


def _tiny_learner(overrides=None):
    config = _resolve_config_profiles(
        _load_configs(), ("smac_vector", "ma_jepa")
    ).update(
        {
            "batch_size": 2,
            "batch_length": 4,
            "replay_context": 4,
            "agent.num_agents": 2,
            "agent.imag_length": 3,
            "agent.ppo.epochs": 2,
            "agent.ppo.actor_epochs": 2,
            "agent.ppo.critic_epochs": 2,
            "agent.enc.simple.layers": 1,
            "agent.enc.simple.units": 8,
            "agent.policy.layers": 1,
            "agent.policy.units": 8,
            "agent.rewhead.units": 8,
            "agent.conhead.units": 8,
            "agent.maskhead.units": 8,
            "agent.rewhead.bins": 5,
            "agent.dyn.parallel_transformer.stoch": 2,
            "agent.dyn.parallel_transformer.classes": 4,
            "agent.dyn.parallel_transformer.heads": 2,
            "agent.dyn.parallel_transformer.ffup": 2,
            "agent.marl.ctde.joint.width": 8,
            "agent.marl.ctde.joint.heads": 2,
            "agent.marl.ctde.joint.agent_layers": 1,
            "agent.marl.ctde.joint.ffup": 2,
            "agent.marl.ctde.head.layers": 1,
            "agent.marl.ctde.head.units": 8,
            "agent.marl.ctde.head.bins": 5,
            "agent.marl.ctde.critic.width": 8,
            "agent.marl.ctde.critic.layers": 1,
            "agent.marl.ctde.critic.heads": 2,
            "agent.marl.ctde.critic.ffup": 2,
            "agent.marl.ctde.critic.value_layers": 1,
            "agent.marl.ctde.critic.value_units": 8,
            "agent.marl.ctde.critic.bins": 5,
            "agent.marl.ctde.multistep_jepa.width": 8,
            "agent.marl.ctde.multistep_jepa.layers": 1,
            "agent.marl.ctde.multistep_jepa.units": 8,
            "agent.marl.ctde.multistep_jepa.plan_units": 8,
            "agent.marl.ctde.teammate_belief.layers": 1,
            "agent.marl.ctde.teammate_belief.units": 8,
            "agent.marl.ctde.teammate_belief.adapter_units": 8,
            "agent.marl.ctde.self_fed.horizons": [2],
            "agent.marl.ctde.self_fed.anchors": 2,
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
    if overrides:
        config = config.update(overrides)
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



import sys
source = "local"
arm = sys.argv[1]
extra = {"agent.simplification.joint_mask": False} if arm == "nojointmask" else ({"agent.simplification.local_outcomes": False} if arm == "nolocaloutcomes" else ({"agent.rewhead.units": 4, "agent.conhead.units": 4, "agent.maskhead.units": 4} if arm == "heads256" else {}))
learner, obs_space, act_space = _tiny_learner({
 "agent.marl.ctde.imagination_mask_source": source,
 "agent.marl.ctde.compare_mask_heads": True,
 "agent.marl.ctde.imagination_mask_sampling": "bernoulli",
 "agent.marl.ctde.teammate_belief.enabled": False,
 "agent.marl.ctde.multistep_jepa.belief_context": False,
 "agent.ppo.factual_value.enabled": False,
 "agent.ppo.factual_value.representation_scale": 0.0,
 "agent.marl.ctde.direct_latent": False,
 "agent.marl.ctde.self_fed.bptt_steps": 2,
 "agent.marl.ctde.self_fed.scale": 0.1,
 "agent.marl.ctde.self_fed.trajectory_kl_scale": 0.1,
 "agent.loss_scales.ctde_posterior_alignment": 0.05,
 **extra,
})
data = _synthetic_replay(learner, obs_space, act_space)
carry = learner.init_train(2)
state = nj.init(learner.train)({}, carry, data, seed=702)
data = dict(data, _environment_step=jnp.full((2,8), 10,jnp.int32))
updated, (carry, _, metrics) = jax.jit(nj.pure(learner.train))(state, carry, data, seed=704)
_assert_finite((updated, metrics))
assert float(metrics['ppo/batch_illegal_action_fraction']) == 0
_, (_, report) = jax.jit(nj.pure(learner.report))(updated, carry, data, seed=706)
_assert_finite(report)
if arm != 'nojointmask':
    assert any('/head_local/' in key for key in report), report.keys()
    assert any('/head_joint/' in key for key in report)
if arm == 'nojointmask':
    assert not any(k.startswith('ctde_mask/') for k in updated)
if arm == 'nolocaloutcomes':
    assert not any(k.startswith(('rew/', 'con/')) for k in updated)
print('PASS',source,'PPO update, legal stored masks, finite paired head diagnostics', flush=True)

import pickle
from pathlib import Path
if len(sys.argv)>2:
    Path(sys.argv[2]).write_bytes(pickle.dumps(jax.device_get((state, updated, metrics))))
