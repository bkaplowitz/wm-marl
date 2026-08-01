"""Emit deterministic M3 or DreaMARL A=1 numerical parity artifacts.

Run the two implementations in separate processes with the same visible GPU,
configuration, synthetic replay, and RNG seed. The resulting JSON files must
match for every digest. This avoids environment randomness and verifies the
complete policy and one-step learner computation, including optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import elements
import embodied.jax.internal as jax_internal
import jax
import numpy as np
import ruamel.yaml as yaml

from world_marl.dreamarl.agent import Agent as DreaMARLAgent
from world_marl.dreamarl.axes import (
    GLOBAL_OBSERVATION_KEYS,
    GLOBAL_REPLAY_KEYS,
)
from world_marl.dreamarl.m3.agent import Agent as M3Agent
from world_marl.dreamarl.runtime import algorithm_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("m3", "dreamarl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--platform", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument(
        "--interaction-context",
        choices=("none", "aligned", "shuffled"),
        default="none",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Use reduced dynamics and heads while preserving the exact axis lift.",
    )
    args = parser.parse_args(argv)
    if args.num_agents != 1 and args.implementation != "dreamarl":
        parser.error("the frozen M3 reference only supports num_agents=1")
    if args.updates < 1:
        parser.error("updates must be positive")
    if args.implementation != "dreamarl" and args.interaction_context != "none":
        parser.error("interaction context is a DreaMARL-only option")

    args.logdir.mkdir(parents=True, exist_ok=True)
    config = _resolve_config(
        args.logdir,
        args.platform,
        args.seed,
        args.mini,
        args.num_agents,
        args.interaction_context,
    )
    local_obs_space, local_act_space = _local_spaces()
    if args.implementation == "dreamarl":
        obs_space = _add_agent_axes(local_obs_space, args.num_agents)
        act_space = _add_agent_axes(local_act_space, args.num_agents)
        agent_type = DreaMARLAgent
    else:
        obs_space = local_obs_space
        act_space = local_act_space
        agent_type = M3Agent

    agent_config = elements.Config(
        **config.agent,
        logdir=str(args.logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )
    if args.implementation == "m3":
        dynamics = dict(agent_config.dyn.jepa_transformer)
        for key in (
            "num_agents",
            "interaction",
            "interaction_units",
            "interaction_heads",
            "interaction_seed",
        ):
            dynamics.pop(key, None)
        dyn_config = dict(agent_config.dyn)
        dyn_config["jepa_transformer"] = elements.Config(dynamics)
        agent_values = dict(agent_config)
        agent_values["dyn"] = elements.Config(dyn_config)
        agent_config = elements.Config(agent_values)
    agent = agent_type(obs_space, act_space, agent_config)
    initial_state = _digest_tree(jax.device_get(agent.params))
    shared_initial_state = _digest_tree(
        jax.device_get(_without_interaction(agent.params))
    )
    interaction_initial = jax.device_get(_only_interaction(agent.params))
    direct_interaction_initial = jax.device_get(
        _direct_interaction(agent.params)
    )

    batch_size = int(config.batch_size)
    policy_seed = agent._seeds(0, agent.policy_mirrored)
    policy_carry = agent._init_policy(
        agent.policy_params, policy_seed, batch_size
    )
    policy_obs = _policy_observations(local_obs_space, batch_size, args.seed)
    if args.implementation == "dreamarl":
        policy_obs = _insert_policy_agent_axes(policy_obs, args.num_agents)
    policy_obs = jax_internal.device_put(policy_obs, agent.policy_sharded)
    policy_carry, actions, policy_outputs = agent._policy(
        agent.policy_params,
        policy_seed,
        policy_carry,
        policy_obs,
        "train",
    )
    if args.implementation == "dreamarl" and args.num_agents == 1:
        policy_carry, actions, policy_outputs = jax.device_get(
            (policy_carry, actions, policy_outputs)
        )
        policy_carry = _squeeze_policy_agent_axes(policy_carry)
        actions = _squeeze_policy_agent_axes(actions)
        policy_outputs = _squeeze_policy_agent_axes(policy_outputs)

    replay = _fixed_replay(
        agent.spaces,
        batch_size,
        int(config.batch_length + config.replay_context),
        args.seed,
        has_agent_axis=args.implementation == "dreamarl",
        num_agents=args.num_agents,
    )
    replay = jax_internal.device_put(replay, agent.train_sharded)
    train_carry = agent.init_train(batch_size)
    updated = agent.params
    for update_index in range(args.updates):
        train_seed = agent._seeds(update_index, agent.train_mirrored)
        allowed = {
            key: value
            for key, value in updated.items()
            if key in agent.policy_keys
        }
        donated = {
            key: value
            for key, value in updated.items()
            if key not in agent.policy_keys
        }
        updated, train_carry, train_outputs, train_metrics = agent._train(
            donated, allowed, train_seed, train_carry, replay
        )
    if args.implementation == "dreamarl" and args.num_agents == 1:
        train_carry, train_outputs = jax.device_get(
            (train_carry, train_outputs)
        )
        train_carry = _squeeze_policy_agent_axes(train_carry)
        train_outputs = _squeeze_train_outputs(train_outputs)

    result = {
        "implementation": args.implementation,
        "mini": args.mini,
        "num_agents": args.num_agents,
        "interaction_context": args.interaction_context,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": batch_size,
        "sequence_length": int(config.batch_length + config.replay_context),
        "initial_state": initial_state,
        "shared_initial_state": shared_initial_state,
        "policy_carry": _digest_tree(policy_carry),
        "policy_actions": _digest_tree(actions),
        "policy_outputs": _digest_tree(policy_outputs),
        "loss_metrics": _digest_tree(train_metrics),
        "core_loss_metrics": _digest_tree(
            _without_interaction(train_metrics)
        ),
        "interaction_metrics": {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in train_metrics.items()
            if "interaction" in key
        },
        "train_carry": _digest_tree(train_carry),
        "train_outputs": _digest_tree(train_outputs),
        "updated_state": _digest_tree(updated),
        "shared_updated_state": _digest_tree(
            _without_interaction(updated)
        ),
        "interaction_update": _tree_change(
            interaction_initial,
            jax.device_get(_only_interaction(updated)),
        ),
        "direct_interaction_update": _tree_change(
            direct_interaction_initial,
            jax.device_get(_direct_interaction(updated)),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _resolve_config(
    logdir: Path,
    platform: str,
    seed: int,
    mini: bool,
    num_agents: int,
    interaction_context: str,
):
    raw = yaml.YAML(typ="safe").load(
        (algorithm_root() / "configs.yaml").read_text(encoding="utf-8")
    )
    config = elements.Config(raw["defaults"])
    config = config.update(raw["dmc_vision"])
    config = config.update(raw["jepa_transformer"])
    jax_config = config.jax.update(platform=platform)
    dynamics = config.agent.dyn.jepa_transformer.update(
        interaction=interaction_context
    )
    agent_config = config.agent.update(
        num_agents=num_agents,
        dyn=config.agent.dyn.update(jepa_transformer=dynamics),
    )
    updates: dict[str, Any] = {
        "logdir": str(logdir),
        "seed": seed,
        "jax": jax_config,
        "agent": agent_config,
    }
    if mini:
        dynamics = agent_config.dyn.jepa_transformer.update(
            deter=32,
            hidden=16,
            stoch=4,
            classes=8,
            model=32,
            layers=1,
            heads=4,
            context=4,
            ffup=2,
        )
        updates.update(
            batch_size=2,
            batch_length=8,
            replay_context=4,
            report_length=8,
            agent=agent_config.update(
                dyn=agent_config.dyn.update(jepa_transformer=dynamics),
                dec=agent_config.dec.update(
                    simple=agent_config.dec.simple.update(depth=2)
                ),
                rewhead=agent_config.rewhead.update(units=32, bins=15),
                conhead=agent_config.conhead.update(units=32),
                policy=agent_config.policy.update(units=32),
                value=agent_config.value.update(units=32, bins=15),
                imag_length=5,
            ),
        )
    return config.update(updates)


def _local_spaces():
    obs_space = {
        "is_first": elements.Space(bool, ()),
        "is_last": elements.Space(bool, ()),
        "is_terminal": elements.Space(bool, ()),
        "reward": elements.Space(np.float32, ()),
        "image": elements.Space(np.uint8, (64, 64, 3), 0, 255),
    }
    act_space = {
        "action": elements.Space(np.float32, (2,), -1.0, 1.0),
    }
    return obs_space, act_space


def _add_agent_axes(spaces, num_agents: int):
    result = {}
    for key, space in spaces.items():
        if key in GLOBAL_OBSERVATION_KEYS:
            result[key] = space
            continue
        shape = (num_agents, *space.shape)
        low = None if space.low is None else np.broadcast_to(space.low, shape)
        high = None if space.high is None else np.broadcast_to(space.high, shape)
        result[key] = elements.Space(space.dtype, shape, low, high)
    return result


def _policy_observations(spaces, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    observations = {}
    for key, space in spaces.items():
        shape = (batch_size, *space.shape)
        if key == "image":
            value = rng.integers(0, 256, shape, dtype=np.uint8)
        else:
            value = np.zeros(shape, space.dtype)
        observations[key] = value
    observations["is_first"][:] = True
    return observations


def _insert_policy_agent_axes(tree, num_agents: int):
    return {
        key: (
            value
            if key in GLOBAL_OBSERVATION_KEYS
            else np.repeat(value[:, None], num_agents, axis=1)
        )
        for key, value in tree.items()
    }


def _squeeze_policy_agent_axes(tree):
    return jax.tree.map(lambda value: value[:, 0], tree)


def _fixed_replay(
    spaces,
    batch_size: int,
    sequence_length: int,
    seed: int,
    *,
    has_agent_axis: bool,
    num_agents: int,
):
    result = {}
    global_keys = GLOBAL_OBSERVATION_KEYS | GLOBAL_REPLAY_KEYS
    for index, (key, space) in enumerate(sorted(spaces.items())):
        local_shape = space.shape
        if has_agent_axis and key not in global_keys:
            local_shape = local_shape[1:]
        shape = (batch_size, sequence_length, *local_shape)
        rng = np.random.default_rng([seed, index])
        if space.dtype == np.uint8:
            value = rng.integers(0, 256, shape, dtype=np.uint8)
        elif space.dtype == bool:
            value = np.zeros(shape, bool)
        elif np.issubdtype(space.dtype, np.integer):
            value = np.zeros(shape, space.dtype)
        else:
            value = rng.normal(0.0, 0.1, shape).astype(space.dtype)
            if key == "action":
                value = np.tanh(value).astype(space.dtype)
        if has_agent_axis and key not in global_keys:
            value = np.repeat(value[:, :, None], num_agents, axis=2)
        result[key] = value
    result["is_first"][:, 0] = True
    result["is_last"][:, -1] = True
    result["consec"] = np.broadcast_to(
        np.arange(sequence_length, dtype=np.int32),
        (batch_size, sequence_length),
    ).copy()
    return result


def _squeeze_train_outputs(outputs):
    if "replay" not in outputs:
        return outputs
    replay = {
        key: value if key == "stepid" else value[:, :, 0]
        for key, value in outputs["replay"].items()
    }
    return {**outputs, "replay": replay}


def _digest_tree(tree) -> dict[str, object]:
    leaves, treedef = jax.tree_util.tree_flatten(jax.device_get(tree))
    digest = hashlib.sha256()
    total_values = 0
    for leaf in leaves:
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
        total_values += int(array.size)
    return {
        "sha256": digest.hexdigest(),
        "treedef": str(treedef),
        "leaves": len(leaves),
        "values": total_values,
    }


def _without_interaction(tree):
    if not isinstance(tree, dict):
        return tree
    return {
        key: _without_interaction(value)
        for key, value in tree.items()
        if "interaction" not in key
        and key
        not in {
            "local_dyn_ent",
            "opt/grad_rms",
            "opt/param_count",
            "opt/param_rms",
            "opt/update_rms",
        }
    }


def _only_interaction(tree):
    if not isinstance(tree, dict):
        return tree
    result = {}
    for key, value in tree.items():
        if "interaction" in key:
            result[key] = value
        elif isinstance(value, dict):
            nested = _only_interaction(value)
            if nested:
                result[key] = nested
    return result


def _direct_interaction(tree):
    return {
        key: value
        for key, value in tree.items()
        if key.startswith("dyn/interaction")
    }


def _tree_change(before, after):
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    if len(before_leaves) != len(after_leaves):
        raise ValueError((len(before_leaves), len(after_leaves)))
    squared = 0.0
    maximum = 0.0
    changed = 0
    for old, new in zip(before_leaves, after_leaves, strict=True):
        delta = np.asarray(new, np.float64) - np.asarray(old, np.float64)
        squared += float(np.square(delta).sum())
        maximum = max(maximum, float(np.abs(delta).max(initial=0.0)))
        changed += int(np.any(delta != 0))
    return {
        "l2": float(np.sqrt(squared)),
        "max_abs": maximum,
        "changed_leaves": changed,
        "leaves": len(before_leaves),
    }


if __name__ == "__main__":
    raise SystemExit(main())
