from __future__ import annotations

import jax
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.evaluation import (
    evaluate_policy,
    evaluate_policy_scan,
    train_state_policy,
)


def _make_coins_state(adapter, seed: int = 5):
    config = IPPOConfig(network_arch="mlp")
    return create_ippo_state(
        jax.random.PRNGKey(seed),
        adapter.observation_shape,
        adapter.action_dim,
        config,
    )


def test_scan_eval_matches_host_eval_coins():
    """A fully-jitted lax.scan eval must reproduce the host loop bit-for-bit.

    Coins episodes are lockstep (all envs complete every max_cycles steps), and a
    deterministic policy makes the action PRNG inert, so the only randomness is the
    env-key stream. Replicating that stream under scan must yield identical returns
    and lengths to ``evaluate_policy``.
    """
    num_envs, max_cycles, seed, episodes = 4, 8, 3, 8

    state = _make_coins_state(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    host_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    policy_fn = train_state_policy(
        state,
        num_envs=num_envs,
        num_agents=host_adapter.num_agents,
        deterministic=True,
        observation_mode="vector",
    )
    host = evaluate_policy(host_adapter, policy_fn, episodes=episodes)

    scan_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    scan = evaluate_policy_scan(
        scan_adapter,
        state,
        episodes=episodes,
        deterministic=True,
        observation_mode="vector",
    )

    assert scan.episodes == host.episodes == episodes
    np.testing.assert_array_equal(scan.lengths, host.lengths)
    np.testing.assert_allclose(scan.returns, host.returns, rtol=0, atol=1e-5)
    assert scan.lengths.tolist() == [max_cycles] * episodes
