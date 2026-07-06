from __future__ import annotations

import jax
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import (
    MAPPOConfig,
    create_train_state as create_mappo_state,
)
from world_marl.config import TrainConfig
from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.evaluation import (
    evaluate_policy,
    evaluate_policy_scan,
    mappo_train_state_policy,
    train_state_policy,
)
from world_marl.training import central_observation_shape


def _make_ippo_state(adapter, seed: int = 5):
    config = IPPOConfig(network_arch="mlp")
    return create_ippo_state(
        jax.random.PRNGKey(seed),
        adapter.observation_shape,
        adapter.action_dim,
        config,
    )


def test_scan_eval_matches_loop_eval_coins():
    """A fully-jitted lax.scan eval must reproduce the Python loop bit-for-bit.

    Coins episodes are lockstep (all envs complete every max_cycles steps), and a
    deterministic policy makes the action PRNG inert, so the only randomness is the
    env-key stream. Replicating that stream under scan must yield identical returns
    and lengths to ``evaluate_policy``.
    """
    num_envs, max_cycles, seed, episodes = 4, 8, 3, 8

    state = _make_ippo_state(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    loop_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    policy_fn = train_state_policy(
        state,
        num_envs=num_envs,
        num_agents=loop_adapter.num_agents,
        deterministic=True,
        observation_mode="vector",
    )
    loop = evaluate_policy(loop_adapter, policy_fn, episodes=episodes)

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

    assert scan.episodes == loop.episodes == episodes
    np.testing.assert_array_equal(scan.lengths, loop.lengths)
    np.testing.assert_allclose(scan.returns, loop.returns, rtol=0, atol=1e-5)
    assert scan.lengths.tolist() == [max_cycles] * episodes


def test_scan_eval_matches_loop_eval_coins_stochastic():
    """Stochastic parity pins the policy-PRNG contract.

    The loop policy splits one action key per step from ``PRNGKey(seed)``; the
    scan must consume the identical stream, so sampled (not argmax) actions --
    and therefore returns -- match bit-for-bit.
    """
    num_envs, max_cycles, seed, episodes = 4, 8, 3, 8

    state = _make_ippo_state(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    loop_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    policy_fn = train_state_policy(
        state,
        num_envs=num_envs,
        num_agents=loop_adapter.num_agents,
        deterministic=False,
        seed=0,
        observation_mode="vector",
    )
    loop = evaluate_policy(loop_adapter, policy_fn, episodes=episodes)

    scan_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    scan = evaluate_policy_scan(
        scan_adapter,
        state,
        episodes=episodes,
        deterministic=False,
        observation_mode="vector",
        seed=0,
    )

    assert scan.episodes == loop.episodes == episodes
    np.testing.assert_array_equal(scan.lengths, loop.lengths)
    np.testing.assert_allclose(scan.returns, loop.returns, rtol=0, atol=1e-5)


def test_scan_eval_matches_loop_eval_mappo_coins():
    """MAPPO scan eval must rebuild the centralized critic input on device and
    reproduce the loop path (``mappo_train_state_policy`` + ``evaluate_policy``)
    bit-for-bit on coins."""
    num_envs, max_cycles, seed, episodes = 4, 8, 3, 8

    template = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    state = create_mappo_state(
        jax.random.PRNGKey(5),
        template.observation_shape,
        central_observation_shape(
            template.observation_shape,
            template.num_agents,
            observation_mode="vector",
        ),
        template.action_dim,
        MAPPOConfig(network_arch="mlp"),
    )

    loop_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    policy_fn = mappo_train_state_policy(
        state,
        num_envs=num_envs,
        num_agents=loop_adapter.num_agents,
        deterministic=True,
        observation_mode="vector",
    )
    loop = evaluate_policy(loop_adapter, policy_fn, episodes=episodes)

    scan_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    scan = evaluate_policy_scan(
        scan_adapter,
        state,
        episodes=episodes,
        deterministic=True,
        observation_mode="vector",
        algorithm="mappo",
    )

    assert scan.episodes == loop.episodes == episodes
    np.testing.assert_array_equal(scan.lengths, loop.lengths)
    np.testing.assert_allclose(scan.returns, loop.returns, rtol=0, atol=1e-5)
    assert scan.lengths.tolist() == [max_cycles] * episodes


def test_random_scan_eval_shapes_and_determinism():
    """The random baseline scan is seed-deterministic and shaped like the loop
    result: returns ``[episodes, num_agents]``, coins lengths all max_cycles."""
    from world_marl.evaluation import evaluate_random_policy_scan

    num_envs, max_cycles, seed, episodes = 4, 8, 3, 8

    first = evaluate_random_policy_scan(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        ),
        episodes=episodes,
        seed=11,
    )
    second = evaluate_random_policy_scan(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        ),
        episodes=episodes,
        seed=11,
    )

    assert first.episodes == episodes
    assert first.returns.shape == (episodes, 2)
    assert first.lengths.tolist() == [max_cycles] * episodes
    np.testing.assert_array_equal(first.returns, second.returns)
    np.testing.assert_array_equal(first.lengths, second.lengths)


def test_random_baseline_uses_scan_on_scannable_adapters(monkeypatch):
    """train_e2e's random baseline must not fall back to the Python-loop eval
    when the adapter supports ``scan_rewards_dones`` (everything but MeltingPot)."""
    from world_marl.scripts import train_e2e

    def _no_loop_eval(*args, **kwargs):
        raise AssertionError(
            "random baseline used the Python-loop eval on a scannable adapter"
        )

    monkeypatch.setattr(train_e2e, "evaluate_policy", _no_loop_eval)
    cfg = TrainConfig(substrate="coins", num_envs=4, max_cycles=8, eval_episodes=8)
    baseline = train_e2e.evaluate_random_baseline(cfg, seed=3)
    assert baseline["episodes"] == 8


def test_scan_eval_matches_loop_eval_gymnax_cartpole():
    """Early-terminating episodes must segment identically to the loop.

    CartPole episodes end at different steps per env, so returns cannot come
    from lockstep block sums -- the scan has to reconstruct episodes from the
    dones and emit them in the loop's (step, env) completion order.
    """
    num_envs, max_cycles, seed, episodes = 4, 32, 3, 8

    state = _make_ippo_state(
        GymnaxVectorAdapter(
            "CartPole-v1", num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    loop_adapter = GymnaxVectorAdapter(
        "CartPole-v1", num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    policy_fn = train_state_policy(
        state,
        num_envs=num_envs,
        num_agents=loop_adapter.num_agents,
        deterministic=True,
        observation_mode="vector",
    )
    loop = evaluate_policy(loop_adapter, policy_fn, episodes=episodes)

    scan_adapter = GymnaxVectorAdapter(
        "CartPole-v1", num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    scan = evaluate_policy_scan(
        scan_adapter,
        state,
        episodes=episodes,
        deterministic=True,
        observation_mode="vector",
    )

    assert scan.episodes == loop.episodes == episodes
    np.testing.assert_array_equal(scan.lengths, loop.lengths)
    np.testing.assert_array_equal(scan.returns, loop.returns)
    assert any(length < max_cycles for length in scan.lengths.tolist())
