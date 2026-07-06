from __future__ import annotations

import jax
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.training import central_observation_shape
from world_marl.world_model_training import (
    collect_policy_transition_batch,
    collect_policy_transition_batch_host,
    collect_random_transition_batch,
    concatenate_transition_batches,
)


def _make_adapter(num_envs=4, max_cycles=8, seed=3):
    return JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )


def _make_coins_state(adapter, seed: int = 5):
    config = IPPOConfig(network_arch="mlp")
    return create_ippo_state(
        jax.random.PRNGKey(seed),
        adapter.observation_shape,
        adapter.action_dim,
        config,
    )


def _make_coins_mappo_state(adapter, seed: int = 5):
    config = MAPPOConfig(network_arch="mlp")
    return create_mappo_state(
        jax.random.PRNGKey(seed),
        adapter.observation_shape,
        central_observation_shape(
            adapter.observation_shape,
            adapter.num_agents,
            observation_mode="vector",
        ),
        adapter.action_dim,
        config,
    )


def test_policy_collect_matches_host_coins():
    """The fused IPPO policy collector must reproduce its host-loop twin.

    Both draw actions from ``jax.random`` splitting policy-key-then-env-key, so
    integer actions are the exact PRNG canary and continuous tensors match to
    float tolerance. ``rollout_steps`` spans a ``max_cycles`` boundary so the
    fused rollout's auto-reset path is exercised. Stats and the
    episode-accumulator writeback must also match, since a chained collector and
    the training loop resume from those accumulators.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary

    state = _make_coins_state(_make_adapter(num_envs, max_cycles, seed))
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_batch, host_obs, _host_rng, host_starts, host_stats = (
        collect_policy_transition_batch_host(
            host_adapter,
            state,
            obs0_host,
            key,
            rollout_steps=rollout_steps,
            algorithm="ippo",
        )
    )

    device_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_device = device_adapter.reset()
    device_batch, device_obs, _device_rng, device_starts, device_stats = (
        collect_policy_transition_batch(
            device_adapter,
            state,
            obs0_device,
            key,
            rollout_steps=rollout_steps,
            algorithm="ippo",
        )
    )

    flat_rows = rollout_steps * num_envs
    assert device_batch.actions.shape == (flat_rows, device_adapter.num_agents)

    # Integer actions: exact PRNG canary (env-key and policy-key streams).
    np.testing.assert_array_equal(
        np.asarray(device_batch.actions), np.asarray(host_batch.actions)
    )
    for field in ("states", "next_states", "rewards", "dones"):
        np.testing.assert_allclose(
            np.asarray(getattr(device_batch, field)),
            np.asarray(getattr(host_batch, field)),
            rtol=0,
            atol=1e-5,
            err_msg=field,
        )
    np.testing.assert_allclose(
        np.asarray(device_starts), np.asarray(host_starts), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(device_obs), np.asarray(host_obs), rtol=0, atol=1e-5
    )

    # Stats + the episode bookkeeping writeback must match the host adapter.
    assert device_stats.real_env_steps == host_stats.real_env_steps
    assert device_stats.completed_episodes == host_stats.completed_episodes
    for attr in ("episode_return_mean", "episode_length_mean"):
        host_value = getattr(host_stats, attr)
        device_value = getattr(device_stats, attr)
        if host_value is None:
            assert device_value is None
        else:
            np.testing.assert_allclose(device_value, host_value, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(
        device_adapter._episode_lengths, host_adapter._episode_lengths
    )
    np.testing.assert_allclose(
        device_adapter._episode_returns,
        host_adapter._episode_returns,
        rtol=0,
        atol=1e-5,
    )


def test_random_collect_is_structurally_valid_and_deterministic():
    """The fused random collector has no host oracle (numpy vs jax PRNG), so
    verify structure: shapes/dtypes, in-range uniform actions, next-state
    consistency (``next_states[t] == states[t+1]``, last row closed by the
    post-rollout obs), determinism given a key, and the real-env-step count.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12

    adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0 = adapter.reset()
    key = jax.random.PRNGKey(seed)
    batch, last_obs, starts, stats = collect_random_transition_batch(
        adapter, obs0, key, rollout_steps=rollout_steps
    )

    flat_rows = rollout_steps * num_envs
    num_agents = adapter.num_agents
    assert batch.states.shape[0] == flat_rows
    assert batch.actions.shape == (flat_rows, num_agents)
    assert np.asarray(batch.actions).dtype == np.int32
    actions = np.asarray(batch.actions)
    assert actions.min() >= 0 and actions.max() < adapter.action_dim

    states = np.asarray(batch.states)
    next_states = np.asarray(batch.next_states)
    # next_states[k] == states[k + num_envs] within the folded [T*E, A, d] layout.
    np.testing.assert_allclose(
        next_states[:-num_envs], states[num_envs:], rtol=0, atol=1e-5
    )
    # The final step's next-states are closed by the post-rollout observations.
    np.testing.assert_allclose(
        next_states[-num_envs:],
        np.asarray(last_obs).reshape((num_envs, num_agents, -1)),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(starts), states)
    assert stats.real_env_steps == rollout_steps * num_envs

    # Determinism: same key + same fresh reset reproduce the batch exactly.
    adapter2 = _make_adapter(num_envs, max_cycles, seed)
    obs0b = adapter2.reset()
    batch2, _last2, _starts2, _stats2 = collect_random_transition_batch(
        adapter2, obs0b, key, rollout_steps=rollout_steps
    )
    np.testing.assert_array_equal(np.asarray(batch2.actions), actions)
    np.testing.assert_array_equal(np.asarray(batch2.states), states)


def test_random_then_policy_handoff_is_continuous():
    """Chaining random -> policy on one adapter (as the prefit block does) must
    thread the carry: the policy collector's first-step states equal the random
    collector's post-rollout observations, and the two batches concatenate.
    """
    num_envs, max_cycles, seed = 4, 8, 3

    adapter = _make_adapter(num_envs, max_cycles, seed)
    state = _make_coins_state(adapter)
    observations = adapter.reset()
    key = jax.random.PRNGKey(seed)

    key, random_key = jax.random.split(key)
    random_batch, observations, _r_starts, _r_stats = collect_random_transition_batch(
        adapter, observations, random_key, rollout_steps=10
    )
    _key, policy_key = jax.random.split(key)
    policy_batch, _p_obs, _p_rng, _p_starts, _p_stats = collect_policy_transition_batch(
        adapter,
        state,
        observations,
        policy_key,
        rollout_steps=10,
        algorithm="ippo",
    )

    # The policy collector must start from the random collector's post-rollout
    # carry: its first-step states (rows [0, num_envs)) equal the handed-off
    # observations.
    np.testing.assert_allclose(
        np.asarray(policy_batch.states[:num_envs]),
        np.asarray(observations),
        rtol=0,
        atol=1e-5,
    )
    combined = concatenate_transition_batches([random_batch, policy_batch])
    assert combined.states.shape[0] == (10 + 10) * num_envs


def test_policy_collect_matches_host_coins_mappo():
    """The fused MAPPO policy collector must reproduce its host-loop twin.

    Same contract as the IPPO case, with the centralized-critic observations
    rebuilt on-device inside ``make_mappo_get_action_and_value``'s closure (host
    builds them in numpy): integer actions are the exact PRNG canary, tensors
    match to float tolerance.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary

    state = _make_coins_mappo_state(_make_adapter(num_envs, max_cycles, seed))
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_batch, host_obs, _host_rng, host_starts, host_stats = (
        collect_policy_transition_batch_host(
            host_adapter,
            state,
            obs0_host,
            key,
            rollout_steps=rollout_steps,
            algorithm="mappo",
        )
    )

    device_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_device = device_adapter.reset()
    device_batch, device_obs, _device_rng, device_starts, device_stats = (
        collect_policy_transition_batch(
            device_adapter,
            state,
            obs0_device,
            key,
            rollout_steps=rollout_steps,
            algorithm="mappo",
        )
    )

    np.testing.assert_array_equal(
        np.asarray(device_batch.actions), np.asarray(host_batch.actions)
    )
    for field in ("states", "next_states", "rewards", "dones"):
        np.testing.assert_allclose(
            np.asarray(getattr(device_batch, field)),
            np.asarray(getattr(host_batch, field)),
            rtol=0,
            atol=1e-5,
            err_msg=field,
        )
    np.testing.assert_allclose(
        np.asarray(device_starts), np.asarray(host_starts), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(device_obs), np.asarray(host_obs), rtol=0, atol=1e-5
    )
    assert device_stats.real_env_steps == host_stats.real_env_steps
    assert device_stats.completed_episodes == host_stats.completed_episodes


def test_policy_collect_rejects_unknown_algorithm():
    adapter = _make_adapter()
    state = _make_coins_state(adapter)
    obs0 = adapter.reset()
    try:
        collect_policy_transition_batch(
            adapter,
            state,
            obs0,
            jax.random.PRNGKey(0),
            rollout_steps=4,
            algorithm="qmix",
        )
    except ValueError as exc:
        assert "algorithm" in str(exc)
    else:  # pragma: no cover - guard must raise
        raise AssertionError(
            "expected ValueError for unknown-algorithm policy collection"
        )
