from __future__ import annotations

import jax
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.training import central_observation_shape
from world_marl.world_model_training import (
    collect_policy_transition_batch,
    collect_policy_transition_batch_scan,
    collect_random_transition_batch_scan,
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


def test_policy_scan_matches_host_collect_policy_coins():
    """The IPPO policy collector's scan twin must reproduce the host loop.

    Both draw actions from ``jax.random`` splitting policy-key-then-env-key, so
    integer actions are the exact PRNG canary and continuous tensors match to
    float tolerance. ``rollout_steps`` spans a ``max_cycles`` boundary so the
    scan's auto-reset path is exercised. Stats and the episode-accumulator
    writeback must also match, since a chained collector and the training loop
    resume from those accumulators.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary

    state = _make_coins_state(_make_adapter(num_envs, max_cycles, seed))
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_batch, host_obs, _host_rng, host_starts, host_stats = (
        collect_policy_transition_batch(
            host_adapter,
            state,
            obs0_host,
            key,
            rollout_steps=rollout_steps,
            algorithm="ippo",
        )
    )

    scan_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_scan = scan_adapter.reset()
    scan_batch, scan_obs, _scan_rng, scan_starts, scan_stats = (
        collect_policy_transition_batch_scan(
            scan_adapter,
            state,
            obs0_scan,
            key,
            rollout_steps=rollout_steps,
            algorithm="ippo",
        )
    )

    flat_rows = rollout_steps * num_envs
    assert scan_batch.actions.shape == (flat_rows, scan_adapter.num_agents)

    # Integer actions: exact PRNG canary (env-key and policy-key streams).
    np.testing.assert_array_equal(
        np.asarray(scan_batch.actions), np.asarray(host_batch.actions)
    )
    for field in ("states", "next_states", "rewards", "dones"):
        np.testing.assert_allclose(
            np.asarray(getattr(scan_batch, field)),
            np.asarray(getattr(host_batch, field)),
            rtol=0,
            atol=1e-5,
            err_msg=field,
        )
    np.testing.assert_allclose(
        np.asarray(scan_starts), np.asarray(host_starts), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(scan_obs), np.asarray(host_obs), rtol=0, atol=1e-5
    )

    # Stats + the episode bookkeeping writeback must match the host adapter.
    assert scan_stats.real_env_steps == host_stats.real_env_steps
    assert scan_stats.completed_episodes == host_stats.completed_episodes
    for attr in ("episode_return_mean", "episode_length_mean"):
        host_value = getattr(host_stats, attr)
        scan_value = getattr(scan_stats, attr)
        if host_value is None:
            assert scan_value is None
        else:
            np.testing.assert_allclose(scan_value, host_value, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(
        scan_adapter._episode_lengths, host_adapter._episode_lengths
    )
    np.testing.assert_allclose(
        scan_adapter._episode_returns,
        host_adapter._episode_returns,
        rtol=0,
        atol=1e-5,
    )


def test_random_scan_is_structurally_valid_and_deterministic():
    """The random collector's scan twin has no host oracle (numpy vs jax PRNG),
    so verify structure: shapes/dtypes, in-range uniform actions, next-state
    consistency (``next_states[t] == states[t+1]``, last row closed by the
    post-rollout obs), determinism given a key, and the real-env-step count.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12

    adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0 = adapter.reset()
    key = jax.random.PRNGKey(seed)
    batch, last_obs, starts, stats = collect_random_transition_batch_scan(
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
    batch2, _last2, _starts2, _stats2 = collect_random_transition_batch_scan(
        adapter2, obs0b, key, rollout_steps=rollout_steps
    )
    np.testing.assert_array_equal(np.asarray(batch2.actions), actions)
    np.testing.assert_array_equal(np.asarray(batch2.states), states)


def test_random_then_policy_scan_handoff_is_continuous():
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
    random_batch, observations, _r_starts, _r_stats = (
        collect_random_transition_batch_scan(
            adapter, observations, random_key, rollout_steps=10
        )
    )
    _key, policy_key = jax.random.split(key)
    policy_batch, _p_obs, _p_rng, _p_starts, _p_stats = (
        collect_policy_transition_batch_scan(
            adapter,
            state,
            observations,
            policy_key,
            rollout_steps=10,
            algorithm="ippo",
        )
    )

    # The policy scan must start from the random scan's post-rollout carry: its
    # first-step states (rows [0, num_envs)) equal the handed-off observations.
    np.testing.assert_allclose(
        np.asarray(policy_batch.states[:num_envs]),
        np.asarray(observations),
        rtol=0,
        atol=1e-5,
    )
    combined = concatenate_transition_batches([random_batch, policy_batch])
    assert combined.states.shape[0] == (10 + 10) * num_envs


def test_policy_scan_matches_host_collect_policy_coins_mappo():
    """The MAPPO policy collector's scan twin must reproduce the host loop.

    Same contract as the IPPO case, with the centralized-critic observations
    rebuilt on-device inside ``_make_mappo_get_action_and_value`` (host builds
    them in numpy): integer actions are the exact PRNG canary, tensors match
    to float tolerance.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary

    state = _make_coins_mappo_state(_make_adapter(num_envs, max_cycles, seed))
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_batch, host_obs, _host_rng, host_starts, host_stats = (
        collect_policy_transition_batch(
            host_adapter,
            state,
            obs0_host,
            key,
            rollout_steps=rollout_steps,
            algorithm="mappo",
        )
    )

    scan_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_scan = scan_adapter.reset()
    scan_batch, scan_obs, _scan_rng, scan_starts, scan_stats = (
        collect_policy_transition_batch_scan(
            scan_adapter,
            state,
            obs0_scan,
            key,
            rollout_steps=rollout_steps,
            algorithm="mappo",
        )
    )

    np.testing.assert_array_equal(
        np.asarray(scan_batch.actions), np.asarray(host_batch.actions)
    )
    for field in ("states", "next_states", "rewards", "dones"):
        np.testing.assert_allclose(
            np.asarray(getattr(scan_batch, field)),
            np.asarray(getattr(host_batch, field)),
            rtol=0,
            atol=1e-5,
            err_msg=field,
        )
    np.testing.assert_allclose(
        np.asarray(scan_starts), np.asarray(host_starts), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(scan_obs), np.asarray(host_obs), rtol=0, atol=1e-5
    )
    assert scan_stats.real_env_steps == host_stats.real_env_steps
    assert scan_stats.completed_episodes == host_stats.completed_episodes


def test_policy_scan_rejects_unknown_algorithm():
    adapter = _make_adapter()
    state = _make_coins_state(adapter)
    obs0 = adapter.reset()
    try:
        collect_policy_transition_batch_scan(
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
            "expected ValueError for unknown-algorithm scan collection"
        )
