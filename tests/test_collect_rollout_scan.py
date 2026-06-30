from __future__ import annotations

import jax
import numpy as np

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.training import collect_rollout, collect_rollout_scan


def _make_coins_state(adapter, seed: int = 5):
    config = IPPOConfig(network_arch="mlp")
    return create_ippo_state(
        jax.random.PRNGKey(seed),
        adapter.observation_shape,
        adapter.action_dim,
        config,
    )


def test_scan_rollout_matches_host_collect_rollout_coins():
    """A jitted lax.scan rollout must reproduce host ``collect_rollout`` outputs.

    Actions are STOCHASTIC here (``_ippo_infer_with_entropy`` samples), so the
    policy PRNG stream is live -- the scan must split it in the host's order
    (policy key first, then env keys). Integer actions are the exact-equal PRNG
    canary; continuous tensors match to float tolerance. ``rollout_steps`` spans
    an episode boundary so the auto-reset path inside the scan is exercised.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary

    state = _make_coins_state(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    host_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    obs0_host = host_adapter.reset()
    rng = jax.random.PRNGKey(seed)
    host = collect_rollout(
        host_adapter, state, obs0_host, rng, rollout_steps=rollout_steps
    )

    scan_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    obs0_scan = scan_adapter.reset()
    scan = collect_rollout_scan(
        scan_adapter, state, obs0_scan, rng, rollout_steps=rollout_steps
    )

    flat_agents = num_envs * scan_adapter.num_agents
    assert scan.batch.actions.shape == (rollout_steps, flat_agents)

    # Integer actions: exact PRNG canary (both env-key and policy-key streams).
    np.testing.assert_array_equal(
        np.asarray(scan.batch.actions), np.asarray(host.batch.actions)
    )
    # Continuous tensors: float tolerance (XLA may fuse the scan body differently).
    for field in ("observations", "log_probs", "rewards", "dones", "values"):
        np.testing.assert_allclose(
            np.asarray(getattr(scan.batch, field)),
            np.asarray(getattr(host.batch, field)),
            rtol=0,
            atol=1e-5,
            err_msg=field,
        )
    np.testing.assert_allclose(
        np.asarray(scan.last_values),
        np.asarray(host.last_values),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(scan.next_observations),
        np.asarray(host.next_observations),
        rtol=0,
        atol=1e-5,
    )


def test_scan_rollout_metrics_match_host_collect_rollout_coins():
    """``collect_rollout_scan`` must be a true drop-in: same metrics dict + the
    same episode bookkeeping writeback as host ``collect_rollout``.

    ``train_e2e`` reads ``metrics['completed_episodes']`` and splats the whole
    metrics dict into logged rows, so a minimal-metrics scan path would silently
    report zero episodes and drop diagnostics. ``rollout_steps`` spans one
    boundary (4 envs each complete once, then a partial episode), exercising both
    completed-episode derivation and the carried-over partial accumulator.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12

    state = _make_coins_state(
        JaxMARLCoinGameVectorAdapter(
            num_envs=num_envs, max_cycles=max_cycles, seed=seed
        )
    )

    host_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    obs0_host = host_adapter.reset()
    rng = jax.random.PRNGKey(seed)
    host = collect_rollout(
        host_adapter, state, obs0_host, rng, rollout_steps=rollout_steps
    )

    scan_adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )
    obs0_scan = scan_adapter.reset()
    scan = collect_rollout_scan(
        scan_adapter, state, obs0_scan, rng, rollout_steps=rollout_steps
    )

    assert set(scan.metrics) == set(host.metrics)
    assert scan.metrics["completed_episodes"] == host.metrics["completed_episodes"]
    assert host.metrics["completed_episodes"] == num_envs  # one boundary crossed
    for key in (
        "rollout_mean_reward",
        "episode_return_mean",
        "episode_length_mean",
        "policy_entropy_mean",
        "value_explained_variance",
        "value_target_mean",
    ):
        np.testing.assert_allclose(
            scan.metrics[key], host.metrics[key], rtol=0, atol=1e-5, err_msg=key
        )

    # Cross-call continuity: the partial-episode accumulator must match the host
    # adapter's, so a subsequent rollout completes episodes at the same boundary.
    np.testing.assert_array_equal(
        scan_adapter._episode_lengths, host_adapter._episode_lengths
    )
    np.testing.assert_allclose(
        scan_adapter._episode_returns,
        host_adapter._episode_returns,
        rtol=0,
        atol=1e-5,
    )
