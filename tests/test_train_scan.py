from __future__ import annotations

import jax
import numpy as np
import pytest

from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.ippo import ppo_update
from world_marl.algs.mappo import (
    MAPPOConfig,
    create_train_state as create_mappo_state,
    mappo_update,
)
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.training import (
    central_observation_shape,
    collect_mappo_rollout,
    collect_rollout,
    train_real_scan,
)
from world_marl.world_model import (
    VectorWorldModelConfig,
    create_world_model_state,
    simulate_ippo_model_rollout,
    simulate_mappo_model_rollout,
    train_imagined_scan,
)
from world_marl.world_model_training import sample_initial_states

ALGORITHMS = ("ippo", "mappo")


def _imagined_reward_done(states, actions, next_states):
    del states, next_states
    return actions.astype(np.float32), (actions == 2).astype(np.float32)


def _make_adapter(num_envs=4, max_cycles=8, seed=3):
    return JaxMARLCoinGameVectorAdapter(
        num_envs=num_envs, max_cycles=max_cycles, seed=seed
    )


def _make_policy_state(adapter, algorithm: str, seed: int = 5):
    if algorithm == "mappo":
        config = MAPPOConfig(network_arch="mlp")
        state = create_mappo_state(
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
    else:
        config = IPPOConfig(network_arch="mlp")
        state = create_ippo_state(
            jax.random.PRNGKey(seed),
            adapter.observation_shape,
            adapter.action_dim,
            config,
        )
    return state, config


def _host_oracle(
    adapter,
    train_state,
    observations,
    rng,
    *,
    num_updates,
    config,
    rollout_steps,
    algorithm,
    freeze_policy,
):
    """The trusted host loop: exactly ``run_training``'s model-free branch.

    Calls the host collector (``collect_rollout`` / ``collect_mappo_rollout``,
    whose PRNG threading the adapter scan reproduces bit-for-bit on integer
    actions) + a standalone jitted update in a Python loop, threading state
    through the (stateful) adapter and the local ``rng``/``observations`` just
    as the production MeltingPot loop does. Returns
    ``(train_state, observations, rng, per_update_metrics)``.
    """
    if algorithm == "mappo":
        update = mappo_update

        def collect(state, obs, key):
            return collect_mappo_rollout(
                adapter,
                state,
                obs,
                key,
                rollout_steps=rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                observation_mode="vector",
            )
    else:
        update = ppo_update

        def collect(state, obs, key):
            return collect_rollout(
                adapter,
                state,
                obs,
                key,
                rollout_steps=rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

    update_fn = jax.jit(
        lambda st, batch, last_values, uk: update(st, batch, last_values, uk, config)
    )
    per_update: list[dict] = []
    for _ in range(num_updates):
        rng, rollout_key, update_key = jax.random.split(rng, 3)
        rollout = collect(train_state, observations, rollout_key)
        observations = rollout.next_observations
        update_metrics: dict = {}
        if not freeze_policy:
            train_state, update_metrics = update_fn(
                train_state, rollout.batch, rollout.last_values, update_key
            )
        row = {
            "rollout_mean_reward": rollout.metrics["rollout_mean_reward"],
            "episode_return_mean": rollout.metrics["episode_return_mean"],
            "episode_length_mean": rollout.metrics["episode_length_mean"],
            "completed_episodes": rollout.metrics["completed_episodes"],
        }
        for key, value in update_metrics.items():
            row[f"ppo/{key}"] = value
        per_update.append(row)
    return train_state, observations, rng, per_update


def _assert_params_close(host_state, scan_state, *, atol):
    host_leaves = jax.tree_util.tree_leaves(host_state.params)
    scan_leaves = jax.tree_util.tree_leaves(scan_state.params)
    assert len(host_leaves) == len(scan_leaves)
    for h, s in zip(host_leaves, scan_leaves):
        np.testing.assert_allclose(np.asarray(s), np.asarray(h), rtol=0, atol=atol)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_real_scan_matches_host_single_update(algorithm):
    """N=1 is the tight logic oracle: same start params -> the folded scan runs
    the *identical* rollout (bit-for-bit), gradient, and update as the host loop.
    The only admissible drift is XLA fusing the update inlined-in-scan vs the
    standalone jit, so params/ppo metrics match to a tight float tolerance while
    the carried ``rng`` (structural), ``completed_episodes`` and
    ``episode_length_mean`` (fixed-horizon coins -> timer-driven) match exactly.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps = 12  # crosses one max_cycles boundary -> reset-on-done

    train_state, config = _make_policy_state(
        _make_adapter(num_envs, max_cycles, seed), algorithm
    )
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_state, host_obs, host_rng, host_rows = _host_oracle(
        host_adapter,
        train_state,
        obs0_host,
        key,
        num_updates=1,
        config=config,
        rollout_steps=rollout_steps,
        algorithm=algorithm,
        freeze_policy=False,
    )

    scan_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_scan = scan_adapter.reset()
    scan_state, scan_obs, scan_rng, scan_metrics = train_real_scan(
        scan_adapter,
        train_state,
        obs0_scan,
        key,
        num_updates=1,
        config=config,
        rollout_steps=rollout_steps,
        algorithm=algorithm,
        freeze_policy=False,
    )

    # Structural PRNG canary: the carried key must be bit-identical.
    np.testing.assert_array_equal(np.asarray(scan_rng), np.asarray(host_rng))

    # Fixed-horizon coins -> episode counts/lengths are timer-driven (exact).
    assert (
        int(scan_metrics["completed_episodes"][0]) == host_rows[0]["completed_episodes"]
    )
    np.testing.assert_allclose(
        np.asarray(scan_metrics["episode_length_mean"][0]),
        host_rows[0]["episode_length_mean"],
        rtol=0,
        atol=1e-5,
    )

    # Reward-derived metrics: identical rollout at N=1 -> tight.
    for field in ("rollout_mean_reward", "episode_return_mean"):
        np.testing.assert_allclose(
            np.asarray(scan_metrics[field][0]), host_rows[0][field], rtol=0, atol=1e-5
        )
    for field in ("ppo/actor_loss", "ppo/value_loss", "ppo/total_loss", "ppo/entropy"):
        np.testing.assert_allclose(
            np.asarray(scan_metrics[field][0]), host_rows[0][field], rtol=0, atol=1e-5
        )

    _assert_params_close(host_state, scan_state, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(scan_obs), np.asarray(host_obs), rtol=0, atol=1e-5
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_real_scan_matches_host_multi_update(algorithm):
    """N>1 pins the carry threading across a boundary. The carried ``rng`` and
    the timer-driven episode counts/lengths must match exactly every update
    (proving the reset-on-done accumulators thread across the max_cycles reset),
    while compounded fusion drift is tolerated on params and reward metrics.
    """
    num_envs, max_cycles, seed = 4, 8, 3
    rollout_steps, num_updates = 12, 3

    train_state, config = _make_policy_state(
        _make_adapter(num_envs, max_cycles, seed), algorithm
    )
    key = jax.random.PRNGKey(seed)

    host_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_host = host_adapter.reset()
    host_state, _host_obs, host_rng, host_rows = _host_oracle(
        host_adapter,
        train_state,
        obs0_host,
        key,
        num_updates=num_updates,
        config=config,
        rollout_steps=rollout_steps,
        algorithm=algorithm,
        freeze_policy=False,
    )

    scan_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0_scan = scan_adapter.reset()
    scan_state, _scan_obs, scan_rng, scan_metrics = train_real_scan(
        scan_adapter,
        train_state,
        obs0_scan,
        key,
        num_updates=num_updates,
        config=config,
        rollout_steps=rollout_steps,
        algorithm=algorithm,
        freeze_policy=False,
    )

    np.testing.assert_array_equal(np.asarray(scan_rng), np.asarray(host_rng))

    for i in range(num_updates):
        assert (
            int(scan_metrics["completed_episodes"][i])
            == host_rows[i]["completed_episodes"]
        )
        np.testing.assert_allclose(
            np.asarray(scan_metrics["episode_length_mean"][i]),
            host_rows[i]["episode_length_mean"],
            rtol=0,
            atol=1e-5,
        )

    # Compounded inlined-vs-standalone update fusion drift -> loose stability check.
    _assert_params_close(host_state, scan_state, atol=1e-3)
    for i in range(num_updates):
        for field in ("ppo/total_loss", "rollout_mean_reward", "episode_return_mean"):
            np.testing.assert_allclose(
                np.asarray(scan_metrics[field][i]),
                host_rows[i][field],
                rtol=0,
                atol=1e-3,
            )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_real_scan_freeze_policy_leaves_params_unchanged(algorithm):
    """With ``freeze_policy`` the scan must skip the update and return the
    start params unchanged, while still advancing the env carry and rollout
    metrics (the warmup path uses this before the world model is fit).
    """
    num_envs, max_cycles, seed = 4, 8, 3

    train_state, config = _make_policy_state(
        _make_adapter(num_envs, max_cycles, seed), algorithm
    )
    key = jax.random.PRNGKey(seed)

    scan_adapter = _make_adapter(num_envs, max_cycles, seed)
    obs0 = scan_adapter.reset()
    scan_state, _obs, _rng, scan_metrics = train_real_scan(
        scan_adapter,
        train_state,
        obs0,
        key,
        num_updates=2,
        config=config,
        rollout_steps=12,
        algorithm=algorithm,
        freeze_policy=True,
    )

    _assert_params_close(train_state, scan_state, atol=0.0)
    assert scan_metrics["rollout_mean_reward"].shape == (2,)


def test_train_real_scan_rejects_unknown_algorithm():
    adapter = _make_adapter()
    train_state, config = _make_policy_state(adapter, "ippo")
    with pytest.raises(ValueError, match="algorithm"):
        train_real_scan(
            adapter,
            train_state,
            adapter.reset(),
            jax.random.PRNGKey(0),
            num_updates=1,
            config=config,
            rollout_steps=4,
            algorithm="qmix",
        )


def _make_imagined_setup(algorithm: str, pool_size=5, num_envs=3):
    wm_config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    if algorithm == "mappo":
        policy_config = MAPPOConfig(network_arch="mlp", num_minibatches=1)
        train_state = create_mappo_state(
            jax.random.PRNGKey(1),
            (wm_config.state_dim,),
            central_observation_shape(
                (wm_config.state_dim,),
                wm_config.num_agents,
                observation_mode="vector",
            ),
            wm_config.action_dim,
            policy_config,
        )
    else:
        policy_config = IPPOConfig(network_arch="mlp", num_minibatches=1)
        train_state = create_ippo_state(
            jax.random.PRNGKey(1),
            (wm_config.state_dim,),
            wm_config.action_dim,
            policy_config,
        )
    model_state = create_world_model_state(jax.random.PRNGKey(0), wm_config)
    model_start_states = jax.random.normal(
        jax.random.PRNGKey(7), (pool_size, wm_config.num_agents, wm_config.state_dim)
    )
    return (
        model_state,
        train_state,
        model_start_states,
        wm_config,
        policy_config,
        num_envs,
    )


def _host_imagined_oracle(
    model_state,
    train_state,
    model_start_states,
    rng,
    *,
    num_updates,
    policy_config,
    world_model_config,
    rollout_steps,
    num_envs,
    algorithm,
    freeze_policy,
):
    """The trusted host loop: exactly ``run_training``'s imagined (prefit) branch.

    Per update splits the key four ways (``rng, rollout_key, start_key,
    update_key``), resamples ``initial_states`` from the fixed pool, runs the
    on-device imagined rollout, and applies a standalone jitted update.
    """
    if algorithm == "mappo":
        update = mappo_update
        simulate = simulate_mappo_model_rollout
    else:
        update = ppo_update
        simulate = simulate_ippo_model_rollout
    update_fn = jax.jit(
        lambda st, batch, last_values, uk: update(
            st, batch, last_values, uk, policy_config
        )
    )
    per_update: list[dict] = []
    for _ in range(num_updates):
        rng, rollout_key, start_key, update_key = jax.random.split(rng, 4)
        initial_states = sample_initial_states(
            model_start_states, start_key, num_envs=num_envs
        )
        rollout = simulate(
            model_state,
            train_state,
            initial_states,
            rollout_key,
            rollout_steps=rollout_steps,
            config=world_model_config,
            reward_done_fn=_imagined_reward_done,
        )
        update_metrics: dict = {}
        if not freeze_policy:
            train_state, update_metrics = update_fn(
                train_state, rollout.batch, rollout.last_values, update_key
            )
        row = {
            "rollout_mean_reward": rollout.metrics["rollout_mean_reward"],
            "model_rollout_mean_reward": rollout.metrics["model_rollout_mean_reward"],
        }
        for key, value in update_metrics.items():
            row[f"ppo/{key}"] = value
        per_update.append(row)
    return train_state, rng, per_update


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_imagined_scan_matches_host_single_update(algorithm):
    """N=1 tight logic oracle for the imagined (prefit) branch: same start
    params + start_key + rollout_key -> identical imagined rollout and gradient,
    so params/ppo/reward metrics match tight and the carried ``rng`` (4-way split
    threaded structurally) matches exactly.
    """
    model_state, train_state, pool, wm_config, policy_config, num_envs = (
        _make_imagined_setup(algorithm)
    )
    key = jax.random.PRNGKey(4)

    host_state, host_rng, host_rows = _host_imagined_oracle(
        model_state,
        train_state,
        pool,
        key,
        num_updates=1,
        policy_config=policy_config,
        world_model_config=wm_config,
        rollout_steps=3,
        num_envs=num_envs,
        algorithm=algorithm,
        freeze_policy=False,
    )
    scan_state, scan_rng, scan_metrics = train_imagined_scan(
        model_state,
        train_state,
        pool,
        key,
        num_updates=1,
        policy_config=policy_config,
        world_model_config=wm_config,
        rollout_steps=3,
        reward_done_fn=_imagined_reward_done,
        num_envs=num_envs,
        algorithm=algorithm,
        freeze_policy=False,
    )

    np.testing.assert_array_equal(np.asarray(scan_rng), np.asarray(host_rng))
    for field in ("rollout_mean_reward", "model_rollout_mean_reward"):
        np.testing.assert_allclose(
            np.asarray(scan_metrics[field][0]), host_rows[0][field], rtol=0, atol=1e-5
        )
    for field in ("ppo/actor_loss", "ppo/value_loss", "ppo/total_loss", "ppo/entropy"):
        np.testing.assert_allclose(
            np.asarray(scan_metrics[field][0]), host_rows[0][field], rtol=0, atol=1e-5
        )
    _assert_params_close(host_state, scan_state, atol=1e-5)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_imagined_scan_matches_host_multi_update(algorithm):
    """N>1 pins carry threading (train_state + 4-way rng across updates). The
    carried ``rng`` matches exactly; compounded fusion drift is tolerated on
    params and reward metrics.
    """
    model_state, train_state, pool, wm_config, policy_config, num_envs = (
        _make_imagined_setup(algorithm)
    )
    key = jax.random.PRNGKey(4)
    num_updates = 3

    host_state, host_rng, host_rows = _host_imagined_oracle(
        model_state,
        train_state,
        pool,
        key,
        num_updates=num_updates,
        policy_config=policy_config,
        world_model_config=wm_config,
        rollout_steps=3,
        num_envs=num_envs,
        algorithm=algorithm,
        freeze_policy=False,
    )
    scan_state, scan_rng, scan_metrics = train_imagined_scan(
        model_state,
        train_state,
        pool,
        key,
        num_updates=num_updates,
        policy_config=policy_config,
        world_model_config=wm_config,
        rollout_steps=3,
        reward_done_fn=_imagined_reward_done,
        num_envs=num_envs,
        algorithm=algorithm,
        freeze_policy=False,
    )

    np.testing.assert_array_equal(np.asarray(scan_rng), np.asarray(host_rng))
    _assert_params_close(host_state, scan_state, atol=1e-3)
    for i in range(num_updates):
        for field in ("ppo/total_loss", "model_rollout_mean_reward"):
            np.testing.assert_allclose(
                np.asarray(scan_metrics[field][i]),
                host_rows[i][field],
                rtol=0,
                atol=1e-3,
            )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_imagined_scan_freeze_policy_leaves_params_unchanged(algorithm):
    model_state, train_state, pool, wm_config, policy_config, num_envs = (
        _make_imagined_setup(algorithm)
    )
    key = jax.random.PRNGKey(4)

    scan_state, _rng, scan_metrics = train_imagined_scan(
        model_state,
        train_state,
        pool,
        key,
        num_updates=2,
        policy_config=policy_config,
        world_model_config=wm_config,
        rollout_steps=3,
        reward_done_fn=_imagined_reward_done,
        num_envs=num_envs,
        algorithm=algorithm,
        freeze_policy=True,
    )
    _assert_params_close(train_state, scan_state, atol=0.0)
    assert scan_metrics["model_rollout_mean_reward"].shape == (2,)
