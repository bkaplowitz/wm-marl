from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from flow_matching.models import MLPVectorField
from world_marl.algs.ippo import IPPOConfig, create_train_state as create_ippo_state
from world_marl.algs.mappo import MAPPOConfig, create_train_state as create_mappo_state
from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from tests.conftest import DummyParallelEnv
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    _pack_cond_vars,
    _pack_transition,
    _transition_dim,
    _unpack_transition,
    create_world_model_state,
    simulate_mappo_model_rollout,
    train_world_model_step,
    world_model_loss,
)
from world_marl.world_model_training import (
    collect_random_transition_batch,
    flatten_state_observations,
)


def test_random_prefit_collection_uses_flat_vector_states(
    dummy_env_factory,
):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        observations = adapter.reset()
        batch, next_observations, start_states, stats = collect_random_transition_batch(
            adapter,
            observations,
            np.random.default_rng(0),
            rollout_steps=2,
        )

        state_dim = int(np.prod(adapter.observation_shape))
        assert batch.states.shape == (
            2 * adapter.num_envs,
            adapter.num_agents,
            state_dim,
        )
        assert batch.actions.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert batch.next_states.shape == batch.states.shape
        assert batch.rewards.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert batch.dones.shape == (2 * adapter.num_envs, adapter.num_agents)
        assert not hasattr(batch, "policy_ids")
        assert next_observations.shape == observations.shape
        assert start_states.shape == (
            2 * adapter.num_envs,
            adapter.num_agents,
            state_dim,
        )
        assert stats.real_env_steps == 2
        assert stats.completed_episodes == 0
    finally:
        adapter.close()


def test_prefit_collection_counts_completed_real_episodes():
    adapter = MeltingPotVectorAdapter(
        num_envs=2,
        env_factory=lambda: DummyParallelEnv(horizon=1),
    )
    try:
        observations = adapter.reset()
        _, _, _, stats = collect_random_transition_batch(
            adapter,
            observations,
            np.random.default_rng(0),
            rollout_steps=1,
        )

        assert stats.real_env_steps == 2
        assert stats.completed_episodes == 2
        assert stats.episode_return_mean is not None
        assert stats.episode_length_mean == 1.0
    finally:
        adapter.close()


def test_flatten_state_observations_preserves_env_agent_axes():
    observations = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))

    states = flatten_state_observations(observations)

    assert states.shape == (2, 3, 4)
    np.testing.assert_allclose(states, observations)


def test_create_world_model_state_uses_mlp_vector_field():
    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )

    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    model = model_state.apply_fn.__self__
    target_dim = config.num_agents * config.state_dim
    cond_dim = config.num_agents * config.state_dim + (
        config.num_agents * config.action_dim
    )
    output = model_state.apply_fn(
        {"params": model_state.params},
        jnp.zeros((1, target_dim), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, cond_dim), dtype=jnp.float32),
    )

    assert isinstance(model, MLPVectorField)
    assert output.shape == (1, target_dim)


def test_world_model_pack_cond_vars_order_is_state_then_agent_major_actions():
    config = VectorWorldModelConfig(state_dim=2, num_agents=2, action_dim=3)
    states = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    actions = jnp.asarray([[0, 2]], dtype=jnp.int32)

    packed = _pack_cond_vars(states, actions, config)

    np.testing.assert_allclose(
        np.asarray(packed),
        np.asarray([[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )


def test_world_model_pack_transition_round_trips_next_state_only():
    config = VectorWorldModelConfig(state_dim=2, num_agents=2, action_dim=3)
    next_states = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    packed = _pack_transition(next_states, config)
    unpacked = _unpack_transition(packed, config)

    assert packed.shape == (2, _transition_dim(config))
    assert packed.shape == (2, 4)
    np.testing.assert_allclose(np.asarray(unpacked), np.asarray(next_states))


def test_world_model_loss_ignores_reward_and_done_fields():
    config = VectorWorldModelConfig(
        state_dim=2,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    batch = VectorTransitionBatch(
        states=jnp.zeros((4, 2, 2), dtype=jnp.float32),
        actions=jnp.zeros((4, 2), dtype=jnp.int32),
        next_states=jnp.ones((4, 2, 2), dtype=jnp.float32),
        rewards=jnp.zeros((4, 2), dtype=jnp.float32),
        dones=jnp.zeros((4, 2), dtype=jnp.float32),
    )
    changed_rewards_dones = batch._replace(
        rewards=jnp.full((4, 2), 100.0, dtype=jnp.float32),
        dones=jnp.ones((4, 2), dtype=jnp.float32),
    )

    key = jax.random.PRNGKey(1)
    base_loss = world_model_loss(
        model_state.params,
        model_state.apply_fn,
        key,
        batch,
        config,
    )
    changed_loss = world_model_loss(
        model_state.params,
        model_state.apply_fn,
        key,
        changed_rewards_dones,
        config,
    )

    np.testing.assert_allclose(np.asarray(base_loss), np.asarray(changed_loss))


def test_train_world_model_step_returns_finite_loss():
    config = VectorWorldModelConfig(
        state_dim=2,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    batch = VectorTransitionBatch(
        states=jnp.zeros((4, 2, 2), dtype=jnp.float32),
        actions=jnp.zeros((4, 2), dtype=jnp.int32),
        next_states=jnp.ones((4, 2, 2), dtype=jnp.float32),
        rewards=jnp.zeros((4, 2), dtype=jnp.float32),
        dones=jnp.zeros((4, 2), dtype=jnp.float32),
    )

    _, loss = train_world_model_step(
        model_state,
        jax.random.PRNGKey(1),
        batch,
        config,
    )

    assert jnp.isfinite(loss)


def test_simulate_mappo_model_rollout_returns_vector_central_batches():
    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_config = MAPPOConfig(network_arch="mlp", num_minibatches=1)
    policy_state = create_mappo_state(
        jax.random.PRNGKey(1),
        (4,),
        (10,),
        3,
        policy_config,
    )
    initial_states = jnp.zeros((3, 2, 4), dtype=jnp.float32)

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)

    rollout = simulate_mappo_model_rollout(
        model_state,
        policy_state,
        initial_states,
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
        reward_done_fn=reward_done_fn,
    )

    assert rollout.batch.observations.shape == (2, 6, 4)
    assert rollout.batch.central_observations.shape == (2, 6, 10)
    assert rollout.batch.actions.shape == (2, 6)
    assert rollout.batch.rewards.shape == (2, 6)
    assert rollout.last_values.shape == (6,)
    assert "rollout_mean_reward" in rollout.metrics
    assert (
        rollout.metrics["rollout_mean_reward"]
        == rollout.metrics["model_rollout_mean_reward"]
    )


def test_simulate_ippo_model_rollout_supports_mlp_vector_policy():
    from world_marl.world_model import simulate_ippo_model_rollout

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        jnp.zeros((3, 2, 4), dtype=jnp.float32),
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
        reward_done_fn=reward_done_fn,
    )

    assert rollout.batch.observations.shape == (2, 6, 4)
    assert rollout.batch.actions.shape == (2, 6)
    assert rollout.last_values.shape == (6,)
    assert "rollout_mean_reward" in rollout.metrics
    assert (
        rollout.metrics["rollout_mean_reward"]
        == rollout.metrics["model_rollout_mean_reward"]
    )


def test_simulate_ippo_model_rollout_uses_reward_done_provider():
    from world_marl.world_model import simulate_ippo_model_rollout

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        jnp.zeros((3, 2, 4), dtype=jnp.float32),
        jax.random.PRNGKey(2),
        rollout_steps=2,
        config=config,
        reward_done_fn=reward_done_fn,
    )

    np.testing.assert_allclose(
        np.asarray(rollout.batch.rewards),
        np.asarray(rollout.batch.actions, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.dones),
        np.asarray(rollout.batch.actions == 2, dtype=np.float32),
    )


def test_simulate_ippo_model_rollout_jits_with_multiple_integration_steps():
    # Guards the jitted lax.scan rollout: with integration_steps > 1 the inner
    # Euler integrator is itself a lax.scan, so a stray host-sync (e.g. a float()
    # left inside the scan body) would raise ConcretizationTypeError here rather
    # than passing silently as it can with integration_steps == 1.
    from world_marl.world_model import simulate_ippo_model_rollout

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=3,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        jnp.zeros((3, 2, 4), dtype=jnp.float32),
        jax.random.PRNGKey(2),
        rollout_steps=4,
        config=config,
        reward_done_fn=reward_done_fn,
    )

    assert rollout.batch.observations.shape == (4, 6, 4)
    assert rollout.batch.actions.shape == (4, 6)
    assert rollout.batch.rewards.shape == (4, 6)
    assert rollout.last_values.shape == (6,)
    assert bool(jnp.all(jnp.isfinite(rollout.batch.observations)))
    assert bool(jnp.all(jnp.isfinite(rollout.batch.rewards)))
    assert bool(jnp.all(jnp.isfinite(rollout.last_values)))
    assert np.isfinite(rollout.metrics["rollout_mean_reward"])


def test_fit_world_model_steps_returns_per_step_loss_history():
    from world_marl.world_model import create_world_model_state
    from world_marl.world_model_training import fit_world_model_steps

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=1,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    n = 6
    batch = VectorTransitionBatch(
        states=jnp.zeros((n, 2, 4), dtype=jnp.float32),
        actions=jnp.zeros((n, 2), dtype=jnp.int32),
        next_states=jnp.ones((n, 2, 4), dtype=jnp.float32),
        rewards=jnp.zeros((n, 2), dtype=jnp.float32),
        dones=jnp.zeros((n, 2), dtype=jnp.float32),
    )
    steps = 5

    _, _, loss, history = fit_world_model_steps(
        model_state,
        jax.random.PRNGKey(1),
        batch,
        config,
        steps=steps,
    )

    history = np.asarray(history, dtype=np.float32)
    assert history.shape == (steps,)
    np.testing.assert_allclose(history[-1], np.asarray(loss, dtype=np.float32))


def test_fit_world_model_steps_matches_explicit_python_loop():
    # Pins the lax.scan fit against an independent Python-loop reference so a
    # silent key-threading break (e.g. reusing one key every step) cannot pass.
    # The flow-matching loss samples a fresh time/noise per key, so splitting the
    # rng inside the scan body must reproduce the loop's exact key sequence,
    # final optimizer state, and per-step loss history.
    from world_marl.world_model import create_world_model_state
    from world_marl.world_model_training import fit_world_model_steps

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=2,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    n, steps = 6, 7
    batch = VectorTransitionBatch(
        states=jax.random.normal(jax.random.PRNGKey(3), (n, 2, 4)),
        actions=jnp.ones((n, 2), dtype=jnp.int32),
        next_states=jax.random.normal(jax.random.PRNGKey(4), (n, 2, 4)),
        rewards=jnp.zeros((n, 2), dtype=jnp.float32),
        dones=jnp.zeros((n, 2), dtype=jnp.float32),
    )
    rng = jax.random.PRNGKey(1)

    new_state, new_rng, new_loss, new_history = fit_world_model_steps(
        model_state, rng, batch, config, steps=steps
    )

    ref_state, ref_rng = model_state, rng
    ref_losses = []
    for _ in range(steps):
        ref_rng, fit_key = jax.random.split(ref_rng)
        ref_state, ref_loss = train_world_model_step(ref_state, fit_key, batch, config)
        ref_losses.append(ref_loss)
    ref_history = jnp.stack(ref_losses)

    np.testing.assert_allclose(
        np.asarray(new_history), np.asarray(ref_history), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(new_loss), np.asarray(ref_history[-1]), atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(new_rng), np.asarray(ref_rng))
    for new_leaf, ref_leaf in zip(
        jax.tree_util.tree_leaves(new_state.params),
        jax.tree_util.tree_leaves(ref_state.params),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(new_leaf), np.asarray(ref_leaf), atol=1e-5
        )

    # Non-vacuity: the fit must actually move the loss, else equality is trivial.
    assert not bool(jnp.allclose(new_history[0], new_history[-1]))


def _reward_done_actions(states, actions, next_states):
    del states, next_states
    return actions.astype(jnp.float32), (actions == 2).astype(jnp.float32)


def _explicit_imagined_unroll(
    model_state,
    policy_state,
    initial_states,
    rng,
    *,
    rollout_steps,
    config,
    is_mappo,
):
    # Plain-Python reference for the jitted lax.scan in _imagined_rollout. Splits
    # the rng into (action_key, model_key) per step exactly as the scan body does,
    # so any divergence in key threading or carry handling shows up as a mismatch.
    from world_marl.training import build_vector_central
    from world_marl.world_model import (
        _apply_vector_policy,
        _reward_done,
        predict_next,
    )

    num_envs = initial_states.shape[0]
    num_actors = num_envs * config.num_agents
    current = initial_states
    keys = ("observations", "actions", "log_probs", "rewards", "dones", "values")
    rows = {key: [] for key in keys}
    if is_mappo:
        rows["central_observations"] = []

    for _ in range(rollout_steps):
        flat = current.reshape((num_actors, config.state_dim))
        central = (
            build_vector_central(current, jnp).reshape((num_actors, -1))
            if is_mappo
            else None
        )
        rng, action_key, model_key = jax.random.split(rng, 3)
        policy, values = _apply_vector_policy(policy_state, flat, central)
        actions = policy.sample(seed=action_key).astype(jnp.int32)
        log_probs = policy.log_prob(actions)
        env_actions = actions.reshape((num_envs, config.num_agents))
        next_states = predict_next(model_state, model_key, current, env_actions, config)
        rewards, dones = _reward_done(
            _reward_done_actions, current, env_actions, next_states
        )
        rows["observations"].append(flat)
        rows["actions"].append(actions)
        rows["log_probs"].append(log_probs)
        rows["rewards"].append(rewards.reshape((num_actors,)))
        rows["dones"].append(dones.reshape((num_actors,)))
        rows["values"].append(values)
        if is_mappo:
            rows["central_observations"].append(central)
        current = next_states

    stacked = {key: jnp.stack(value, axis=0) for key, value in rows.items()}
    last_flat = current.reshape((num_actors, config.state_dim))
    last_central = (
        build_vector_central(current, jnp).reshape((num_actors, -1))
        if is_mappo
        else None
    )
    last_values = _apply_vector_policy(policy_state, last_flat, last_central)[1]
    return stacked, current, last_values


def test_simulate_ippo_model_rollout_matches_explicit_python_loop():
    from world_marl.world_model import (
        create_world_model_state,
        simulate_ippo_model_rollout,
    )

    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=2,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )
    initial_states = jax.random.normal(jax.random.PRNGKey(5), (3, 2, 4))
    rng = jax.random.PRNGKey(2)
    rollout_steps = 4

    rollout = simulate_ippo_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        reward_done_fn=_reward_done_actions,
    )
    stacked, final_states, last_values = _explicit_imagined_unroll(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        is_mappo=False,
    )

    np.testing.assert_array_equal(
        np.asarray(rollout.batch.actions), np.asarray(stacked["actions"])
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.observations),
        np.asarray(stacked["observations"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.log_probs),
        np.asarray(stacked["log_probs"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.rewards), np.asarray(stacked["rewards"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.dones), np.asarray(stacked["dones"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.values), np.asarray(stacked["values"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.last_values), np.asarray(last_values), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.next_observations), np.asarray(final_states), atol=1e-5
    )

    # Non-vacuity: the world model must actually evolve the states over the
    # rollout, otherwise matching a no-op reference proves nothing.
    assert not bool(jnp.allclose(rollout.next_observations, initial_states))


def test_simulate_mappo_model_rollout_matches_explicit_python_loop():
    config = VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(8,),
        integration_steps=2,
    )
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_mappo_state(
        jax.random.PRNGKey(1),
        (4,),
        (10,),
        3,
        MAPPOConfig(network_arch="mlp", num_minibatches=1),
    )
    initial_states = jax.random.normal(jax.random.PRNGKey(5), (3, 2, 4))
    rng = jax.random.PRNGKey(2)
    rollout_steps = 4

    rollout = simulate_mappo_model_rollout(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        reward_done_fn=_reward_done_actions,
    )
    stacked, final_states, last_values = _explicit_imagined_unroll(
        model_state,
        policy_state,
        initial_states,
        rng,
        rollout_steps=rollout_steps,
        config=config,
        is_mappo=True,
    )

    np.testing.assert_array_equal(
        np.asarray(rollout.batch.actions), np.asarray(stacked["actions"])
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.central_observations),
        np.asarray(stacked["central_observations"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.observations),
        np.asarray(stacked["observations"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.rewards), np.asarray(stacked["rewards"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.batch.values), np.asarray(stacked["values"]), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.last_values), np.asarray(last_values), atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(rollout.next_observations), np.asarray(final_states), atol=1e-5
    )

    assert not bool(jnp.allclose(rollout.next_observations, initial_states))
