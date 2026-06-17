from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.algs.ippo import IPPOConfig
from world_marl.algs.ippo import create_train_state as create_ippo_state
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.world_model import (
    VectorTransitionBatch,
    VectorWorldModelConfig,
    _num_factors,
    _pack_discrete_tokens,
    _unpack_discrete_onehot,
    create_world_model_state,
    predict_next,
    simulate_ippo_model_rollout,
    train_world_model_step,
)
from world_marl.world_model_training import fit_world_model_steps


def _coins_config() -> VectorWorldModelConfig:
    return VectorWorldModelConfig(
        state_dim=36,
        num_agents=2,
        action_dim=5,
        num_categories=9,
    )


def test_pack_discrete_tokens_matches_real_env_argmax_decode():
    # Lock the strided (3, 3, 4) layout against the canonical per-channel argmax
    # decode that coin_game_reward_done trusts, on REAL env observations rather
    # than synthesized one-hots, so a reshape bug cannot masquerade as a modeling
    # failure later. The oracle appends agent-major / channel-minor to match the
    # (B, A, C) -> (B, d) flatten in _pack_discrete_tokens.
    adapter = JaxMARLCoinGameVectorAdapter(num_envs=4, max_cycles=50, seed=0)
    try:
        config = _coins_config()
        rng = np.random.default_rng(0)
        observations = adapter.reset()
        for _ in range(10):
            states = jnp.asarray(observations)  # (B, 2, 36)
            tokens = _pack_discrete_tokens(states, config)  # (B, d=8)

            num_envs = states.shape[0]
            expected = []
            for agent in range(config.num_agents):
                grid = np.asarray(states[:, agent]).reshape((num_envs, 3, 3, 4))
                for channel in range(config.state_dim // config.num_categories):
                    expected.append(
                        np.argmax(grid[..., channel].reshape((num_envs, 9)), axis=-1)
                    )
            expected_tokens = np.stack(expected, axis=1)  # (B, 8)

            np.testing.assert_array_equal(np.asarray(tokens), expected_tokens)
            observations = adapter.step(adapter.sample_actions(rng)).observations
    finally:
        adapter.close()


def test_discrete_pack_unpack_round_trips_on_real_states():
    # Valid one-hot env states must survive tokens -> one-hot exactly, confirming
    # _unpack_discrete_onehot rebuilds the strided position-major layout.
    adapter = JaxMARLCoinGameVectorAdapter(num_envs=4, max_cycles=50, seed=1)
    try:
        config = _coins_config()
        states = jnp.asarray(adapter.reset())
        tokens = _pack_discrete_tokens(states, config)
        rebuilt = _unpack_discrete_onehot(tokens, config)

        assert tokens.shape == (states.shape[0], _num_factors(config))
        assert rebuilt.shape == states.shape
        np.testing.assert_allclose(np.asarray(rebuilt), np.asarray(states))
    finally:
        adapter.close()


def test_unpack_discrete_onehot_produces_valid_onehot_grids():
    # Arbitrary tokens (not from the env) must still yield exactly one active cell
    # per channel in the strided layout.
    config = _coins_config()
    tokens = jnp.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=jnp.int32,
    )
    grids = _unpack_discrete_onehot(tokens, config)
    per_channel = np.asarray(grids).reshape((2, config.num_agents, 9, 4))
    np.testing.assert_array_equal(per_channel.sum(axis=2), np.ones((2, 2, 4)))
    np.testing.assert_array_equal(
        _pack_discrete_tokens(grids, config), np.asarray(tokens)
    )


def _toy_discrete_world_config() -> VectorWorldModelConfig:
    # state_dim = V*C = 2*2, so transition_dim = num_agents*state_dim = 8 = d*V.
    return VectorWorldModelConfig(
        state_dim=4,
        num_agents=2,
        action_dim=3,
        hidden_dims=(16,),
        integration_steps=2,
        flow_type="discrete",
        num_categories=2,
    )


def test_discrete_predict_next_returns_valid_onehot_grids():
    config = _toy_discrete_world_config()
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    states = jax.random.normal(jax.random.PRNGKey(2), (5, config.num_agents, 4))
    actions = jnp.zeros((5, config.num_agents), dtype=jnp.int32)

    next_states = predict_next(
        model_state, jax.random.PRNGKey(3), states, actions, config
    )

    assert next_states.shape == (5, config.num_agents, 4)
    channels = config.state_dim // config.num_categories
    per_channel = np.asarray(next_states).reshape(
        (5, config.num_agents, config.num_categories, channels)
    )
    np.testing.assert_array_equal(
        per_channel.sum(axis=2), np.ones((5, config.num_agents, channels))
    )


def test_discrete_fit_world_model_steps_matches_python_loop():
    # Mirrors the continuous reference pin: the jitted lax.scan fit must reproduce
    # an independent Python loop's per-step loss, final rng, and params exactly, so
    # a key-threading break in the discrete path cannot pass silently.
    config = _toy_discrete_world_config()
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    n, steps = 6, 7
    states = jax.random.normal(jax.random.PRNGKey(3), (n, config.num_agents, 4))
    batch = VectorTransitionBatch(
        states=states,
        actions=jnp.ones((n, config.num_agents), dtype=jnp.int32),
        next_states=states,  # cond encodes the target -> a learnable, moving loss
        rewards=jnp.zeros((n, config.num_agents), dtype=jnp.float32),
        dones=jnp.zeros((n, config.num_agents), dtype=jnp.float32),
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
        np.testing.assert_allclose(np.asarray(new_leaf), np.asarray(ref_leaf), atol=1e-5)

    assert not bool(jnp.allclose(new_history[0], new_history[-1]))


def test_discrete_imagined_rollout_is_valid_onehot_and_deterministic():
    # Discrete predict_next must slot into the jitted lax.scan rollout, feed valid
    # one-hot states back as conditioning, and stay deterministic under a fixed rng
    # (a stray host-sync with integration_steps>1 would raise here instead).
    config = _toy_discrete_world_config()
    model_state = create_world_model_state(jax.random.PRNGKey(0), config)
    policy_state = create_ippo_state(
        jax.random.PRNGKey(1),
        (4,),
        3,
        IPPOConfig(network_arch="mlp", num_minibatches=1),
    )
    initial_states = _unpack_discrete_onehot(
        jnp.zeros((3, _num_factors(config)), dtype=jnp.int32), config
    )

    def reward_done_fn(states, actions, next_states):
        del states, next_states
        return actions.astype(jnp.float32), jnp.zeros_like(actions, dtype=jnp.float32)

    def _run(seed):
        return simulate_ippo_model_rollout(
            model_state,
            policy_state,
            initial_states,
            jax.random.PRNGKey(seed),
            rollout_steps=4,
            config=config,
            reward_done_fn=reward_done_fn,
        )

    first, second = _run(2), _run(2)
    np.testing.assert_array_equal(
        np.asarray(first.batch.observations), np.asarray(second.batch.observations)
    )

    obs = np.asarray(first.batch.observations)  # (T, num_actors, state_dim)
    channels = config.state_dim // config.num_categories
    grids = obs.reshape((obs.shape[0], obs.shape[1], config.num_categories, channels))
    np.testing.assert_array_equal(grids.sum(axis=2), np.ones_like(grids.sum(axis=2)))

    assert not bool(
        jnp.allclose(first.next_observations, initial_states)
    )  # states must evolve
