from __future__ import annotations

import jax
import numpy as np

from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter


def test_jaxmarl_coin_adapter_reset_step_shapes_and_random_actions():
    adapter = JaxMARLCoinGameVectorAdapter(num_envs=3, max_cycles=5, seed=0)
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(1))
        step = adapter.step(actions)

        assert adapter.num_agents == 2
        assert adapter.action_dim == 5
        assert adapter.observation_shape == (36,)
        assert observations.shape == (3, 2, 36)
        assert actions.shape == (3, 2)
        assert np.all(actions >= 0)
        assert np.all(actions < 5)
        assert step.observations.shape == (3, 2, 36)
        assert step.rewards.shape == (3, 2)
        assert step.dones.shape == (3, 2)
    finally:
        adapter.close()


def test_jaxmarl_coin_adapter_rewards_and_dones_match_direct_env_step():
    adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=1,
        max_cycles=5,
        seed=0,
        auto_reset=False,
    )
    try:
        adapter.reset()
        actions = np.asarray([[1, 4]], dtype=np.int32)
        pre_state = jax.tree.map(lambda leaf: leaf[0], adapter._state)
        pre_key = adapter._keys[0]
        next_key, step_key = jax.random.split(pre_key)
        _, _, expected_rewards, expected_dones, _ = adapter.env.step(
            step_key,
            pre_state,
            {"0": actions[0, 0], "1": actions[0, 1]},
        )

        step = adapter.step(actions)

        np.testing.assert_array_equal(np.asarray(adapter._keys[0]), np.asarray(next_key))
        np.testing.assert_allclose(
            step.rewards,
            np.asarray([[expected_rewards["0"], expected_rewards["1"]]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            step.dones,
            np.asarray([[expected_dones["0"], expected_dones["1"]]], dtype=np.float32),
        )
    finally:
        adapter.close()


def test_jaxmarl_coin_adapter_completes_episode_and_trusts_internal_reset():
    max_cycles = 4
    adapter = JaxMARLCoinGameVectorAdapter(num_envs=2, max_cycles=max_cycles, seed=0)
    try:
        adapter.reset()
        rng = np.random.default_rng(0)
        for _ in range(max_cycles - 1):
            interior = adapter.step(adapter.sample_actions(rng))
            assert interior.completed_returns == ()
            assert interior.completed_lengths == ()

        boundary_actions = adapter.sample_actions(rng)
        pre_state = adapter._state
        pre_keys = adapter._keys
        step_keys = jax.vmap(jax.random.split)(pre_keys)[:, 1]
        action_dict = {
            agent: np.asarray(boundary_actions[:, agent_index], dtype=np.int32)
            for agent_index, agent in enumerate(adapter.agents)
        }
        ref_obs, _, _, ref_done, _ = jax.vmap(adapter.env.step)(
            step_keys, pre_state, action_dict
        )

        step = adapter.step(boundary_actions)

        assert np.all(np.asarray(ref_done["__all__"]))
        assert np.all(step.dones == 1.0)
        assert len(step.completed_returns) == adapter.num_envs
        assert step.completed_lengths == (max_cycles,) * adapter.num_envs

        expected_obs = np.stack(
            [np.asarray(ref_obs[a], dtype=np.float32) for a in adapter.agents], axis=1
        )
        np.testing.assert_allclose(step.observations, expected_obs)
    finally:
        adapter.close()
