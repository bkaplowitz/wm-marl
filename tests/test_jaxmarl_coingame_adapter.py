from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from world_marl.envs.jaxmarl_coin_adapter import (
    JaxMARLCoinGameVectorAdapter,
    coin_game_reward_done,
)

_STAY = 4  # MOVES[4] == [0, 0]; post-move cell equals the player's cell


def _coin_state(red, blue, red_coin, blue_coin) -> jnp.ndarray:
    """Build a ``(1, 2, 36)`` CoinGame state from four ``(row, col)`` positions.

    Only agent 0's grid is populated; ``coin_game_reward_done`` reads
    ``states[:, 0]`` and ignores agent 1's view, so it is left zeroed.
    Channel order mirrors ``_abs_position``: [red_player, blue_player,
    red_coin, blue_coin].
    """
    grid = np.zeros((3, 3, 4), dtype=np.float32)
    grid[red[0], red[1], 0] = 1.0
    grid[blue[0], blue[1], 1] = 1.0
    grid[red_coin[0], red_coin[1], 2] = 1.0
    grid[blue_coin[0], blue_coin[1], 3] = 1.0
    state = np.zeros((1, 2, 36), dtype=np.float32)
    state[0, 0] = grid.reshape(-1)
    return jnp.asarray(state)


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

        np.testing.assert_array_equal(
            np.asarray(adapter._keys[0]), np.asarray(next_key)
        )
        np.testing.assert_allclose(
            step.rewards,
            np.asarray(
                [[expected_rewards["0"], expected_rewards["1"]]], dtype=np.float32
            ),
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


def test_coin_game_reward_done_payoff_branches_are_exact():
    # Both players "stay" so the post-move cell is the player's own cell; each
    # scenario triggers exactly one pickup branch with all other entities on
    # distinct cells. Expected rewards read straight off [[1, 1, -2], [1, 1, -2]].
    actions = jnp.asarray([[_STAY, _STAY]], dtype=jnp.int32)
    cases = {
        # red picks up red coin -> red +1, blue 0
        "red_takes_red": (
            _coin_state(red=(0, 0), blue=(1, 1), red_coin=(0, 0), blue_coin=(2, 2)),
            [1.0, 0.0],
        ),
        # red picks up blue coin -> red +1 (rb), blue -2 (penalty)
        "red_takes_blue": (
            _coin_state(red=(0, 0), blue=(1, 1), red_coin=(2, 2), blue_coin=(0, 0)),
            [1.0, -2.0],
        ),
        # blue picks up red coin -> red -2 (penalty), blue +1 (br)
        "blue_takes_red": (
            _coin_state(red=(1, 1), blue=(0, 0), red_coin=(0, 0), blue_coin=(2, 2)),
            [-2.0, 1.0],
        ),
        # blue picks up blue coin -> red 0, blue +1
        "blue_takes_blue": (
            _coin_state(red=(1, 1), blue=(0, 0), red_coin=(2, 2), blue_coin=(0, 0)),
            [0.0, 1.0],
        ),
        # no overlap -> no reward
        "neither": (
            _coin_state(red=(0, 0), blue=(1, 1), red_coin=(2, 2), blue_coin=(0, 2)),
            [0.0, 0.0],
        ),
    }
    for name, (state, expected) in cases.items():
        rewards, dones = coin_game_reward_done(state, actions, state)
        np.testing.assert_allclose(
            np.asarray(rewards), np.asarray([expected]), err_msg=name
        )
        np.testing.assert_array_equal(np.asarray(dones), np.zeros((1, 2)))


def test_coin_game_reward_done_applies_move_before_matching():
    # Red is NOT on the coin initially; action 0 (right -> [0, 1]) moves it onto
    # the coin. A fn that ignored env_actions would score this 0 and fail.
    state = _coin_state(red=(0, 0), blue=(1, 1), red_coin=(0, 1), blue_coin=(2, 2))
    actions = jnp.asarray([[0, _STAY]], dtype=jnp.int32)  # red moves right, blue stays
    rewards, _ = coin_game_reward_done(state, actions, state)
    np.testing.assert_allclose(np.asarray(rewards), np.asarray([[1.0, 0.0]]))


def test_coin_game_reward_done_matches_real_env_reward():
    # Validate the analytic oracle against CoinGame's emitted reward over real
    # transitions, away from the episode boundary (large max_cycles) where the
    # env zeros reward but the analytic fn does not. This pins channel order,
    # the agent-0 absolute frame, and the action-column convention.
    adapter = JaxMARLCoinGameVectorAdapter(
        num_envs=1, max_cycles=100, seed=0, auto_reset=False
    )
    try:
        observations = adapter.reset()
        rng = np.random.default_rng(0)
        nonzero_seen = False
        for _ in range(50):
            actions = adapter.sample_actions(rng)
            step = adapter.step(actions)
            rewards, dones = coin_game_reward_done(
                jnp.asarray(observations),
                jnp.asarray(actions),
                jnp.asarray(step.observations),
            )
            np.testing.assert_allclose(
                np.asarray(rewards), step.rewards, err_msg="reward mismatch"
            )
            np.testing.assert_array_equal(np.asarray(dones), step.dones)
            nonzero_seen = nonzero_seen or bool(np.any(step.rewards != 0.0))
            observations = step.observations
        assert nonzero_seen, "test never exercised a nonzero reward"
    finally:
        adapter.close()
