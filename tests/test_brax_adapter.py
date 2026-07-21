from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from world_marl.envs.brax_adapter import BraxVectorAdapter, brax_env_name
from world_marl.world_model_foundation.collect import collect_adapter_sequence


@struct.dataclass
class _FakeBraxState:
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    count: jax.Array


class _FakeBraxEnv:
    action_size = 2

    def reset(self, rng):
        seed_value = jax.random.randint(rng, (), minval=0, maxval=1000).astype(
            jnp.float32
        )
        return _FakeBraxState(
            obs=jnp.stack([seed_value, jnp.asarray(0.0), jnp.asarray(0.0)]),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            done=jnp.asarray(0.0, dtype=jnp.float32),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def step(self, state, action):
        count = state.count + 1
        reward = jnp.sum(action)
        done = (count >= 2).astype(jnp.float32)
        return state.replace(
            obs=jnp.stack([state.obs[0], count.astype(jnp.float32), reward]),
            reward=reward,
            done=done,
            count=count,
        )


class _NoDoneFakeBraxEnv(_FakeBraxEnv):
    """Never signals done itself, so only the adapter's max_cycles truncation
    can end an episode."""

    def step(self, state, action):
        stepped = super().step(state, action)
        return stepped.replace(done=jnp.zeros_like(stepped.done))


def _constant_gaav(train_state, action_key, obs_flat):
    del train_state, action_key
    num_rows = obs_flat.shape[0]
    actions = jnp.full((num_rows, 2), 0.25, dtype=jnp.float32)
    zeros = jnp.zeros((num_rows,), dtype=jnp.float32)
    return actions, zeros, zeros, zeros


def _recurrent_constant_policy(policy_state, carry, obs_flat, is_first):
    del policy_state, obs_flat
    carry = jnp.where(is_first[:, None], 0, carry) + 1
    actions = jnp.full((carry.shape[0], 2), 0.25, dtype=jnp.float32)
    return carry, actions


def test_brax_env_name_parses_name():
    assert brax_env_name("brax:reacher") == "reacher"
    with pytest.raises(ValueError, match="brax:<env_name>"):
        brax_env_name("brax:")


def test_brax_adapter_reset_step_and_completion():
    adapter = BraxVectorAdapter(
        "fake",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=_FakeBraxEnv,
        backend="mjx",
    )
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(0))
        first = adapter.step(actions)
        second = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))

        assert adapter.num_agents == 1
        assert adapter.action_dim == 2
        assert adapter.observation_shape == (3,)
        assert adapter.environment_metadata == {
            "environment_backend": "brax",
            "physics_backend": "mjx",
            "observation_mode": "vector",
        }
        assert observations.shape == (2, 1, 3)
        assert actions.shape == (2, 1, 2)
        assert first.observations.shape == (2, 1, 3)
        assert first.rewards.shape == (2, 1)
        assert second.dones.tolist() == [[1.0], [1.0]]
        assert len(second.completed_returns) == 2
        assert second.completed_lengths == (2, 2)
    finally:
        adapter.close()


def test_brax_scan_rollout_matches_host_loop_and_resets_in_scan():
    """``scan_rollout`` vs the host ``step`` loop under a constant policy. The
    fake env's dynamics channels (count, reward) are reset-key independent, so
    rewards, dones, and ``obs[..., 1:]`` must match the host exactly for all
    steps, and the full obs must match up to the first in-scan reset (reset
    keys are distribution-equivalent by design, so the seeded obs channel
    diverges after it). done=(count>=2) yields the pattern [0,1,0,1,0], which
    proves the in-scan auto-reset actually fires and restarts episodes.
    """
    num_steps = 5

    def make_adapter():
        return BraxVectorAdapter(
            "fake", num_envs=2, max_cycles=5, seed=10, env_factory=_FakeBraxEnv
        )

    scan_adapter = make_adapter()
    host_adapter = make_adapter()
    try:
        obs0_scan = scan_adapter.reset()
        obs0_host = host_adapter.reset()
        np.testing.assert_array_equal(obs0_scan, obs0_host)

        counter_before = scan_adapter._reset_counter
        ys, last_obs_flat = scan_adapter.scan_rollout(
            _constant_gaav,
            None,
            num_steps,
            policy_key=jax.random.PRNGKey(0),
            observations=obs0_scan,
        )
        obs_seq, action_seq, _log_probs, _values, _entropies, reward_seq, done_seq = ys

        host_obs = [obs0_host.reshape((2, -1))]
        host_rewards = []
        host_dones = []
        for _ in range(num_steps):
            step = host_adapter.step(np.full((2, 1, 2), 0.25, dtype=np.float32))
            host_rewards.append(step.rewards.reshape((2,)))
            host_dones.append(step.dones.reshape((2,)))
            host_obs.append(step.observations.reshape((2, -1)))

        assert np.asarray(action_seq).shape == (num_steps, 2, 2)
        np.testing.assert_allclose(np.asarray(action_seq), 0.25)
        np.testing.assert_array_equal(np.asarray(reward_seq), np.stack(host_rewards))
        np.testing.assert_array_equal(
            np.asarray(done_seq, dtype=np.float32), np.stack(host_dones)
        )
        assert np.asarray(done_seq, dtype=np.float32).T.tolist() == [
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
        ]
        np.testing.assert_array_equal(np.asarray(obs_seq)[:2], np.stack(host_obs[:2]))
        np.testing.assert_array_equal(
            np.asarray(obs_seq)[..., 1:], np.stack(host_obs[:-1])[..., 1:]
        )
        np.testing.assert_array_equal(
            np.asarray(last_obs_flat)[..., 1:], host_obs[-1][..., 1:]
        )

        # The scan advanced the env carry but not the episode accumulators.
        assert scan_adapter._reset_counter == counter_before + 1
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(scan_adapter._state.count)), [1, 1]
        )
        assert scan_adapter._episode_lengths.tolist() == [0, 0]
        assert scan_adapter._episode_returns.tolist() == [[0.0], [0.0]]
    finally:
        scan_adapter.close()
        host_adapter.close()


def test_brax_scan_rollout_truncates_at_max_cycles():
    """With an env that never signals done, only the in-scan truncation timer
    (seeded from ``_episode_lengths``) can end episodes: max_cycles=3 over 7
    steps must produce dones exactly at steps 3 and 6, with the timer zeroed
    by the in-scan reset in between.
    """
    adapter = BraxVectorAdapter(
        "fake", num_envs=2, max_cycles=3, seed=10, env_factory=_NoDoneFakeBraxEnv
    )
    try:
        observations = adapter.reset()
        ys, _last_obs_flat = adapter.scan_rollout(
            _constant_gaav,
            None,
            7,
            policy_key=jax.random.PRNGKey(0),
            observations=observations,
        )
        _obs, _actions, _log_probs, _values, _entropies, _rewards, done_seq = ys

        expected = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
        assert np.asarray(done_seq, dtype=np.float32).T.tolist() == [
            expected,
            expected,
        ]
    finally:
        adapter.close()


def test_brax_recurrent_scan_and_random_collection_stay_on_device():
    adapter = BraxVectorAdapter(
        "fake", num_envs=2, max_cycles=3, seed=10, env_factory=_NoDoneFakeBraxEnv
    )
    try:
        observations = adapter.reset()
        ys, _last_obs, final_policy_carry = adapter.scan_recurrent_rollout(
            _recurrent_constant_policy,
            None,
            jnp.zeros((2, 1), dtype=jnp.int32),
            4,
            observations=observations,
        )
        scanned_obs, actions, rewards, terminals, lasts = ys

        assert np.asarray(actions).shape == (4, 2, 2)
        assert np.asarray(rewards).shape == (4, 2)
        assert np.asarray(terminals, dtype=np.float32).T.tolist() == [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        assert np.asarray(lasts, dtype=np.float32).T.tolist() == [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        assert np.asarray(final_policy_carry).tolist() == [[4], [4]]
        assert np.asarray(rewards)[0].tolist() == [0.0, 0.0]
        assert bool(
            np.allclose(np.asarray(scanned_obs)[1:, :, -1], np.asarray(rewards)[1:])
        )
        assert bool(np.all(np.asarray(actions)[-1] == 0.0))

        batch = collect_adapter_sequence(adapter, time_steps=4, seed=4)
        assert batch.metadata["collection_execution"] == "jax_scan"
        assert bool(np.all(batch.is_first[0]))
        assert not bool(np.any(batch.is_terminal[3]))
        assert bool(np.all(batch.is_last[3]))
        assert bool(np.all(batch.continues[3] == 1.0))
    finally:
        adapter.close()


def test_brax_online_scan_passes_arrivals_through_learner_carry():
    adapter = BraxVectorAdapter(
        "fake", num_envs=2, max_cycles=3, seed=12, env_factory=_NoDoneFakeBraxEnv
    )
    try:
        observations = adapter.reset()

        def learner_step(carry, obs, reward, is_terminal, is_last, is_first):
            del obs, is_terminal
            count = carry + 1
            actions = jnp.ones((2, 2), dtype=jnp.float32)
            metrics = {
                "count": count,
                "reward_sum": reward.sum(),
                "first_count": is_first.sum(),
                "last_count": is_last.sum(),
            }
            return count, actions, metrics

        ys, _last_obs, final_carry = adapter.scan_online_rollout(
            learner_step,
            jnp.zeros((), dtype=jnp.int32),
            4,
            observations=observations,
        )
        obs, actions, rewards, terminals, lasts, firsts, metrics = ys

        assert obs.shape == (4, 2, int(np.prod(adapter.observation_shape)))
        assert actions.shape == (4, 2, 2)
        assert rewards.shape == terminals.shape == lasts.shape == firsts.shape
        assert int(final_carry) == 4
        assert np.asarray(metrics["count"]).tolist() == [1, 2, 3, 4]
        assert np.asarray(metrics["first_count"]).tolist() == [2, 0, 0, 0]
        assert np.asarray(metrics["last_count"]).tolist() == [0, 0, 0, 2]
    finally:
        adapter.close()


def test_default_factory_forwards_episode_length_to_brax():
    pytest.importorskip("brax")

    adapter = BraxVectorAdapter("fast", num_envs=1, max_cycles=7)
    try:
        assert adapter._env.episode_length == 7
    finally:
        adapter.close()


def test_brax_adapter_can_reset_selected_vector_members():
    adapter = BraxVectorAdapter(
        "fake",
        num_envs=2,
        max_cycles=5,
        seed=10,
        env_factory=_FakeBraxEnv,
    )
    try:
        adapter.reset()
        adapter.step(np.zeros((2, 1, 2), dtype=np.float32))
        reset_observations = adapter.reset_indices(np.asarray([0]))
        following = adapter.step(np.zeros((2, 1, 2), dtype=np.float32))
    finally:
        adapter.close()

    assert reset_observations.shape == (1, 1, 3)
    assert reset_observations[0, 0, 1] == 0.0
    assert following.dones.tolist() == [[0.0], [1.0]]
