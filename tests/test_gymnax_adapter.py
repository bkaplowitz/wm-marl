from __future__ import annotations

from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.envs.gymnax_adapter import GymnaxVectorAdapter
from world_marl.evaluation import constant_policy, evaluate_policy
from world_marl.scripts.train_e2e import (
    _make_training_adapter,
    algorithm_config_from_args,
    create_algorithm_train_state,
)
from world_marl.training import collect_mappo_rollout, collect_rollout
from world_marl.world_model_foundation.collect import collect_adapter_sequence


def _args(algorithm: str = "ippo") -> Namespace:
    return Namespace(
        algorithm=algorithm,
        substrate="gymnax:CartPole-v1",
        num_envs=2,
        max_cycles=2,
        observation_size=None,
        include_observation_scalars=False,
        append_agent_id=False,
        prefit_world_model=False,
        learning_rate=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=1,
        num_minibatches=1,
        activation="relu",
    )


def test_gymnax_adapter_reset_step_and_episode_completion():
    adapter = GymnaxVectorAdapter(
        "CartPole-v1",
        num_envs=2,
        max_cycles=1,
        seed=0,
    )
    try:
        observations = adapter.reset()
        actions = adapter.sample_actions(np.random.default_rng(0))
        step = adapter.step(actions)

        assert adapter.num_agents == 1
        assert adapter.action_dim == 2
        assert adapter.observation_shape == (4,)
        assert observations.shape == (2, 1, 4)
        assert actions.shape == (2, 1)
        assert step.observations.shape == (2, 1, 4)
        assert step.rewards.shape == (2, 1)
        assert step.dones.shape == (2, 1)
        assert len(step.completed_returns) == 2
        assert step.completed_lengths == (1, 1)
    finally:
        adapter.close()


def test_gymnax_adapter_evaluates_through_common_loop():
    adapter = GymnaxVectorAdapter(
        "CartPole-v1",
        num_envs=2,
        max_cycles=1,
        seed=0,
    )
    try:
        result = evaluate_policy(
            adapter,
            constant_policy(0),
            episodes=4,
        )

        assert result.returns.shape == (4, 1)
        assert result.lengths.tolist() == [1, 1, 1, 1]
        assert result.steps == 4
    finally:
        adapter.close()


def test_gymnax_scan_rollout_matches_host_step_loop():
    """The public ``scan_rollout`` wrapper must reproduce the host ``step``
    loop bit-for-bit given the same actions: both consume the same per-env key
    stream in the same order. A key-independent constant policy removes the
    action-sampling variable, and ``num_steps > max_cycles`` crosses gymnax's
    internal auto-reset inside the scan. The final ``_keys`` must match the
    host adapter's exactly, while the episode accumulators stay untouched
    (callers replay them from the recorded dones).
    """
    num_envs, max_cycles, num_steps = 2, 3, 5

    def make_adapter():
        return GymnaxVectorAdapter(
            "CartPole-v1", num_envs=num_envs, max_cycles=max_cycles, seed=7
        )

    scan_adapter = make_adapter()
    host_adapter = make_adapter()
    try:
        obs0_scan = scan_adapter.reset()
        obs0_host = host_adapter.reset()
        np.testing.assert_array_equal(obs0_scan, obs0_host)

        def zero_action(train_state, action_key, obs_flat):
            del train_state, action_key
            num_rows = obs_flat.shape[0]
            zeros = jnp.zeros((num_rows,), dtype=jnp.float32)
            return jnp.zeros((num_rows,), dtype=jnp.int32), zeros, zeros, zeros

        ys, last_obs_flat = scan_adapter.scan_rollout(
            zero_action,
            None,
            num_steps,
            policy_key=jax.random.PRNGKey(0),
            observations=obs0_scan,
        )
        obs_seq, action_seq, _log_probs, _values, _entropies, reward_seq, done_seq = ys

        host_obs = [obs0_host.reshape((num_envs, -1))]
        host_rewards = []
        host_dones = []
        for _ in range(num_steps):
            step = host_adapter.step(np.zeros((num_envs, 1), dtype=np.int32))
            host_rewards.append(step.rewards.reshape((num_envs,)))
            host_dones.append(step.dones.reshape((num_envs,)))
            host_obs.append(step.observations.reshape((num_envs, -1)))

        assert np.asarray(action_seq).shape == (num_steps, num_envs)
        np.testing.assert_allclose(
            np.asarray(obs_seq), np.stack(host_obs[:-1]), rtol=0, atol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(reward_seq), np.stack(host_rewards))
        np.testing.assert_array_equal(
            np.asarray(done_seq, dtype=np.float32), np.stack(host_dones)
        )
        assert np.asarray(done_seq).any()  # max_cycles=3 < num_steps -> a reset ran
        np.testing.assert_allclose(
            np.asarray(last_obs_flat), host_obs[-1], rtol=0, atol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(scan_adapter._keys), np.asarray(host_adapter._keys)
        )
        assert scan_adapter._episode_lengths.tolist() == [0, 0]
        assert scan_adapter._episode_returns.tolist() == [[0.0], [0.0]]
    finally:
        scan_adapter.close()
        host_adapter.close()


def test_gymnax_recurrent_scan_and_random_collection_stay_on_device():
    adapter = GymnaxVectorAdapter("CartPole-v1", num_envs=2, max_cycles=3, seed=11)
    try:
        observations = adapter.reset()

        def policy_step(policy_state, carry, obs_flat, is_first):
            del policy_state, obs_flat, is_first
            actions = jnp.zeros((2,), dtype=jnp.int32)
            return carry + 1, actions

        ys, _last_obs, final_policy_carry = adapter.scan_recurrent_rollout(
            policy_step,
            None,
            jnp.zeros((), dtype=jnp.int32),
            4,
            observations=observations,
        )
        _obs, actions, rewards, terminals, lasts = ys
        assert actions.shape == (4, 2)
        assert rewards.shape == (4, 2)
        assert terminals.shape == (4, 2)
        assert lasts.shape == (4, 2)
        assert np.asarray(rewards, dtype=np.float32).T.tolist() == [
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]
        assert np.asarray(terminals, dtype=bool).T.tolist() == [
            [False, False, False, True],
            [False, False, False, True],
        ]
        assert np.array_equal(np.asarray(terminals), np.asarray(lasts))
        assert bool(np.all(np.asarray(actions)[-1] == 0))
        assert int(final_policy_carry) == 4

        batch = collect_adapter_sequence(adapter, time_steps=4, seed=12)
        assert batch.metadata["collection_execution"] == "jax_scan"
        assert batch.actions.shape == (4, 2)
    finally:
        adapter.close()


def test_gymnax_online_scan_passes_arrivals_through_learner_carry():
    adapter = GymnaxVectorAdapter("CartPole-v1", num_envs=2, max_cycles=3, seed=13)
    try:
        observations = adapter.reset()

        def learner_step(carry, obs, reward, is_terminal, is_last, is_first):
            del obs, is_terminal
            count = carry + 1
            actions = jnp.zeros((2,), dtype=jnp.int32)
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
        assert actions.shape == (4, 2)
        assert rewards.shape == terminals.shape == lasts.shape == firsts.shape
        assert int(final_carry) == 4
        assert np.asarray(metrics["count"]).tolist() == [1, 2, 3, 4]
        assert np.asarray(metrics["first_count"]).tolist() == [2, 0, 0, 0]
        assert np.asarray(metrics["last_count"]).tolist() == [0, 0, 0, 2]
    finally:
        adapter.close()


@pytest.mark.parametrize("algorithm", ["ippo", "mappo"])
def test_gymnax_substrate_selects_vector_mlp_policy_and_rollout_path(algorithm):
    args = _args(algorithm)
    adapter = _make_training_adapter(args, seed=0)
    try:
        assert isinstance(adapter, GymnaxVectorAdapter)
        assert algorithm_config_from_args(args).network_arch == "mlp"

        config = algorithm_config_from_args(args)
        train_state = create_algorithm_train_state(
            args.algorithm,
            jax.random.PRNGKey(0),
            adapter,
            config,
            observation_mode="vector",
        )
        if algorithm == "mappo":
            rollout = collect_mappo_rollout(
                adapter,
                train_state,
                adapter.reset(),
                jax.random.PRNGKey(1),
                rollout_steps=2,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                observation_mode="vector",
            )
            assert rollout.batch.central_observations.shape == (2, 2, 5)
        else:
            rollout = collect_rollout(
                adapter,
                train_state,
                adapter.reset(),
                jax.random.PRNGKey(1),
                rollout_steps=2,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

        assert rollout.batch.observations.shape == (2, 2, 4)
        assert rollout.batch.actions.shape == (2, 2)
        assert rollout.next_observations.shape == (2, 1, 4)
    finally:
        adapter.close()
