from __future__ import annotations

from argparse import Namespace

import jax
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
