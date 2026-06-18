from __future__ import annotations

from argparse import Namespace

import jax

from world_marl.envs.meltingpot_adapter import MeltingPotVectorAdapter
from world_marl.envs.jaxmarl_coin_adapter import JaxMARLCoinGameVectorAdapter
from world_marl.scripts.train_e2e import (
    _make_training_adapter,
    algorithm_config_from_args,
    create_algorithm_train_state,
    policy_from_train_state,
)


def _args(*, algorithm: str) -> Namespace:
    return Namespace(
        algorithm=algorithm,
        substrate="coins",
        num_envs=1,
        max_cycles=5,
        observation_size=None,
        include_observation_scalars=False,
        append_agent_id=False,
        prefit_world_model=True,
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


def test_prefit_coins_uses_jaxmarl_vector_adapter():
    adapter = _make_training_adapter(_args(algorithm="ippo"), seed=0)
    try:
        assert isinstance(adapter, JaxMARLCoinGameVectorAdapter)
        assert adapter.observation_shape == (36,)
        assert adapter.action_dim == 5
    finally:
        adapter.close()


def test_prefit_world_model_selects_mlp_policy_config():
    assert algorithm_config_from_args(_args(algorithm="ippo")).network_arch == "mlp"
    assert algorithm_config_from_args(_args(algorithm="mappo")).network_arch == "mlp"


def test_coins_selects_mlp_policy_config_without_prefit():
    args = _args(algorithm="ippo")
    args.prefit_world_model = False

    assert algorithm_config_from_args(args).network_arch == "mlp"


def test_prefit_ippo_policy_uses_flat_vector_observations(dummy_env_factory):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        args = _args(algorithm="ippo")
        config = algorithm_config_from_args(args)
        state = create_algorithm_train_state(
            args.algorithm,
            jax.random.PRNGKey(0),
            adapter,
            config,
            observation_mode="vector",
        )
        policy = policy_from_train_state(
            args.algorithm,
            state,
            adapter=adapter,
            deterministic=True,
            seed=0,
            observation_mode="vector",
        )

        actions = policy(adapter.reset())

        assert actions.shape == (adapter.num_envs, adapter.num_agents)
    finally:
        adapter.close()


def test_prefit_mappo_policy_uses_flat_vector_observations(dummy_env_factory):
    adapter = MeltingPotVectorAdapter(num_envs=1, env_factory=dummy_env_factory)
    try:
        args = _args(algorithm="mappo")
        config = algorithm_config_from_args(args)
        state = create_algorithm_train_state(
            args.algorithm,
            jax.random.PRNGKey(0),
            adapter,
            config,
            observation_mode="vector",
        )
        policy = policy_from_train_state(
            args.algorithm,
            state,
            adapter=adapter,
            deterministic=True,
            seed=0,
            observation_mode="vector",
        )

        actions = policy(adapter.reset())

        assert actions.shape == (adapter.num_envs, adapter.num_agents)
    finally:
        adapter.close()
