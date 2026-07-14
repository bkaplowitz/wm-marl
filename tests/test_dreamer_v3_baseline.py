from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from flax.traverse_util import flatten_dict

from world_marl.dreamer_v3_baseline.config import DreamerV3Config
from world_marl.dreamer_v3_baseline.imagination import (
    imagination_weights,
    imagine_dreamer_rollout,
    train_dreamer_actor_critic,
)
from world_marl.dreamer_v3_baseline.losses import (
    categorical_kl_loss,
    symexp,
    symlog,
    two_hot,
)
from world_marl.dreamer_v3_baseline.models import (
    ContinueHead,
    DreamerActor,
    DreamerCritic,
    DreamerDecoder,
    DreamerEncoder,
    RewardHead,
)
from world_marl.dreamer_v3_baseline.rssm import (
    DreamerRSSM,
    categorical_straight_through,
    flatten_rssm_state,
    initial_rssm_state,
)
from world_marl.dreamer_v3_baseline.training import (
    create_dreamer_train_state,
    dreamer_train_step,
    observe_dreamer_sequence,
    scan_dreamer_world_model_updates,
)
from world_marl.scripts.train_dreamer_v3_baseline import main as train_dreamer_main
from world_marl.world_model_foundation.collect import synthetic_sequence_collector
from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    sequence_batch_to_jax,
)


def test_config_defaults_lock_categorical_rssm_contract() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(8, 8, 3))

    assert config.rssm.deterministic_size > 0
    assert config.rssm.stochastic_size > 0
    assert config.rssm.discrete_classes > 1
    assert config.rssm.latent_size == (
        config.rssm.deterministic_size
        + config.rssm.stochastic_size * config.rssm.discrete_classes
    )
    assert config.reward_head.distribution == "symlog_two_hot"
    assert config.continue_head.distribution == "bernoulli"


def test_config_defaults_match_dreamerv3_12m_and_table4() -> None:
    config = DreamerV3Config(action_dim=4, observation_shape=(64, 64, 3))

    assert config.rssm.deterministic_size == 2048
    assert config.rssm.hidden_size == 256
    assert config.rssm.stochastic_size == 32
    assert config.rssm.discrete_classes == 16
    assert config.rssm.blocks == 8
    assert config.encoder.cnn_depth == 16
    assert config.encoder.hidden_dims == (256, 256, 256)
    assert config.replay.capacity == 5_000_000
    assert config.replay.batch_size == 16
    assert config.replay.batch_length == 64
    assert config.optimizer.learning_rate == pytest.approx(4e-5)
    assert config.optimizer.agc == pytest.approx(0.3)
    assert config.optimizer.epsilon == pytest.approx(1e-20)
    assert config.optimizer.beta1 == pytest.approx(0.9)
    assert config.optimizer.beta2 == pytest.approx(0.99)
    assert config.kl_free_nats == pytest.approx(1.0)
    assert config.dynamics_kl_scale == pytest.approx(1.0)
    assert config.representation_kl_scale == pytest.approx(0.1)
    assert config.actor_critic.imagination_horizon == 15
    assert config.actor_critic.discount_horizon == 333
    assert config.actor_critic.discount_lambda == pytest.approx(0.95)
    assert config.actor_critic.critic_replay_scale == pytest.approx(0.3)
    assert config.actor_critic.entropy_scale == pytest.approx(3e-4)


def test_imagination_weights_mask_terminal_replay_starts() -> None:
    predicted_continues = jnp.asarray(
        [
            [0.9, 0.9, 0.9],
            [0.8, 0.8, 0.8],
            [0.7, 0.7, 0.7],
        ],
        dtype=jnp.float32,
    )
    start_continues = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)

    weights = imagination_weights(predicted_continues, start_continues)

    np.testing.assert_allclose(weights[:, 0], [1.0, 0.9, 0.72])
    np.testing.assert_allclose(weights[:, 1], 0.0)
    np.testing.assert_allclose(weights[:, 2], [1.0, 0.9, 0.72])


def test_categorical_straight_through_returns_one_hot_forward_values() -> None:
    logits = jnp.asarray([[[0.0, 1.0, -1.0], [2.0, 0.0, -2.0]]], dtype=jnp.float32)

    stoch, probs = categorical_straight_through(
        logits,
        jax.random.PRNGKey(42),
        unimix=0.01,
    )

    assert stoch.shape == logits.shape
    assert probs.shape == logits.shape
    assert bool(jnp.allclose(jnp.sum(stoch, axis=-1), 1.0))
    assert bool(jnp.allclose(jnp.sum(probs, axis=-1), 1.0))


def test_rssm_prior_and_posterior_shapes_and_finite_kl() -> None:
    config = DreamerV3Config.debug(action_dim=4, observation_shape=(5,))
    rssm = DreamerRSSM(config.rssm, action_dim=config.action_dim)
    prev_state = initial_rssm_state(batch_size=3, config=config.rssm)
    actions = jax.nn.one_hot(jnp.asarray([0, 1, 2]), config.action_dim)
    embed = jnp.ones((3, config.encoder.hidden_dims[-1]), dtype=jnp.float32)
    params = rssm.init(
        jax.random.PRNGKey(0),
        prev_state,
        actions,
        embed,
        jax.random.PRNGKey(1),
    )

    prior, posterior = rssm.apply(
        params,
        prev_state,
        actions,
        embed,
        jax.random.PRNGKey(2),
    )
    kl = categorical_kl_loss(posterior.logits, prior.logits, free_nats=0.0)

    assert prior.deterministic.shape == (3, config.rssm.deterministic_size)
    assert posterior.stochastic.shape == (
        3,
        config.rssm.stochastic_size,
        config.rssm.discrete_classes,
    )
    assert flatten_rssm_state(posterior).shape == (3, config.rssm.latent_size)
    assert bool(jnp.isfinite(kl))
    parameter_paths = {"/".join(path) for path in flatten_dict(params["params"]).keys()}
    assert any("block_gru_hidden" in path for path in parameter_paths)


def test_encoder_decoder_reward_continue_heads_match_world_model_shapes() -> None:
    config = DreamerV3Config.debug(action_dim=4, observation_shape=(16, 16, 3))
    observations = jnp.ones((2, *config.observation_shape), dtype=jnp.float32)
    features = jnp.ones((2, config.rssm.latent_size), dtype=jnp.float32)

    encoder = DreamerEncoder(
        config.observation_shape,
        hidden_dims=config.encoder.hidden_dims,
        cnn_depth=config.encoder.cnn_depth,
    )
    encoder_params = encoder.init(jax.random.PRNGKey(1), observations)
    embeddings = encoder.apply(encoder_params, observations)

    decoder = DreamerDecoder(
        config.observation_shape,
        hidden_dims=config.encoder.hidden_dims,
        cnn_depth=config.encoder.cnn_depth,
        deterministic_size=config.rssm.deterministic_size,
        stochastic_size=config.rssm.stochastic_size,
        discrete_classes=config.rssm.discrete_classes,
        blocks=config.rssm.blocks,
        hidden_size=config.rssm.hidden_size,
    )
    decoder_params = decoder.init(jax.random.PRNGKey(2), features)
    reconstructions = decoder.apply(decoder_params, features)

    reward_head = RewardHead(config.reward_head.bins)
    reward_params = reward_head.init(jax.random.PRNGKey(3), features)
    reward_logits = reward_head.apply(reward_params, features)

    continue_head = ContinueHead()
    continue_params = continue_head.init(jax.random.PRNGKey(4), features)
    continue_logits = continue_head.apply(continue_params, features)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0
    assert reconstructions.shape == observations.shape
    assert reward_logits.shape == (2, config.reward_head.bins)
    assert continue_logits.shape == (2,)
    assert bool(jnp.all((reconstructions >= 0.0) & (reconstructions <= 1.0)))


def test_decoder_vector_observations_are_unbounded() -> None:
    decoder = DreamerDecoder((5,), hidden_dims=())
    features = jnp.zeros((2, 4), dtype=jnp.float32)
    params = decoder.init(jax.random.PRNGKey(41), features)
    mutable = unfreeze(params)
    mutable["params"]["vector_output"]["kernel"] = jnp.zeros_like(
        mutable["params"]["vector_output"]["kernel"]
    )
    mutable["params"]["vector_output"]["bias"] = jnp.full_like(
        mutable["params"]["vector_output"]["bias"], -2.0
    )

    reconstructions = decoder.apply(freeze(mutable), features)

    assert bool(jnp.allclose(reconstructions, -2.0))


def test_symlog_symexp_and_two_hot_reward_targets() -> None:
    values = jnp.asarray([-2.0, 0.0, 3.0], dtype=jnp.float32)
    encoded = symlog(values)
    decoded = symexp(encoded)
    targets = two_hot(encoded, num_bins=9, lower=-4.0, upper=4.0)

    assert bool(jnp.allclose(decoded, values, atol=1e-5))
    assert targets.shape == (3, 9)
    assert bool(jnp.allclose(jnp.sum(targets, axis=-1), 1.0))


def test_world_model_train_step_updates_params_and_returns_finite_metrics() -> None:
    config = DreamerV3Config.debug(
        action_dim=3,
        observation_shape=(16, 16, 3),
    )
    batch = synthetic_sequence_collector(
        env_name="synthetic:image-grid",
        time_steps=4,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=config.action_dim,
    )
    state = create_dreamer_train_state(
        jax.random.PRNGKey(5), config, learning_rate=1e-3
    )

    updated, metrics = dreamer_train_step(
        state,
        batch,
        config,
        jax.random.PRNGKey(6),
    )

    assert updated.step == state.step + 1
    for key in (
        "loss",
        "reconstruction_loss",
        "reward_loss",
        "continue_loss",
        "kl_loss",
    ):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key]))


def test_dreamer_recurrence_and_optimizer_updates_lower_to_scan() -> None:
    config = DreamerV3Config.debug(action_dim=2, observation_shape=(3,))
    batch = synthetic_sequence_collector(
        env_name="synthetic:scan",
        time_steps=3,
        batch_size=1,
        observation_shape=config.observation_shape,
        action_dim=config.action_dim,
    )
    replay = sequence_batch_to_jax(batch)
    state = create_dreamer_train_state(
        jax.random.PRNGKey(51), config, learning_rate=1e-3
    )

    recurrent_jaxpr = jax.make_jaxpr(
        lambda params, observations, actions, is_first, key: observe_dreamer_sequence(
            params,
            observations,
            actions,
            is_first,
            config,
            key,
        )
    )(
        state.params,
        replay.observations,
        replay.actions,
        replay.is_first,
        jax.random.PRNGKey(53),
    )
    module_jaxpr = jax.make_jaxpr(
        lambda params, observations, actions, is_first, key: state.apply_fn(
            params,
            observations,
            actions,
            is_first,
            key,
        )
    )(
        state.params,
        replay.observations,
        replay.actions,
        replay.is_first,
        jax.random.PRNGKey(54),
    )
    update_jaxpr = jax.make_jaxpr(
        lambda train_state, key, replay_batch: scan_dreamer_world_model_updates(
            train_state,
            replay_batch,
            key,
            config=config,
            train_steps=2,
            sequence_length=3,
            batch_size=1,
        )
    )(state, jax.random.PRNGKey(52), replay)

    assert "scan[" in str(recurrent_jaxpr)
    assert "scan[" in str(module_jaxpr)
    assert str(update_jaxpr).count("scan[") >= 2


def test_world_model_train_step_accepts_continuous_adapter_actions() -> None:
    config = DreamerV3Config.debug(
        action_dim=2,
        action_mode="continuous",
        observation_shape=(5,),
    )
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=np.zeros((4, 2, config.action_dim), dtype=np.float32),
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:continuous"},
    )
    state = create_dreamer_train_state(
        jax.random.PRNGKey(9), config, learning_rate=1e-3
    )

    updated, metrics = dreamer_train_step(
        state,
        batch,
        config,
        jax.random.PRNGKey(10),
    )

    assert updated.step == state.step + 1
    assert bool(jnp.isfinite(metrics["loss"]))


@pytest.mark.parametrize("action_mode", ["discrete", "continuous"])
def test_imagined_actor_critic_training_returns_finite_policy_rollout(
    action_mode: str,
) -> None:
    config = DreamerV3Config.debug(
        action_dim=3 if action_mode == "discrete" else 2,
        action_mode=action_mode,
        observation_shape=(5,),
    )
    if action_mode == "discrete":
        actions = np.zeros((4, 2), dtype=np.int32)
    else:
        actions = np.zeros((4, 2, config.action_dim), dtype=np.float32)
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=actions,
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": action_mode, "env": "fake:policy"},
    )
    world_model_state = create_dreamer_train_state(
        jax.random.PRNGKey(20), config, learning_rate=1e-3
    )

    actor_state, critic_state, metrics, rollout = train_dreamer_actor_critic(
        world_model_state=world_model_state,
        batch=batch,
        config=config,
        train_steps=2,
        learning_rate=1e-3,
        imagination_horizon=3,
        seed=21,
    )

    actor = DreamerActor(
        config.action_dim,
        config.action_mode,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    critic = DreamerCritic(
        config.actor_critic.value_bins,
        hidden_dims=config.actor_critic.hidden_dims,
    )
    actor_outputs = actor.apply({"params": actor_state.params}, rollout.features[0])
    critic_logits = critic.apply({"params": critic_state.params}, rollout.features[0])
    assert actor_state.step == 2
    assert critic_state.step == 2
    assert ("logits" in actor_outputs) == (action_mode == "discrete")
    assert critic_logits.shape == (8, config.actor_critic.value_bins)
    assert len(metrics) == 2
    assert rollout.actions.shape[:2] == (3, 8)
    assert rollout.features.shape == (3, 8, config.rssm.latent_size)
    start_state = initial_rssm_state(batch_size=8, config=config.rssm)
    rollout_jaxpr = jax.make_jaxpr(
        lambda key: imagine_dreamer_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_state=start_state,
            config=config,
            horizon=3,
            key=key,
        )
    )(jax.random.PRNGKey(22))
    assert "scan[" in str(rollout_jaxpr)
    for row in metrics:
        for key in ("actor_loss", "critic_loss", "imagined_reward", "imagined_value"):
            assert np.isfinite(row[key])


def test_dreamer_cli_smoke_writes_expected_artifacts(tmp_path) -> None:
    exit_code = train_dreamer_main(
        [
            "--env",
            "synthetic:image-grid",
            "--out-dir",
            str(tmp_path),
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--time-steps",
            "4",
            "--batch-size",
            "2",
            "--image-size",
            "16",
            "--model-size",
            "debug",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "sources.json",
        "world_model_metrics.jsonl",
        "actor_critic_metrics.jsonl",
        "agent_metrics.jsonl",
        "open_loop_reconstruction.png",
        "imagined_rollout.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    outcome = json.loads((tmp_path / "outcome.json").read_text())
    assert outcome["status"] in {"ok", "learning_gate_failed"}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["environment_backend"] == "synthetic"
    assert summary["observation_mode"] == "pixels"
    assert summary["real_env_transitions"] == 0
    assert summary["model_updates"] == 2
    assert summary["policy_updates"] == 2
    assert summary["imagined_transitions"] == 180


def test_dreamer_cli_brax_smoke_writes_real_env_artifacts(tmp_path) -> None:
    pytest.importorskip("brax")

    exit_code = train_dreamer_main(
        [
            "--env",
            "brax:reacher",
            "--out-dir",
            str(tmp_path),
            "--num-envs",
            "2",
            "--collect-steps",
            "4",
            "--max-cycles",
            "4",
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--eval-episodes",
            "1",
            "--model-size",
            "debug",
            "--brax-backend",
            "mjx",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "world_model_metrics.jsonl",
        "actor_critic_metrics.jsonl",
        "real_env_metrics.jsonl",
        "open_loop_reconstruction.png",
        "imagined_rollout.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["env"] == "brax:reacher"
    assert summary["action_mode"] == "continuous"
    assert summary["policy_source"] == "imagined_actor"
    assert summary["environment_backend"] == "brax"
    assert summary["physics_backend"] == "mjx"
    assert summary["observation_mode"] == "vector"
    assert summary["collection_policy"] == "dreamer_actor"
    assert summary["training_execution"] == "nested_jax_scan"
    assert summary["evaluation_execution"] == "jax_scan"
    assert summary["real_env_transitions"] >= 8
    assert summary["model_updates"] == 2
    assert summary["policy_updates"] == 2
    assert "real_env_return" in summary
    real_env_row = json.loads(
        (tmp_path / "real_env_metrics.jsonl").read_text().splitlines()[0]
    )
    assert real_env_row["policy_source"] == "imagined_actor"
    assert real_env_row["evaluation_execution"] == "jax_scan"


@pytest.mark.integration
def test_dreamer_cli_mjx_dmc_smoke_writes_accelerator_provenance(
    tmp_path,
) -> None:
    pytest.importorskip("mujoco_playground")

    exit_code = train_dreamer_main(
        [
            "--env",
            "dmc:cartpole/swingup",
            "--out-dir",
            str(tmp_path),
            "--num-envs",
            "1",
            "--collect-steps",
            "4",
            "--max-cycles",
            "4",
            "--train-steps",
            "2",
            "--policy-train-steps",
            "2",
            "--eval-episodes",
            "1",
            "--model-size",
            "debug",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["env"] == "dmc:cartpole/swingup"
    assert summary["seed"] == 0
    assert summary["environment_backend"] == "mujoco_playground"
    assert summary["physics_backend"] == "mjx"
    assert summary["observation_mode"] == "vector"
    assert summary["collection_execution"] == "jax_scan"
    assert summary["collection_policy"] == "dreamer_actor"
    assert summary["training_execution"] == "nested_jax_scan"
    assert summary["evaluation_execution"] == "jax_scan"
    assert summary["real_env_transitions"] >= 4
    assert summary["evaluation_env_transitions"] == 4
    assert summary["policy_source"] == "imagined_actor"
    assert np.isfinite(summary["real_env_return"])


@pytest.mark.integration
def test_dreamer_cli_rejects_host_rendered_dmc_pixels() -> None:
    pytest.importorskip("dm_control")

    with pytest.raises(
        RuntimeError,
        match="host-loop collection is not supported",
    ):
        train_dreamer_main(
            [
                "--env",
                "dmc-pixels:point_mass/easy",
                "--num-envs",
                "1",
                "--collect-steps",
                "4",
                "--max-cycles",
                "4",
                "--train-steps",
                "2",
                "--policy-train-steps",
                "2",
                "--eval-episodes",
                "1",
                "--image-size",
                "16",
                "--model-size",
                "debug",
            ]
        )
