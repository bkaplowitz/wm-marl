from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import freeze, unfreeze

from world_marl.genie2_continuous_jax.action_bridge import (
    fit_linear_action_bridge,
)
from world_marl.genie2_continuous_jax.autoencoder import (
    ContinuousLatentAutoencoder,
    reconstruction_loss,
)
from world_marl.genie2_continuous_jax.config import Genie2ContinuousConfig
from world_marl.genie2_continuous_jax.dynamics import (
    CausalLatentDynamics,
    classifier_free_guidance,
    dynamics_mse_loss,
)
from world_marl.genie2_continuous_jax.lam import (
    ContinuousLAM,
    lam_kl_loss,
    sample_latent_actions,
)
from world_marl.genie2_continuous_jax.rl_heads import RewardContinueHead
from world_marl.genie2_continuous_jax.sampling import sample_next_observation
from world_marl.genie2_continuous_jax.policy import (
    Genie2PolicyRollout,
    _score_candidate_rollouts,
    _critic_loss,
    simulate_latent_policy_rollout,
    train_genie2_latent_policy,
)
from world_marl.genie2_continuous_jax.training import (
    create_genie2_train_state,
    genie2_train_step,
    sample_genie2_latents,
    scan_genie2_training_phases,
    scan_genie2_world_model_updates,
    train_genie2_world_model,
)
from world_marl.scripts.train_genie2_continuous_jax import main as train_genie2_main
from world_marl.world_model_foundation.collect import synthetic_sequence_collector
from world_marl.world_model_foundation.replay import (
    WorldModelSequenceBatch,
    sequence_batch_to_jax,
)


def test_config_defaults_use_continuous_latents_not_vq_primary() -> None:
    config = Genie2ContinuousConfig()

    assert config.specification == "genie2_public_latent_diffusion"
    assert config.implementation_profile == "jasmine_diffusion_paper"
    assert config.representation == "continuous_latent_patch_grid"
    assert config.conditioning_mode == "real_action"
    assert config.lam.enabled is False
    assert config.lam.kind == "continuous_extension"
    assert config.dynamics.objective == "diffusion_forcing_x_prediction"
    assert config.dynamics.denoising_steps == 25
    assert config.vq_maskgit_ablation_enabled is False
    assert config.autoencoder.latent_patch_dim > 0


def test_vector_adapter_autoencoder_returns_continuous_latents() -> None:
    observations = jnp.ones((2, 5), dtype=jnp.float32) * 0.5
    model = ContinuousLatentAutoencoder(latent_dim=12, hidden_dims=(32,))
    params = model.init(jax.random.PRNGKey(0), observations)

    latents, reconstructions = model.apply(params, observations)
    loss = reconstruction_loss(observations, reconstructions)

    assert latents.shape == (2, 12)
    assert reconstructions.shape == observations.shape
    assert bool(jnp.isfinite(loss))


def test_autoencoder_vector_decoder_is_unbounded() -> None:
    observations = jnp.zeros((2, 5), dtype=jnp.float32)
    model = ContinuousLatentAutoencoder(latent_dim=4, hidden_dims=(8,))
    params = model.init(jax.random.PRNGKey(40), observations)
    mutable = unfreeze(params)
    mutable["params"]["decoder"]["kernel"] = jnp.zeros_like(
        mutable["params"]["decoder"]["kernel"]
    )
    mutable["params"]["decoder"]["bias"] = jnp.full_like(
        mutable["params"]["decoder"]["bias"], -2.0
    )

    _, reconstructions = model.apply(freeze(mutable), observations)

    assert bool(jnp.allclose(reconstructions, -2.0))


def test_optional_continuous_lam_extension_samples_latent_actions() -> None:
    prev_latents = jnp.zeros((3, 10), dtype=jnp.float32)
    next_latents = jnp.ones((3, 10), dtype=jnp.float32)
    model = ContinuousLAM(latent_action_dim=5, hidden_dims=(32,))
    params = model.init(jax.random.PRNGKey(1), prev_latents, next_latents)

    mean, log_std = model.apply(params, prev_latents, next_latents)
    actions = sample_latent_actions(jax.random.PRNGKey(2), mean, log_std)
    loss = lam_kl_loss(mean, log_std)

    assert mean.shape == (3, 5)
    assert log_std.shape == (3, 5)
    assert actions.shape == (3, 5)
    assert bool(jnp.all(jnp.isfinite(actions)))
    assert bool(jnp.isfinite(loss))


def test_legacy_flat_dynamics_extension_predicts_next_latent() -> None:
    latent_history = jnp.zeros((2, 4, 10), dtype=jnp.float32)
    latent_actions = jnp.ones((2, 4, 5), dtype=jnp.float32)
    noise_level = jnp.full((2,), 0.25, dtype=jnp.float32)
    model = CausalLatentDynamics(
        latent_dim=10,
        latent_action_dim=5,
        model_dim=32,
        num_heads=4,
        num_layers=1,
    )
    params = model.init(
        jax.random.PRNGKey(3), latent_history, latent_actions, noise_level
    )

    conditioned = model.apply(params, latent_history, latent_actions, noise_level)
    unconditioned = model.apply(
        params, latent_history, jnp.zeros_like(latent_actions), noise_level
    )
    guided = classifier_free_guidance(
        conditioned=conditioned,
        unconditioned=unconditioned,
        guidance_scale=1.5,
    )
    loss = dynamics_mse_loss(conditioned, jnp.ones_like(conditioned))

    assert conditioned.shape == (2, 10)
    assert guided.shape == (2, 10)
    assert bool(jnp.all(jnp.isfinite(guided)))
    assert bool(jnp.isfinite(loss))


def test_optional_linear_action_bridge_recovers_known_mapping() -> None:
    latent_actions = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    real_actions = np.asarray([[0.5], [2.5], [-0.5], [1.5]], dtype=np.float32)

    bridge = fit_linear_action_bridge(latent_actions, real_actions, ridge=1e-6)
    predicted = bridge.predict(latent_actions)

    np.testing.assert_allclose(predicted, real_actions, atol=1e-4)
    assert bridge.latent_action_dim == 2
    assert bridge.real_action_dim == 1


def test_optional_vector_sampler_and_reward_continue_head_are_finite() -> None:
    latents = jnp.ones((3, 12), dtype=jnp.float32)
    latent_actions = jnp.zeros((3, 5), dtype=jnp.float32)
    head = RewardContinueHead(hidden_dims=(16,))
    params = head.init(jax.random.PRNGKey(4), latents, latent_actions)

    reward, continue_logit = head.apply(params, latents, latent_actions)

    assert reward.shape == (3,)
    assert continue_logit.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(reward)))

    autoencoder = ContinuousLatentAutoencoder(latent_dim=12, hidden_dims=(16,))
    obs = jnp.ones((3, 6), dtype=jnp.float32) * 0.25
    ae_params = autoencoder.init(jax.random.PRNGKey(5), obs)
    decoded = sample_next_observation(
        autoencoder.apply,
        ae_params,
        latents,
        observation_shape=obs.shape[1:],
    )
    decoded_zeros = sample_next_observation(
        autoencoder.apply,
        ae_params,
        jnp.zeros_like(latents),
        observation_shape=obs.shape[1:],
    )

    assert decoded.shape == obs.shape
    assert not bool(jnp.allclose(decoded, decoded_zeros))


def test_genie2_train_step_updates_params_and_returns_finite_metrics() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(16, 16, 3),
        action_mode="discrete",
    )
    batch = synthetic_sequence_collector(
        env_name="synthetic:image-grid",
        time_steps=4,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=3,
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(6),
        observation_shape=config.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    updated, metrics = genie2_train_step(
        state,
        sequence_batch_to_jax(batch),
        config,
        jax.random.PRNGKey(7),
    )

    assert updated.tokenizer.step == state.tokenizer.step + 1
    assert updated.dynamics.step == state.dynamics.step + 1
    assert updated.heads.step == state.heads.step + 1
    for key in (
        "loss",
        "tokenizer_loss",
        "reconstruction_loss",
        "dynamics_loss",
        "diffusion_x_loss",
        "reward_continue_loss",
        "reward_loss",
        "continue_loss",
    ):
        assert key in metrics
        assert bool(jnp.isfinite(metrics[key]))


def test_genie2_optimizer_scan_samples_windows_within_dynamics_context() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(5,),
        action_mode="discrete",
    )
    batch = synthetic_sequence_collector(
        env_name="synthetic:long-context",
        time_steps=6,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=3,
    )
    replay = sequence_batch_to_jax(batch)
    state = create_genie2_train_state(
        jax.random.PRNGKey(61),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    update_jaxpr = jax.make_jaxpr(
        lambda train_state, key, replay_batch: scan_genie2_world_model_updates(
            train_state,
            replay_batch,
            key,
            config=config,
            train_steps=2,
            sequence_length=4,
            batch_size=2,
        )
    )(state, jax.random.PRNGKey(62), replay)
    updated, metrics = train_genie2_world_model(
        batch=batch,
        config=config,
        train_steps=2,
        learning_rate=1e-3,
        seed=63,
        sequence_length=4,
        batch_size=2,
    )

    assert "scan[" in str(update_jaxpr)
    assert updated.tokenizer.step == 2
    assert updated.dynamics.step == 2
    assert updated.heads.step == 2
    assert len(metrics) == 2
    assert all(np.isfinite(row["loss"]) for row in metrics)


def test_genie2_staged_scan_returns_fixed_validation_losses() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=3,
        observation_shape=(5,),
        action_mode="discrete",
    )
    batch = synthetic_sequence_collector(
        env_name="synthetic:validation",
        time_steps=6,
        batch_size=2,
        observation_shape=config.observation_shape,
        action_dim=3,
    )
    replay = sequence_batch_to_jax(batch)
    state = create_genie2_train_state(
        jax.random.PRNGKey(71),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    updated, metrics, validation = jax.jit(
        scan_genie2_training_phases,
        static_argnames=(
            "config",
            "tokenizer_steps",
            "dynamics_steps",
            "reward_continue_steps",
            "sequence_length",
            "batch_size",
        ),
    )(
        state,
        replay,
        jax.random.PRNGKey(72),
        config=config,
        tokenizer_steps=2,
        dynamics_steps=2,
        reward_continue_steps=2,
        sequence_length=4,
        batch_size=2,
    )

    assert updated.tokenizer.step == 2
    assert set(metrics) == {"tokenizer", "dynamics", "reward_continue"}
    assert set(validation) == {"tokenizer", "dynamics", "reward_continue"}
    for stage in validation.values():
        assert set(stage) == {"initial_loss", "final_loss"}
        assert np.isfinite(np.asarray(stage["initial_loss"]))
        assert np.isfinite(np.asarray(stage["final_loss"]))


def test_genie2_diffusion_sampler_lowers_to_nested_scans() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(5,),
        action_mode="continuous",
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(71),
        observation_shape=config.observation_shape,
        config=config,
        learning_rate=1e-3,
    )
    latent_history = jnp.zeros(
        (2, 2, 1, config.autoencoder.latent_patch_dim), dtype=jnp.float32
    )
    actions = jnp.zeros((2, 3, config.action_dim), dtype=jnp.float32)

    sampler_jaxpr = jax.make_jaxpr(
        lambda key, history, action_sequence: sample_genie2_latents(
            state,
            history,
            action_sequence,
            config,
            key,
            num_future_frames=2,
        )
    )(jax.random.PRNGKey(72), latent_history, actions)

    assert str(sampler_jaxpr).count("scan[") >= 2


def test_genie2_train_step_accepts_vector_adapter_replay() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(5,),
        action_mode="continuous",
    )
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=np.zeros((4, 2, 2), dtype=np.float32),
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:continuous"},
    )
    state = create_genie2_train_state(
        jax.random.PRNGKey(7),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    updated, metrics = genie2_train_step(
        state,
        sequence_batch_to_jax(batch),
        config,
        jax.random.PRNGKey(8),
    )

    assert updated.tokenizer.step == state.tokenizer.step + 1
    assert updated.dynamics.step == state.dynamics.step + 1
    assert updated.heads.step == state.heads.step + 1
    assert bool(jnp.isfinite(metrics["loss"]))


def test_latent_policy_training_returns_finite_learned_simulator_rollout() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(5,),
        action_mode="continuous",
    )
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=4 * 2 * 5, dtype=np.float32).reshape(
            (4, 2, 5)
        ),
        actions=np.zeros((4, 2, 2), dtype=np.float32),
        rewards=np.zeros((4, 2), dtype=np.float32),
        continues=np.ones((4, 2), dtype=np.float32),
        is_first=np.array(
            [[True, True], [False, False], [False, False], [False, False]]
        ),
        is_terminal=np.zeros((4, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:policy"},
    )
    world_model_state = create_genie2_train_state(
        jax.random.PRNGKey(30),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    actor_state, critic_state, metrics, rollout = train_genie2_latent_policy(
        world_model_state=world_model_state,
        batch=batch,
        observation_shape=batch.observation_shape,
        config=config,
        train_steps=1,
        learning_rate=1e-3,
        imagination_horizon=2,
        seed=31,
    )

    assert actor_state.step == 1
    assert critic_state.step == 1
    assert len(metrics) == 1
    policy_batch_size = config.latent_policy.batch_size
    assert rollout.states.shape == (
        2,
        policy_batch_size,
        1,
        config.autoencoder.latent_patch_dim,
    )
    assert rollout.latents.shape == (
        2,
        policy_batch_size,
        1,
        config.autoencoder.latent_patch_dim,
    )
    assert rollout.environment_actions.shape == (
        2,
        policy_batch_size,
        config.action_dim,
    )
    assert rollout.model_actions.shape == (2, policy_batch_size, config.action_dim)
    rollout_jaxpr = jax.make_jaxpr(
        lambda key: simulate_latent_policy_rollout(
            world_model_state=world_model_state,
            actor_state=actor_state,
            critic_state=critic_state,
            start_latents=jnp.zeros(
                (2, 1, 1, config.autoencoder.latent_patch_dim),
                dtype=jnp.float32,
            ),
            start_actions=jnp.zeros((2, 0, config.action_dim), dtype=jnp.float32),
            observation_shape=batch.observation_shape,
            config=config,
            horizon=2,
            key=key,
        )
    )(jax.random.PRNGKey(32))
    assert "scan[" in str(rollout_jaxpr)
    for row in metrics:
        for key in ("actor_loss", "critic_loss", "imagined_reward", "imagined_value"):
            assert np.isfinite(row[key])


def test_critic_loss_uses_current_state_not_generated_next_latent() -> None:
    states = jnp.zeros((2, 1, 1, 1), dtype=jnp.float32)
    next_latents = jnp.full((2, 1, 1, 1), 10.0, dtype=jnp.float32)
    rollout = Genie2PolicyRollout(
        states=states,
        latents=next_latents,
        environment_actions=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        model_actions=jnp.zeros((2, 1, 1), dtype=jnp.float32),
        rewards=jnp.zeros((2, 1), dtype=jnp.float32),
        continues=jnp.ones((2, 1), dtype=jnp.float32),
        values=jnp.zeros((2, 1), dtype=jnp.float32),
        returns=jnp.zeros((2, 1), dtype=jnp.float32),
        log_probabilities=jnp.zeros((2, 1), dtype=jnp.float32),
        entropies=jnp.zeros((2, 1), dtype=jnp.float32),
        weights=jnp.ones((2, 1), dtype=jnp.float32),
    )
    critic_state = SimpleNamespace(apply_fn=lambda _params, values: values[..., 0])

    loss = _critic_loss({}, critic_state, rollout)

    assert float(loss) == pytest.approx(0.0)


def test_candidate_distill_policy_updates_actor_without_scalar_critic() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=2,
        observation_shape=(5,),
        action_mode="continuous",
    )
    batch = WorldModelSequenceBatch(
        observations=np.linspace(0.0, 1.0, num=6 * 2 * 5, dtype=np.float32).reshape(
            (6, 2, 5)
        ),
        actions=np.zeros((6, 2, 2), dtype=np.float32),
        rewards=np.ones((6, 2), dtype=np.float32),
        continues=np.ones((6, 2), dtype=np.float32),
        is_first=np.array(
            [
                [True, True],
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False],
            ]
        ),
        is_terminal=np.zeros((6, 2), dtype=bool),
        metadata={"action_mode": "continuous", "env": "fake:candidate-policy"},
    )
    world_model_state = create_genie2_train_state(
        jax.random.PRNGKey(40),
        observation_shape=batch.observation_shape,
        config=config,
        learning_rate=1e-3,
    )

    actor_state, critic_state, metrics, rollout = train_genie2_latent_policy(
        world_model_state=world_model_state,
        batch=batch,
        observation_shape=batch.observation_shape,
        config=config,
        train_steps=2,
        learning_rate=1e-3,
        imagination_horizon=2,
        objective="candidate-distill",
        num_candidates=8,
        candidate_min_gap=0.0,
        seed=41,
    )

    assert actor_state.step == 2
    assert critic_state.step == 0
    assert len(metrics) == 2
    assert rollout.environment_actions.shape[1] == config.latent_policy.batch_size
    for row in metrics:
        for key in (
            "actor_loss",
            "candidate_best_score",
            "candidate_mean_score",
            "candidate_active_fraction",
            "action_saturation_fraction",
        ):
            assert np.isfinite(row[key])


def test_candidate_scores_roll_forward_through_dynamics_with_scan() -> None:
    config = Genie2ContinuousConfig.debug(
        action_dim=1,
        observation_shape=(5,),
        action_mode="continuous",
    )
    config = replace(
        config,
        dynamics=replace(config.dynamics, denoising_steps=25),
    )

    def dynamics_apply(
        _params,
        noised_latents,
        actions,
        _denoising_steps,
        *,
        condition_keep_mask,
        training,
        method,
    ):
        del training, method
        context_steps_match_sampler = jnp.all(
            _denoising_steps[:, :-1] == config.dynamics.denoising_steps - 1,
            axis=1,
        ) & (_denoising_steps[:, -1] == 0)
        action_effect = (
            actions[:, -1, 0] * condition_keep_mask[:, -1] * context_steps_match_sampler
        )
        next_latent = jnp.broadcast_to(
            action_effect[:, None, None],
            noised_latents[:, -1].shape,
        )
        return noised_latents.at[:, -1].set(next_latent)

    def heads_apply(_params, pooled_latents, _actions):
        return pooled_latents[:, 0], jnp.full(
            (pooled_latents.shape[0],),
            12.0,
            dtype=jnp.float32,
        )

    def actor_apply(_params, pooled_latents):
        return {
            "mean": jnp.zeros((pooled_latents.shape[0], 1), dtype=jnp.float32),
            "log_std": jnp.zeros((pooled_latents.shape[0], 1), dtype=jnp.float32),
        }

    state = SimpleNamespace(
        dynamics=SimpleNamespace(apply_fn=dynamics_apply, params={}),
        heads=SimpleNamespace(apply_fn=heads_apply, params={}),
    )
    actor = SimpleNamespace(apply_fn=actor_apply, params={})
    start_latents = jnp.zeros(
        (1, 2, 1, config.autoencoder.latent_patch_dim),
        dtype=jnp.float32,
    )
    start_actions = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    candidates = jnp.asarray([[[-0.5], [0.5]]], dtype=jnp.float32)

    one_step = _score_candidate_rollouts(
        world_model_state=state,
        actor_state=actor,
        actor_params=actor.params,
        start_latents=start_latents,
        start_actions=start_actions,
        normalized_candidates=candidates,
        config=config,
        key=jax.random.PRNGKey(81),
        horizon=1,
    )
    two_step = _score_candidate_rollouts(
        world_model_state=state,
        actor_state=actor,
        actor_params=actor.params,
        start_latents=start_latents,
        start_actions=start_actions,
        normalized_candidates=candidates,
        config=config,
        key=jax.random.PRNGKey(81),
        horizon=2,
    )
    planning_jaxpr = jax.make_jaxpr(
        lambda candidate_actions: _score_candidate_rollouts(
            world_model_state=state,
            actor_state=actor,
            actor_params=actor.params,
            start_latents=start_latents,
            start_actions=start_actions,
            normalized_candidates=candidate_actions,
            config=config,
            key=jax.random.PRNGKey(81),
            horizon=2,
        )
    )(candidates)

    np.testing.assert_allclose(one_step[0, 0], one_step[0, 1], atol=1e-6)
    assert two_step[0, 1] > two_step[0, 0]
    assert "scan[" in str(planning_jaxpr)


def test_genie2_cli_smoke_writes_expected_artifacts(tmp_path) -> None:
    exit_code = train_genie2_main(
        [
            "--env",
            "synthetic:image-grid",
            "--out-dir",
            str(tmp_path),
            "--model-size",
            "debug",
            "--train-steps",
            "1",
            "--policy-train-steps",
            "1",
            "--imagination-horizon",
            "1",
            "--time-steps",
            "4",
            "--sequence-length",
            "4",
            "--batch-size",
            "2",
            "--train-batch-size",
            "2",
            "--image-size",
            "16",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "sources.json",
        "tokenizer_metrics.jsonl",
        "autoencoder_metrics.jsonl",
        "dynamics_metrics.jsonl",
        "reward_continue_metrics.jsonl",
        "validation_metrics.json",
        "policy_metrics.jsonl",
        "real_env_metrics.jsonl",
        "conditioning.json",
        "open_loop_rollout.png",
        "action_grid.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["model"] == "genie2_continuous_jax"
    assert summary["policy_source"] == "direct_action_policy"
    assert summary["environment_backend"] == "synthetic"
    assert summary["observation_mode"] == "pixels"
    assert summary["real_env_transitions"] == 0
    assert summary["model_updates"] == 3
    assert summary["policy_updates"] == 1
    assert summary["policy_imagination_batch_size"] == 4
    assert summary["imagined_transitions"] == 4
    assert summary["candidate_action_evaluations"] == 0
    assert np.isfinite(summary["real_env_return"])
    validation = json.loads((tmp_path / "validation_metrics.json").read_text())
    assert set(validation) == {"tokenizer", "dynamics", "reward_continue"}
    assert not (tmp_path / "lam_metrics.jsonl").exists()
    assert not (tmp_path / "latent_action_bridge.json").exists()
    assert not (tmp_path / "latent_action_usage.json").exists()


def test_genie2_cli_brax_smoke_uses_direct_action_policy(tmp_path) -> None:
    pytest.importorskip("brax")

    exit_code = train_genie2_main(
        [
            "--env",
            "brax:reacher",
            "--out-dir",
            str(tmp_path),
            "--model-size",
            "debug",
            "--num-envs",
            "2",
            "--collect-steps",
            "4",
            "--max-cycles",
            "4",
            "--train-steps",
            "1",
            "--policy-train-steps",
            "1",
            "--imagination-horizon",
            "1",
            "--sequence-length",
            "4",
            "--train-batch-size",
            "2",
            "--eval-episodes",
            "1",
            "--brax-backend",
            "mjx",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    for name in (
        "config.json",
        "tokenizer_metrics.jsonl",
        "autoencoder_metrics.jsonl",
        "dynamics_metrics.jsonl",
        "reward_continue_metrics.jsonl",
        "validation_metrics.json",
        "policy_metrics.jsonl",
        "real_env_metrics.jsonl",
        "conditioning.json",
        "open_loop_rollout.png",
        "action_grid.png",
        "outcome.json",
        "summary.json",
    ):
        assert (tmp_path / name).exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["env"] == "brax:reacher"
    assert summary["action_mode"] == "continuous"
    assert summary["policy_source"] == "direct_action_policy"
    assert summary["environment_backend"] == "brax"
    assert summary["physics_backend"] == "mjx"
    assert summary["observation_mode"] == "vector"
    assert summary["evaluation_execution"] == "jax_scan"
    assert summary["real_env_transitions"] == 8
    assert summary["model_updates"] == 3
    assert summary["policy_updates"] == 1
    assert np.isfinite(summary["real_env_return"])
    assert not (tmp_path / "lam_metrics.jsonl").exists()
    assert not (tmp_path / "latent_action_bridge.json").exists()
    real_env_row = json.loads(
        (tmp_path / "real_env_metrics.jsonl").read_text().splitlines()[0]
    )
    assert real_env_row["policy_source"] == "direct_action_policy"
    assert real_env_row["evaluation_execution"] == "jax_scan"


def test_genie2_cli_mjx_dmc_smoke_uses_direct_action_policy(tmp_path) -> None:
    pytest.importorskip("mujoco_playground")

    exit_code = train_genie2_main(
        [
            "--env",
            "dmc:cartpole/swingup",
            "--out-dir",
            str(tmp_path),
            "--model-size",
            "debug",
            "--num-envs",
            "1",
            "--collect-steps",
            "4",
            "--max-cycles",
            "4",
            "--train-steps",
            "1",
            "--policy-train-steps",
            "1",
            "--policy-objective",
            "candidate-distill",
            "--num-policy-candidates",
            "8",
            "--imagination-horizon",
            "1",
            "--sequence-length",
            "4",
            "--train-batch-size",
            "1",
            "--eval-episodes",
            "1",
            "--allow-fail",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["env"] == "dmc:cartpole/swingup"
    assert summary["policy_source"] == "direct_action_policy"
    assert summary["policy_objective"] == "candidate-distill"
    assert summary["imagined_transitions"] == 32
    assert summary["candidate_action_evaluations"] == 32
    assert summary["environment_backend"] == "mujoco_playground"
    assert summary["physics_backend"] == "mjx"
    assert summary["observation_mode"] == "vector"
    assert summary["collection_execution"] == "jax_scan"
    assert summary["training_execution"] == "nested_jax_scan"
    assert summary["evaluation_execution"] == "jax_scan"
    assert np.isfinite(summary["real_env_return"])
