import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.world_model_foundation.replay import JaxSequenceBatch


def _sequence_batch() -> JaxSequenceBatch:
    observations = jnp.arange(4 * 2 * 2 * 3 * 3, dtype=jnp.float32).reshape(
        4, 2, 2, 3, 3
    )
    actions = jnp.arange(8, dtype=jnp.int32).reshape(4, 2)
    rewards = jnp.arange(8, dtype=jnp.float32).reshape(4, 2) + 10
    continues = jnp.ones((4, 2), dtype=jnp.float32)
    is_first = jnp.asarray(
        [[True, True], [False, False], [True, False], [False, False]]
    )
    return JaxSequenceBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continues=continues,
        is_first=is_first,
        is_terminal=~continues.astype(bool),
        is_last=~continues.astype(bool),
    )


def test_backend_sequence_conversion_and_transition_pairing_are_exact() -> None:
    from world_marl.latent_action_world_model.replay import (
        pair_valid_transitions,
        to_backend_sequence,
    )

    sequence = to_backend_sequence(_sequence_batch())
    pairs = pair_valid_transitions(sequence)

    assert sequence.observations.shape == (2, 4, 2, 3, 3)
    assert jnp.array_equal(
        sequence.observations[1, 2], _sequence_batch().observations[2, 1]
    )
    assert sequence.valid_transitions.shape == (2, 3)
    assert jnp.array_equal(
        sequence.valid_transitions,
        jnp.asarray([[True, False, True], [True, True, True]]),
    )
    assert pairs.observations.shape == (5, 2, 3, 3)
    assert pairs.next_observations.shape == pairs.observations.shape
    assert jnp.array_equal(pairs.actions, jnp.asarray([0, 4, 1, 3, 5]))
    assert jnp.array_equal(pairs.rewards, jnp.asarray([12, 16, 13, 15, 17]))
    assert jnp.array_equal(pairs.continues, jnp.ones((5,)))


def test_backend_sequence_rejects_vector_and_non_rgb_observations() -> None:
    from world_marl.latent_action_world_model.replay import to_backend_sequence

    batch = _sequence_batch()
    with pytest.raises(ValueError, match="HWC RGB"):
        to_backend_sequence(batch._replace(observations=jnp.zeros((4, 2, 8))))
    with pytest.raises(ValueError, match="HWC RGB"):
        to_backend_sequence(batch._replace(observations=jnp.zeros((4, 2, 2, 3, 1))))


def _write_calibration(path, *, missing_code: bool = False) -> None:
    time_steps = 13 if missing_code else 14
    observations = np.arange(time_steps * 1 * 2 * 2 * 3, dtype=np.float32).reshape(
        time_steps, 1, 2, 2, 3
    )
    actions = np.arange(time_steps, dtype=np.int32).reshape(time_steps, 1)
    is_first = np.zeros((time_steps, 1), dtype=bool)
    is_first[0] = True
    np.savez(
        path,
        observations=observations,
        actions=actions,
        is_first=is_first,
        environment=np.asarray("synthetic:bridge"),
        provenance=np.asarray(
            json.dumps({"collector": "expert-v1", "seed": 7}, sort_keys=True)
        ),
    )


def test_expert_bridge_requires_provenance_and_all_six_codes(tmp_path) -> None:
    from world_marl.latent_action_world_model.bridge import load_expert_bridge

    calibration_path = tmp_path / "calibration.npz"
    _write_calibration(calibration_path)

    def infer_codes(videos: jax.Array) -> jax.Array:
        return jnp.arange(videos.shape[0]) % 6

    bridge = load_expert_bridge(calibration_path, infer_codes=infer_codes)
    assert bridge.environment == "synthetic:bridge"
    assert bridge.provenance == {"collector": "expert-v1", "seed": 7}
    assert jnp.array_equal(bridge.counts, jnp.asarray([3, 2, 2, 2, 2, 2]))

    missing_path = tmp_path / "missing-code.npz"
    _write_calibration(missing_path, missing_code=True)

    def missing_code(videos: jax.Array) -> jax.Array:
        return jnp.arange(videos.shape[0]) % 5

    with pytest.raises(ValueError, match="all six latent-action codes"):
        load_expert_bridge(missing_path, infer_codes=missing_code)

    no_provenance = tmp_path / "no-provenance.npz"
    np.savez(
        no_provenance,
        observations=np.zeros((2, 1, 2, 2, 3)),
        actions=np.zeros((2, 1)),
        is_first=np.asarray([[True], [False]]),
        environment=np.asarray("synthetic:bridge"),
    )
    with pytest.raises(ValueError, match="provenance"):
        load_expert_bridge(no_provenance, infer_codes=infer_codes)


@pytest.mark.parametrize("continuous", [False, True])
def test_expert_bridge_uniformly_samples_every_recorded_action(
    tmp_path, continuous: bool
) -> None:
    from world_marl.latent_action_world_model.bridge import (
        load_expert_bridge,
        sample_real_actions,
    )

    calibration_path = tmp_path / "calibration.npz"
    _write_calibration(calibration_path)
    if continuous:
        with np.load(calibration_path) as data:
            arrays = {name: data[name] for name in data.files}
        scalar_actions = arrays["actions"]
        arrays["actions"] = np.stack(
            [scalar_actions, scalar_actions + 100], axis=-1
        ).astype(np.float32)
        np.savez(calibration_path, **arrays)

    bridge = load_expert_bridge(
        calibration_path,
        infer_codes=lambda videos: jnp.arange(videos.shape[0]) % 6,
    )
    samples = jax.vmap(
        lambda key: sample_real_actions(
            key, bridge.actions, bridge.counts, jnp.asarray([0])
        )[0]
    )(jax.random.split(jax.random.PRNGKey(0), 2048))
    observed = np.unique(np.asarray(samples), axis=0)

    expected = np.asarray([[0], [6], [12]])
    if continuous:
        expected = np.concatenate([expected, expected + 100], axis=1)
    else:
        expected = expected[:, 0]
    assert np.array_equal(observed, expected)


def test_reward_continue_heads_use_two_hot_bce_and_stop_model_gradients() -> None:
    from world_marl.latent_action_world_model.heads import (
        RewardContinueHeads,
        decode_reward,
        reward_continue_loss,
    )

    module = RewardContinueHeads(hidden_dims=(16,))
    features = jax.random.normal(jax.random.PRNGKey(1), (6, 8))
    variables = module.init(jax.random.PRNGKey(2), features)
    rewards = jnp.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    continues = jnp.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    outputs = module.apply(variables, features)
    loss, metrics = reward_continue_loss(outputs, rewards, continues)
    feature_gradients = jax.grad(
        lambda values: reward_continue_loss(
            module.apply(variables, values), rewards, continues
        )[0]
    )(features)

    assert outputs.reward_logits.shape == (6, 255)
    assert outputs.continue_logits.shape == (6,)
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["reward_loss"])
    assert jnp.isfinite(metrics["continue_loss"])
    assert jnp.all(feature_gradients == 0)
    assert decode_reward(outputs.reward_logits).shape == rewards.shape
    assert jnp.allclose(
        outputs.continue_probability,
        jax.nn.sigmoid(outputs.continue_logits),
    )


def test_reward_continue_heads_tiny_overfit_runs_in_scan() -> None:
    from world_marl.latent_action_world_model.heads import (
        RewardContinueHeads,
        create_head_train_state,
        scan_head_updates,
    )

    features = jax.random.normal(jax.random.PRNGKey(3), (8, 6))
    rewards = jnp.linspace(-2.0, 2.0, 8)
    continues = jnp.asarray([0.0, 1.0] * 4)
    state = create_head_train_state(
        jax.random.PRNGKey(4),
        RewardContinueHeads(hidden_dims=(32,)),
        features,
        learning_rate=3e-2,
    )
    updates = 24
    state, metrics = scan_head_updates(
        state,
        jnp.repeat(features[None], updates, axis=0),
        jnp.repeat(rewards[None], updates, axis=0),
        jnp.repeat(continues[None], updates, axis=0),
    )
    scan_jaxpr = jax.make_jaxpr(
        lambda current: scan_head_updates(
            current,
            jnp.repeat(features[None], 2, axis=0),
            jnp.repeat(rewards[None], 2, axis=0),
            jnp.repeat(continues[None], 2, axis=0),
        )
    )(state)

    assert metrics["loss"][-1] < metrics["loss"][0]
    assert "scan[" in str(scan_jaxpr)


def _terminating_head_state(feature_dim: int):
    from flax.core import freeze, unfreeze
    from world_marl.latent_action_world_model.heads import (
        RewardContinueHeads,
        create_head_train_state,
    )

    state = create_head_train_state(
        jax.random.PRNGKey(30),
        RewardContinueHeads(hidden_dims=(8,)),
        jnp.zeros((1, feature_dim)),
        learning_rate=1e-3,
    )
    params = unfreeze(state.params)
    params["continue_logits"]["kernel"] = jnp.zeros_like(
        params["continue_logits"]["kernel"]
    )
    params["continue_logits"]["bias"] = jnp.full_like(
        params["continue_logits"]["bias"], -100.0
    )
    return state.replace(params=freeze(params))


def test_jafar_simulator_runs_complete_sampler_and_resets_from_replay() -> None:
    from world_marl.jafar.config import DynamicsConfig, LAMConfig, TokenizerConfig
    from world_marl.jafar.model import JafarWorldModel
    from world_marl.latent_action_world_model.simulator import (
        create_jafar_replay_pool,
        initialize_jafar_state,
        jafar_simulator_step,
    )

    model = JafarWorldModel(
        tokenizer_config=TokenizerConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            codebook_dropout=0.0,
        ),
        lam_config=LAMConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
        ),
        dynamics_config=DynamicsConfig(
            model_dim=8,
            num_latents=8,
            num_blocks=1,
            num_heads=2,
        ),
    )
    init_videos = jax.random.uniform(jax.random.PRNGKey(31), (2, 2, 4, 4, 3))
    params = model.init(
        jax.random.PRNGKey(32),
        {"videos": init_videos, "mask_rng": jax.random.PRNGKey(33)},
        training=True,
    )["params"]
    context = init_videos[:, :1]
    replay_pixels = jnp.stack(
        [jnp.zeros_like(context[0]), jnp.ones_like(context[0])], axis=0
    )
    state = initialize_jafar_state(model, params, context)
    pool = create_jafar_replay_pool(model, params, replay_pixels)
    head_state = _terminating_head_state(feature_dim=10)
    latent_codes = jnp.asarray([1, 2], dtype=jnp.int32)
    next_state, reward, done, continue_probability = jafar_simulator_step(
        model,
        params,
        head_state,
        pool,
        state,
        latent_codes,
        jax.random.PRNGKey(34),
        sampler_steps=2,
        sample_argmax=True,
    )
    step_jaxpr = jax.make_jaxpr(
        lambda rng: jafar_simulator_step(
            model,
            params,
            head_state,
            pool,
            state,
            latent_codes,
            rng,
            sampler_steps=2,
            sample_argmax=True,
        )
    )(jax.random.PRNGKey(35))

    assert next_state.token_history.shape == state.token_history.shape
    assert next_state.pixels.shape == state.pixels.shape
    assert reward.shape == (2,)
    assert jnp.array_equal(done, jnp.ones((2,), dtype=bool))
    assert jnp.all(continue_probability < 1e-20)
    assert jnp.all(
        (next_state.pixels == 0).all(axis=(1, 2, 3, 4))
        | (next_state.pixels == 1).all(axis=(1, 2, 3, 4))
    )
    assert str(step_jaxpr).count("scan[") >= 2


def test_jasmine_simulator_runs_complete_sampler_and_resets_from_replay() -> None:
    from world_marl.jasmine.config import DynamicsConfig, LAMConfig, TokenizerConfig
    from world_marl.jasmine.model import JasmineWorldModel
    from world_marl.latent_action_world_model.simulator import (
        create_jasmine_replay_pool,
        initialize_jasmine_state,
        jasmine_simulator_step,
    )

    model = JasmineWorldModel(
        tokenizer_config=TokenizerConfig(
            model_dim=8,
            ffn_dim=16,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        lam_config=LAMConfig(
            model_dim=8,
            ffn_dim=16,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        dynamics_config=DynamicsConfig(
            model_dim=8,
            ffn_dim=16,
            latent_patch_dim=4,
            latent_action_dim=4,
            num_blocks=1,
            num_heads=2,
            denoise_steps=4,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        lam_co_train=True,
    )
    init_videos = jax.random.uniform(jax.random.PRNGKey(36), (2, 2, 4, 4, 3))
    params = model.init(
        jax.random.PRNGKey(37),
        {"videos": init_videos, "rng": jax.random.PRNGKey(38)},
    )["params"]
    context = init_videos[:, :1].astype(jnp.bfloat16)
    replay_pixels = jnp.stack(
        [jnp.zeros_like(context[0]), jnp.ones_like(context[0])], axis=0
    )
    state = initialize_jasmine_state(model, params, context)
    pool = create_jasmine_replay_pool(model, params, replay_pixels)
    head_state = _terminating_head_state(feature_dim=10)
    latent_codes = jnp.asarray([3, 4], dtype=jnp.int32)
    next_state, reward, done, continue_probability = jasmine_simulator_step(
        model,
        params,
        head_state,
        pool,
        state,
        latent_codes,
        jax.random.PRNGKey(39),
        diffusion_steps=4,
        context_corruption=0.1,
    )
    step_jaxpr = jax.make_jaxpr(
        lambda rng: jasmine_simulator_step(
            model,
            params,
            head_state,
            pool,
            state,
            latent_codes,
            rng,
            diffusion_steps=4,
            context_corruption=0.1,
        )
    )(jax.random.PRNGKey(40))

    assert next_state.latent_history.shape == state.latent_history.shape
    assert next_state.pixels.shape == state.pixels.shape
    assert reward.shape == (2,)
    assert jnp.array_equal(done, jnp.ones((2,), dtype=bool))
    assert jnp.all(continue_probability < 1e-20)
    assert jnp.all(
        (next_state.pixels == 0).all(axis=(1, 2, 3, 4))
        | (next_state.pixels == 1).all(axis=(1, 2, 3, 4))
    )
    assert str(step_jaxpr).count("scan[") >= 2


def test_scanned_simulator_uses_existing_cnn_ppo_update_unchanged() -> None:
    from world_marl.algs.ippo import IPPOConfig, create_train_state
    from world_marl.jafar.config import DynamicsConfig, LAMConfig, TokenizerConfig
    from world_marl.jafar.model import JafarWorldModel
    from world_marl.latent_action_world_model.policy import (
        scan_simulator_ppo_updates,
    )
    from world_marl.latent_action_world_model.simulator import (
        create_jafar_replay_pool,
        initialize_jafar_state,
        jafar_simulator_step,
    )

    model = JafarWorldModel(
        tokenizer_config=TokenizerConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            codebook_dropout=0.0,
        ),
        lam_config=LAMConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
        ),
        dynamics_config=DynamicsConfig(
            model_dim=8,
            num_latents=8,
            num_blocks=1,
            num_heads=2,
        ),
    )
    videos = jax.random.uniform(jax.random.PRNGKey(41), (2, 2, 4, 4, 3))
    model_params = model.init(
        jax.random.PRNGKey(42),
        {"videos": videos, "mask_rng": jax.random.PRNGKey(43)},
        training=True,
    )["params"]
    state = initialize_jafar_state(model, model_params, videos[:, :1])
    pool = create_jafar_replay_pool(model, model_params, videos[:, :1])
    head_state = _terminating_head_state(feature_dim=10)
    config = IPPOConfig(update_epochs=1, num_minibatches=1, network_arch="cnn")
    policy_state = create_train_state(
        jax.random.PRNGKey(44), (4, 4, 3), action_dim=6, config=config
    )

    def step_fn(simulator_state, latent_codes, rng):
        return jafar_simulator_step(
            model,
            model_params,
            head_state,
            pool,
            simulator_state,
            latent_codes,
            rng,
            sampler_steps=2,
            sample_argmax=True,
        )

    updated_policy, final_state, rollouts, metrics = scan_simulator_ppo_updates(
        policy_state,
        state,
        step_fn,
        jax.random.PRNGKey(45),
        updates=1,
        horizon=2,
        config=config,
    )
    update_jaxpr = jax.make_jaxpr(
        lambda current, rng: scan_simulator_ppo_updates(
            current,
            state,
            step_fn,
            rng,
            updates=1,
            horizon=2,
            config=config,
        )
    )(policy_state, jax.random.PRNGKey(46))

    assert rollouts.observations.shape == (1, 2, 2, 4, 4, 3)
    assert rollouts.actions.shape == (1, 2, 2)
    assert jnp.all((rollouts.actions >= 0) & (rollouts.actions < 6))
    assert final_state.token_history.shape == state.token_history.shape
    assert jnp.isfinite(metrics["total_loss"])
    assert any(
        not jnp.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(policy_state.params),
            jax.tree.leaves(updated_policy.params),
            strict=True,
        )
    )
    assert str(update_jaxpr).count("scan[") >= 5
