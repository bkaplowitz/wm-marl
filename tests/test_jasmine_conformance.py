import importlib
from pathlib import Path

import jax
import jax.numpy as jnp


def test_jasmine_source_defaults_are_locked() -> None:
    config_module = importlib.import_module("world_marl.jasmine.config")

    config = config_module.JasmineConfig()

    assert config.sequence_length == 16
    assert config.image_height == 64
    assert config.image_width == 64
    assert config.image_channels == 3

    assert config.tokenizer.patch_size == 16
    assert config.tokenizer.model_dim == 512
    assert config.tokenizer.ffn_dim == 2048
    assert config.tokenizer.latent_dim == 32
    assert config.tokenizer.num_blocks == 4
    assert config.tokenizer.num_heads == 8
    assert config.tokenizer.max_mask_ratio == 0.9
    assert config.tokenizer.param_dtype == jnp.float32
    assert config.tokenizer.dtype == jnp.bfloat16
    assert config.tokenizer.use_flash_attention is True

    assert config.lam.patch_size == 16
    assert config.lam.model_dim == 512
    assert config.lam.ffn_dim == 2048
    assert config.lam.latent_dim == 32
    assert config.lam.num_latents == 6
    assert config.lam.num_blocks == 4
    assert config.lam.num_heads == 8
    assert config.lam.reset_inactive_after == 50

    assert config.dynamics.model_dim == 512
    assert config.dynamics.ffn_dim == 2048
    assert config.dynamics.num_blocks == 6
    assert config.dynamics.num_heads == 8
    assert config.dynamics.denoise_steps == 64
    assert config.dynamics.context_corruption == 0.1
    assert config.dynamics.use_gt_actions is False
    assert config.dynamics.use_cfg is False

    assert config.tokenizer_training.updates == 300_000
    assert config.tokenizer_training.batch_size == 48
    assert config.tokenizer_training.warmup_steps == 10_000
    assert config.tokenizer_training.wsd_decay_steps == 30_000
    assert config.tokenizer_training.peak_learning_rate == 3e-4
    assert config.lam_training.updates == 200_000
    assert config.lam_training.batch_size == 36
    assert config.lam_training.warmup_steps == 5_000
    assert config.lam_training.wsd_decay_steps == 20_000
    assert config.lam_training.peak_learning_rate == 3e-5
    assert config.dynamics_training.updates == 200_000
    assert config.dynamics_training.batch_size == 36
    assert config.dynamics_training.warmup_steps == 5_000
    assert config.dynamics_training.wsd_decay_steps == 20_000
    assert config.dynamics_training.peak_learning_rate == 1e-4


def test_jasmine_patchify_unpatchify_preserves_source_layout_and_crop() -> None:
    preprocess = importlib.import_module("world_marl.jasmine.preprocess")
    videos = jnp.arange(1 * 2 * 5 * 7 * 3, dtype=jnp.float32).reshape(1, 2, 5, 7, 3)

    patches = preprocess.patchify(videos, size=4)
    reconstructed = preprocess.unpatchify(patches, size=4, h_out=5, w_out=7)

    assert patches.shape == (1, 2, 4, 48)
    assert jnp.array_equal(patches[0, 0, 0], videos[0, 0, :4, :4].reshape(-1))
    assert reconstructed.shape == videos.shape
    assert jnp.array_equal(reconstructed, videos)


def test_jasmine_axial_transformer_preserves_causality_remat_and_dtypes() -> None:
    nn_module = importlib.import_module("world_marl.jasmine.nn")
    transformer = nn_module.AxialTransformer(
        input_dim=4,
        model_dim=8,
        ffn_dim=16,
        out_dim=4,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
        spatial_causal=False,
        temporal_causal=True,
    )
    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 2, 4))
    variables = transformer.init(jax.random.PRNGKey(1), inputs)
    baseline = transformer.apply(variables, inputs)

    future_inputs = inputs.at[:, 2, 0].add(jnp.asarray([10.0, -7.0, 3.0, 1.0]))
    future_outputs = transformer.apply(variables, future_inputs)
    spatial_inputs = inputs.at[:, 1, 1].add(jnp.asarray([2.0, -1.0, 3.0, -2.0]))
    spatial_outputs = transformer.apply(variables, spatial_inputs)
    apply_jaxpr = jax.make_jaxpr(lambda values: transformer.apply(variables, values))(
        inputs
    )

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
    assert baseline.dtype == jnp.bfloat16
    assert jnp.allclose(
        future_outputs[:, :2].astype(jnp.float32),
        baseline[:, :2].astype(jnp.float32),
        atol=1e-5,
    )
    assert not jnp.allclose(future_outputs[:, 2], baseline[:, 2])
    assert not jnp.allclose(spatial_outputs[:, 1, 0], baseline[:, 1, 0])
    assert "remat" in str(apply_jaxpr)


def test_jasmine_vq_uses_source_initializer_cosine_codes_and_st_gradient() -> None:
    nn_module = importlib.import_module("world_marl.jasmine.nn")
    quantizer = nn_module.VectorQuantizer(
        latent_dim=2,
        num_latents=2,
        dropout=0.0,
        dtype=jnp.float32,
    )
    inputs = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    variables = quantizer.init(jax.random.PRNGKey(2), inputs, training=False)
    variables["params"]["codebook"] = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )

    z_q, codes, embeddings, indices = quantizer.apply(variables, inputs, training=False)
    input_gradient = jax.grad(
        lambda value: quantizer.apply(variables, value, training=False)[0].sum()
    )(inputs)

    assert variables["params"]["codebook"].dtype == jnp.float32
    assert jnp.array_equal(indices, jnp.asarray([0, 1]))
    assert jnp.allclose(codes, jnp.eye(2))
    assert jnp.allclose(embeddings, jnp.eye(2))
    assert jnp.allclose(z_q, codes)
    assert jnp.allclose(
        input_gradient,
        jnp.asarray([[0.0, 0.5], [1 / 3, 0.0]]),
        atol=1e-7,
    )


def test_jasmine_mae_masks_per_frame_and_bounds_continuous_latents() -> None:
    tokenizer_module = importlib.import_module("world_marl.jasmine.tokenizer")
    mask, probabilities = tokenizer_module.sample_patch_mask(
        jax.random.PRNGKey(3),
        batch_size=2,
        sequence_length=3,
        num_patches=16,
        max_mask_ratio=0.9,
    )
    assert mask.shape == (2, 3, 16)
    assert probabilities.shape == (2, 3)
    assert jnp.all((probabilities >= 0.0) & (probabilities <= 0.9))
    assert jnp.unique(probabilities).size > 1

    tokenizer = tokenizer_module.TokenizerMAE(
        in_dim=3,
        model_dim=8,
        ffn_dim=16,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        max_mask_ratio=0.9,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
    )
    videos = jax.random.uniform(jax.random.PRNGKey(4), (2, 3, 4, 4, 3))
    batch = {"videos": videos, "rng": jax.random.PRNGKey(5)}
    variables = tokenizer.init(jax.random.PRNGKey(6), batch, training=True)
    outputs = tokenizer.apply(variables, batch, training=True)
    decoded = tokenizer.apply(
        variables,
        outputs["z"],
        (4, 4),
        method=tokenizer.decode,
    )

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
    assert outputs["z"].shape == (2, 3, 4, 4)
    assert outputs["z"].dtype == jnp.bfloat16
    assert jnp.all((outputs["z"] >= -1.0) & (outputs["z"] <= 1.0))
    assert outputs["recon"].shape == videos.shape
    assert outputs["recon"].dtype == jnp.bfloat16
    assert jnp.all((outputs["recon"] >= 0.0) & (outputs["recon"] <= 1.0))
    assert jnp.allclose(decoded, outputs["recon"])


def test_jasmine_lam_preserves_action_layout_causality_and_dtypes() -> None:
    lam_module = importlib.import_module("world_marl.jasmine.lam")
    lam = lam_module.LatentActionModel(
        in_dim=3,
        model_dim=8,
        ffn_dim=16,
        latent_dim=4,
        num_latents=6,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
    )
    videos = jax.random.uniform(jax.random.PRNGKey(7), (1, 3, 4, 4, 3))
    variables = lam.init(jax.random.PRNGKey(8), {"videos": videos}, training=False)
    outputs = lam.apply(variables, {"videos": videos}, training=False)
    changed_videos = videos.at[:, 2, :, :, 0].set(1.0 - videos[:, 2, :, :, 0])
    changed = lam.apply(
        variables,
        changed_videos,
        training=False,
        method=lam.vq_encode,
    )

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
    assert variables["params"]["action_in"].shape == (1, 1, 1, 12)
    assert outputs["z_q"].shape == (1, 2, 1, 4)
    assert outputs["z"].shape == (2, 4)
    assert outputs["emb"].shape == (2, 4)
    assert outputs["indices"].shape == (2,)
    assert outputs["recon"].shape == (1, 2, 4, 4, 3)
    assert outputs["recon"].dtype == jnp.bfloat16
    assert jnp.all((outputs["recon"] >= 0.0) & (outputs["recon"] <= 1.0))
    assert jnp.allclose(
        changed["emb"][0].astype(jnp.float32),
        outputs["emb"][0].astype(jnp.float32),
        atol=1e-5,
    )
    assert not jnp.allclose(changed["emb"][1], outputs["emb"][1])


def test_jasmine_diffusion_noising_weighting_levels_and_dtypes() -> None:
    dynamics_module = importlib.import_module("world_marl.jasmine.dynamics")
    clean = jnp.asarray([[[[2.0]]]])
    noise = jnp.asarray([[[[-1.0]]]])
    signal = jnp.asarray([[0.25]])
    mixed = dynamics_module.linear_noise_mix(clean, noise, signal)
    expected = (1.0 - (1.0 - 1e-5) * 0.25) * noise + 0.25 * clean
    assert jnp.allclose(mixed, expected)
    assert jnp.allclose(
        dynamics_module.ramp_weight(jnp.asarray([0.0, 0.5, 1.0])),
        jnp.asarray([0.1, 0.55, 1.0]),
    )

    model = dynamics_module.DynamicsDiffusion(
        model_dim=8,
        ffn_dim=16,
        latent_patch_dim=4,
        latent_action_dim=4,
        num_blocks=1,
        num_heads=2,
        denoise_steps=8,
        dropout=0.0,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
    )
    batch = {
        "token_latents": jax.random.normal(
            jax.random.PRNGKey(9), (2, 3, 4, 4), dtype=jnp.bfloat16
        ),
        "latent_actions": jax.random.normal(
            jax.random.PRNGKey(10), (2, 2, 1, 4), dtype=jnp.bfloat16
        ),
        "rng": jax.random.PRNGKey(11),
    }
    variables = model.init(jax.random.PRNGKey(12), batch)
    predicted, levels = model.apply(variables, batch)

    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(variables))
    assert predicted.shape == batch["token_latents"].shape
    assert predicted.dtype == jnp.bfloat16
    assert levels.shape == (2, 3)
    assert jnp.all((levels >= 0.0) & (levels < 1.0))
    assert jnp.unique(levels).size > 1


def test_jasmine_sampler_context_corruption_and_nested_scans() -> None:
    config_module = importlib.import_module("world_marl.jasmine.config")
    model_module = importlib.import_module("world_marl.jasmine.model")
    sampling_module = importlib.import_module("world_marl.jasmine.sampling")
    assert jnp.allclose(
        sampling_module.snapped_context_signal_level(64, 0.1),
        58 / 64,
    )

    model = model_module.JasmineWorldModel(
        tokenizer_config=config_module.TokenizerConfig(
            model_dim=8,
            ffn_dim=16,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            max_mask_ratio=0.9,
            dtype=jnp.bfloat16,
            use_flash_attention=False,
        ),
        lam_config=config_module.LAMConfig(
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
        dynamics_config=config_module.DynamicsConfig(
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
    videos = jax.random.uniform(jax.random.PRNGKey(13), (1, 3, 4, 4, 3))
    variables = model.init(
        jax.random.PRNGKey(14),
        {"videos": videos, "rng": jax.random.PRNGKey(15)},
    )
    sample_batch = {
        "videos": videos[:, :1].astype(jnp.bfloat16),
        "latent_actions": jnp.asarray([[0, 1]], dtype=jnp.int32),
        "rng": jax.random.PRNGKey(16),
    }
    generated = model.apply(
        variables,
        sample_batch,
        seq_len=3,
        diffusion_steps=4,
        context_corruption=0.1,
        method=model.sample,
    )
    sample_jaxpr = jax.make_jaxpr(
        lambda rng: model.apply(
            variables,
            {**sample_batch, "rng": rng},
            seq_len=3,
            diffusion_steps=4,
            context_corruption=0.1,
            method=model.sample,
        )
    )(jax.random.PRNGKey(17))

    assert generated.shape == (1, 3, 4, 4, 3)
    assert generated.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(generated))
    assert jnp.all((generated >= 0.0) & (generated <= 1.0))
    assert str(sample_jaxpr).count("scan[") >= 2


def test_jasmine_wsd_schedule_and_source_losses_are_exact() -> None:
    training_module = importlib.import_module("world_marl.jasmine.training")
    schedule = training_module.wsd_schedule(
        initial_learning_rate=0.0,
        peak_learning_rate=1e-3,
        decay_end=0.0,
        total_steps=10,
        warmup_steps=2,
        decay_steps=3,
    )
    values = jax.vmap(schedule)(jnp.asarray([0, 2, 7, 10]))
    assert jnp.allclose(values, jnp.asarray([0.0, 1e-3, 1e-3, 0.0]))

    tokenizer_loss, tokenizer_metrics = training_module.tokenizer_loss(
        jnp.asarray([0.0, 1.0]),
        {"recon": jnp.asarray([0.5, 0.5])},
    )
    assert jnp.allclose(tokenizer_loss, 0.25)
    assert jnp.allclose(tokenizer_metrics["mse"], tokenizer_loss)

    dynamics_loss, dynamics_metrics = training_module.diffusion_loss(
        {
            "x_pred": jnp.zeros((1, 2, 1, 1)),
            "x_gt": jnp.asarray([[[[1.0]], [[2.0]]]]),
            "signal_level": jnp.asarray([[0.0, 1.0]]),
            "lam_indices": jnp.asarray([0, 1, 1]),
        },
        num_actions=6,
    )
    assert jnp.allclose(dynamics_loss, 2.05)
    assert jnp.allclose(dynamics_metrics["mse"], 2.5)
    assert jnp.allclose(dynamics_metrics["codebook_usage_lam"], 2 / 6)


def test_jasmine_scanned_training_freezes_mae_and_cotrains_lam() -> None:
    config_module = importlib.import_module("world_marl.jasmine.config")
    model_module = importlib.import_module("world_marl.jasmine.model")
    training_module = importlib.import_module("world_marl.jasmine.training")
    model = model_module.JasmineWorldModel(
        tokenizer_config=config_module.TokenizerConfig(
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
        lam_config=config_module.LAMConfig(
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
        dynamics_config=config_module.DynamicsConfig(
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
    videos = jax.random.uniform(jax.random.PRNGKey(18), (2, 3, 4, 4, 3))
    batch = {"videos": videos, "rng": jax.random.PRNGKey(19)}
    state = training_module.create_dynamics_train_state(
        jax.random.PRNGKey(20),
        model,
        batch,
        learning_rate=1e-3,
    )
    batches = {
        "videos": jnp.repeat(videos[None], 2, axis=0),
        "rng": jax.random.split(jax.random.PRNGKey(21), 2),
    }
    updated, metrics = training_module.scan_dynamics_updates(state, batches)
    scan_jaxpr = jax.make_jaxpr(
        lambda current: training_module.scan_dynamics_updates(current, batches)
    )(state)

    def tree_equal(left, right) -> bool:
        return all(
            bool(jnp.array_equal(x, y))
            for x, y in zip(
                jax.tree.leaves(left),
                jax.tree.leaves(right),
                strict=True,
            )
        )

    assert "decoder" not in state.params["lam"]
    assert tree_equal(updated.params["tokenizer"], state.params["tokenizer"])
    assert not tree_equal(updated.params["lam"], state.params["lam"])
    assert not tree_equal(updated.params["dynamics"], state.params["dynamics"])
    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert "scan[" in str(scan_jaxpr)


def test_jasmine_tokenizer_lam_and_dynamics_tiny_overfit() -> None:
    config_module = importlib.import_module("world_marl.jasmine.config")
    lam_module = importlib.import_module("world_marl.jasmine.lam")
    model_module = importlib.import_module("world_marl.jasmine.model")
    tokenizer_module = importlib.import_module("world_marl.jasmine.tokenizer")
    training_module = importlib.import_module("world_marl.jasmine.training")
    videos = jnp.zeros((1, 3, 4, 4, 3), dtype=jnp.bfloat16)
    updates = 12
    repeated_videos = jnp.repeat(videos[None], updates, axis=0)

    tokenizer = tokenizer_module.TokenizerMAE(
        in_dim=3,
        model_dim=8,
        ffn_dim=16,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        max_mask_ratio=0.9,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
    )
    tokenizer_state = training_module.create_tokenizer_train_state(
        jax.random.PRNGKey(22), tokenizer, videos, learning_rate=1e-2
    )
    _, tokenizer_metrics = training_module.scan_tokenizer_updates(
        tokenizer_state,
        repeated_videos,
        jax.random.split(jax.random.PRNGKey(23), updates),
    )
    assert tokenizer_metrics["loss"][-1] < tokenizer_metrics["loss"][0]

    lam = lam_module.LatentActionModel(
        in_dim=3,
        model_dim=8,
        ffn_dim=16,
        latent_dim=4,
        num_latents=6,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
        param_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        use_flash_attention=False,
    )
    lam_state = training_module.create_lam_train_state(
        jax.random.PRNGKey(24), lam, videos, learning_rate=1e-2
    )
    _, _, lam_metrics = training_module.scan_lam_updates(
        lam_state,
        jnp.zeros((6,), dtype=jnp.int32),
        repeated_videos,
        jax.random.split(jax.random.PRNGKey(25), updates),
        beta=0.25,
        reset_threshold=50,
    )
    assert lam_metrics["loss"][-1] < lam_metrics["loss"][0]

    model = model_module.JasmineWorldModel(
        tokenizer_config=config_module.TokenizerConfig(
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
        lam_config=config_module.LAMConfig(
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
        dynamics_config=config_module.DynamicsConfig(
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
    dynamics_batch = {"videos": videos, "rng": jax.random.PRNGKey(26)}
    dynamics_state = training_module.create_dynamics_train_state(
        jax.random.PRNGKey(27), model, dynamics_batch, learning_rate=1e-2
    )
    dynamics_batches = {
        "videos": repeated_videos,
        "rng": jnp.repeat(dynamics_batch["rng"][None], updates, axis=0),
    }
    _, dynamics_metrics = training_module.scan_dynamics_updates(
        dynamics_state, dynamics_batches
    )
    assert dynamics_metrics["loss"][-1] < dynamics_metrics["loss"][0]


def test_jasmine_package_exports_source_arm_components() -> None:
    package = importlib.import_module("world_marl.jasmine")

    assert package.JasmineWorldModel.__name__ == "JasmineWorldModel"
    assert package.TokenizerMAE.__name__ == "TokenizerMAE"
    assert package.LatentActionModel.__name__ == "LatentActionModel"
    assert package.DynamicsDiffusion.__name__ == "DynamicsDiffusion"


def test_jasmine_adapted_modules_record_pinned_source_and_changes() -> None:
    source_dir = Path(__file__).parents[1] / "src" / "world_marl" / "jasmine"
    adapted_modules = (
        "config.py",
        "preprocess.py",
        "nn.py",
        "tokenizer.py",
        "lam.py",
        "dynamics.py",
        "sampling.py",
        "training.py",
        "model.py",
    )

    for module_name in adapted_modules:
        contents = (source_dir / module_name).read_text()
        assert "p-doom/jasmine" in contents
        assert "420859bc99eecf6b07a7e9edf65d5d145935f1e1" in contents
        assert "path" in contents.lower()
        assert "Integration changes" in contents
