import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import optax


def test_jafar_source_defaults_are_locked() -> None:
    config_module = importlib.import_module("world_marl.jafar.config")

    config = config_module.JafarConfig()

    assert config.sequence_length == 16
    assert config.image_height == 64
    assert config.image_width == 64
    assert config.image_channels == 3

    assert config.tokenizer.patch_size == 4
    assert config.tokenizer.model_dim == 512
    assert config.tokenizer.latent_dim == 32
    assert config.tokenizer.num_latents == 1024
    assert config.tokenizer.num_blocks == 8
    assert config.tokenizer.num_heads == 8
    assert config.tokenizer.codebook_dropout == 0.01

    assert config.lam.patch_size == 16
    assert config.lam.model_dim == 512
    assert config.lam.latent_dim == 32
    assert config.lam.num_latents == 6
    assert config.lam.num_blocks == 8
    assert config.lam.num_heads == 8
    assert config.lam.reset_inactive_after == 50

    assert config.dynamics.model_dim == 512
    assert config.dynamics.num_latents == 1024
    assert config.dynamics.num_blocks == 12
    assert config.dynamics.num_heads == 8
    assert config.dynamics.mask_limit == 0.5
    assert config.dynamics.maskgit_steps == 25

    assert config.tokenizer_training.updates == 300_000
    assert config.tokenizer_training.batch_size == 48
    assert config.tokenizer_training.warmup_steps == 10_000
    assert config.tokenizer_training.peak_learning_rate == 3e-4
    assert config.lam_training.updates == 200_000
    assert config.lam_training.batch_size == 36
    assert config.lam_training.warmup_steps == 5_000
    assert config.lam_training.peak_learning_rate == 3e-5
    assert config.dynamics_training.updates == 200_000
    assert config.dynamics_training.batch_size == 36
    assert config.dynamics_training.warmup_steps == 5_000
    assert config.dynamics_training.peak_learning_rate == 3e-5


def test_jafar_patchify_unpatchify_preserves_source_layout_and_crop() -> None:
    preprocess = importlib.import_module("world_marl.jafar.preprocess")
    videos = jnp.arange(1 * 2 * 5 * 7 * 3, dtype=jnp.float32).reshape(1, 2, 5, 7, 3)

    patches = preprocess.patchify(videos, size=4)
    reconstructed = preprocess.unpatchify(patches, size=4, h_out=5, w_out=7)

    assert patches.shape == (1, 2, 4, 48)
    assert jnp.array_equal(patches[0, 0, 0], videos[0, 0, :4, :4].reshape(-1))
    assert reconstructed.shape == videos.shape
    assert jnp.array_equal(reconstructed, videos)


def test_jafar_vq_uses_cosine_codes_and_source_straight_through_gradient() -> None:
    nn_module = importlib.import_module("world_marl.jafar.nn")
    quantizer = nn_module.VectorQuantizer(
        latent_dim=2,
        num_latents=2,
        dropout=0.0,
    )
    inputs = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    variables = quantizer.init(jax.random.PRNGKey(0), inputs, training=False)
    variables["params"]["codebook"] = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )

    z_q, codes, embeddings, indices = quantizer.apply(variables, inputs, training=False)
    input_gradient = jax.grad(
        lambda value: quantizer.apply(variables, value, training=False)[0].sum()
    )(inputs)

    assert jnp.array_equal(indices, jnp.asarray([0, 1]))
    assert jnp.allclose(codes, jnp.eye(2))
    assert jnp.allclose(embeddings, jnp.eye(2))
    assert jnp.allclose(z_q, codes)
    assert jnp.allclose(
        input_gradient,
        jnp.asarray([[0.0, 0.5], [1 / 3, 0.0]]),
        atol=1e-7,
    )


def test_jafar_transformer_is_temporally_causal_spatial_and_rematerialized() -> None:
    nn_module = importlib.import_module("world_marl.jafar.nn")
    transformer = nn_module.STTransformer(
        model_dim=8,
        out_dim=4,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
    )
    inputs = jax.random.normal(jax.random.PRNGKey(1), (1, 3, 2, 4))
    variables = transformer.init(jax.random.PRNGKey(2), inputs)
    baseline = transformer.apply(variables, inputs)

    future_inputs = inputs.at[:, 2, 0].add(jnp.asarray([10.0, -7.0, 3.0, 1.0]))
    future_outputs = transformer.apply(variables, future_inputs)
    spatial_inputs = inputs.at[:, 1, 1].add(jnp.asarray([2.0, -1.0, 3.0, -2.0]))
    spatial_outputs = transformer.apply(variables, spatial_inputs)
    apply_jaxpr = jax.make_jaxpr(lambda values: transformer.apply(variables, values))(
        inputs
    )

    assert jnp.allclose(future_outputs[:, :2], baseline[:, :2], atol=1e-6)
    assert not jnp.allclose(future_outputs[:, 2], baseline[:, 2])
    assert not jnp.allclose(spatial_outputs[:, 1, 0], baseline[:, 1, 0])
    assert "remat" in str(apply_jaxpr)


def test_jafar_tokenizer_preserves_source_token_grid_and_sigmoid_decode() -> None:
    tokenizer_module = importlib.import_module("world_marl.jafar.tokenizer")
    tokenizer = tokenizer_module.TokenizerVQVAE(
        in_dim=3,
        model_dim=8,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
    )
    videos = jax.random.uniform(jax.random.PRNGKey(3), (1, 2, 4, 4, 3))
    variables = tokenizer.init(
        jax.random.PRNGKey(4), {"videos": videos}, training=False
    )

    outputs = tokenizer.apply(variables, {"videos": videos}, training=False)
    decoded = tokenizer.apply(
        variables,
        outputs["indices"],
        (4, 4),
        method=tokenizer.decode,
    )

    assert outputs["z_q"].shape == (1, 2, 4, 4)
    assert outputs["z"].shape == (8, 4)
    assert outputs["emb"].shape == (8, 4)
    assert outputs["indices"].shape == (1, 2, 4)
    assert outputs["recon"].shape == videos.shape
    assert jnp.all((outputs["recon"] >= 0.0) & (outputs["recon"] <= 1.0))
    assert jnp.allclose(decoded, outputs["recon"])


def test_jafar_lam_uses_future_frame_action_tokens_and_next_frame_decode() -> None:
    lam_module = importlib.import_module("world_marl.jafar.lam")
    lam = lam_module.LatentActionModel(
        in_dim=3,
        model_dim=8,
        latent_dim=4,
        num_latents=6,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
    )
    videos = jax.random.uniform(jax.random.PRNGKey(5), (1, 3, 4, 4, 3))
    variables = lam.init(jax.random.PRNGKey(6), {"videos": videos}, training=False)
    outputs = lam.apply(variables, {"videos": videos}, training=False)

    changed_videos = videos.at[:, 2, :, :, 0].set(1.0 - videos[:, 2, :, :, 0])
    changed = lam.apply(
        variables,
        changed_videos,
        training=False,
        method=lam.vq_encode,
    )

    assert variables["params"]["action_in"].shape == (1, 1, 1, 12)
    assert outputs["z_q"].shape == (1, 2, 1, 4)
    assert outputs["z"].shape == (2, 4)
    assert outputs["emb"].shape == (2, 4)
    assert outputs["indices"].shape == (2,)
    assert outputs["recon"].shape == (1, 2, 4, 4, 3)
    assert jnp.all((outputs["recon"] >= 0.0) & (outputs["recon"] <= 1.0))
    assert jnp.allclose(changed["emb"][0], outputs["emb"][0], atol=1e-6)
    assert not jnp.allclose(changed["emb"][1], outputs["emb"][1])


def test_jafar_maskgit_mask_limit_first_frame_and_action_alignment() -> None:
    dynamics_module = importlib.import_module("world_marl.jafar.dynamics")
    keys = jax.random.split(jax.random.PRNGKey(7), 128)
    probabilities = jax.vmap(
        lambda key: dynamics_module.sample_mask_probability(key, 0.5)
    )(keys)
    assert jnp.all((probabilities >= 0.0) & (probabilities <= 0.5))

    dynamics = dynamics_module.DynamicsMaskGIT(
        model_dim=8,
        num_latents=8,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        mask_limit=0.5,
    )
    tokens = jnp.zeros((2, 3, 32), dtype=jnp.int32)
    actions = jax.random.normal(jax.random.PRNGKey(8), (2, 2, 1, 4))
    batch = {
        "video_tokens": tokens,
        "latent_actions": actions,
        "mask_rng": jax.random.PRNGKey(9),
    }
    variables = dynamics.init(jax.random.PRNGKey(10), batch, training=True)
    training_outputs = dynamics.apply(variables, batch, training=True)
    baseline = dynamics.apply(variables, batch, training=False)
    changed_actions = actions.at[:, 0].add(jnp.asarray([[[3.0, -2.0, 1.0, -4.0]]]))
    changed = dynamics.apply(
        variables,
        {**batch, "latent_actions": changed_actions},
        training=False,
    )

    assert training_outputs["token_logits"].shape == (2, 3, 32, 8)
    assert training_outputs["mask"].shape == tokens.shape
    assert not jnp.any(training_outputs["mask"][:, 0])
    assert jnp.any(training_outputs["mask"][:, 1:])
    assert baseline["mask"] is None
    assert jnp.allclose(
        changed["token_logits"][:, 0], baseline["token_logits"][:, 0], atol=1e-6
    )
    assert not jnp.allclose(
        changed["token_logits"][:, 1], baseline["token_logits"][:, 1]
    )


def test_jafar_maskgit_uses_source_25_step_cosine_schedule() -> None:
    sampling_module = importlib.import_module("world_marl.jafar.sampling")
    steps = jnp.arange(25)
    ratios = jax.vmap(lambda step: sampling_module.unmasked_ratio(step, 25))(steps)

    assert jnp.allclose(ratios[0], jnp.cos(jnp.pi / 50))
    assert jnp.allclose(ratios[-1], 0.0, atol=1e-7)
    assert jnp.all(jnp.diff(ratios) < 0.0)


def test_jafar_losses_and_inactive_code_reset_match_source_equations() -> None:
    training_module = importlib.import_module("world_marl.jafar.training")
    targets = jnp.asarray([[[[[0.25]]]]])
    outputs = {
        "recon": jnp.asarray([[[[[0.5]]]]]),
        "z": jnp.asarray([[0.0, 1.0]]),
        "emb": jnp.asarray([[1.0, 1.0]]),
    }
    total, metrics = training_module.vqvae_loss(targets, outputs, beta=0.25)

    assert jnp.allclose(metrics["mse"], 0.0625)
    assert jnp.allclose(metrics["q_loss"], 0.5)
    assert jnp.allclose(metrics["commitment_loss"], 0.5)
    assert jnp.allclose(total, 0.6875)

    logits = jnp.asarray([[[[3.0, -1.0], [-2.0, 2.0]]]])
    labels = jnp.asarray([[[0, 0]]])
    mask = jnp.asarray([[[True, False]]])
    masked_loss, masked_accuracy = training_module.masked_token_metrics(
        logits, labels, mask
    )
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    assert jnp.allclose(masked_loss, expected[0, 0, 0])
    assert jnp.allclose(masked_accuracy, 1.0)

    codebook = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    counts = jnp.asarray([4, 0, 0])
    last_active = jnp.asarray([0, 49, 48])
    new_codebook, new_last_active = training_module.reset_inactive_codes(
        jax.random.PRNGKey(11),
        codebook,
        counts,
        last_active,
        threshold=50,
    )
    assert jnp.array_equal(new_codebook[0], codebook[0])
    assert jnp.array_equal(new_codebook[1], codebook[0])
    assert jnp.array_equal(new_codebook[2], codebook[2])
    assert jnp.array_equal(new_last_active, jnp.asarray([0, 0, 49]))


def test_jafar_sampler_decodes_pixels_and_lowers_to_nested_scans() -> None:
    config_module = importlib.import_module("world_marl.jafar.config")
    model_module = importlib.import_module("world_marl.jafar.model")
    tokenizer_config = config_module.TokenizerConfig(
        model_dim=8,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        codebook_dropout=0.0,
    )
    lam_config = config_module.LAMConfig(
        model_dim=8,
        latent_dim=4,
        num_latents=6,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
    )
    dynamics_config = config_module.DynamicsConfig(
        model_dim=8,
        num_latents=8,
        num_blocks=1,
        num_heads=2,
        maskgit_steps=2,
    )
    model = model_module.JafarWorldModel(
        tokenizer_config=tokenizer_config,
        lam_config=lam_config,
        dynamics_config=dynamics_config,
    )
    full_videos = jax.random.uniform(jax.random.PRNGKey(12), (1, 3, 4, 4, 3))
    init_batch = {
        "videos": full_videos,
        "mask_rng": jax.random.PRNGKey(13),
    }
    variables = model.init(jax.random.PRNGKey(14), init_batch, training=True)
    sample_batch = {
        "videos": full_videos[:, :1],
        "latent_actions": jnp.asarray([[0, 1]], dtype=jnp.int32),
        "rng": jax.random.PRNGKey(15),
    }

    generated = model.apply(
        variables,
        sample_batch,
        seq_len=3,
        steps=2,
        sample_argmax=True,
        method=model.sample,
    )
    sample_jaxpr = jax.make_jaxpr(
        lambda rng: model.apply(
            variables,
            {**sample_batch, "rng": rng},
            seq_len=3,
            steps=2,
            sample_argmax=True,
            method=model.sample,
        )
    )(jax.random.PRNGKey(16))

    assert generated.shape == (1, 3, 4, 4, 3)
    assert jnp.all(jnp.isfinite(generated))
    assert jnp.all((generated >= 0.0) & (generated <= 1.0))
    assert str(sample_jaxpr).count("scan[") >= 2


def test_jafar_dynamics_training_scan_freezes_tokenizer_and_lam() -> None:
    config_module = importlib.import_module("world_marl.jafar.config")
    model_module = importlib.import_module("world_marl.jafar.model")
    training_module = importlib.import_module("world_marl.jafar.training")
    model = model_module.JafarWorldModel(
        tokenizer_config=config_module.TokenizerConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            codebook_dropout=0.0,
        ),
        lam_config=config_module.LAMConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
        ),
        dynamics_config=config_module.DynamicsConfig(
            model_dim=8,
            num_latents=8,
            num_blocks=1,
            num_heads=2,
        ),
    )
    batch = {
        "videos": jax.random.uniform(jax.random.PRNGKey(17), (2, 3, 8, 8, 3)),
        "mask_rng": jax.random.PRNGKey(18),
    }
    state = training_module.create_dynamics_train_state(
        jax.random.PRNGKey(19),
        model,
        batch,
        learning_rate=1e-3,
    )
    batches = {
        "videos": jnp.repeat(batch["videos"][None], 2, axis=0),
        "mask_rng": jax.random.split(jax.random.PRNGKey(20), 2),
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

    assert tree_equal(updated.params["tokenizer"], state.params["tokenizer"])
    assert tree_equal(updated.params["lam"], state.params["lam"])
    assert not tree_equal(updated.params["dynamics"], state.params["dynamics"])
    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert "scan[" in str(scan_jaxpr)


def test_jafar_tokenizer_tiny_overfit_reduces_source_loss() -> None:
    tokenizer_module = importlib.import_module("world_marl.jafar.tokenizer")
    training_module = importlib.import_module("world_marl.jafar.training")
    tokenizer = tokenizer_module.TokenizerVQVAE(
        in_dim=3,
        model_dim=8,
        latent_dim=4,
        num_latents=8,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
    )
    videos = jnp.zeros((1, 2, 4, 4, 3), dtype=jnp.float32)
    state = training_module.create_tokenizer_train_state(
        jax.random.PRNGKey(21),
        tokenizer,
        videos,
        learning_rate=1e-2,
    )
    updates = 12
    batches = jnp.repeat(videos[None], updates, axis=0)
    rngs = jax.random.split(jax.random.PRNGKey(22), updates)

    _, metrics = training_module.scan_tokenizer_updates(
        state,
        batches,
        rngs,
        beta=0.25,
    )

    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert metrics["loss"][-1] < metrics["loss"][0]


def test_jafar_lam_tiny_overfit_reduces_source_loss() -> None:
    lam_module = importlib.import_module("world_marl.jafar.lam")
    training_module = importlib.import_module("world_marl.jafar.training")
    lam = lam_module.LatentActionModel(
        in_dim=3,
        model_dim=8,
        latent_dim=4,
        num_latents=6,
        patch_size=2,
        num_blocks=1,
        num_heads=2,
        dropout=0.0,
        codebook_dropout=0.0,
    )
    videos = jnp.zeros((1, 3, 4, 4, 3), dtype=jnp.float32)
    state = training_module.create_lam_train_state(
        jax.random.PRNGKey(23),
        lam,
        videos,
        learning_rate=1e-2,
    )
    updates = 12
    batches = jnp.repeat(videos[None], updates, axis=0)
    rngs = jax.random.split(jax.random.PRNGKey(24), updates)
    last_active = jnp.zeros((6,), dtype=jnp.int32)

    _, last_active, metrics = training_module.scan_lam_updates(
        state,
        last_active,
        batches,
        rngs,
        beta=0.25,
        reset_threshold=50,
    )

    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert metrics["loss"][-1] < metrics["loss"][0]
    assert last_active.shape == (6,)


def test_jafar_dynamics_tiny_overfit_reduces_masked_cross_entropy() -> None:
    config_module = importlib.import_module("world_marl.jafar.config")
    model_module = importlib.import_module("world_marl.jafar.model")
    training_module = importlib.import_module("world_marl.jafar.training")
    model = model_module.JafarWorldModel(
        tokenizer_config=config_module.TokenizerConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=8,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
            codebook_dropout=0.0,
        ),
        lam_config=config_module.LAMConfig(
            model_dim=8,
            latent_dim=4,
            num_latents=6,
            patch_size=2,
            num_blocks=1,
            num_heads=2,
        ),
        dynamics_config=config_module.DynamicsConfig(
            model_dim=8,
            num_latents=8,
            num_blocks=1,
            num_heads=2,
        ),
    )
    batch = {
        "videos": jnp.zeros((2, 3, 8, 8, 3), dtype=jnp.float32),
        "mask_rng": jax.random.PRNGKey(25),
    }
    state = training_module.create_dynamics_train_state(
        jax.random.PRNGKey(26),
        model,
        batch,
        learning_rate=1e-2,
    )
    updates = 12
    batches = {
        "videos": jnp.repeat(batch["videos"][None], updates, axis=0),
        "mask_rng": jnp.repeat(batch["mask_rng"][None], updates, axis=0),
    }

    _, metrics = training_module.scan_dynamics_updates(state, batches)

    assert jnp.all(jnp.isfinite(metrics["loss"]))
    assert metrics["loss"][-1] < metrics["loss"][0]


def test_jafar_package_exports_source_arm_components() -> None:
    package = importlib.import_module("world_marl.jafar")

    assert package.JafarWorldModel.__name__ == "JafarWorldModel"
    assert package.TokenizerVQVAE.__name__ == "TokenizerVQVAE"
    assert package.LatentActionModel.__name__ == "LatentActionModel"
    assert package.DynamicsMaskGIT.__name__ == "DynamicsMaskGIT"


def test_jafar_adapted_modules_record_pinned_source_and_changes() -> None:
    source_dir = Path(__file__).parents[1] / "src" / "world_marl" / "jafar"
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
        assert "FLAIROx/jafar" in contents
        assert "5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665" in contents
        assert "path" in contents.lower()
        assert "Integration changes" in contents
