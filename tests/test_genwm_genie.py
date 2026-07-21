"""Genie-style transformer VQ-VAE tokenizer whose codes are llada2's targets."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from world_marl.genwm import (
    CodebookTokenizer,
    GenieTokenizer,
    GenWMConfig,
    PPOConfig,
    create_genie_state,
    create_genwm_state,
    create_head_state,
    create_policy_state,
    decode_tokens,
    encode_tokens,
    fit_quantile_tokenizer,
    genie_train_step,
    genwm_predict_next,
    genwm_train_step,
    make_genie_encode,
)
from world_marl.scripts.train_single_genwm import (
    _make_scan_action_fns,
    _resolve_genie,
    parse_args,
)

OBS_DIM = 5
CODE_DIM = 3
CODEBOOK_SIZE = 7


class _ScanAdapterStub:
    action_dim = 2
    action_low = np.array([-1.0, -1.0], dtype=np.float32)
    action_high = np.array([1.0, 1.0], dtype=np.float32)
    observation_shape = (OBS_DIM,)

    def scan_rollout(self):
        raise NotImplementedError


def _tiny_module() -> GenieTokenizer:
    return GenieTokenizer(
        obs_dim=OBS_DIM,
        codebook_size=CODEBOOK_SIZE,
        code_dim=CODE_DIM,
        model_dim=16,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
    )


def _wide_config() -> GenWMConfig:
    return GenWMConfig(
        arm="llada2",
        obs_dim=OBS_DIM,
        action_dim=2,
        action_mode="continuous",
        obs_bins=CODEBOOK_SIZE,
        action_bins=4,
        code_dim=CODE_DIM,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        mlp_ratio=2,
        block_size=1,
        steps_per_block=1,
    )


def _observations(batch: int, seed: int) -> jax.Array:
    return jnp.asarray(
        np.random.default_rng(seed).normal(size=(batch, OBS_DIM)), dtype=jnp.float32
    )


def test_codebook_tokenizer_roundtrip():
    codebook = jnp.asarray(
        np.random.default_rng(0).normal(size=(CODEBOOK_SIZE, CODE_DIM)),
        dtype=jnp.float32,
    )
    tokenizer = CodebookTokenizer(codebook=codebook)
    assert tokenizer.num_bins == CODEBOOK_SIZE
    assert tokenizer.code_dim == CODE_DIM
    ids = jnp.asarray(
        np.random.default_rng(1).integers(0, CODEBOOK_SIZE, size=(6, OBS_DIM)),
        dtype=jnp.int32,
    )
    values = decode_tokens(tokenizer, ids)
    assert values.shape == (6, OBS_DIM * CODE_DIM)
    np.testing.assert_allclose(
        np.asarray(values).reshape(6, OBS_DIM, CODE_DIM),
        np.asarray(codebook)[np.asarray(ids)],
        rtol=1e-6,
    )
    recovered = encode_tokens(tokenizer, values)
    assert recovered.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(recovered), np.asarray(ids))


def test_genwm_config_code_dim():
    base = GenWMConfig(
        arm="llada2", obs_dim=OBS_DIM, action_dim=2, action_mode="continuous"
    )
    assert base.code_dim == 1
    assert base.float_obs_dim == OBS_DIM
    wide = _wide_config()
    assert wide.float_obs_dim == OBS_DIM * CODE_DIM
    assert wide.cond_dim == OBS_DIM * CODE_DIM + 2


def test_policy_and_head_use_float_obs_dim():
    config = _wide_config()
    policy_state = create_policy_state(jax.random.PRNGKey(0), config, PPOConfig())
    head_state = create_head_state(jax.random.PRNGKey(1), config)
    latents = jnp.zeros((2, OBS_DIM * CODE_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, 2), dtype=jnp.float32)
    policy, values = policy_state.apply_fn({"params": policy_state.params}, latents)
    assert values.shape == (2,)
    assert policy.mode().shape == (2, 2)
    reward, continue_logit = head_state.apply_fn(
        {"params": head_state.params}, latents, actions
    )
    assert reward.shape == (2,)
    assert continue_logit.shape == (2,)


def test_genie_encode_outputs_codebook_rows_with_input_gradient():
    module = _tiny_module()
    state = create_genie_state(jax.random.PRNGKey(0), module, learning_rate=1e-3)
    encode = make_genie_encode(module)
    observations = _observations(4, 2)
    latents = encode(state.params, observations)
    assert latents.shape == (4, OBS_DIM * CODE_DIM)
    codebook = np.asarray(state.params["codebook"])
    rows = np.asarray(latents).reshape(4, OBS_DIM, CODE_DIM)
    distances = np.linalg.norm(rows[..., None, :] - codebook[None, None], axis=-1)
    np.testing.assert_allclose(distances.min(axis=-1), 0.0, atol=1e-5)
    gradient = jax.grad(
        lambda obs: jnp.sum(
            module.apply({"params": state.params}, obs, method=GenieTokenizer.encode)
        )
    )(observations)
    assert np.any(np.abs(np.asarray(gradient)) > 0.0)


def test_genie_train_step_reduces_loss_and_updates_all_parts():
    module = _tiny_module()
    state = create_genie_state(jax.random.PRNGKey(0), module, learning_rate=1e-3)
    observations = _observations(128, 3)
    initial_params = state.params
    _, initial_metrics = genie_train_step(state, observations)
    metrics = initial_metrics
    for _ in range(300):
        state, metrics = genie_train_step(state, observations)
    assert np.isfinite(float(metrics["genie_total_loss"]))
    assert float(metrics["genie_recon_loss"]) < 0.5 * float(
        initial_metrics["genie_recon_loss"]
    )
    for name in ("encoder", "decoder", "codebook"):
        before = jax.tree_util.tree_leaves(initial_params[name])
        after = jax.tree_util.tree_leaves(state.params[name])
        assert any(
            not np.allclose(np.asarray(a), np.asarray(b))
            for a, b in zip(before, after, strict=True)
        )


def test_genwm_llada2_with_codebook_tokenizer():
    config = _wide_config()
    codebook = jnp.asarray(
        np.random.default_rng(4).normal(size=(CODEBOOK_SIZE, CODE_DIM)),
        dtype=jnp.float32,
    )
    tokenizer = CodebookTokenizer(codebook=codebook)
    wm_state = create_genwm_state(jax.random.PRNGKey(0), config)
    ids = jnp.asarray(
        np.random.default_rng(5).integers(0, CODEBOOK_SIZE, size=(8, OBS_DIM)),
        dtype=jnp.int32,
    )
    next_ids = jnp.asarray(
        np.random.default_rng(6).integers(0, CODEBOOK_SIZE, size=(8, OBS_DIM)),
        dtype=jnp.int32,
    )
    observations = decode_tokens(tokenizer, ids)
    actions = jnp.asarray(
        np.random.default_rng(7).uniform(-1.0, 1.0, size=(8, 2)), dtype=jnp.float32
    )
    action_tokenizer = fit_quantile_tokenizer(np.asarray(actions), 4)
    wm_state, loss = genwm_train_step(
        wm_state,
        jax.random.PRNGKey(1),
        observations,
        actions,
        decode_tokens(tokenizer, next_ids),
        tokenizer,
        action_tokenizer,
        config,
    )
    assert np.isfinite(float(loss))
    predicted = genwm_predict_next(
        wm_state,
        jax.random.PRNGKey(2),
        observations,
        actions,
        tokenizer,
        action_tokenizer,
        config,
    )
    assert predicted.shape == (8, OBS_DIM * CODE_DIM)
    rows = np.asarray(predicted).reshape(8, OBS_DIM, CODE_DIM)
    distances = np.linalg.norm(
        rows[..., None, :] - np.asarray(codebook)[None, None], axis=-1
    )
    np.testing.assert_allclose(distances.min(axis=-1), 0.0, atol=1e-5)


def test_scan_action_fns_thread_encoder_params():
    module = _tiny_module()
    state_a = create_genie_state(jax.random.PRNGKey(0), module, learning_rate=1e-3)
    state_b = create_genie_state(jax.random.PRNGKey(1), module, learning_rate=1e-3)
    encode = make_genie_encode(module)
    policy_state = create_policy_state(
        jax.random.PRNGKey(2), _wide_config(), PPOConfig()
    )
    fns = _make_scan_action_fns(
        _ScanAdapterStub(), "continuous", encode_fn=encode, encoder_in_state=True
    )
    observations = jnp.asarray(
        np.random.default_rng(4).normal(size=(3, OBS_DIM)) * 2.0, dtype=jnp.float32
    )
    actions_a, _, values_a, _ = fns["mode"](
        (policy_state, state_a.params), jax.random.PRNGKey(3), observations
    )
    policy, expected_values = policy_state.apply_fn(
        {"params": policy_state.params}, encode(state_a.params, observations)
    )
    np.testing.assert_allclose(
        np.asarray(actions_a), np.asarray(policy.mode()), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(values_a), np.asarray(expected_values), rtol=1e-6
    )
    _, _, values_b, _ = fns["mode"](
        (policy_state, state_b.params), jax.random.PRNGKey(3), observations
    )
    assert not np.allclose(np.asarray(values_a), np.asarray(values_b))
    random_actions, _, _, _ = fns["random"](None, jax.random.PRNGKey(5), observations)
    assert random_actions.shape == (3, 2)


def _genie_args(*extra: str) -> argparse.Namespace:
    return parse_args(
        ["--env", "brax:reacher", "--arm", "llada2", "--tokenizer", "genie", *extra]
    )


def test_parse_args_genie_flag_defaults():
    args = _genie_args()
    assert args.tokenizer == "genie"
    assert args.genie_code_dim == 16
    assert args.genie_model_dim == 64
    assert args.genie_heads == 4
    assert args.genie_layers == 2
    assert args.genie_learning_rate == pytest.approx(3e-4)
    assert args.genie_train_steps == 2000
    assert args.genie_online_train_steps == 500
    assert parse_args(["--env", "brax:reacher", "--arm", "llada2"]).tokenizer == (
        "quantile"
    )


def test_resolve_genie_returns_none_for_quantile():
    args = parse_args(["--env", "brax:reacher", "--arm", "llada2"])
    assert _resolve_genie(args, _ScanAdapterStub()) is None


def test_resolve_genie_builds_module_from_args():
    args = _genie_args(
        "--genie-code-dim", "4", "--genie-model-dim", "32", "--obs-bins", "16"
    )
    module = _resolve_genie(args, _ScanAdapterStub())
    assert isinstance(module, GenieTokenizer)
    assert module.obs_dim == OBS_DIM
    assert module.codebook_size == 16
    assert module.code_dim == 4
    assert module.model_dim == 32


def test_resolve_genie_rejects_non_token_arms():
    for arm in ("model-free", "continuous-transformer"):
        args = parse_args(
            ["--env", "brax:reacher", "--arm", arm, "--tokenizer", "genie"]
        )
        with pytest.raises(ValueError, match="token arm"):
            _resolve_genie(args, _ScanAdapterStub())


def test_resolve_genie_rejects_latent_encoder_combo(tmp_path):
    args = _genie_args("--latent-encoder", str(tmp_path))
    with pytest.raises(ValueError, match="latent-encoder"):
        _resolve_genie(args, _ScanAdapterStub())


def test_resolve_genie_rejects_loop_adapters():
    class _LoopAdapter:
        observation_shape = (OBS_DIM,)

    with pytest.raises(ValueError, match="scan_rollout"):
        _resolve_genie(_genie_args(), _LoopAdapter())
