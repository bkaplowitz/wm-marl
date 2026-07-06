"""Faithful-LLaDA2.0 arm: paths, block-diffusion mask, loss, WSD, sampler, e2e.

Covers the arXiv 2512.15745 components wired into the world model: absorbing
masking + complementary pair (§5.1), the eq-3 block-diffusion attention mask, the
BDLM loss (eq 1) reweighting and its CAP/MoE-aux terms (eq 6), the WSD block-size
curriculum (§4.1), top-k checkpoint merge (§4.3), the block-by-block sampler (§5.4),
and the world_model.py integration (create -> train-with-WSD -> predict_next).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flow_matching.llada2 import (
    BlockDiffusionTransformer,
    apply_rope,
    block_diffusion_attention_mask,
    complementary_absorbing_pair,
    create_llada2_train_state,
    llada2_bdlm_loss,
    llada2_train_step,
    mask_schedule,
    sample_absorbing_path,
    sample_llada2_block_diffusion,
    topk_checkpoint_merge,
    wsd_block_size_schedule,
)
from world_marl.world_model import (
    LLaDA2WorldModelConfig,
    VectorTransitionBatch,
    _num_factors,
    _unpack_discrete_onehot,
    create_world_model_state,
    predict_next,
    train_world_model_step,
)
from world_marl.world_model_training import fit_world_model_steps

# CoinGame-shaped: d=8 factors, V=9 ([MASK]=9, vocab 10), 2 agents, 4 actions.
D, V, NUM_AGENTS, NUM_ACTIONS, STATE_DIM = 8, 9, 2, 5, 36
UNIFORM_CE = float(D * np.log(V))  # ~17.58


# --------------------------------------------------------------------------- #
# absorbing path + complementary masking (§5.1)
# --------------------------------------------------------------------------- #
def test_absorbing_path_masks_to_mask_token_and_keeps_clean():
    x0 = jax.random.randint(jax.random.PRNGKey(0), (32, D), 0, V, dtype=jnp.int32)
    t = jnp.full((32, 1), 0.5)
    x_t, masked = sample_absorbing_path(jax.random.PRNGKey(1), x0, t, V)
    masked = np.asarray(masked)
    x_t, x0_np = np.asarray(x_t), np.asarray(x0)
    assert np.all(x_t[masked] == V)  # corrupted positions -> [MASK]=V
    assert np.all(x_t[~masked] == x0_np[~masked])  # kept positions unchanged
    assert masked.any() and (~masked).any()  # band actually masks some, keeps some


def test_complementary_pair_covers_every_position_exactly_once():
    x0 = jax.random.randint(jax.random.PRNGKey(2), (32, D), 0, V, dtype=jnp.int32)
    t = jnp.full((32, 1), 0.4)
    x_t, x_t_comp, masked, masked_comp = complementary_absorbing_pair(
        jax.random.PRNGKey(3), x0, t, V
    )
    masked, masked_comp = np.asarray(masked), np.asarray(masked_comp)
    # logical inverse: every position masked in exactly one member of the pair.
    assert np.all(masked ^ masked_comp)
    # each position presents its clean token in exactly one member.
    x_t, x_t_comp, x0_np = np.asarray(x_t), np.asarray(x_t_comp), np.asarray(x0)
    clean_in_primary = x_t == x0_np
    clean_in_comp = x_t_comp == x0_np
    assert np.all(clean_in_primary ^ clean_in_comp)


# --------------------------------------------------------------------------- #
# block-diffusion attention mask (eq 3 / eq 4)
# --------------------------------------------------------------------------- #
def test_block_diffusion_training_mask_structure():
    # prefix=2, response=4, block=2 -> response blocks [0,0,1,1].
    # layout: prefix 0,1 | x_t 2,3,4,5 | x_0 6,7,8,9 (L=10).
    m = np.asarray(block_diffusion_attention_mask(2, 2, 4, include_clean_copy=True))
    assert m.shape == (10, 10)
    # prefix attends prefix only.
    assert m[0, 0] and m[0, 1] and not m[0, 2:].any()
    # x_t block 0 (row 2): prefix + own noisy block; NO earlier clean (none); no later.
    assert list(np.where(m[2])[0]) == [0, 1, 2, 3]
    # x_t block 1 (row 4): prefix + own noisy block (4,5) + earlier clean block (6,7).
    assert list(np.where(m[4])[0]) == [0, 1, 4, 5, 6, 7]
    # x_0 (clean) never attends x_t (noisy): row 6 has nothing in [2..5].
    assert not m[6, 2:6].any()
    # x_0 block 0 (row 6): prefix + own clean block only.
    assert list(np.where(m[6])[0]) == [0, 1, 6, 7]
    # x_0 block 1 (row 8): prefix + clean blocks <= 1 (block-causal).
    assert list(np.where(m[8])[0]) == [0, 1, 6, 7, 8, 9]


def test_block_diffusion_inference_mask_is_block_causal():
    # include_clean_copy=False -> drop the x_0 copy; L = prefix + response = 6.
    m = np.asarray(block_diffusion_attention_mask(2, 2, 4, include_clean_copy=False))
    assert m.shape == (6, 6)
    assert list(np.where(m[2])[0]) == [0, 1, 2, 3]  # block 0: prefix + own block
    assert list(np.where(m[4])[0]) == [0, 1, 2, 3, 4, 5]  # block 1: prefix + <= block


def test_block_diffusion_traced_block_size_keeps_shape():
    # block_size may be a traced scalar; only contents change, never shape L.
    m = jax.jit(
        lambda bs: block_diffusion_attention_mask(2, bs, 4, include_clean_copy=True)
    )(jnp.int32(4))
    assert np.asarray(m).shape == (10, 10)


def test_doc_level_mask_blocks_cross_doc_attention():
    # eq-4: distinct doc_ids must zero out cross-document attention the eq-3 mask
    # otherwise allows. The coin-game default is single-doc, so this is the only
    # check that exercises the restriction (direction, not mere presence).
    base = np.asarray(block_diffusion_attention_mask(2, 2, 4, include_clean_copy=True))
    length = base.shape[0]
    doc_ids = jnp.asarray(
        [0] * (length // 2) + [1] * (length - length // 2), dtype=jnp.int32
    )
    restricted = np.asarray(
        block_diffusion_attention_mask(
            2, 2, 4, include_clean_copy=True, doc_ids=doc_ids
        )
    )
    same_doc = np.asarray(doc_ids)[:, None] == np.asarray(doc_ids)[None, :]
    assert np.all(restricted <= base)  # restriction only removes edges, never adds
    assert np.all(~restricted | same_doc)  # every surviving edge is intra-document
    assert np.any(base & ~same_doc)  # baseline DID allow cross-doc (sanity)
    assert np.any(base & ~same_doc & ~restricted)  # ...and eq-4 removed it


# --------------------------------------------------------------------------- #
# RoPE (+YaRN) position rotation (§ backbone)
# --------------------------------------------------------------------------- #
def test_rope_is_a_genuine_position_rotation():
    # RoPE must be a real position-dependent rotation, not a no-op: position 0 is
    # identity, nonzero positions actually move the vector, the per-token norm is
    # preserved (orthogonal), and the SAME vector at different positions diverges.
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2, 8))  # (B, L, H, head_dim)
    out = np.asarray(apply_rope(x, jnp.arange(4)))
    x_np = np.asarray(x)
    assert np.allclose(out[:, 0], x_np[:, 0], atol=1e-5)  # pos 0 -> angle 0 -> identity
    assert not np.allclose(out[:, 1], x_np[:, 1], atol=1e-3)  # pos 1 actually rotates
    assert np.allclose(  # rotation preserves the per-token norm
        np.linalg.norm(out, axis=-1), np.linalg.norm(x_np, axis=-1), atol=1e-4
    )
    same = jnp.broadcast_to(x[:, :1], (1, 4, 2, 8))  # one vector at every position
    rot = np.asarray(apply_rope(same, jnp.arange(4)))
    assert not np.allclose(rot[:, 1], rot[:, 2], atol=1e-3)  # positions are distinct


# --------------------------------------------------------------------------- #
# noise schedule + WSD curriculum (§4.1)
# --------------------------------------------------------------------------- #
def test_mask_schedule_endpoints_and_nonlinearity():
    for name in ("linear", "cosine"):
        alpha, _ = mask_schedule(name)
        np.testing.assert_allclose(float(alpha(jnp.array(0.0))), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(alpha(jnp.array(1.0))), 0.0, atol=1e-6)
    # cosine reweight differs from linear's constant alpha'=-1 mid-schedule.
    _, lin_dt = mask_schedule("linear")
    _, cos_dt = mask_schedule("cosine")
    assert float(lin_dt(jnp.array(0.5))) != float(cos_dt(jnp.array(0.5)))


def test_wsd_schedule_grows_full_shrinks_and_divides_d():
    divisors = tuple(s for s in range(1, D + 1) if D % s == 0)  # (1,2,4,8)
    total = 100
    sched = [wsd_block_size_schedule(s, total, divisors=divisors) for s in range(total)]
    assert all(b in divisors for b in sched)  # every block size divides d
    assert max(sched) == D  # stable phase reaches the full sequence (MDLM)
    assert sched[0] <= sched[total // 2]  # warmup does not shrink
    assert sched[-1] <= sched[total // 2]  # decay does not grow


# --------------------------------------------------------------------------- #
# checkpoint merge (§4.3 WSM)
# --------------------------------------------------------------------------- #
def test_topk_checkpoint_merge_is_leafwise_mean():
    a = {"w": jnp.ones((3, 2)), "b": 2.0 * jnp.ones((2,))}
    c = {"w": 3.0 * jnp.ones((3, 2)), "b": 4.0 * jnp.ones((2,))}
    merged = topk_checkpoint_merge([a, c])
    np.testing.assert_allclose(np.asarray(merged["w"]), 2.0)  # (1+3)/2
    np.testing.assert_allclose(np.asarray(merged["b"]), 3.0)  # (2+4)/2
    with pytest.raises(ValueError):
        topk_checkpoint_merge([])


# --------------------------------------------------------------------------- #
# BDLM loss (eq 1) + CAP / MoE-aux (eq 6)
# --------------------------------------------------------------------------- #
def _make_model(num_experts=4, expert_top_k=2):
    return BlockDiffusionTransformer(
        num_categories=V,
        num_actions=NUM_ACTIONS,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_hidden_dim=64,
        num_experts=num_experts,
        expert_top_k=expert_top_k,
    )


def _make_state(key, **kw):
    model = _make_model(**kw)
    return create_llada2_train_state(
        key, model, 1e-3, num_factors=D, num_action_tokens=NUM_AGENTS
    )


def _make_tokens(key, batch=16):
    k1, k2, k3 = jax.random.split(key, 3)
    x0 = jax.random.randint(k1, (batch, D), 0, V, dtype=jnp.int32)
    prev = jax.random.randint(k2, (batch, D), 0, V, dtype=jnp.int32)
    actions = jax.random.randint(
        k3, (batch, NUM_AGENTS), 0, NUM_ACTIONS, dtype=jnp.int32
    )
    return x0, prev, actions


def test_init_loss_near_uniform_ce_not_double():
    # Regression guard for the complementary sum-vs-average bug: an unbiased eq-1
    # NLL bound starts at ~d*ln(V); SUMMING the antithetic pair would land at ~2x.
    state = _make_state(jax.random.PRNGKey(0))
    x0, prev, actions = _make_tokens(jax.random.PRNGKey(1))
    loss = float(
        llada2_bdlm_loss(
            state.params,
            state.apply_fn,
            jax.random.PRNGKey(2),
            x0,
            prev,
            actions,
            V,
            block_size=4,
            complementary=True,
        )
    )
    assert np.isfinite(loss)
    assert 0.5 * UNIFORM_CE < loss < 1.6 * UNIFORM_CE, loss  # ~17.6, well below 2x=35


def test_loss_counts_only_masked_positions():
    # Isolate the masked-only sft term (experts=1 -> aux=0, cap_lambda=0). Pinning the
    # band to a near-zero vs near-one mask rate makes the loss scale with how much is
    # masked: ~0 when nothing is masked, large when everything is.
    state = _make_state(jax.random.PRNGKey(3), num_experts=1, expert_top_k=1)
    x0, prev, actions = _make_tokens(jax.random.PRNGKey(4))

    def loss_at(band, key):
        return float(
            llada2_bdlm_loss(
                state.params,
                state.apply_fn,
                key,
                x0,
                prev,
                actions,
                V,
                block_size=D,
                complementary=False,
                cap_lambda=0.0,
                alpha_min=band,
                alpha_max=band + 1e-3,
            )
        )

    low = loss_at(0.001, jax.random.PRNGKey(5))  # ~no positions masked
    high = loss_at(0.95, jax.random.PRNGKey(6))  # almost all positions masked
    assert low < 1e-2  # unmasked positions contribute nothing
    assert high > 5.0  # masked positions drive the loss
    assert low < high


def test_cap_and_moe_aux_are_finite_and_contribute():
    state = _make_state(jax.random.PRNGKey(7), num_experts=4, expert_top_k=2)
    x0, prev, actions = _make_tokens(jax.random.PRNGKey(8))
    key = jax.random.PRNGKey(9)

    def loss(cap_lambda, moe_aux_coeff):
        return float(
            llada2_bdlm_loss(
                state.params,
                state.apply_fn,
                key,
                x0,
                prev,
                actions,
                V,
                block_size=4,
                complementary=True,
                cap_lambda=cap_lambda,
                moe_aux_coeff=moe_aux_coeff,
            )
        )

    full = loss(0.1, 0.01)
    no_cap = loss(0.0, 0.01)
    no_aux = loss(0.1, 0.0)
    assert all(np.isfinite(v) for v in (full, no_cap, no_aux))
    # Sign matters, not just presence: CAP (eq 6) is +lambda*entropy on correctly
    # predicted masked tokens (>=0), so it must *raise* the loss it penalizes -- a
    # flipped sign would lower it and still pass a mere `!=` check.
    assert full > no_cap  # CAP confidence penalty (eq 6) adds a non-negative term
    assert full > no_aux  # MoE load-balancing aux (>=0) adds a non-negative term


def test_nonlinear_schedule_reweight_differs_from_linear():
    state = _make_state(jax.random.PRNGKey(10), num_experts=1, expert_top_k=1)
    x0, prev, actions = _make_tokens(jax.random.PRNGKey(11))
    key = jax.random.PRNGKey(12)
    args = (state.params, state.apply_fn, key, x0, prev, actions, V)
    lin = float(llada2_bdlm_loss(*args, block_size=D, mask_schedule_name="linear"))
    cos = float(llada2_bdlm_loss(*args, block_size=D, mask_schedule_name="cosine"))
    assert np.isfinite(lin) and np.isfinite(cos)
    assert lin != cos


def test_llada2_train_step_decreases_loss():
    state = _make_state(jax.random.PRNGKey(13), num_experts=4, expert_top_k=2)
    x0, prev, actions = _make_tokens(jax.random.PRNGKey(14))
    key = jax.random.PRNGKey(15)
    losses = []
    for _ in range(60):
        key, sk = jax.random.split(key)
        state, loss = llada2_train_step(
            state, sk, x0, prev, actions, V, block_size=4, complementary=True
        )
        losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]


# --------------------------------------------------------------------------- #
# block-by-block hybrid-confidence sampler (§5.4)
# --------------------------------------------------------------------------- #
def test_sampler_output_valid_and_no_mask_leftover():
    state = _make_state(jax.random.PRNGKey(16))
    pk, ak = jax.random.split(jax.random.PRNGKey(17))
    prev = jax.random.randint(pk, (8, D), 0, V, dtype=jnp.int32)
    actions = jax.random.randint(ak, (8, NUM_AGENTS), 0, NUM_ACTIONS, dtype=jnp.int32)
    # block_size/steps_per_block are static inference settings (§5.4): they drive
    # range/floor-div/quota, so jit must treat them (and the token counts) as static.
    sampler = jax.jit(
        sample_llada2_block_diffusion,
        static_argnums=(0,),
        static_argnames=(
            "num_factors",
            "num_categories",
            "block_size",
            "steps_per_block",
        ),
    )
    out = sampler(
        state.apply_fn,
        state.params,
        jax.random.PRNGKey(18),
        prev,
        actions,
        num_factors=D,
        num_categories=V,
        block_size=4,
        steps_per_block=4,
        confidence_threshold=0.9,
    )
    out = np.asarray(out)
    assert out.shape == (8, D)
    assert out.min() >= 0 and out.max() < V  # tokens in [0, V); no [MASK]=V leftover


# --------------------------------------------------------------------------- #
# world_model.py integration: create -> train (WSD) -> predict_next
# --------------------------------------------------------------------------- #
def _wm_config(**overrides):
    base = LLaDA2WorldModelConfig(
        state_dim=STATE_DIM,
        num_agents=NUM_AGENTS,
        action_dim=NUM_ACTIONS,
        num_categories=V,
        model_dim=32,
        num_heads=4,
        ffn_hidden_dims=(64, 64),
        num_experts=4,
        expert_top_k=2,
        block_size=4,
        steps_per_block=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _wm_batch(key, config, batch=32):
    d = _num_factors(config)
    k1, k2, k3 = jax.random.split(key, 3)
    states = _unpack_discrete_onehot(
        jax.random.randint(k1, (batch, d), 0, V, dtype=jnp.int32), config
    )
    next_states = _unpack_discrete_onehot(
        jax.random.randint(k2, (batch, d), 0, V, dtype=jnp.int32), config
    )
    actions = jax.random.randint(
        k3, (batch, NUM_AGENTS), 0, NUM_ACTIONS, dtype=jnp.int32
    )
    return VectorTransitionBatch(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=jnp.zeros((batch, NUM_AGENTS)),
        dones=jnp.zeros((batch, NUM_AGENTS)),
    )


def test_world_model_llada2_train_then_predict_one_hot():
    config = _wm_config()
    key = jax.random.PRNGKey(20)
    key, ik, bk = jax.random.split(key, 3)
    state = create_world_model_state(ik, config)
    batch = _wm_batch(bk, config)

    losses = []
    for i in range(60):
        key, tk, nk = jax.random.split(key, 3)
        bs = jnp.int32([1, 2, 4, 8][min(i // 15, 3)])  # traced WSD-style block size
        state, loss = train_world_model_step(
            state, tk, batch, config, block_size=bs, mask_noise_std=0.0, noise_rng=nk
        )
        losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]

    key, pk = jax.random.split(key)
    pred = predict_next(state, pk, batch.states, batch.actions, config)
    assert pred.shape == (32, NUM_AGENTS, STATE_DIM)
    # each (agent, channel) block of V must be exactly one-hot.
    grid = pred.reshape((32, NUM_AGENTS, V, STATE_DIM // V))
    sums = jnp.sum(grid, axis=2)
    np.testing.assert_allclose(np.asarray(sums), 1.0, atol=1e-5)


@pytest.mark.parametrize("wsd_enabled", [True, False])
def test_fit_world_model_steps_wsd_toggle(wsd_enabled):
    config = _wm_config(wsd_enabled=wsd_enabled)
    key = jax.random.PRNGKey(21)
    key, ik, bk, fk = jax.random.split(key, 4)
    state = create_world_model_state(ik, config)
    batch = _wm_batch(bk, config)
    state, _, final_loss, history = fit_world_model_steps(
        state, fk, batch, config, steps=60
    )
    assert history.shape[0] == 60
    assert jnp.isfinite(history).all()
    assert float(final_loss) < float(history[0])


def test_fit_world_model_steps_checkpoint_merge():
    config = _wm_config(wsd_merge_k=3)
    key = jax.random.PRNGKey(22)
    key, ik, bk, fk = jax.random.split(key, 4)
    state = create_world_model_state(ik, config)
    batch = _wm_batch(bk, config)
    state, _, _, history = fit_world_model_steps(state, fk, batch, config, steps=60)
    # segmented histories concatenate to `steps`; merged params stay finite.
    assert history.shape[0] == 60
    assert jnp.isfinite(history).all()
    assert all(
        bool(jnp.all(jnp.isfinite(p))) for p in jax.tree_util.tree_leaves(state.params)
    )
