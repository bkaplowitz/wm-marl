"""Training helpers for the LLaDA2.0 block-diffusion arm (arXiv 2512.15745):
the BDLM / conditional-SFT loss (eqs 1, 5, 6), the WSD block-size curriculum
(§4.1), and Weight-Space Merge (§4.3).
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from flow_matching.llada2.paths import (
    complementary_absorbing_pair,
    mask_schedule,
    sample_absorbing_path,
    sample_t_in_bandwidth,
)


def create_llada2_train_state(
    key: jax.Array,
    model: nn.Module,
    learning_rate: float,
    *,
    num_factors: int,
    num_action_tokens: int,
) -> TrainState:
    """Initialize a LLaDA2.0 block-diffusion model and its Adam optimizer.

    The masked model conditions on a clean prefix (``num_factors`` prev-state tokens
    + ``num_action_tokens`` action tokens) and the noisy ``x_t`` block. Init traces
    the *training* forward (``x0_target`` supplied, full block) so every parameter —
    including the clean-copy path — is created; parameters are independent of
    ``block_size`` (it only reshapes the attention-mask contents, never a weight).
    """
    key, init_key = jax.random.split(key)
    tokens = jnp.zeros((1, num_factors), dtype=jnp.int32)
    action_ids = jnp.zeros((1, num_action_tokens), dtype=jnp.int32)
    params = model.init(
        init_key,
        tokens,
        tokens,
        action_ids,
        tokens,
        block_size=num_factors,
    )["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def llada2_bdlm_loss(
    params: Any,
    apply_fn: Any,  # model.apply
    key: jax.Array,
    x0: jax.Array,
    prev_tokens: jax.Array,
    action_ids: jax.Array,
    num_categories: int,
    *,
    block_size,
    mask_schedule_name: str = "linear",
    alpha_min: float = 0.15,
    alpha_max: float = 0.95,
    complementary: bool = True,
    cap_lambda: float = 0.1,
    moe_aux_coeff: float = 0.01,
    mask_noise_std: float = 0.0,
    noise_rng: jax.Array | None = None,
) -> jax.Array:
    """LLaDA2.0 block-diffusion / conditional-SFT loss (eqs 1, 5, 6; §5.1).

    Sample a mask ratio ``t`` in the bandwidth ``[alpha_min, alpha_max]``, absorb
    ``x0`` -> ``[MASK]`` per the schedule, run the block-diffusion forward (clean
    prefix + noisy block + clean copy), and accumulate masked-only integer-label
    cross-entropy reweighted by ``w = -alpha'(t)/(1-alpha(t))`` (eq 1). With
    ``complementary`` the logical-inverse mask is scored too — its members are masked
    at the *keep* rate ``alpha(t)``, so the matching reweight is ``-alpha'(t)/alpha(t)``
    and every position is supervised exactly once (§5.1). Adds the MoE load-balancing
    aux and the CAP confidence loss (eq 6: minimize the entropy of correctly-predicted
    masked tokens; correctness is stop-gradiented).
    """
    alpha, alpha_dt = mask_schedule(mask_schedule_name)
    key_t, key_path = jax.random.split(key)
    t = sample_t_in_bandwidth(key_t, (x0.shape[0], 1), alpha_min, alpha_max)
    neg_dalpha = -alpha_dt(t)  # > 0 for a decreasing keep schedule
    w_primary = neg_dalpha / jnp.maximum(1.0 - alpha(t), 1e-4)  # denom = mask rate
    w_comp = neg_dalpha / jnp.maximum(alpha(t), 1e-4)  # denom = complement mask rate

    if noise_rng is None:
        noise_primary = noise_comp = None
    else:
        noise_primary, noise_comp = jax.random.split(noise_rng)

    def member_loss(x_t, masked, weight, member_noise_rng):
        logits, aux = apply_fn(
            {"params": params},
            x_t,
            prev_tokens,
            action_ids,
            x0,
            block_size=block_size,
            mask_noise_std=mask_noise_std,
            noise_rng=member_noise_rng,
        )
        masked_f = masked.astype(logits.dtype)
        token_ce = optax.softmax_cross_entropy_with_integer_labels(logits, x0)
        sft = jnp.mean(jnp.sum(weight * masked_f * token_ce, axis=-1))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
        correct = jax.lax.stop_gradient(
            (jnp.argmax(logits, axis=-1) == x0).astype(logits.dtype)
        )
        conf = jnp.mean(jnp.sum(masked_f * correct * entropy, axis=-1))
        return sft, conf, aux

    if complementary:
        x_t, x_t_comp, masked, masked_comp = complementary_absorbing_pair(
            key_path, x0, t, num_categories, alpha
        )
        sft_p, conf_p, aux_p = member_loss(x_t, masked, w_primary, noise_primary)
        sft_c, conf_c, aux_c = member_loss(x_t_comp, masked_comp, w_comp, noise_comp)
        sft = 0.5 * (sft_p + sft_c)
        conf = 0.5 * (conf_p + conf_c)
        aux = 0.5 * (aux_p + aux_c)
    else:
        x_t, masked = sample_absorbing_path(key_path, x0, t, num_categories, alpha)
        sft, conf, aux = member_loss(x_t, masked, w_primary, noise_primary)

    return sft + cap_lambda * conf + moe_aux_coeff * aux


@partial(
    jax.jit,
    static_argnames=("num_categories", "mask_schedule_name", "complementary"),
)
def llada2_train_step(
    state: TrainState,
    key: jax.Array,
    x0: jax.Array,
    prev_tokens: jax.Array,
    action_ids: jax.Array,
    num_categories: int,
    *,
    block_size,
    mask_schedule_name: str = "linear",
    alpha_min: float = 0.15,
    alpha_max: float = 0.95,
    complementary: bool = True,
    cap_lambda: float = 0.1,
    moe_aux_coeff: float = 0.01,
    mask_noise_std: float = 0.0,
    noise_rng: jax.Array | None = None,
) -> tuple[TrainState, jax.Array]:
    """One optimizer update for the LLaDA2.0 block-diffusion model.

    ``block_size`` stays a *traced* argument so the WSD curriculum can thread a
    different block size every step through one fused ``scan`` with no recompiles —
    only the mask contents depend on it, not its shape. ``num_categories``,
    ``mask_schedule_name``, and ``complementary`` are static (they pick Python
    branches / schedules / vocab).
    """
    loss, grads = jax.value_and_grad(llada2_bdlm_loss)(
        state.params,
        state.apply_fn,
        key,
        x0,
        prev_tokens,
        action_ids,
        num_categories,
        block_size=block_size,
        mask_schedule_name=mask_schedule_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        complementary=complementary,
        cap_lambda=cap_lambda,
        moe_aux_coeff=moe_aux_coeff,
        mask_noise_std=mask_noise_std,
        noise_rng=noise_rng,
    )
    return state.apply_gradients(grads=grads), loss


def wsd_block_size_schedule(
    step: int,
    total_steps: int,
    *,
    divisors: tuple[int, ...] = (1, 2, 4, 8),
    warmup_frac: float = 0.3,
    stable_frac: float = 0.4,
) -> int:
    """WSD block-size curriculum (§4.1), host-side and returning a plain ``int``.

    Warmup grows the block size through ``divisors``; Stable holds it at the maximum
    (``L_B = L`` -> MDLM, eq 2); Decay shrinks it back for fast block-by-block
    inference. Every returned value is one of ``divisors`` (each a divisor of ``d``),
    non-decreasing in warmup, flat in stable, non-increasing in decay.
    """
    divs = sorted(set(divisors))
    max_block = divs[-1]
    frac = step / max(total_steps - 1, 1)
    stable_end = warmup_frac + stable_frac
    if frac < warmup_frac:
        idx = min(int(frac / max(warmup_frac, 1e-9) * len(divs)), len(divs) - 1)
        return divs[idx]
    if frac < stable_end:
        return max_block
    desc = divs[::-1]
    idx = min(
        int((frac - stable_end) / max(1.0 - stable_end, 1e-9) * len(desc)),
        len(desc) - 1,
    )
    return desc[idx]


def topk_checkpoint_merge(param_trees: list[Any]) -> Any:
    """Weight-Space Merge (§4.3): leaf-wise arithmetic mean of parameter pytrees.

    The caller selects which top-k checkpoints (by validation loss) to pass in; this
    averages them into a single merged parameter tree of identical structure.
    """
    if not param_trees:
        raise ValueError("expected at least one checkpoint to merge")
    return jax.tree_util.tree_map(
        lambda *leaves: sum(leaves) / len(leaves), *param_trees
    )
