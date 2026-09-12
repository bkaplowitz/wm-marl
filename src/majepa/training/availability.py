"""Read-only diagnostics for probability quality and liveness-gated support."""

import jax
import jax.numpy as jnp


def availability_metrics(
    logits, target, gated_mask, predicted_alive, actual_alive, valid
):
    """Factual-action report; no sampling or changes to the simulator's state.

    Raw head probabilities are evaluated separately from hard threshold support
    and its liveness gate. Expected sampled error includes empty-mask fallback.
    Reliability bins carry counts so consumers can combine them correctly.
    """
    logits = logits.astype(jnp.float32)
    target = target.astype(bool)
    weight = jnp.broadcast_to(valid[..., None], target.shape).astype(jnp.float32)
    probability = jax.nn.sigmoid(logits)
    label = target.astype(jnp.float32)
    nll = jax.nn.softplus(logits) - label * logits

    def mean(value, selected=weight):
        return (value.astype(jnp.float32) * selected).sum() / jnp.maximum(
            selected.sum(), 1
        )

    legal = weight * label
    illegal = weight * (1 - label)
    raw = probability >= 0.5
    noop = jnp.zeros_like(raw).at[..., 0].set(True)
    supported = jnp.where(raw.any(-1, keepdims=True), raw, noop)
    alive = predicted_alive >= 0.5
    # Independent Bernoulli draws followed by the exact deployed no-op fallback.
    empty_probability = jnp.prod(1 - probability, axis=-1, keepdims=True)
    sampled_probability = probability + noop * empty_probability
    sampled_probability = jnp.where(alive[..., None], sampled_probability, noop)
    live_valid = valid.astype(jnp.float32) * actual_alive.astype(jnp.float32)
    metrics = {
        "raw_mask_nll": mean(nll),
        "raw_mask_brier": mean((probability - label) ** 2),
        "raw_mask_false_positive": mean(raw, illegal),
        "raw_mask_false_negative": mean(~raw, legal),
        "gated_mask_false_negative": mean(~gated_mask, legal),
        "mask_fn_added_by_death": mean(supported & ~gated_mask, legal),
        "sampled_mask_expected_error": mean(
            jnp.where(target, 1 - sampled_probability, sampled_probability)
        ),
        "sampled_mask_empty_probability": mean(
            empty_probability[..., 0], valid.astype(jnp.float32)
        ),
        "false_death_rate": mean(~alive, live_valid),
        "mask_event_count": weight.sum(),
        "mask_legal_count": legal.sum(),
        "mask_illegal_count": illegal.sum(),
        "actual_alive_count": live_valid.sum(),
    }
    # SMAC action-index groups; index >=6 includes both attacks and healing.
    for name, sl in (
        ("noop", slice(0, 1)),
        ("stop", slice(1, 2)),
        ("move", slice(2, 6)),
        ("target", slice(6, None)),
    ):
        selected = weight[..., sl]
        metrics[f"raw_mask_{name}_nll"] = mean(nll[..., sl], selected)
        metrics[f"raw_mask_{name}_brier"] = mean(
            (probability[..., sl] - label[..., sl]) ** 2, selected
        )
        metrics[f"raw_mask_{name}_count"] = selected.sum()
    for index in range(10):
        selected = weight * (
            (probability >= index / 10)
            & (probability < (index + 1) / 10 if index < 9 else probability <= 1)
        )
        metrics[f"mask_reliability_bin{index}_count"] = selected.sum()
        metrics[f"mask_reliability_bin{index}_prediction"] = mean(probability, selected)
        metrics[f"mask_reliability_bin{index}_frequency"] = mean(label, selected)
    return jax.lax.stop_gradient(metrics)
