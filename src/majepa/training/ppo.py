"""Clipped PPO objectives over frozen MA-JEPA imagined trajectories."""

from __future__ import annotations

import jax
import jax.numpy as jnp


f32 = jnp.float32
sg = jax.lax.stop_gradient


def scheduled_entropy_coefficient(
    environment_step,
    *,
    initial,
    final,
    decay_steps,
    schedule="cosine",
):
    """Anneal the PPO entropy bonus against total environment transitions."""

    initial = float(initial)
    final = float(final)
    decay_steps = int(decay_steps)
    schedule = str(schedule)
    if initial < 0.0 or final < 0.0:
        raise ValueError("entropy coefficients must be nonnegative")
    if decay_steps < 1:
        raise ValueError("entropy decay_steps must be positive")
    progress = jnp.clip(
        jnp.asarray(environment_step, f32) / float(decay_steps), 0.0, 1.0
    )
    if schedule == "linear":
        weight = 1.0 - progress
    elif schedule == "cosine":
        weight = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    else:
        raise ValueError("entropy schedule must be 'linear' or 'cosine'")
    return jnp.asarray(final, f32) + (initial - final) * weight


def _same_shape(name, *values):
    shapes = {tuple(jnp.shape(value)) for value in values}
    if len(shapes) != 1:
        raise ValueError(f"{name} tensors must have one shape, got {sorted(shapes)}")


def masked_weighted_mean(value, valid, weight=None):
    """Average over valid imagined decisions with optional trajectory weights."""

    value = jnp.asarray(value, f32)
    valid = jnp.asarray(valid, bool)
    if value.shape != valid.shape:
        raise ValueError(
            f"value and validity shapes differ: {value.shape} and {valid.shape}"
        )
    if weight is None:
        weight = jnp.ones_like(value)
    weight = jnp.asarray(weight, f32)
    if weight.shape != value.shape:
        raise ValueError(
            f"weight and value shapes differ: {weight.shape} and {value.shape}"
        )
    selected = valid.astype(f32) * sg(weight)
    return (value * selected).sum() / jnp.maximum(selected.sum(), 1.0)


def generalized_advantage_estimate(
    reward,
    continuation,
    target_value,
    state_valid,
    *,
    lam=0.95,
):
    """Build frozen GAE targets from one JEPA imagination.

    Inputs are state-aligned ``[B, H + 1]`` arrays. ``reward[:, t + 1]`` and
    ``continuation[:, t + 1]`` describe the transition leaving decision state
    ``t``. Per-agent death is represented by ``state_valid`` and therefore
    cuts bootstrapping even when the team episode itself continues.
    """

    reward = jnp.asarray(reward, f32)
    continuation = jnp.asarray(continuation, f32)
    target_value = jnp.asarray(target_value, f32)
    state_valid = jnp.asarray(state_valid, bool)
    _same_shape(
        "PPO reward, continuation, target value, and state validity",
        reward,
        continuation,
        target_value,
        state_valid,
    )
    if reward.ndim != 2 or reward.shape[1] < 2:
        raise ValueError(
            "PPO trajectories must be rank-2 with at least one transition, got "
            f"{reward.shape}"
        )
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("PPO lambda must be in [0, 1]")

    decision_valid = state_valid[:, :-1]
    next_valid = state_valid[:, 1:]
    discount = continuation[:, 1:] * next_valid.astype(f32)
    delta = reward[:, 1:] + discount * target_value[:, 1:] - target_value[:, :-1]

    def reverse_step(carry, inputs):
        current_delta, current_discount, current_valid = inputs
        advantage = current_delta + lam * current_discount * carry
        advantage = jnp.where(current_valid, advantage, 0.0)
        return advantage, advantage

    _, reversed_advantage = jax.lax.scan(
        reverse_step,
        jnp.zeros_like(delta[:, 0]),
        (
            delta[:, ::-1].T,
            discount[:, ::-1].T,
            decision_valid[:, ::-1].T,
        ),
    )
    advantage = reversed_advantage[::-1].T
    returns = advantage + target_value[:, :-1]

    if discount.shape[1] == 1:
        trajectory_weight = jnp.ones_like(discount)
    else:
        trajectory_weight = jnp.concatenate(
            [
                jnp.ones_like(discount[:, :1]),
                jnp.cumprod(discount[:, :-1], axis=1),
            ],
            axis=1,
        )
    trajectory_weight *= decision_valid.astype(f32)
    return (
        sg(returns),
        sg(advantage),
        sg(decision_valid),
        sg(trajectory_weight),
    )


def normalize_advantage(advantage, valid, weight=None, epsilon=1e-8):
    """Normalize one immutable PPO advantage batch over effective decisions."""

    advantage = jnp.asarray(advantage, f32)
    valid = jnp.asarray(valid, bool)
    if advantage.shape != valid.shape:
        raise ValueError(
            "PPO advantage and validity must have identical shapes, got "
            f"{advantage.shape} and {valid.shape}"
        )
    if weight is None:
        weight = jnp.ones_like(advantage)
    weight = jnp.asarray(weight, f32)
    if weight.shape != advantage.shape:
        raise ValueError(
            "PPO advantage weight must match the advantage shape, got "
            f"{weight.shape} and {advantage.shape}"
        )
    selected = valid.astype(f32) * weight
    count = jnp.maximum(selected.sum(), 1.0)
    mean = (advantage * selected).sum() / count
    variance = (jnp.square(advantage - mean) * selected).sum() / count
    normalized = (advantage - mean) / jnp.sqrt(variance + float(epsilon))
    effective = valid & (weight > 0.0)
    return sg(jnp.where(effective, normalized, 0.0))


def clipped_policy_objective(
    new_logits,
    old_logits,
    action,
    advantage,
    valid,
    trajectory_weight,
    *,
    clip_epsilon=0.2,
    entropy_coefficient=1e-2,
    normalize_entropy=False,
):
    """Categorical clipped PPO on the exact policy support used in imagination."""

    new_logits = jnp.asarray(new_logits, f32)
    old_logits = sg(jnp.asarray(old_logits, f32))
    action = sg(jnp.asarray(action, jnp.int32))
    advantage = sg(jnp.asarray(advantage, f32))
    valid = sg(jnp.asarray(valid, bool))
    trajectory_weight = sg(jnp.asarray(trajectory_weight, f32))
    if new_logits.shape != old_logits.shape:
        raise ValueError(
            "PPO old and new logits must have identical shapes, got "
            f"{old_logits.shape} and {new_logits.shape}"
        )
    if new_logits.shape[:-1] != action.shape:
        raise ValueError(
            "PPO action shape must match the logit batch axes, got "
            f"{action.shape} and {new_logits.shape}"
        )
    _same_shape(
        "PPO action, advantage, validity, and trajectory weight",
        action,
        advantage,
        valid,
        trajectory_weight,
    )
    clip_epsilon = float(clip_epsilon)
    entropy_coefficient = jnp.asarray(entropy_coefficient, f32)
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("PPO clip epsilon must be in (0, 1)")

    old_logprob_all = jax.nn.log_softmax(old_logits, axis=-1)
    new_logprob_all = jax.nn.log_softmax(new_logits, axis=-1)
    selected_action = jax.nn.one_hot(action, new_logits.shape[-1], dtype=f32)
    old_logprob = (old_logprob_all * selected_action).sum(axis=-1)
    new_logprob = (new_logprob_all * selected_action).sum(axis=-1)
    logratio = new_logprob - old_logprob
    ratio = jnp.exp(jnp.clip(logratio, -20.0, 20.0))
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = jnp.minimum(ratio * advantage, clipped_ratio * advantage)

    new_probability = jnp.exp(new_logprob_all)
    old_probability = jnp.exp(old_logprob_all)
    raw_entropy = -(new_probability * new_logprob_all).sum(axis=-1)
    legal_count = (new_logits > -1e20).sum(axis=-1).astype(f32)
    max_entropy = jnp.log(jnp.maximum(legal_count, 1.0))
    normalized_entropy = jnp.where(
        max_entropy > 0.0,
        raw_entropy / jnp.maximum(max_entropy, 1e-8),
        0.0,
    )
    entropy = normalized_entropy if normalize_entropy else raw_entropy
    exact_kl = (old_probability * (old_logprob_all - new_logprob_all)).sum(axis=-1)
    approx_kl = (ratio - 1.0) - logratio
    clipped = jnp.abs(ratio - 1.0) > clip_epsilon
    loss = masked_weighted_mean(
        -surrogate - entropy_coefficient * entropy,
        valid,
        trajectory_weight,
    )

    effective = valid & (trajectory_weight > 0.0)
    metrics = {
        "loss": loss,
        "surrogate": masked_weighted_mean(surrogate, valid, trajectory_weight),
        "entropy": masked_weighted_mean(entropy, valid, trajectory_weight),
        "raw_entropy": masked_weighted_mean(raw_entropy, valid, trajectory_weight),
        "normalized_entropy": masked_weighted_mean(
            normalized_entropy, valid, trajectory_weight
        ),
        "entropy_coefficient": entropy_coefficient,
        "exact_kl": masked_weighted_mean(exact_kl, valid, trajectory_weight),
        "approx_kl": masked_weighted_mean(approx_kl, valid, trajectory_weight),
        "clip_fraction": masked_weighted_mean(
            clipped.astype(f32), valid, trajectory_weight
        ),
        "ratio": masked_weighted_mean(ratio, valid, trajectory_weight),
        "ratio_min": jnp.where(
            effective.any(),
            jnp.min(jnp.where(effective, ratio, jnp.inf)),
            1.0,
        ),
        "ratio_max": jnp.where(
            effective.any(),
            jnp.max(jnp.where(effective, ratio, -jnp.inf)),
            1.0,
        ),
        "valid_fraction": valid.astype(f32).mean(),
        "effective_weight": (valid.astype(f32) * trajectory_weight).mean(),
        "advantage_mean": masked_weighted_mean(advantage, valid, trajectory_weight),
        "advantage_rms": jnp.sqrt(
            masked_weighted_mean(jnp.square(advantage), valid, trajectory_weight)
        ),
    }
    return loss, metrics


def value_objective(value_output, target_return, valid, trajectory_weight):
    """Fit the maintained distributional critic to frozen imagined returns."""

    target_return = sg(jnp.asarray(target_return, f32))
    valid = sg(jnp.asarray(valid, bool))
    trajectory_weight = sg(jnp.asarray(trajectory_weight, f32))
    if (
        target_return.shape != valid.shape
        or target_return.shape != trajectory_weight.shape
    ):
        raise ValueError(
            "PPO return, validity, and trajectory weight shapes must match, got "
            f"{target_return.shape}, {valid.shape}, and {trajectory_weight.shape}"
        )
    prediction = value_output.pred().astype(f32)
    if prediction.shape != target_return.shape:
        raise ValueError(
            "PPO value prediction and return shapes must match, got "
            f"{prediction.shape} and {target_return.shape}"
        )
    per_entry_loss = value_output.loss(target_return).astype(f32)
    loss = masked_weighted_mean(per_entry_loss, valid, trajectory_weight)
    error = prediction - target_return
    target_mean = masked_weighted_mean(target_return, valid, trajectory_weight)
    error_mean = masked_weighted_mean(error, valid, trajectory_weight)
    target_variance = masked_weighted_mean(
        jnp.square(target_return - target_mean), valid, trajectory_weight
    )
    error_variance = masked_weighted_mean(
        jnp.square(error - error_mean), valid, trajectory_weight
    )
    return loss, {
        "loss": loss,
        "value_mean": masked_weighted_mean(prediction, valid, trajectory_weight),
        "return_mean": target_mean,
        "rmse": jnp.sqrt(
            masked_weighted_mean(jnp.square(error), valid, trajectory_weight)
        ),
        "bias": error_mean,
        "explained_variance": 1.0 - error_variance / jnp.maximum(target_variance, 1e-8),
    }


__all__ = [
    "clipped_policy_objective",
    "generalized_advantage_estimate",
    "masked_weighted_mean",
    "normalize_advantage",
    "value_objective",
]
