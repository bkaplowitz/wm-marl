"""Bounded offline interventions for multi-agent representation diagnosis.

The functions in this module operate only on tensors extracted from a frozen
DreaMARL checkpoint. They are deliberately independent of the online learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax


Array = jax.Array


class Intervention(StrEnum):
    BASELINE = "baseline"
    ACTIONS_PREDICTOR = "actions_predictor"
    LATENTS_PREDICTOR = "latents_predictor"
    PAIRED_PREDICTOR = "paired_predictor"
    PAIRED_SHUFFLED_PREDICTOR = "paired_shuffled_predictor"
    PAIRED_PRIOR = "paired_prior"
    PAIRED_TEMPORAL = "paired_temporal"


INTERVENTIONS = tuple(Intervention)


class Control(StrEnum):
    CORRECT = "correct"
    SHUFFLE_AGENTS = "shuffle_agents"
    SHUFFLE_ENVIRONMENTS = "shuffle_environments"
    NULL = "null"


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    hidden: int = 128
    learning_rate: float = 3e-4
    steps: int = 2_000
    batch_size: int = 8
    seed: int = 0
    grad_clip: float = 10.0


def source_for(intervention: Intervention) -> str:
    if intervention is Intervention.ACTIONS_PREDICTOR:
        return "actions"
    if intervention is Intervention.LATENTS_PREDICTOR:
        return "latents"
    if intervention is Intervention.BASELINE:
        return "none"
    return "paired"


def target_for(intervention: Intervention) -> str:
    if intervention is Intervention.PAIRED_PRIOR:
        return "prior"
    if intervention is Intervention.PAIRED_TEMPORAL:
        return "temporal"
    return "predictor"


def leave_one_out_mean(values: Array, agent_mask: Array | None = None) -> Array:
    """Pool all other agents while preserving the focal agent axis.

    Args:
        values: ``[batch, time, agent, feature]``.
        agent_mask: Optional ``[batch, time, agent]`` validity mask.
    """

    if values.ndim != 4:
        raise ValueError(f"expected [B,T,A,D], got {values.shape}")
    if agent_mask is None:
        agent_mask = jnp.ones(values.shape[:3], bool)
    if agent_mask.shape != values.shape[:3]:
        raise ValueError((agent_mask.shape, values.shape))
    weights = agent_mask.astype(values.dtype)[..., None]
    total = (values * weights).sum(axis=2, keepdims=True)
    count = weights.sum(axis=2, keepdims=True)
    others_total = total - values * weights
    others_count = count - weights
    pooled = others_total / jnp.maximum(others_count, 1)
    focal_valid = agent_mask[..., None]
    return jnp.where(focal_valid & (others_count > 0), pooled, 0)


def apply_control(values: Array, control: Control, key: Array) -> Array:
    """Apply an information-use control without changing tensor shape."""

    if control is Control.CORRECT:
        return values
    if control is Control.NULL:
        return jnp.zeros_like(values)
    if control is Control.SHUFFLE_AGENTS:
        permutation = jax.random.permutation(key, values.shape[2])
        return values[:, :, permutation]
    if control is Control.SHUFFLE_ENVIRONMENTS:
        permutation = jax.random.permutation(key, values.shape[0])
        return values[permutation]
    raise ValueError(control)


def build_source(
    tensors: Mapping[str, Array],
    intervention: Intervention,
    *,
    control: Control = Control.CORRECT,
    key: Array | None = None,
) -> Array:
    """Build aligned per-agent source elements before leave-one-out pooling."""

    source = source_for(intervention)
    if source == "none":
        shape = (*tensors["stoch"].shape[:3], 0)
        return jnp.zeros(shape, jnp.float32)
    latent_width = int(np.prod(tensors["stoch"].shape[-2:]))
    stoch = tensors["pair"][..., :latent_width]
    action = tensors["pair"][..., latent_width:]
    if source == "actions":
        values = jnp.concatenate([jnp.zeros_like(stoch), action], -1)
    elif source == "latents":
        values = jnp.concatenate([stoch, jnp.zeros_like(action)], -1)
    elif source == "paired":
        values = jnp.concatenate([stoch, action], -1)
    else:
        raise ValueError(source)
    if intervention is Intervention.PAIRED_SHUFFLED_PREDICTOR:
        control = Control.SHUFFLE_ENVIRONMENTS
    key = jax.random.key(0) if key is None else key
    return apply_control(values, control, key)


def init_adapter(
    key: Array,
    input_dim: int,
    output_dim: int,
    hidden: int,
    *,
    recurrent: bool = False,
) -> dict[str, Array]:
    """Initialize a zero-output residual adapter at exact baseline behavior."""

    key_in, key_rec = jax.random.split(key)
    scale = np.sqrt(2.0 / max(input_dim, 1))
    params = {
        "input_kernel": jax.random.normal(key_in, (input_dim, hidden)) * scale,
        "input_bias": jnp.zeros((hidden,)),
        "output_kernel": jnp.zeros((hidden, output_dim)),
        "output_bias": jnp.zeros((output_dim,)),
    }
    if recurrent:
        params.update(
            recurrent_kernel=jax.random.normal(key_rec, (hidden, hidden))
            / np.sqrt(hidden),
            recurrent_bias=jnp.zeros((hidden,)),
        )
    return params


def adapter_residual(
    params: Mapping[str, Array], source: Array, *, recurrent: bool = False
) -> Array:
    """Apply the shared adapter to grouped ``[B,T,A,D]`` elements."""

    elements = jnp.einsum("btad,dh->btah", source, params["input_kernel"])
    elements = jax.nn.silu(elements + params["input_bias"])
    context = leave_one_out_mean(elements)
    if recurrent:
        initial = jnp.zeros(
            (source.shape[0], source.shape[2], context.shape[-1]),
            context.dtype,
        )

        def step(carry, current):
            carry = jnp.tanh(
                current
                + jnp.einsum("bah,hk->bak", carry, params["recurrent_kernel"])
                + params["recurrent_bias"]
            )
            return carry, carry

        _, hidden = jax.lax.scan(step, initial, context.swapaxes(0, 1))
        hidden = hidden.swapaxes(0, 1)
    else:
        hidden = context
    return (
        jnp.einsum("btah,hd->btad", hidden, params["output_kernel"])
        + params["output_bias"]
    )


def cosine_error(prediction: Array, target: Array) -> Array:
    prediction = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8
    )
    target = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-8
    )
    return 1.0 - (prediction * target).sum(-1)


def categorical_kl(
    post_logit: Array, prior_logit: Array, unimix: float = 0.01
) -> Array:
    """Raw unclipped model KL summed over categorical stochastic variables."""

    classes = post_logit.shape[-1]
    post_prob = jax.nn.softmax(post_logit, -1)
    prior_prob = jax.nn.softmax(prior_logit, -1)
    post_prob = (1 - unimix) * post_prob + unimix / classes
    prior_prob = (1 - unimix) * prior_prob + unimix / classes
    post_logprob = jnp.log(post_prob)
    prior_logprob = jnp.log(prior_prob)
    return (post_prob * (post_logprob - prior_logprob)).sum((-1, -2))


def intervention_prediction(
    tensors: Mapping[str, Array],
    intervention: Intervention,
    params: Mapping[str, Array] | None,
    *,
    control: Control = Control.CORRECT,
    key: Array | None = None,
) -> tuple[Array, Array]:
    """Return adjusted predictor tokens and prior logits."""

    prediction = tensors["pred_token"]
    prior = tensors["prior_logit"]
    if intervention is Intervention.BASELINE:
        return prediction, prior
    if params is None:
        raise ValueError("non-baseline intervention requires adapter parameters")
    source = build_source(tensors, intervention, control=control, key=key)
    recurrent = intervention is Intervention.PAIRED_TEMPORAL
    residual = adapter_residual(params, source, recurrent=recurrent)
    if target_for(intervention) == "prior":
        prior = prior + residual.reshape(prior.shape)
    else:
        prediction = prediction + residual
    return prediction, prior


def loss_and_metrics(
    params: Mapping[str, Array] | None,
    tensors: Mapping[str, Array],
    intervention: Intervention,
    *,
    control: Control = Control.CORRECT,
    key: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    prediction, prior = intervention_prediction(
        tensors, intervention, params, control=control, key=key
    )
    valid = tensors.get("valid", ~tensors["reset"])
    cosine = cosine_error(prediction, tensors["target_token"])
    kl = categorical_kl(tensors["post_logit"], prior)
    denominator = jnp.maximum(valid.sum(), 1)
    cosine_mean = (cosine * valid).sum() / denominator
    kl_mean = (kl * valid).sum() / denominator
    objective = kl_mean if target_for(intervention) == "prior" else cosine_mean
    return objective, {
        "cosine": cosine_mean,
        "raw_kl": kl_mean,
        "valid_transitions": valid.sum(),
    }


def train_adapter(
    tensors: Mapping[str, Array],
    intervention: Intervention,
    config: AdapterConfig,
) -> tuple[dict[str, Array] | None, list[dict[str, float]]]:
    """Fit only the diagnostic adapter on a frozen tensor dataset."""

    if intervention is Intervention.BASELINE:
        _, metrics = loss_and_metrics(None, tensors, intervention)
        return None, [{key: float(value) for key, value in metrics.items()}]
    source = build_source(tensors, intervention)
    output_dim = (
        int(np.prod(tensors["prior_logit"].shape[-2:]))
        if target_for(intervention) == "prior"
        else tensors["target_token"].shape[-1]
    )
    params = init_adapter(
        jax.random.key(config.seed),
        source.shape[-1],
        output_dim,
        config.hidden,
        recurrent=intervention is Intervention.PAIRED_TEMPORAL,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    state = optimizer.init(params)

    @jax.jit
    def update(params, state, batch):
        (loss, metrics), grads = jax.value_and_grad(
            loss_and_metrics, has_aux=True
        )(params, batch, intervention)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, metrics

    history = []
    generator = np.random.default_rng(config.seed)
    trajectories = next(iter(tensors.values())).shape[0]
    batch_size = min(config.batch_size, trajectories)
    for step in range(config.steps):
        indices = generator.choice(trajectories, batch_size, replace=False)
        batch = jax.tree.map(lambda value: value[indices], tensors)
        params, state, loss, metrics = update(params, state, batch)
        if step in {0, config.steps - 1} or (step + 1) % 100 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )
    return params, history
