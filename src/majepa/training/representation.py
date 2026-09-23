"""Representation objectives used by canonical decoder-free MA-JEPA."""

import jax
import jax.numpy as jnp


def sigreg_loss(
    embeddings,
    key,
    *,
    knots=17,
    num_proj=256,
    aggregation="pooled",
    team_size=1,
    valid=None,
):
    values = embeddings.astype(jnp.float32)
    if team_size < 1:
        raise ValueError("team_size must be positive")
    if aggregation not in {"pooled", "per_agent", "per_timestep"}:
        raise ValueError("aggregation must be pooled, per_agent, or per_timestep")
    directions = jax.random.normal(
        key, (values.shape[-1], num_proj), dtype=values.dtype
    )
    directions /= jnp.linalg.norm(directions, axis=0, keepdims=True) + 1e-6
    points = jnp.linspace(0.0, 3.0, knots, dtype=values.dtype)
    step = jnp.asarray(3.0 / (knots - 1), dtype=values.dtype)
    weights = jnp.full((knots,), 2.0 * step, dtype=values.dtype)
    weights = weights.at[0].set(step).at[-1].set(step)
    gaussian = jnp.exp(-jnp.square(points) / 2.0)
    if aggregation == "per_agent":
        if values.ndim != 3 or values.shape[0] % team_size:
            raise ValueError(
                "per_agent SIGReg expects [batch * agent, time, embedding] "
                f"with a batch divisible by team_size, got {values.shape} and "
                f"team_size={team_size}"
            )
        batch = values.shape[0] // team_size
        values = values.reshape((batch, team_size, values.shape[1], values.shape[2]))
        values = values.transpose((1, 0, 2, 3)).reshape(
            (team_size, batch * values.shape[2], values.shape[3])
        )
        if valid is None:
            valid = jnp.ones((team_size, values.shape[1]), bool)
        else:
            valid = jnp.asarray(valid, bool)
            if valid.shape != embeddings.shape[:-1]:
                raise ValueError(
                    f"SIGReg validity {valid.shape} does not match embeddings "
                    f"{embeddings.shape[:-1]}"
                )
            valid = valid.reshape((batch, team_size, valid.shape[1]))
            valid = valid.transpose((1, 0, 2)).reshape((team_size, -1))
        sample_axis = 1
        sample_count = valid.sum(axis=1)
    elif aggregation == "per_timestep":
        if values.ndim != 3:
            raise ValueError(
                "per_timestep SIGReg expects [batch, time, embedding], got "
                f"{values.shape}"
            )
        if valid is None:
            valid = jnp.ones(values.shape[:-1], bool)
        else:
            valid = jnp.asarray(valid, bool)
            if valid.shape != values.shape[:-1]:
                raise ValueError(
                    f"SIGReg validity {valid.shape} does not match embeddings "
                    f"{values.shape[:-1]}"
                )
        sample_axis = 0
        sample_count = valid.sum(axis=0)
    else:
        values = values.reshape((-1, values.shape[-1]))
        if valid is None:
            valid = jnp.ones((values.shape[0],), bool)
        else:
            valid = jnp.asarray(valid, bool).reshape((-1,))
            if valid.shape[0] != values.shape[0]:
                raise ValueError(
                    f"SIGReg validity {valid.shape} does not match flattened "
                    f"embeddings {values.shape}"
                )
        sample_axis = 0
        sample_count = valid.sum()
    samples = values @ directions
    phases = samples[..., None] * points
    expanded_valid = valid[..., None, None]
    denominator = jnp.maximum(sample_count, 1)
    if aggregation in {"per_agent", "per_timestep"}:
        denominator = denominator[:, None, None]
    cosine = (jnp.cos(phases) * expanded_valid).sum(axis=sample_axis) / denominator
    sine = (jnp.sin(phases) * expanded_valid).sum(axis=sample_axis) / denominator
    error = jnp.square(cosine - gaussian) + jnp.square(sine)
    integrated = jnp.mean(error @ (weights * gaussian), axis=-1)
    if aggregation in {"per_agent", "per_timestep"}:
        weighted = integrated * sample_count
        return weighted.sum() / jnp.maximum((sample_count > 0).sum(), 1)
    return integrated * sample_count


def embedding_prediction_loss(prediction, target, *, distance, stop_target):
    prediction = prediction.astype(jnp.float32)
    target = target.astype(jnp.float32)
    if stop_target:
        target = jax.lax.stop_gradient(target)
    prediction_norm = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6
    )
    target_norm = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6
    )
    cosine = (prediction_norm * target_norm).sum(axis=-1)
    mse = jnp.square(prediction - target).mean(axis=-1)
    loss = 1.0 - cosine if distance == "cosine" else mse
    return loss, cosine, mse


def embedding_std(embeddings):
    values = embeddings.astype(jnp.float32).reshape((-1, embeddings.shape[-1]))
    return values.std(axis=0).mean()
