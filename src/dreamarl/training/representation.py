"""Representation objectives used by canonical decoder-free DreaMARL."""

import jax
import jax.numpy as jnp


def sigreg_loss(embeddings, key, *, knots=17, num_proj=256, aggregation="pooled"):
    values = embeddings.astype(jnp.float32)
    directions = jax.random.normal(
        key, (values.shape[-1], num_proj), dtype=values.dtype
    )
    directions /= jnp.linalg.norm(directions, axis=0, keepdims=True) + 1e-6
    points = jnp.linspace(0.0, 3.0, knots, dtype=values.dtype)
    step = jnp.asarray(3.0 / (knots - 1), dtype=values.dtype)
    weights = jnp.full((knots,), 2.0 * step, dtype=values.dtype)
    weights = weights.at[0].set(step).at[-1].set(step)
    gaussian = jnp.exp(-jnp.square(points) / 2.0)
    if aggregation == "pooled":
        values = values.reshape((-1, values.shape[-1]))
    samples = values @ directions
    phases = samples[..., None] * points
    error = jnp.square(jnp.mean(jnp.cos(phases), axis=0) - gaussian)
    error += jnp.square(jnp.mean(jnp.sin(phases), axis=0))
    return jnp.mean(error @ (weights * gaussian)) * values.shape[0]


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


def spatial_patch_mask(key, leading_shape, grid_shape, ratio):
    """Sample exactly the configured number of target patches."""

    patch_count = grid_shape[0] * grid_shape[1]
    target_count = min(max(round(ratio * patch_count), 1), patch_count - 1)
    scores = jax.random.uniform(key, (*leading_shape, patch_count))
    indices = jax.lax.top_k(scores, target_count)[1]
    flattened = (
        indices[..., :, None] == jnp.arange(patch_count, dtype=indices.dtype)
    ).any(axis=-2)
    return flattened.reshape((*leading_shape, *grid_shape))


def mask_image_patches(image, mask, *, fill_value=128):
    grid_height, grid_width = mask.shape[-2:]
    image_height, image_width = image.shape[-3:-1]
    expanded = jnp.repeat(mask, image_height // grid_height, axis=-2)
    expanded = jnp.repeat(expanded, image_width // grid_width, axis=-1)
    return jnp.where(
        expanded[..., None], jnp.asarray(fill_value, dtype=image.dtype), image
    )


def masked_spatial_loss(prediction, target, mask):
    prediction = prediction.astype(jnp.float32)
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    prediction /= jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6)
    target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    cosine = (prediction * target).sum(axis=-1)
    flat_mask = mask.reshape(prediction.shape[:-1]).astype(jnp.float32)
    target_count = jnp.maximum(flat_mask.sum(axis=-1), 1.0)
    loss = ((1.0 - cosine) * flat_mask).sum(axis=-1) / target_count
    mean_cosine = (cosine * flat_mask).sum() / jnp.maximum(flat_mask.sum(), 1.0)
    return loss, mean_cosine, flat_mask.mean()
