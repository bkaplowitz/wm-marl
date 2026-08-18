"""Alternative representation objectives and masking recipes."""

import math

import jax
import jax.numpy as jnp


def sigreg_loss(
    embeddings: jax.Array,
    key: jax.Array,
    *,
    knots: int = 17,
    num_proj: int = 256,
    aggregation: str = "pooled",
) -> jax.Array:
    """Penalize deviations from an isotropic Gaussian over random projections."""

    if knots < 2:
        raise ValueError("knots must be at least 2")
    if num_proj < 1:
        raise ValueError("num_proj must be positive")
    if aggregation not in {"pooled", "per_timestep"}:
        raise ValueError("aggregation must be pooled or per_timestep")

    values = embeddings.astype(jnp.float32)
    directions = jax.random.normal(
        key, (values.shape[-1], num_proj), dtype=values.dtype
    )
    directions /= jnp.linalg.norm(directions, axis=0, keepdims=True) + 1e-6

    integration_points = jnp.linspace(0.0, 3.0, knots, dtype=values.dtype)
    step = jnp.asarray(3.0 / (knots - 1), dtype=values.dtype)
    weights = jnp.full((knots,), 2.0 * step, dtype=values.dtype)
    weights = weights.at[0].set(step)
    weights = weights.at[-1].set(step)
    gaussian_cf = jnp.exp(-jnp.square(integration_points) / 2.0)

    if aggregation == "pooled":
        values = values.reshape((-1, values.shape[-1]))
        samples = values @ directions
        sample_axis = 0
        sample_count = values.shape[0]
    else:
        if values.ndim != 3:
            raise ValueError(
                "per_timestep SIGReg expects [batch, time, embedding] values"
            )
        samples = values @ directions
        sample_axis = 0
        sample_count = values.shape[0]

    phases = samples[..., None] * integration_points
    error = jnp.square(jnp.mean(jnp.cos(phases), axis=sample_axis) - gaussian_cf)
    error += jnp.square(jnp.mean(jnp.sin(phases), axis=sample_axis))
    statistic = error @ (weights * gaussian_cf)
    return jnp.mean(statistic) * sample_count


def embedding_prediction_loss(
    prediction: jax.Array,
    target: jax.Array,
    *,
    distance: str,
    stop_target: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the configured embedding loss with common cosine and MSE metrics."""

    if prediction.shape != target.shape:
        raise ValueError(
            "embedding prediction and target shapes must match, got "
            f"{prediction.shape} and {target.shape}"
        )
    if distance not in {"cosine", "mse"}:
        raise ValueError("distance must be cosine or mse")

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


def embedding_std(embeddings: jax.Array) -> jax.Array:
    """Mean per-coordinate standard deviation over batch and time."""

    values = embeddings.astype(jnp.float32)
    values = values.reshape((-1, values.shape[-1]))
    return values.std(axis=0).mean()


def spatial_patch_mask(
    key: jax.Array,
    leading_shape: tuple[int, ...],
    grid_shape: tuple[int, int],
    ratio: float,
    topology: str = "bernoulli",
) -> jax.Array:
    """Sample a spatial target mask with a controlled topology.

    ``fixed_count`` removes Bernoulli count variance. ``multiblock`` uses two
    randomly located compact regions while preserving the same exact count.
    """

    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be strictly between zero and one")
    if topology not in {"bernoulli", "fixed_count", "multiblock"}:
        raise ValueError("topology must be bernoulli, fixed_count, or multiblock")
    patch_count = grid_shape[0] * grid_shape[1]
    if patch_count < 2:
        raise ValueError("spatial masking requires at least two image patches")
    if topology == "bernoulli":
        mask = jax.random.bernoulli(key, ratio, (*leading_shape, *grid_shape))
        flattened = mask.reshape((*leading_shape, -1))
        needs_target = ~flattened.any(axis=-1)
        flattened = flattened.at[..., 0].set(flattened[..., 0] | needs_target)
        needs_context = flattened.all(axis=-1)
        flattened = flattened.at[..., -1].set(flattened[..., -1] & ~needs_context)
        return flattened.reshape((*leading_shape, *grid_shape))

    target_count = min(max(round(ratio * patch_count), 1), patch_count - 1)
    if topology == "fixed_count":
        scores = jax.random.uniform(key, (*leading_shape, patch_count))
    else:
        anchor_key, jitter_key = jax.random.split(key)
        anchors = jax.random.uniform(
            anchor_key,
            (*leading_shape, 2, 2),
            minval=0.0,
            maxval=1.0,
        )
        rows = (jnp.arange(grid_shape[0], dtype=jnp.float32) + 0.5) / grid_shape[0]
        cols = (jnp.arange(grid_shape[1], dtype=jnp.float32) + 0.5) / grid_shape[1]
        coordinates = jnp.stack(jnp.meshgrid(rows, cols, indexing="ij"), axis=-1)
        distances = jnp.square(
            coordinates.reshape((1,) * len(leading_shape) + (*grid_shape, 1, 2))
            - anchors[..., None, None, :, :]
        ).sum(axis=-1)
        nearest_anchor = distances.min(axis=-1).reshape((*leading_shape, patch_count))
        jitter = jax.random.uniform(jitter_key, nearest_anchor.shape) * 1e-4
        scores = -(nearest_anchor + jitter)

    indices = jax.lax.top_k(scores, target_count)[1]
    flattened = (
        indices[..., :, None] == jnp.arange(patch_count, dtype=indices.dtype)
    ).any(axis=-2)
    return flattened.reshape((*leading_shape, *grid_shape))


def vjepa21_multiblock_masks(
    key: jax.Array,
    leading_shape: tuple[int, int],
    grid_shape: tuple[int, int] = (16, 16),
) -> tuple[jax.Array, jax.Array]:
    """Sample the exact two tube-mask families from V-JEPA 2.1 pretraining.

    V-JEPA uses eight small target blocks with spatial scale 0.15 and two
    large target blocks with spatial scale 0.7. Both families use aspect
    ratios in [0.75, 1.5]. A block is constant over the temporal dimension,
    so a sequence observes a spatial tube rather than unrelated frame masks.

    As in the official collator, target and complement index lists are each
    truncated to the minimum count in the batch. This preserves static shapes
    without converting dropped positions into visible context.
    """

    if len(leading_shape) != 2:
        raise ValueError(
            "V-JEPA tube masks require leading_shape=(batch, time), got "
            f"{leading_shape}"
        )
    batch, time = leading_shape
    if time != 16:
        raise ValueError(
            f"V-JEPA 2.1 video masking requires exactly 16 frames, got {time}"
        )
    duration = time // 2  # Official tubelet size is two frames.
    group_keys = jax.random.split(key, 2)

    def group_mask(group_key, blocks, scale):
        shape_key, position_key = jax.random.split(group_key)
        aspect = jax.random.uniform(
            shape_key, (), minval=0.75, maxval=1.5, dtype=jnp.float32
        )
        spatial_keep = int(math.prod(grid_shape) * scale)
        block_height = jnp.clip(
            jnp.rint(jnp.sqrt(spatial_keep * aspect)).astype(jnp.int32),
            1,
            grid_shape[0],
        )
        block_width = jnp.clip(
            jnp.rint(jnp.sqrt(spatial_keep / aspect)).astype(jnp.int32),
            1,
            grid_shape[1],
        )
        row_key, col_key = jax.random.split(position_key)
        row_limit = grid_shape[0] - block_height + 1
        col_limit = grid_shape[1] - block_width + 1
        row_starts = jax.random.randint(row_key, (batch, blocks), 0, row_limit)
        col_starts = jax.random.randint(col_key, (batch, blocks), 0, col_limit)
        rows = jnp.arange(grid_shape[0])[None, None, :, None]
        cols = jnp.arange(grid_shape[1])[None, None, None, :]
        inside_rows = (rows >= row_starts[..., None, None]) & (
            rows < row_starts[..., None, None] + block_height
        )
        inside_cols = (cols >= col_starts[..., None, None]) & (
            cols < col_starts[..., None, None] + block_width
        )
        mask = (inside_rows & inside_cols).any(axis=1)

        mask = jnp.broadcast_to(
            mask[:, None], (batch, duration, *grid_shape)
        )
        flat_target = mask.reshape((batch, -1))
        needs_target = ~flat_target.any(axis=-1)
        flat_target = flat_target.at[:, 0].set(flat_target[:, 0] | needs_target)
        needs_context = flat_target.all(axis=-1)
        flat_target = flat_target.at[:, -1].set(
            flat_target[:, -1] & ~needs_context
        )
        flat_context = ~flat_target

        min_target = flat_target.sum(axis=-1).min()
        min_context = flat_context.sum(axis=-1).min()
        flat_target &= jnp.cumsum(flat_target, axis=-1) <= min_target
        flat_context &= jnp.cumsum(flat_context, axis=-1) <= min_context
        target = flat_target.reshape((batch, duration, *grid_shape))
        context = flat_context.reshape((batch, duration, *grid_shape))
        target = jnp.repeat(target, 2, axis=1)
        context = jnp.repeat(context, 2, axis=1)
        return context, target

    small_context, small_target = group_mask(
        group_keys[0], blocks=8, scale=0.15
    )
    large_context, large_target = group_mask(
        group_keys[1], blocks=2, scale=0.7
    )
    return (
        jnp.stack([small_context, large_context], axis=0),
        jnp.stack([small_target, large_target], axis=0),
    )


def vjepa_multiblock_masks(key, leading_shape, grid_shape=(14, 14)):
    """Compatibility wrapper for the superseded V-JEPA ablation."""

    _, targets = vjepa21_multiblock_masks(key, leading_shape, grid_shape)
    return targets


def mask_image_patches(
    image: jax.Array,
    mask: jax.Array,
    *,
    fill_value: int = 128,
) -> jax.Array:
    """Replace masked image-grid cells without changing image shape or dtype."""

    grid_height, grid_width = mask.shape[-2:]
    image_height, image_width = image.shape[-3:-1]
    if image_height % grid_height or image_width % grid_width:
        raise ValueError(
            "image dimensions must be divisible by the spatial mask grid: "
            f"image={(image_height, image_width)}, grid={(grid_height, grid_width)}"
        )
    expanded = jnp.repeat(mask, image_height // grid_height, axis=-2)
    expanded = jnp.repeat(expanded, image_width // grid_width, axis=-1)
    fill = jnp.asarray(fill_value, dtype=image.dtype)
    return jnp.where(expanded[..., None], fill, image)


def masked_spatial_loss(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Cosine loss over masked patches with a fixed target representation."""

    if prediction.shape != target.shape:
        raise ValueError(
            "spatial prediction and target shapes must match, got "
            f"{prediction.shape} and {target.shape}"
        )
    expected_mask_shape = prediction.shape[:-1]
    valid_mask_shape = (
        mask.ndim >= 2
        and mask.shape[:-2] == prediction.shape[:-2]
        and math.prod(mask.shape[-2:]) == prediction.shape[-2]
    )
    if not valid_mask_shape:
        raise ValueError(
            "spatial mask must contain one value per predicted patch, got "
            f"mask={mask.shape}, prediction={prediction.shape}"
        )

    prediction = prediction.astype(jnp.float32)
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    prediction /= jnp.maximum(jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6)
    target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    cosine = (prediction * target).sum(axis=-1)
    flat_mask = mask.reshape(expected_mask_shape).astype(jnp.float32)
    target_count = jnp.maximum(flat_mask.sum(axis=-1), 1.0)
    loss = ((1.0 - cosine) * flat_mask).sum(axis=-1) / target_count
    mean_cosine = (cosine * flat_mask).sum() / jnp.maximum(flat_mask.sum(), 1.0)
    return loss, mean_cosine, flat_mask.mean()


def vjepa_token_loss(
    prediction: jax.Array,
    target: jax.Array,
    mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """V-JEPA L1 target loss after per-token target normalization."""

    if prediction.shape != target.shape:
        raise ValueError(
            "V-JEPA prediction and target shapes must match, got "
            f"{prediction.shape} and {target.shape}"
        )
    expected_mask_shape = prediction.shape[:-1]
    if (
        mask.shape[:-2] != prediction.shape[:-2]
        or math.prod(mask.shape[-2:]) != prediction.shape[-2]
    ):
        raise ValueError(
            "V-JEPA mask must contain one value per image token, got "
            f"mask={mask.shape}, prediction={prediction.shape}"
        )

    prediction = prediction.astype(jnp.float32)
    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    target_mean = target.mean(axis=-1, keepdims=True)
    target_var = jnp.square(target - target_mean).mean(axis=-1, keepdims=True)
    target = (target - target_mean) * jax.lax.rsqrt(target_var + 1e-6)

    flat_mask = mask.reshape(expected_mask_shape).astype(jnp.float32)
    target_count = jnp.maximum(flat_mask.sum(axis=-1), 1.0)
    per_token = jnp.abs(prediction - target).mean(axis=-1)
    loss = (per_token * flat_mask).sum(axis=-1) / target_count

    prediction_norm = prediction / jnp.maximum(
        jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6
    )
    target_norm = target / jnp.maximum(
        jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6
    )
    cosine = (prediction_norm * target_norm).sum(axis=-1)
    mean_cosine = (cosine * flat_mask).sum() / jnp.maximum(flat_mask.sum(), 1.0)
    return loss, mean_cosine, flat_mask.mean()


def vjepa21_dense_token_loss(
    target_prediction: jax.Array,
    context_prediction: jax.Array,
    target: jax.Array,
    target_mask: jax.Array,
    context_mask: jax.Array,
    *,
    context_weight: float = 0.5,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """V-JEPA 2.1 masked plus distance-weighted visible-token L1 loss."""

    if target_prediction.shape != target.shape or context_prediction.shape != target.shape:
        raise ValueError("V-JEPA 2.1 predictions and targets must have equal shapes")
    leading = target.shape[:-2]
    if len(leading) != 2:
        raise ValueError("V-JEPA 2.1 dense loss expects [batch, time, token, dim]")
    if target_mask.shape[:-2] != leading or context_mask.shape[:-2] != leading:
        raise ValueError("V-JEPA 2.1 masks must match target leading dimensions")
    token_count = target.shape[-2]
    if math.prod(target_mask.shape[-2:]) != token_count:
        raise ValueError("V-JEPA 2.1 masks must contain one entry per token")

    target = jax.lax.stop_gradient(target.astype(jnp.float32))
    target_prediction = target_prediction.astype(jnp.float32)
    context_prediction = context_prediction.astype(jnp.float32)

    # The official 2.1 target normalizer treats the four hierarchical ViT
    # outputs independently before concatenating them for prediction.
    if target.shape[-1] % 4:
        raise ValueError("V-JEPA 2.1 hierarchical target width must divide by four")
    chunks = target.reshape((*target.shape[:-1], 4, target.shape[-1] // 4))
    mean = chunks.mean(axis=-1, keepdims=True)
    var = jnp.square(chunks - mean).mean(axis=-1, keepdims=True)
    target = ((chunks - mean) * jax.lax.rsqrt(var + 1e-6)).reshape(target.shape)

    batch, time = leading
    flat_count = time * token_count
    target_flat = target_mask.reshape((batch, flat_count)).astype(jnp.float32)
    context_flat = context_mask.reshape((batch, flat_count)).astype(jnp.float32)
    flat_target = target.reshape((batch, flat_count, target.shape[-1]))
    flat_target_prediction = target_prediction.reshape(flat_target.shape)
    flat_context_prediction = context_prediction.reshape(flat_target.shape)
    target_l1 = jnp.abs(flat_target_prediction - flat_target).mean(axis=-1)
    target_loss = (target_l1 * target_flat).sum(axis=-1) / jnp.maximum(
        target_flat.sum(axis=-1), 1.0
    )

    grid_height, grid_width = target_mask.shape[-2:]
    ids = jnp.arange(flat_count)
    frame = ids // token_count
    spatial = ids % token_count
    rows = spatial // grid_width
    cols = spatial % grid_width
    coordinates = jnp.stack(
        [frame // 2, rows, cols], axis=-1
    ).astype(jnp.float32)
    euclidean = jnp.sqrt(
        jnp.square(coordinates[:, None] - coordinates[None, :]).sum(axis=-1)
    )
    distances = jnp.where(
        target_flat[..., None, :] > 0,
        euclidean,
        jnp.asarray(jnp.inf, jnp.float32),
    ).min(axis=-1)
    # V-JEPA 2.1 intentionally square-roots the Euclidean distance once more.
    distance_weight = jax.lax.rsqrt(jnp.maximum(distances, 1e-6))
    context_l1 = jnp.abs(flat_context_prediction - flat_target).mean(axis=-1)
    context_loss = (context_l1 * distance_weight * context_flat).sum(axis=-1)
    context_loss /= jnp.maximum(context_flat.sum(axis=-1), 1.0)

    pred_norm = flat_target_prediction / jnp.maximum(
        jnp.linalg.norm(flat_target_prediction, axis=-1, keepdims=True), 1e-6
    )
    target_norm = flat_target / jnp.maximum(
        jnp.linalg.norm(flat_target, axis=-1, keepdims=True), 1e-6
    )
    cosine = (pred_norm * target_norm).sum(axis=-1)
    mean_cosine = (cosine * target_flat).sum() / jnp.maximum(
        target_flat.sum(), 1.0
    )
    clip_loss = target_loss + context_weight * context_loss
    return (
        jnp.broadcast_to(clip_loss[:, None], (batch, time)),
        mean_cosine,
        target_flat.mean(),
    )
