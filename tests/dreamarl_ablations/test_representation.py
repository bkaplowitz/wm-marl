import jax
import jax.numpy as jnp
import numpy as np

from dreamarl.ablations.representation import (
    spatial_patch_mask,
    vjepa21_dense_token_loss,
    vjepa21_multiblock_masks,
)


def test_alternative_mask_topologies_preserve_coverage() -> None:
    for topology in ("fixed_count", "multiblock"):
        mask = spatial_patch_mask(jax.random.key(18), (4, 5), (4, 4), 0.5, topology)
        np.testing.assert_array_equal(
            np.asarray(mask).reshape((20, -1)).sum(axis=-1), np.full(20, 8)
        )


def test_vjepa_masks_use_temporal_tubes() -> None:
    contexts, targets = vjepa21_multiblock_masks(jax.random.key(21), (3, 16))
    for masks in (contexts, targets):
        values = np.asarray(masks)
        assert values.shape == (2, 3, 16, 16, 16)
        np.testing.assert_array_equal(
            values[:, :, 0::2], values[:, :, 1::2]
        )
        counts = values[:, :, 0].reshape((2, 3, -1)).sum(axis=-1)
        total_counts = values.reshape((2, 3, -1)).sum(axis=-1)
        np.testing.assert_array_equal(
            total_counts, np.repeat(total_counts[:, :1], 3, axis=1)
        )
    assert not np.logical_and(np.asarray(contexts), np.asarray(targets)).any()


def test_vjepa_loss_stops_target_gradients() -> None:
    prediction = jax.random.normal(jax.random.key(22), (2, 3, 256, 8))
    context_prediction = jax.random.normal(jax.random.key(25), prediction.shape)
    target = jax.random.normal(jax.random.key(23), prediction.shape)
    contexts, targets = vjepa21_multiblock_masks(jax.random.key(24), (2, 16))
    prediction = jnp.repeat(prediction[:, :1], 16, axis=1)
    context_prediction = jnp.repeat(context_prediction[:, :1], 16, axis=1)
    target = jnp.repeat(target[:, :1], 16, axis=1)
    context, mask = contexts[0], targets[0]

    def objective(pred, context_pred, fixed):
        return vjepa21_dense_token_loss(
            pred, context_pred, fixed, mask, context
        )[0].mean()

    prediction_grad, context_grad, target_grad = jax.grad(
        objective, argnums=(0, 1, 2)
    )(
        prediction, context_prediction, target
    )
    assert np.linalg.norm(np.asarray(prediction_grad)) > 0
    assert np.linalg.norm(np.asarray(context_grad)) > 0
    np.testing.assert_array_equal(np.asarray(target_grad), np.zeros(target.shape))
