import jax
import numpy as np

from dreamarl.ablations.representation import (
    spatial_patch_mask,
    vjepa_multiblock_masks,
    vjepa_token_loss,
)


def test_alternative_mask_topologies_preserve_coverage() -> None:
    for topology in ("fixed_count", "multiblock"):
        mask = spatial_patch_mask(jax.random.key(18), (4, 5), (4, 4), 0.5, topology)
        np.testing.assert_array_equal(
            np.asarray(mask).reshape((20, -1)).sum(axis=-1), np.full(20, 8)
        )


def test_vjepa_masks_use_temporal_tubes() -> None:
    masks = vjepa_multiblock_masks(jax.random.key(21), (3, 5))
    values = np.asarray(masks)
    assert values.shape == (2, 3, 5, 14, 14)
    np.testing.assert_array_equal(
        values[:, :, 1:], np.repeat(values[:, :, :1], 4, axis=2)
    )


def test_vjepa_loss_stops_target_gradients() -> None:
    prediction = jax.random.normal(jax.random.key(22), (2, 3, 196, 8))
    target = jax.random.normal(jax.random.key(23), prediction.shape)
    mask = vjepa_multiblock_masks(jax.random.key(24), (2, 3))[0]

    def objective(pred, fixed):
        return vjepa_token_loss(pred, fixed, mask)[0].mean()

    prediction_grad, target_grad = jax.grad(objective, argnums=(0, 1))(
        prediction, target
    )
    assert np.linalg.norm(np.asarray(prediction_grad)) > 0
    np.testing.assert_array_equal(np.asarray(target_grad), np.zeros(target.shape))
