import jax
import jax.numpy as jnp
import numpy as np

from dreamarl.training.representation import (
    embedding_prediction_loss,
    embedding_std,
    mask_image_patches,
    masked_spatial_loss,
    sigreg_loss,
    spatial_patch_mask,
)


def test_embedding_target_gradient_matches_selected_recipe() -> None:
    prediction = jax.random.normal(jax.random.key(1), (2, 3, 8))
    target = jax.random.normal(jax.random.key(2), prediction.shape)

    def objective(predicted, encoded, *, distance, stop_target):
        return embedding_prediction_loss(
            predicted,
            encoded,
            distance=distance,
            stop_target=stop_target,
        )[0].mean()

    _, ema_target_grad = jax.grad(objective, argnums=(0, 1))(
        prediction,
        target,
        distance="cosine",
        stop_target=True,
    )
    online_prediction_grad, online_target_grad = jax.grad(objective, argnums=(0, 1))(
        prediction,
        target,
        distance="mse",
        stop_target=False,
    )

    np.testing.assert_array_equal(np.asarray(ema_target_grad), np.zeros(target.shape))
    assert np.linalg.norm(np.asarray(online_prediction_grad)) > 0
    assert np.linalg.norm(np.asarray(online_target_grad)) > 0


def test_sigreg_penalizes_collapsed_embeddings_more_than_gaussian_embeddings():
    key = jax.random.key(7)
    gaussian = jax.random.normal(key, (16, 32, 64))
    collapsed = jnp.zeros_like(gaussian)

    collapsed_loss = sigreg_loss(collapsed, key, knots=9, num_proj=32)
    gaussian_loss = sigreg_loss(gaussian, key, knots=9, num_proj=32)

    assert float(collapsed_loss) > float(gaussian_loss)
    assert float(embedding_std(collapsed)) == 0.0
    assert float(embedding_std(gaussian)) > 0.9


def test_per_agent_sigreg_is_invariant_to_replicated_team_size() -> None:
    key = jax.random.key(8)
    embeddings = jax.random.normal(key, (3, 5, 16))
    baseline = sigreg_loss(
        embeddings,
        key,
        knots=9,
        num_proj=32,
        aggregation="per_agent",
        team_size=1,
    )

    for team_size in (2, 5, 8):
        replicated = jnp.repeat(
            embeddings[:, None],
            team_size,
            axis=1,
        ).reshape((embeddings.shape[0] * team_size, *embeddings.shape[1:]))
        candidate = sigreg_loss(
            replicated,
            key,
            knots=9,
            num_proj=32,
            aggregation="per_agent",
            team_size=team_size,
        )
        np.testing.assert_allclose(candidate, baseline, rtol=1e-6, atol=1e-6)


def test_inactive_embeddings_do_not_change_sigreg() -> None:
    key = jax.random.key(81)
    embeddings = jax.random.normal(key, (4, 5, 16))
    valid = jnp.array(
        [
            [True, True, True, False, False],
            [True, True, False, False, False],
            [True, True, True, True, False],
            [True, False, False, False, False],
        ]
    )
    changed = jnp.where(valid[..., None], embeddings, embeddings + 10_000)
    baseline = sigreg_loss(
        embeddings,
        key,
        knots=9,
        num_proj=32,
        aggregation="per_agent",
        team_size=2,
        valid=valid,
    )
    candidate = sigreg_loss(
        changed,
        key,
        knots=9,
        num_proj=32,
        aggregation="per_agent",
        team_size=2,
        valid=valid,
    )
    np.testing.assert_allclose(candidate, baseline, rtol=1e-6, atol=1e-6)


def test_spatial_mask_hides_complete_image_patches() -> None:
    mask = jnp.array([[[[True, False], [False, True]]]])
    image = jnp.arange(1 * 1 * 4 * 6 * 1, dtype=jnp.uint8).reshape((1, 1, 4, 6, 1))
    masked = mask_image_patches(image, mask, fill_value=128)

    np_masked = np.asarray(masked)
    assert (np_masked[:, :, :2, :3] == 128).all()
    assert (np_masked[:, :, 2:, 3:] == 128).all()
    np.testing.assert_array_equal(
        np_masked[:, :, :2, 3:], np.asarray(image)[:, :, :2, 3:]
    )


def test_fixed_spatial_mask_preserves_exact_coverage() -> None:
    mask = spatial_patch_mask(jax.random.key(18), (4, 5), (4, 4), 0.5)
    flattened = np.asarray(mask).reshape((20, -1))
    np.testing.assert_array_equal(flattened.sum(axis=-1), np.full(20, 8))


def test_masked_spatial_loss_only_updates_online_prediction() -> None:
    prediction = jax.random.normal(jax.random.key(9), (2, 3, 4, 8))
    target = jax.random.normal(jax.random.key(10), prediction.shape)
    first = jnp.array([[True, False], [False, True]])
    second = jnp.array([[False, True], [True, False]])
    mask = jnp.stack([jnp.stack([first] * 3), jnp.stack([second] * 3)])

    def objective(predicted, fixed_target):
        return masked_spatial_loss(predicted, fixed_target, mask)[0].mean()

    prediction_grad, target_grad = jax.grad(objective, argnums=(0, 1))(
        prediction, target
    )
    assert np.isfinite(np.asarray(prediction_grad)).all()
    assert np.linalg.norm(np.asarray(prediction_grad)) > 0
    np.testing.assert_array_equal(np.asarray(target_grad), np.zeros(target.shape))
