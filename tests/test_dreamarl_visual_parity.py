import elements
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from dreamerv3 import rssm as reference
from dreamarl.models import visual
from dreamarl.models.latent import CategoricalLatent


OBS_SPACE = {"image": elements.Space(np.uint8, (64, 64, 3))}
ACTION_SPACE = {"action": elements.Space(np.int32, (), 0, 4)}


def test_visual_encoder_matches_pinned_dreamerv3() -> None:
    kwargs = dict(depth=2, mults=(1, 1), kernel=3, act="silu", norm="rms", name="enc")
    official = reference.Encoder(OBS_SPACE, **kwargs)
    candidate = visual.Encoder(OBS_SPACE, **kwargs)
    images = jax.random.randint(
        jax.random.key(1), (2, 3, 64, 64, 3), 0, 256, dtype=jnp.uint8
    )
    observations = {"image": images}
    resets = jnp.zeros((2, 3), bool).at[:, 0].set(True)

    def official_fn(obs, first):
        return official({}, obs, first, training=False)

    def candidate_fn(obs, first):
        return candidate({}, obs, first, training=False)

    params = nj.init(official_fn)({}, observations, resets, seed=10)
    expected = nj.pure(official_fn)(params, observations, resets, seed=11)[1]
    actual = nj.pure(candidate_fn)(params, observations, resets, seed=11)[1]
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))


def test_categorical_prior_matches_pinned_dreamerv3() -> None:
    kwargs = dict(
        deter=16,
        hidden=8,
        stoch=2,
        classes=4,
        imglayers=2,
        act="silu",
        norm="rms",
    )
    official = reference.RSSM(ACTION_SPACE, blocks=2, name="dyn", **kwargs)
    candidate = CategoricalLatent(ACTION_SPACE, enc_output=12, name="dyn", **kwargs)
    feature = jax.random.normal(jax.random.key(2), (3, 16), dtype=jnp.bfloat16)

    def official_fn(value):
        return official._prior(value)

    def candidate_fn(value):
        return candidate._prior(value)

    params = nj.init(official_fn)({}, feature, seed=20)
    expected = nj.pure(official_fn)(params, feature, seed=21)[1]
    actual = nj.pure(candidate_fn)(params, feature, seed=21)[1]
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(np.asarray(right), np.asarray(left))
