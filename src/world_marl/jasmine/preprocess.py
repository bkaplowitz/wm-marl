"""Jasmine pixel patch conversion.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``, path
``jasmine/utils/preprocess.py``. Integration changes: package-local public API;
the padding, rearrangement, and output crop are preserved.
"""

import einops
import jax
import jax.numpy as jnp


def patchify(videos: jax.Array, size: int) -> jax.Array:
    _, _, height, width, _ = videos.shape
    padded = jnp.pad(
        videos,
        ((0, 0), (0, 0), (0, -height % size), (0, -width % size), (0, 0)),
    )
    return einops.rearrange(
        padded,
        "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)",
        hp=size,
        wp=size,
    )


def unpatchify(
    patches: jax.Array,
    size: int,
    h_out: int,
    w_out: int,
) -> jax.Array:
    h_pad = -h_out % size
    height_patches = (h_out + h_pad) // size
    videos = einops.rearrange(
        patches,
        "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
        hp=size,
        wp=size,
        hn=height_patches,
    )
    return videos[:, :, :h_out, :w_out]
