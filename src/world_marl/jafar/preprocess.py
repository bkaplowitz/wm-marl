"""Jafar pixel patch conversion.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``, path
``utils/preprocess.py``. Integration changes: package-qualified imports, type
annotations, and removal of the upstream file's unused imports.
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
