import jax
import numpy as np

from dreamarl.ablations.representation import spatial_patch_mask


def test_alternative_mask_topologies_preserve_coverage() -> None:
    for topology in ("fixed_count", "multiblock"):
        mask = spatial_patch_mask(jax.random.key(18), (4, 5), (4, 4), 0.5, topology)
        np.testing.assert_array_equal(
            np.asarray(mask).reshape((20, -1)).sum(axis=-1), np.full(20, 8)
        )
