import json

import jax.numpy as jnp
import numpy as np

from world_marl.logging import to_jsonable


def test_to_jsonable_handles_scalar_and_vector_jax_arrays():
    payload = {
        "scalar": jnp.asarray(1.5),
        "vector": jnp.arange(3, dtype=jnp.float32),
        "nested": {"matrix": jnp.ones((2, 2))},
        "numpy": np.arange(2),
        "plain": [1, "a", None],
    }
    result = to_jsonable(payload)
    assert result["scalar"] == 1.5
    assert result["vector"] == [0.0, 1.0, 2.0]
    assert result["nested"]["matrix"] == [[1.0, 1.0], [1.0, 1.0]]
    assert result["numpy"] == [0, 1]
    assert result["plain"] == [1, "a", None]
    json.dumps(result)
