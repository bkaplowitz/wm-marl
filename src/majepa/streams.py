"""First-party stream adapter with a separate diagnostic seed sequence."""

import elements
import embodied
from embodied.jax import internal
import numpy as np


def isolated_report_stream(agent, source):
    """Match the pinned JAX stream transport without consuming train seeds.

    The report counter is intentionally not part of the learner checkpoint:
    diagnostics cannot affect training, including after a checkpoint reload.
    The offset gives diagnostics a disjoint practical counter range.
    """
    counter = elements.Counter()

    def prepare(data):
        for key, value in data.items():
            if np.issubdtype(value.dtype, np.floating) and np.isnan(value).any():
                raise ValueError(f"NaN in report field {key}")
        data = internal.device_put(data, agent.train_sharded)
        with counter.lock:
            index = counter.value
            counter.value += 1
        seed = agent._seeds((1 << 32) + index, agent.train_mirrored)
        return {**data, "seed": seed}

    return embodied.streams.Prefetch(source, prepare)
