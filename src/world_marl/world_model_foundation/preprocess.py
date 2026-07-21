from __future__ import annotations

import numpy as np


def normalize_observations(
    observations: np.ndarray, *, clip: bool = True
) -> np.ndarray:
    arr = np.asarray(observations)
    if arr.dtype == np.uint8:
        normalized = arr.astype(np.float32) / np.float32(255.0)
    else:
        normalized = arr.astype(np.float32)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


normalize_images = normalize_observations
