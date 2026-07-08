from __future__ import annotations

from collections.abc import Mapping

import math


def finite_metric_check(metrics: Mapping[str, float]) -> None:
    bad = [name for name, value in metrics.items() if not math.isfinite(float(value))]
    if bad:
        raise ValueError(f"non-finite Dreamer metrics: {bad}")


def loss_decreased(metrics: list[Mapping[str, float]], key: str = "loss") -> bool:
    if len(metrics) < 2:
        return True
    return float(metrics[-1][key]) <= float(metrics[0][key])
