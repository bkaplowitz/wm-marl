from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearActionBridge:
    weights: np.ndarray
    bias: np.ndarray

    @property
    def latent_action_dim(self) -> int:
        return int(self.weights.shape[0])

    @property
    def real_action_dim(self) -> int:
        return int(self.weights.shape[1])

    def predict(self, latent_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(latent_actions, dtype=np.float32)
        if actions.shape[-1] != self.latent_action_dim:
            raise ValueError(
                "latent_actions last dimension must match bridge latent_action_dim"
            )
        return actions @ self.weights + self.bias


def fit_linear_action_bridge(
    latent_actions: np.ndarray,
    real_actions: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> LinearActionBridge:
    latents = np.asarray(latent_actions, dtype=np.float32)
    actions = np.asarray(real_actions, dtype=np.float32)
    if latents.ndim != 2:
        raise ValueError("latent_actions must have shape (batch, latent_action_dim)")
    if actions.ndim != 2:
        raise ValueError("real_actions must have shape (batch, real_action_dim)")
    if latents.shape[0] != actions.shape[0]:
        raise ValueError(
            "latent_actions and real_actions must have the same batch size"
        )
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    design = np.concatenate(
        [latents, np.ones((latents.shape[0], 1), dtype=np.float32)], axis=-1
    )
    penalty = np.eye(design.shape[1], dtype=np.float32) * np.float32(ridge)
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ actions)
    return LinearActionBridge(
        weights=solution[:-1].astype(np.float32),
        bias=solution[-1].astype(np.float32),
    )
