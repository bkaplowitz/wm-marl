"""Jasmine source-default configuration.

Adapted from ``p-doom/jasmine`` at commit
``420859bc99eecf6b07a7e9edf65d5d145935f1e1``. Source paths:
``jasmine/baselines/diffusion/train_tokenizer_mae.py``,
``jasmine/baselines/train_lam.py``,
``jasmine/baselines/diffusion/train_dynamics_diffusion.py``, and
``jasmine/baselines/diffusion/sample_diffusion.py``. Integration changes:
immutable nested dataclasses replace Tyro script arguments and set the approved
64-step source sampler as the dynamics default.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class StageTrainingConfig:
    updates: int
    batch_size: int
    warmup_steps: int
    wsd_decay_steps: int
    peak_learning_rate: float
    initial_learning_rate: float = 0.0
    decay_end: float = 0.0
    adam_b1: float = 0.9
    adam_b2: float = 0.9
    weight_decay: float = 1e-4


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    in_dim: int = 3
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_dim: int = 32
    num_latents: int = 1024
    patch_size: int = 16
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    max_mask_ratio: float = 0.9
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.bfloat16
    use_flash_attention: bool = True


@dataclass(frozen=True, slots=True)
class LAMConfig:
    in_dim: int = 3
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_dim: int = 32
    num_latents: int = 6
    patch_size: int = 16
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    vq_beta: float = 0.25
    reset_inactive_after: int = 50
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.bfloat16
    use_flash_attention: bool = True


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    model_dim: int = 512
    ffn_dim: int = 2048
    latent_patch_dim: int = 32
    latent_action_dim: int = 32
    num_blocks: int = 6
    num_heads: int = 8
    dropout: float = 0.0
    denoise_steps: int = 64
    context_corruption: float = 0.1
    use_gt_actions: bool = False
    use_cfg: bool = False
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.bfloat16
    use_flash_attention: bool = True


def _tokenizer_training() -> StageTrainingConfig:
    return StageTrainingConfig(300_000, 48, 10_000, 30_000, 3e-4)


def _lam_training() -> StageTrainingConfig:
    return StageTrainingConfig(200_000, 36, 5_000, 20_000, 3e-5)


def _dynamics_training() -> StageTrainingConfig:
    return StageTrainingConfig(200_000, 36, 5_000, 20_000, 1e-4)


@dataclass(frozen=True, slots=True)
class JasmineConfig:
    sequence_length: int = 16
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    lam: LAMConfig = field(default_factory=LAMConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    tokenizer_training: StageTrainingConfig = field(default_factory=_tokenizer_training)
    lam_training: StageTrainingConfig = field(default_factory=_lam_training)
    dynamics_training: StageTrainingConfig = field(default_factory=_dynamics_training)
