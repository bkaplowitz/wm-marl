"""Jafar source-default configuration.

Adapted from ``FLAIROx/jafar`` at commit
``5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665``. Source paths:
``train_tokenizer.py``, ``train_lam.py``, ``train_dynamics.py``, and
``sample.py``. Integration changes: immutable nested dataclasses replace Tyro
script arguments and group stage defaults for repository training entrypoints.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StageTrainingConfig:
    updates: int
    batch_size: int
    warmup_steps: int
    initial_learning_rate: float
    peak_learning_rate: float
    adam_b1: float = 0.9
    adam_b2: float = 0.9
    weight_decay: float = 1e-4


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    in_dim: int = 3
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 1024
    patch_size: int = 4
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.01
    vq_beta: float = 0.25


@dataclass(frozen=True, slots=True)
class LAMConfig:
    in_dim: int = 3
    model_dim: int = 512
    latent_dim: int = 32
    num_latents: int = 6
    patch_size: int = 16
    num_blocks: int = 8
    num_heads: int = 8
    dropout: float = 0.0
    codebook_dropout: float = 0.0
    vq_beta: float = 0.25
    reset_inactive_after: int = 50


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    model_dim: int = 512
    num_latents: int = 1024
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.0
    mask_limit: float = 0.5
    maskgit_steps: int = 25
    temperature: float = 1.0


def _tokenizer_training() -> StageTrainingConfig:
    return StageTrainingConfig(300_000, 48, 10_000, 3e-4, 3e-4)


def _lam_training() -> StageTrainingConfig:
    return StageTrainingConfig(200_000, 36, 5_000, 3e-6, 3e-5)


@dataclass(frozen=True, slots=True)
class JafarConfig:
    sequence_length: int = 16
    image_height: int = 64
    image_width: int = 64
    image_channels: int = 3
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    lam: LAMConfig = field(default_factory=LAMConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    tokenizer_training: StageTrainingConfig = field(default_factory=_tokenizer_training)
    lam_training: StageTrainingConfig = field(default_factory=_lam_training)
    dynamics_training: StageTrainingConfig = field(default_factory=_lam_training)
