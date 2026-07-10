from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutoencoderConfig:
    latent_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256)

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")


@dataclass(frozen=True)
class LAMConfig:
    kind: str = "continuous"
    latent_action_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    kl_scale: float = 1e-2

    def __post_init__(self) -> None:
        if self.kind != "continuous":
            raise ValueError("Genie2ContinuousConfig requires a continuous LAM")
        if self.latent_action_dim <= 0:
            raise ValueError("latent_action_dim must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if self.kl_scale <= 0.0:
            raise ValueError("kl_scale must be positive")


@dataclass(frozen=True)
class DynamicsConfig:
    objective: str = "diffusion_velocity"
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_context: int = 32
    classifier_free_dropout: float = 0.1
    guidance_scale: float = 1.5
    sampling_steps: int = 4

    def __post_init__(self) -> None:
        if self.objective not in ("diffusion_velocity", "flow_matching"):
            raise ValueError("objective must be diffusion_velocity or flow_matching")
        if self.model_dim <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError("model_dim, num_heads, and num_layers must be positive")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if not 0.0 <= self.classifier_free_dropout < 1.0:
            raise ValueError("classifier_free_dropout must be in [0, 1)")


@dataclass(frozen=True)
class LatentPolicyConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    imagination_horizon: int = 15
    discount_lambda: float = 0.95
    entropy_scale: float = 3e-4
    action_penalty: float = 1e-3

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.imagination_horizon <= 0:
            raise ValueError("imagination_horizon must be positive")
        if not 0.0 <= self.discount_lambda <= 1.0:
            raise ValueError("discount_lambda must be in [0, 1]")
        if self.entropy_scale < 0.0 or self.action_penalty < 0.0:
            raise ValueError("policy regularization scales must be non-negative")


@dataclass(frozen=True)
class Genie2ContinuousConfig:
    representation: str = "continuous_latent"
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lam: LAMConfig = field(default_factory=LAMConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    latent_policy: LatentPolicyConfig = field(default_factory=LatentPolicyConfig)
    reward_continue_hidden_dims: tuple[int, ...] = (256, 256)
    bridge_hidden_dims: tuple[int, ...] = (128,)
    vq_maskgit_ablation_enabled: bool = False

    def __post_init__(self) -> None:
        if self.representation != "continuous_latent":
            raise ValueError("primary Genie2 arm must use continuous_latent")
        if self.vq_maskgit_ablation_enabled:
            raise ValueError("VQ/MaskGIT belongs in a separate ablation package")
