from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AutoencoderConfig:
    representation: str = "continuous_patch_grid"
    patch_size: int = 16
    latent_patch_dim: int = 32
    model_dim: int = 512
    ffn_dim: int = 2048
    num_blocks: int = 4
    num_heads: int = 8
    max_mask_ratio: float = 0.9
    dropout: float = 0.0
    compute_dtype: str = "bfloat16"
    parameter_dtype: str = "float32"
    vector_hidden_dims: tuple[int, ...] = (256, 256)

    @property
    def latent_dim(self) -> int:
        return self.latent_patch_dim

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self.vector_hidden_dims

    def __post_init__(self) -> None:
        if self.representation != "continuous_patch_grid":
            raise ValueError(
                "the primary Genie2 representation is a continuous patch grid"
            )
        positive = (
            self.patch_size,
            self.latent_patch_dim,
            self.model_dim,
            self.ffn_dim,
            self.num_blocks,
            self.num_heads,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("tokenizer dimensions must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.max_mask_ratio < 1.0:
            raise ValueError("max_mask_ratio must be in [0, 1)")


@dataclass(frozen=True)
class LAMConfig:
    enabled: bool = False
    kind: str = "continuous_extension"
    latent_action_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    kl_scale: float = 1e-2

    def __post_init__(self) -> None:
        if self.kind != "continuous_extension":
            raise ValueError("continuous LAM is an explicitly named extension")
        if self.latent_action_dim <= 0:
            raise ValueError("latent_action_dim must be positive")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")


@dataclass(frozen=True)
class DynamicsConfig:
    objective: str = "diffusion_forcing_x_prediction"
    model_dim: int = 512
    ffn_dim: int = 2048
    num_heads: int = 8
    num_blocks: int = 6
    max_context: int = 32
    classifier_free_dropout: float = 0.1
    guidance_scale: float = 1.5
    denoising_steps: int = 25
    ramp_weight: bool = True
    context_corruption: float = 0.1
    dropout: float = 0.0
    compute_dtype: str = "bfloat16"
    parameter_dtype: str = "float32"

    @property
    def num_layers(self) -> int:
        return self.num_blocks

    @property
    def sampling_steps(self) -> int:
        return self.denoising_steps

    def __post_init__(self) -> None:
        if self.objective != "diffusion_forcing_x_prediction":
            raise ValueError(
                "the Jasmine-derived baseline uses diffusion-forcing x-prediction"
            )
        positive = (
            self.model_dim,
            self.ffn_dim,
            self.num_heads,
            self.num_blocks,
            self.max_context,
            self.denoising_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("dynamics dimensions must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.classifier_free_dropout < 1.0:
            raise ValueError("classifier_free_dropout must be in [0, 1)")
        if not 0.0 <= self.context_corruption < 1.0:
            raise ValueError("context_corruption must be in [0, 1)")


@dataclass(frozen=True)
class StageOptimizerConfig:
    steps: int
    batch_size: int
    max_learning_rate: float
    warmup_steps: int
    wsd_decay_steps: int
    beta1: float = 0.9
    beta2: float = 0.9
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class LatentPolicyConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    batch_size: int = 16
    imagination_horizon: int = 15
    discount_horizon: int = 333
    discount_lambda: float = 0.95
    entropy_scale: float = 3e-4
    action_penalty: float = 1e-3

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.imagination_horizon <= 0:
            raise ValueError("imagination_horizon must be positive")
        if self.discount_horizon <= 1:
            raise ValueError("discount_horizon must be greater than one")
        if not 0.0 <= self.discount_lambda <= 1.0:
            raise ValueError("discount_lambda must be in [0, 1]")

    @property
    def discount(self) -> float:
        return 1.0 - 1.0 / self.discount_horizon


@dataclass(frozen=True)
class Genie2ContinuousConfig:
    specification: str = "genie2_public_latent_diffusion"
    implementation_profile: str = "jasmine_diffusion_paper"
    action_dim: int = 6
    action_mode: str = "continuous"
    action_low: tuple[float, ...] | None = None
    action_high: tuple[float, ...] | None = None
    observation_shape: tuple[int, ...] = (64, 64, 3)
    representation: str = "continuous_latent_patch_grid"
    conditioning_mode: str = "real_action"
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    lam: LAMConfig = field(default_factory=LAMConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    tokenizer_optimizer: StageOptimizerConfig = field(
        default_factory=lambda: StageOptimizerConfig(300_000, 48, 3e-4, 10_000, 30_000)
    )
    dynamics_optimizer: StageOptimizerConfig = field(
        default_factory=lambda: StageOptimizerConfig(200_000, 36, 1e-4, 5_000, 20_000)
    )
    latent_policy: LatentPolicyConfig = field(default_factory=LatentPolicyConfig)
    reward_continue_hidden_dims: tuple[int, ...] = (256, 256)
    bridge_hidden_dims: tuple[int, ...] = (128,)
    vq_maskgit_ablation_enabled: bool = False

    def __post_init__(self) -> None:
        if self.specification != "genie2_public_latent_diffusion":
            raise ValueError("unknown Genie2 public specification")
        if self.implementation_profile != "jasmine_diffusion_paper":
            raise ValueError("unknown Genie2 implementation profile")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be continuous or discrete")
        if self.action_mode == "continuous":
            if self.action_low is not None and len(self.action_low) != self.action_dim:
                raise ValueError("action_low must match action_dim")
            if (
                self.action_high is not None
                and len(self.action_high) != self.action_dim
            ):
                raise ValueError("action_high must match action_dim")
        if self.representation != "continuous_latent_patch_grid":
            raise ValueError(
                "primary Genie2 arm must preserve a continuous latent patch grid"
            )
        if self.conditioning_mode not in {"real_action", "continuous_lam_extension"}:
            raise ValueError("unknown conditioning_mode")
        if self.conditioning_mode == "real_action" and self.lam.enabled:
            raise ValueError(
                "LAM cannot be enabled in the public Genie2-conformant arm"
            )
        if (
            self.conditioning_mode == "continuous_lam_extension"
            and not self.lam.enabled
        ):
            raise ValueError("continuous_lam_extension requires lam.enabled")
        if self.vq_maskgit_ablation_enabled:
            raise ValueError("VQ/MaskGIT belongs in a separate Genie1 ablation")

    @property
    def is_image_observation(self) -> bool:
        return len(self.observation_shape) == 3

    @classmethod
    def debug(
        cls,
        *,
        action_dim: int,
        observation_shape: tuple[int, ...],
        action_mode: str = "continuous",
        action_low: tuple[float, ...] | None = None,
        action_high: tuple[float, ...] | None = None,
    ) -> Genie2ContinuousConfig:
        return cls(
            action_dim=action_dim,
            action_mode=action_mode,
            action_low=action_low,
            action_high=action_high,
            observation_shape=observation_shape,
            autoencoder=AutoencoderConfig(
                patch_size=4,
                latent_patch_dim=8,
                model_dim=16,
                ffn_dim=32,
                num_blocks=1,
                num_heads=2,
                max_mask_ratio=0.5,
                compute_dtype="float32",
                vector_hidden_dims=(16, 16),
            ),
            dynamics=DynamicsConfig(
                model_dim=16,
                ffn_dim=32,
                num_heads=2,
                num_blocks=1,
                max_context=8,
                denoising_steps=2,
                compute_dtype="float32",
            ),
            tokenizer_optimizer=StageOptimizerConfig(8, 2, 3e-4, 0, 0),
            dynamics_optimizer=StageOptimizerConfig(8, 2, 1e-4, 0, 0),
            reward_continue_hidden_dims=(16, 16),
            bridge_hidden_dims=(16,),
            latent_policy=LatentPolicyConfig(hidden_dims=(16, 16), batch_size=4),
        )
