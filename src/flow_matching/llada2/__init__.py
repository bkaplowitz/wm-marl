"""Faithful LLaDA2.0 block-diffusion arm (arXiv 2512.15745).

Mirrors the parent package layout: :mod:`flow_matching.llada2.models` holds the
RoPE (+YaRN) / RMSNorm / SwiGLU-MoE backbone, :mod:`flow_matching.llada2.paths`
the absorbing (masked) forward process and the eq-3 block-diffusion attention
mask, :mod:`flow_matching.llada2.train` the BDLM/SFT loss and the WSD block-size
curriculum, and :mod:`flow_matching.llada2.simulate` the block-by-block
hybrid-confidence sampler.
"""

from flow_matching.llada2.models import (
    BlockDiffusionTransformer,
    MoELayer,
    RoPEAttention,
    SwiGLUExpert,
    apply_rope,
    rotate_half,
)
from flow_matching.llada2.paths import (
    absorbing_loss_weight,
    block_diffusion_attention_mask,
    complementary_absorbing_pair,
    mask_schedule,
    sample_absorbing_path,
    sample_t_in_bandwidth,
)
from flow_matching.llada2.simulate import sample_llada2_block_diffusion
from flow_matching.llada2.train import (
    create_llada2_train_state,
    llada2_bdlm_loss,
    llada2_train_step,
    topk_checkpoint_merge,
    wsd_block_size_schedule,
)

__all__ = [
    "BlockDiffusionTransformer",
    "MoELayer",
    "RoPEAttention",
    "SwiGLUExpert",
    "absorbing_loss_weight",
    "apply_rope",
    "block_diffusion_attention_mask",
    "complementary_absorbing_pair",
    "create_llada2_train_state",
    "llada2_bdlm_loss",
    "llada2_train_step",
    "mask_schedule",
    "rotate_half",
    "sample_absorbing_path",
    "sample_llada2_block_diffusion",
    "sample_t_in_bandwidth",
    "topk_checkpoint_merge",
    "wsd_block_size_schedule",
]
