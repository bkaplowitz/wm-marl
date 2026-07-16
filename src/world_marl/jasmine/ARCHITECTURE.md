# Jasmine World Model Architecture

## Scope

`world_marl.jasmine` is a source-derived JAX/Flax Linen port of the Jasmine
continuous-tokenizer diffusion baseline. The source is `p-doom/jasmine` at
commit `420859bc99eecf6b07a7e9edf65d5d145935f1e1`. The upstream implementation
uses Flax NNX; this repository translates it to Linen on JAX 0.4.36 and Flax
0.10.4 without changing equations, layouts, initializers, dtypes, or
stop-gradient boundaries.

Jasmine is an independent world-model arm. It is not an alias for Jafar or the
removed `genie2_continuous_jax` package.

## Shared axial transformer

All model inputs are batch-major `B,T,N,D`. Each block applies non-causal
spatial self-attention, causal temporal self-attention after swapping the time
and patch axes, and a two-layer GELU feed-forward network. Sinusoidal temporal
and spatial positions are added separately. Blocks are rematerialized.

Parameters remain float32. Dense and attention compute use bfloat16. Layer
normalization executes in float32, including its statistics and parameters.
The GPU path uses `jax.nn.dot_product_attention(..., implementation="cudnn")`,
with source-equivalent padding to a multiple of four and temporal causality.
The CPU test path preserves the same attention semantics without requesting
cuDNN.

### TokenizerMAE

- RGB patch size: 16.
- Transformer width: 512; FFN width: 2,048.
- Continuous latent patch width: 32.
- Encoder and decoder: 4 axial blocks, 8 heads.
- During training, every frame receives an independently sampled mask ratio
  uniformly below 0.9; patch masks are then sampled independently.
- Masked input patches use one learned LeCun-uniform mask patch.
- Encoder latents are bounded by tanh.
- Decoder pixels use float32 sigmoid before returning to the configured compute
  dtype and unpatchifying.
- Objective: pixel MSE only.
- Source defaults: sequence length 16, 300,000 updates, batch 48, 10,000
  warmup updates, peak AdamW learning rate `3e-4`, and 30,000-step WSD decay.

The tokenizer is frozen while training diffusion dynamics.

### LatentActionModel

- RGB patch size: 16.
- Transformer width: 512; FFN width: 2,048; latent width: 32.
- Encoder and decoder: 4 axial blocks, 8 heads.
- A learned action token is prepended per frame; future-frame action-token
  states encode the preceding transitions.
- Six cosine-normalized codes are selected with nearest-code lookup and
  `x + stop_gradient(code - x)`.
- The reconstruction path projects and broadcasts each action over the source
  frame patches, then predicts the next frame through a sigmoid decoder.
- Standalone LAM defaults are 200,000 updates, batch 36, 5,000 warmup updates,
  peak `3e-5`, 20,000-step WSD decay, and inactive-code reset after 50 updates.

The default dynamics configuration has `use_gt_actions=False`. It removes the
LAM decoder, freezes the tokenizer, and jointly trains the LAM encoder/codebook
with diffusion dynamics when no pretrained LAM checkpoint is supplied. It has
no direct real-action conditioning.

### DynamicsDiffusion

- Transformer width: 512; FFN width: 2,048; 6 axial blocks; 8 heads.
- Each frame receives an independently sampled integer diffusion level.
- With signal level `t = level / denoise_steps`, source noising is
  `(1 - (1 - 1e-5) * t) * noise + t * clean_latent`.
- Projected action and diffusion-level tokens are prepended to every frame's
  latent patch sequence.
- The model predicts the clean latent (`x` prediction), not noise or velocity.
- Per-frame latent MSE is weighted by `0.9 * t + 0.1`.
- Source defaults: sequence length 16, 200,000 updates, batch 36, 5,000
  warmup updates, peak `1e-4`, and 20,000-step WSD decay.

Sampling uses 64 denoising steps, autoregressive future frames, and context
corruption factor 0.1. It performs no classifier-free guidance and introduces
no direct real-action conditioning. Denoising and autoregressive frame loops
are expressed with nested `jax.lax.scan`.

## Repository integration

Repository-specific replay conversion, reward/continuation prediction,
expert-action calibration, simulator/PPO, artifacts, and evaluation live only
in `world_marl.latent_action_world_model`. They are not Jasmine source
behavior. The full tokenizer, LAM, and dynamics are frozen during PPO, and PPO
observes decoded HWC pixels from the complete 64-step sampler.

## Execution invariants

Tokenizer/LAM/dynamics updates, diffusion denoising, autoregressive sampling,
simulator rollouts, PPO minibatches/epochs, and JAX-native evaluation use
`jax.lax.scan`. There is no Python-loop fallback, and metrics are materialized
only for logging or at phase boundaries.
