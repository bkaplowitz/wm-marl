# Jafar World Model Architecture

## Scope

`world_marl.jafar` is a source-derived JAX/Flax Linen port of the Jafar
Jafar VQ-VAE tokenizer, latent-action model (LAM), and MaskGIT dynamics stack. The source is
`FLAIROx/jafar` at commit `5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665`.
The port preserves the model equations, tensor layouts, initializers,
straight-through boundaries, losses, schedules, and source-sized defaults.

The package is one independent world-model arm. It is not a compatibility
layer and makes only the source-conformance claims documented here.

## Source model

All image tensors are batch-major `B,T,H,W,C` RGB values in `[0, 1]`. Spatial
patches are flattened into `B,T,N,P`, where `N` is the number of patches and
`P = patch_size**2 * 3`. The source transformer applies non-causal spatial
attention followed by causal temporal attention, then a feed-forward residual,
with sinusoidal position encodings and rematerialized blocks.

### TokenizerVQVAE

- RGB patch size: 4.
- Transformer width: 512.
- Latent width: 32.
- Codebook: 1,024 cosine-normalized vectors.
- Transformer: 8 blocks, 8 heads.
- Encoder output is normalized before nearest-code lookup.
- Quantization uses `x + stop_gradient(code - x)`.
- Decoder output uses sigmoid and is unpatchified to HWC pixels.
- Loss is pixel MSE plus codebook loss plus `0.25 * commitment_loss`.
- Source defaults are 300,000 updates, batch 48, 10,000 warmup updates, and
  AdamW with peak learning rate `3e-4`, `b1=b2=0.9`, and weight decay `1e-4`.

The tokenizer is trained first. Its parameters are frozen for downstream
dynamics training.

### LatentActionModel

- RGB patch size: 16.
- Transformer width: 512; latent width: 32.
- Codebook: 6 cosine-normalized latent actions.
- Transformer: 8 blocks, 8 heads.
- A learned action token is prepended to the spatial patches of every frame.
- Causal temporal attention makes the action state at frame `t+1` represent
  the transition from frame `t` to frame `t+1`.
- The future-frame action states are quantized with the same straight-through
  estimator as the tokenizer.
- The decoder adds the projected action to every projected source-frame patch
  and reconstructs the next frame with a sigmoid output.
- Loss is next-frame pixel MSE plus codebook loss plus
  `0.25 * commitment_loss`.
- Source defaults are 200,000 updates, batch 36, 5,000 warmup updates, and
  AdamW with peak learning rate `3e-5`.
- A code unused for 50 consecutive updates is replaced with a sampled active
  code and its inactivity counter is reset.

The LAM is trained second and is frozen with the tokenizer during Jafar
dynamics training.

### DynamicsMaskGIT

- Frame tokens are the tokenizer's 1,024 discrete codes.
- Latent-action codes are embedded and added to the following frame position.
- Transformer width: 512; 12 blocks; 8 heads.
- During training, one scalar mask probability is sampled uniformly below
  `0.5`, Bernoulli masks are drawn for all frame tokens, and the first frame is
  forcibly unmasked.
- The objective is 1,024-way cross entropy averaged only over masked tokens.
- Tokenizer and LAM outputs are stop-gradient inputs to dynamics.

Sampling generates each future frame with 25 MaskGIT refinement steps. At
refinement step `s`, the source schedule uses
`cos(pi * (s + 1) / (2 * steps))`; confidence-ranked tokens are retained while
the remainder stay masked. Sampling temperature is multiplied by one minus
that cosine value, and the final stochastic step is replaced by argmax. The
repository port expresses both the refinement loop and the autoregressive
frame loop with `jax.lax.scan`.

## Repository integration

Repository-specific behavior lives in `world_marl.latent_action_world_model`,
not in the source-derived Jafar modules. It provides:

- conversion from time-major replay to valid batch-major RGB transitions;
- learned reward and continuation heads;
- expert latent-code-to-real-action calibration;
- decoded-pixel simulator state and PPO integration;
- checkpoint, metric, provenance, media, and evaluation artifacts.

These extensions are not attributed to Jafar. Gradients from the reward head,
continuation head, simulator, and PPO are stopped at the tokenizer, LAM, and
dynamics boundary. PPO consumes decoded HWC pixels and reuses the repository's
existing `CNNActorCritic`, GAE, and `ppo_update` implementation.

## Execution invariants

Training updates, MaskGIT refinement, autoregressive generation, simulator
rollouts, PPO minibatches/epochs, and JAX-native evaluation use
`jax.lax.scan`. Metrics remain on device until logging or a phase boundary.
There is no Python-loop fallback in these runtime paths.
