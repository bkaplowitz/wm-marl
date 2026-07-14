# Genie2 Public-Source Conformance

Primary public source: [Genie 2: A large-scale foundation world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/).

Google DeepMind has not published a Genie2 paper, model code, weights, parameter counts, optimizer settings, data recipe, or complete architecture. Consequently, this package cannot honestly be called an exact Genie2 reproduction. Its primary arm is constrained to match every architecture fact in the public disclosure, and every undisclosed choice is named by source.

## Publicly Disclosed Contract

| Component | Required behavior | Public evidence |
| --- | --- | --- |
| Input | A prompt image initializes an interactive world | Genie2 announcement and architecture diagram |
| Representation | Frames pass through an autoencoder into latent frame grids | `Diffusion world model` section and diagram |
| Dynamics | A large transformer dynamics model uses a causal temporal mask | `Diffusion world model` section |
| Objective | Autoregressive latent diffusion | `Diffusion world model` section |
| Conditioning | Individual keyboard, mouse, agent, or environment actions plus past latent frames | Architecture section and diagram |
| Inference | Generate one next latent frame at a time and decode it | Architecture section and diagram |
| Control | Classifier-free guidance improves action controllability; dropout rate and guidance scale are not disclosed | `Diffusion world model` section |

The conformant primary mode is therefore `conditioning_mode="real_action"`. A latent action model or latent-to-real bridge is not on this path.

## Jasmine Substitutions

Undisclosed implementation details follow the continuous diffusion baseline in [Jasmine](https://arxiv.org/abs/2510.27002) and its [Apache-2.0 repository](https://github.com/p-doom/jasmine):

| Component | Jasmine-derived choice |
| --- | --- |
| Tokenizer | ST-ViViT masked autoencoder over RGB patches |
| Latent representation | Continuous per-frame patch grid, bounded with `tanh`; no VQ codebook |
| Tokenizer profile | Patch 16, latent patch width 32, model width 512, FFN 2048, 4 axial blocks, 8 heads, maximum mask ratio 0.9 |
| Dynamics profile | Width 512, FFN 2048, 6 axial blocks, 8 heads |
| Activation memory | Rematerialize each axial transformer block during backpropagation |
| Conditioning | Project each action and prepend it as a per-frame spatial token |
| Diffusion forcing | Independently sample a signal level for every frame, linearly mix data and Gaussian noise, and predict clean latent patches (`x`-prediction) |
| Loss | Per-frame latent MSE weighted by `0.9 * signal_level + 0.1` |
| Sampling | Target context corruption 0.1, quantize its signal to the nearest denoising level, and perform frame-by-frame denoising; Jasmine's current sampler default is 64 denoising steps |
| Precision | Float32 parameters and bfloat16 compute |
| Optimization | Staged tokenizer and dynamics training with independent AdamW optimizers (`beta1=beta2=0.9`, weight decay `1e-4`) and WSD schedules |

These are Jasmine choices, not facts disclosed by Google DeepMind. The Linen implementation cites Jasmine but does not import its JAX 0.7/Flax NNX stack.

## Explicit Extensions

- Vector observations use a continuous MLP adapter and a one-patch latent grid. This enables Brax/Gymnax comparisons but is not a visual Genie2 architecture claim.
- Reward and continue heads and controller training are repository RL extensions. They do not alter the latent diffusion objective.
- The `reinforce` controller uses Dreamer-style discounting and a tanh-squashed Gaussian for bounded continuous actions. The `candidate-distill` controller treats each bounded candidate as the first action of a scanned latent rollout through the frozen dynamics, reward head, and continue head, then distills the highest-return action. Both are extension choices, not Genie2 or Jasmine claims.
- Candidate planning uses a single target-level-zero clean-latent prediction per transition instead of the full 25-step visual sampler. It preserves Jasmine's context corruption and maximum context-timestep embedding, uses shared rollout noise across candidates, and does not train a scalar critic.
- Candidate rollout transitions are reported as imagined transitions. The number of first-action candidate evaluations is reported separately.
- `continuous_lam_extension` is an optional unlabeled-video experiment. It is neither disclosed Genie2 nor faithful Jasmine, whose published LAM is discrete VQ.
- VQ/MaskGIT belongs only to a separately named Genie1 ablation.
- Jafar is a Genie1/DreamerV3 implementation reference and is not a Genie2 specification.

## Execution Gates

- Image latents must remain rank-4 `[batch,time,patch,latent]` through tokenizer, dynamics, and sampling.
- Primary dynamics must receive environment/user actions, never inferred latent actions.
- Tokenizer, dynamics, and RL heads train in separate compiled phases.
- Update scans, policy imagination, denoising, frame generation, and JAX-native environment evaluation use `jax.lax.scan`.
- Visual real-environment evaluation carries reset-aware frame history inside that scan; vector adapters use a one-step history.
- No primary artifact may report a latent bridge, LAM usage, VQ codebook, or MaskGIT metric.
