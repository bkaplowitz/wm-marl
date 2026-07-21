# Genie2 Continuous JAX Architecture

The primary specification is Google DeepMind's public [Genie 2 architecture disclosure](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/). There is no public Genie2 paper or code. This package is therefore a transparent public-source implementation, not an exact reproduction claim.

Undisclosed details are taken from the continuous diffusion baseline in the [Jasmine paper](https://arxiv.org/abs/2510.27002) and [Jasmine repository](https://github.com/p-doom/jasmine). [Jafar](https://github.com/FLAIROx/jafar) and the [Genie 1 paper](https://arxiv.org/abs/2402.15391) are references for separately named Genie1 ablations only.

## Primary Contract

```text
RGB frames
  -> ST-ViViT masked autoencoder
  -> continuous latent patch grids z_t

(past latent grids, actual user/environment actions, per-frame diffusion levels)
  -> action-prepended causal axial transformer
  -> clean latent-grid predictions

predicted latent grid
  -> ST-ViViT decoder
  -> next RGB observation
```

The model does not use a VQ-VAE or token codebook. The primary dynamics is not conditioned on an inferred LAM code. It predicts the next continuous latent frame grid from past grids and actual actions.

## Tokenizer

The tokenizer patchifies each RGB frame and alternates spatial and causal temporal attention. During tokenizer training, each sample receives a uniformly sampled patch-mask ratio up to 0.9. The encoder output is bounded by `tanh`; the decoder reconstructs RGB patches through a sigmoid output. Tokenizer reconstruction uses mean pixel MSE. As in Jasmine, each axial block is rematerialized during backpropagation.

The Jasmine profile is patch size 16, latent patch width 32, model width 512, FFN width 2048, four axial blocks, and eight heads. A debug profile changes capacity only; it does not change the architecture or objective.

## Diffusion Dynamics

For each frame independently, training samples an integer diffusion level and constructs

```text
x_noised = (1 - (1 - 1e-5) * signal) * noise + signal * x
```

An action projection and diffusion-level embedding are prepended to every frame's spatial patch sequence. A transformer applies non-causal spatial attention and causal temporal attention. It predicts the clean latent grid directly. The loss is latent MSE with Jasmine's ramp weight `0.9 * signal + 0.1`.

Classifier-free training drops the complete action sequence for a sampled training sequence, so the unconditional branch used during sampling is trained directly. Sampling evaluates conditional and unconditional predictions and combines them as

```text
x_guided = x_unconditional + guidance_scale * (x_conditional - x_unconditional)
```

Genie2 publicly discloses classifier-free guidance but not the conditioning-drop probability or guidance scale; those values are local, explicit configuration choices. Interactive generation uses nested scans: the outer scan generates one frame at a time, and the inner scan performs the 25 denoising steps reported for the diffusion baseline in Jasmine Appendix C. Jasmine targets 0.1 context corruption and quantizes the corresponding signal to the nearest denoising level, which is `22 / 25` in the paper profile. Jasmine's current sampling script defaults to 64 steps; that source/paper discrepancy is not silently folded into the paper profile.

## Training Stages

1. Train the masked autoencoder with its own AdamW/WSD optimizer.
2. Freeze tokenizer gradients and train action-conditioned latent diffusion with a separate AdamW/WSD optimizer.
3. Freeze the generative model and fit reward/continue heads from environment labels.
4. As a repository RL extension, train a controller without changing the frozen world model. The `reinforce` baseline trains a policy and critic from scanned simulator rollouts using Dreamer-style discount horizon 333, lambda 0.95, and a tanh-squashed Gaussian for bounded continuous actions. The separately labeled `candidate-distill` extension samples bounded first actions at replay latent histories and rolls each candidate forward through the frozen action-conditioned dynamics, reward head, and continue head. Later rollout actions come from the current policy. The planner uses one clean-latent prediction at target diffusion level zero per transition, with Jasmine's context corruption and timestep convention, rather than the full 25-step visual sampler. It distills the highest-return first action without training a scalar critic. These planner transitions are counted as imagined transitions; first-action candidate evaluations are reported separately.
5. Evaluate that policy's actual actions directly in the real adapter. Visual evaluation carries a reset-aware observation history through the environment scan so the causal tokenizer receives temporal context.

The real-action policy does not require a latent-to-real bridge.

## Non-Visual Adapters

Brax, most Gymnax tasks, and the MJX-backed MuJoCo Playground DMC tasks used here return vectors. For those comparisons, a symlog-free continuous MLP adapter produces a one-patch continuous latent grid. The diffusion model, direct-action conditioning, reward/continue heads, policy training, and evaluation remain the same. Results from this path are labeled `observation_mode="vector"`; they are not evidence of visual-generation quality.

## Optional Experiments

`continuous_lam_extension`, reward/continue heads, simulator actor-critic training, and candidate distillation are explicit extensions. Candidate distillation and its fast one-prediction planning transition are repository control experiments, not disclosed Genie2 or Jasmine components. VQ/MaskGIT is a Genie1 ablation. LeWM/LeJEPA additions remain separately named ablations and never enter this baseline silently.
