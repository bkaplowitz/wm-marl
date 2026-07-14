# Genie 2 Public-Spec Conformance Matrix

Primary source: Google DeepMind's [Genie 2 architecture disclosure](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/). Google DeepMind has not published a Genie 2 paper, training recipe, code, or weights. Exact reproduction is therefore not a defensible claim.

Undisclosed implementation choices use the continuous diffusion baseline in the [Jasmine paper](https://arxiv.org/abs/2510.27002) and [Jasmine repository](https://github.com/p-doom/jasmine). Every such choice is labeled as a substitution rather than attributed to Genie 2.

| Requirement | Provenance | Local implementation | Executable check |
| --- | --- | --- | --- |
| Autoencoder maps RGB frames to continuous latent frame grids | Official Genie 2 disclosure | `autoencoder.py` | visual tokenizer shape/reconstruction tests |
| Large transformer dynamics with temporal causal masking | Official Genie 2 disclosure | `st_transformer.py`, `dynamics.py` | causal architecture and rollout tests |
| Autoregressive frame-by-frame latent generation from past frames and actions | Official Genie 2 disclosure | `dynamics.py` | nested `lax.scan` sampler test |
| Direct user/environment action conditioning | Official Genie 2 disclosure | `dynamics.py`, `training.py` | config and direct-policy tests |
| Classifier-free guidance for action controllability | Official Genie 2 disclosure; standard whole-sequence dropout is a local choice | `dynamics.py` | guidance and whole-sequence dropout tests |
| ST-transformer masked autoencoder, tanh latent bound, pixel MSE | Jasmine Appendix C | `autoencoder.py` | mask, bound, and reconstruction tests |
| Diffusion forcing with per-frame levels, x-prediction, ramp weighting | Jasmine Appendix C | `dynamics.py` | noising and loss tests |
| Four tokenizer blocks and six dynamics blocks, width 512, FFN 2048, eight heads | Jasmine Table 6 and source | `config.py`, `st_transformer.py` | exact profile and rematerialization tests |
| 25 denoising steps and context corruption targeted at 0.1, then quantized to the nearest diffusion level | Jasmine Appendix C and source | `config.py`, `dynamics.py` | exact profile, quantization, and sampler tests |

Jasmine's current sampling script defaults to 64 denoising steps, while Appendix C reports 25. The `jasmine_diffusion_paper` profile uses 25. A source-script profile, if added, must have a separate name.

The continuous LAM, reward/continue heads, controller training, and any action bridge are repo extensions. The `reinforce` controller uses Dreamer-style discounting and a bounded tanh-squashed Gaussian. The separately labeled `candidate-distill` controller uses scanned, multi-step candidate rollouts through the frozen dynamics and reward/continue heads, then distills the best first action without a scalar critic. Its planning transition uses one target-level-zero clean-latent prediction rather than the full visual denoising sampler. Neither controller is attributed to Genie2 or Jasmine, and both remain separately configured and reported. VQ-VAE, a discrete LAM codebook, and MaskGIT belong to a separately named Genie 1 ablation and are not part of this baseline.
