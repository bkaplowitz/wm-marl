# DreamerV3 Source Conformance

Primary specification: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104), especially equations 1-12, the supplementary `Networks`, `Optimizer`, `Distributions`, and `Experience replay` sections, and Tables 2-4.

Implementation cross-check: [official DreamerV3 JAX repository](https://github.com/danijar/dreamerv3). Where the current repository differs from the paper, this baseline follows the paper and records the difference in tests.

## Required Baseline

| Component | Required behavior | Source |
| --- | --- | --- |
| Image observations | Stride-2 CNN encoder to a 4x4 grid for 64x64 inputs; transposed stride-2 CNN decoder with sigmoid output | Networks supplement |
| Vector observations | Symlog input and target transform; three-layer MLP encoder and decoder | Equations 9-11; Networks supplement |
| RSSM | Deterministic Block GRU plus 32 categorical stochastic latents; straight-through categorical sampling | Equation 1; Tables 3-4 |
| 12M profile | 256 hidden units, 2048 deterministic units in 8 blocks, 16 base CNN channels, 16 classes per latent | Model-size supplement and the stated `8d` construction |
| Categorical distributions | 99% predicted probabilities plus 1% uniform probabilities | World model learning; Distributions supplement |
| Prediction losses | Reconstruction, symexp two-hot reward, and Bernoulli continue losses | Equations 2-3 and 10-11 |
| KL losses | Dynamics KL scale 1.0, representation KL scale 0.1, separate stop gradients, one free nat | Equations 2-3; Table 4 |
| Networks | RMSNorm followed by SiLU; one hidden layer in reward/continue heads; three hidden layers in actor/critic | Networks supplement |
| Initialization | Zero output weights for reward and critic distributions | Critic learning; Distributions supplement |
| Imagination | Horizon 15; discount `gamma = 1 - 1/333`; lambda 0.95; predicted continuation multiplies gamma | Equations 4-5; Table 4 |
| Actor | REINFORCE estimator for discrete and continuous actions, 1% unimix, entropy scale 3e-4, 5th-95th percentile return normalization with lower scale bound 1 | Equation 6; Table 4 |
| Critic | Symexp two-hot distribution, imagined loss scale 1, replay loss scale 0.3, EMA regularizer scale 1 with decay 0.98 | Equation 5 and Critic learning; Table 4 |
| Optimizer | AGC 0.3, LaProp with RMS normalization before momentum, learning rate 4e-5, epsilon 1e-20, no weight decay or annealing | Optimizer supplement; Table 4 |
| Replay | Uniform replay, capacity 5M, batch size 16, batch length 64, online queue semantics, default train ratio 32 | Experience replay supplement; Table 4; author default config |
| Execution | Actor collection, replay insertion, joint learner updates, recurrence, and imagination use nested `jax.lax.scan`; stacked metrics transfer after the compiled phase | Author train-loop ordering, expressed as a JAX-native scheduler |

No LeWM, Genie, Jasmine, prioritized replay, ensemble, target-encoder, or alternative optimizer feature belongs in this baseline.

## Recorded Source Erratum

The 12M row in Table 3 prints 1024 recurrent units, but the surrounding text
defines the recurrent width as `8d`, which is 2048 for `d = 256`. The official
DreamerV3 configuration that accompanies this version of the architecture also
uses 2048. This baseline therefore uses 2048 and keeps this discrepancy visible
instead of silently selecting one of the conflicting values.
