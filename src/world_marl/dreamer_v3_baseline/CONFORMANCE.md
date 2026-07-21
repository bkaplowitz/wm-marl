# DreamerV3 Conformance Matrix

Primary source: [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). Implementation details not fully specified in the text are checked against the authors' [DreamerV3 repository](https://github.com/danijar/dreamerv3).

The baseline configuration is the paper's 12M control profile. Debug configurations may reduce widths and batch sizes only; they must not change the algorithm, distributions, loss definitions, replay semantics, or update ordering.

| Requirement | Source | Local implementation | Executable check |
| --- | --- | --- | --- |
| CNN for image observations; symlog MLP for vectors | Methods, Networks | `models.py` | image/vector encoder and decoder tests |
| 8-block GRU, 2048 deterministic units, 32 categorical latents with 16 classes | Tables 3 and 4, 12M profile | `config.py`, `rssm.py` | exact profile and RSSM shape tests |
| 1% categorical uniform mixture and straight-through samples | World model, Distributions | `rssm.py`, `imagination.py` | categorical and actor distribution tests |
| Reconstruction, symexp two-hot reward, Bernoulli continue, balanced KL, 1 free nat | Equations 2 and 3, Table 4 | `losses.py`, `training.py` | loss-target and finite-gradient tests |
| Imagined REINFORCE actor, lambda returns, percentile return scale, entropy 3e-4 | Equations 5 and 6, Table 4 | `imagination.py`, `training.py` | actor/critic and return tests |
| Distributional critic, replay critic scale 0.3, EMA regularizer and decay 0.98 | Critic learning, Table 4 | `models.py`, `training.py` | critic and EMA tests |
| Uniform replay with online queue and stored latent-state refresh | Methods, Experience replay | `replay.py` | online-first sampling and write-back tests |
| Actor collection, replay insertion, and joint updates interleave at train ratio 32 | Author `embodied/run/train.py`, default config | `training.py`, real-environment CLI path | online scheduler and Brax/Gymnax/MJX smoke tests |
| AGC(0.3) followed by LaProp with beta1 0.9, beta2 0.99, epsilon 1e-20 | Methods, Optimizer | `training.py` | optimizer lowering and exact-default tests |
| Terminal replay states do not seed actor/critic gradients | Author implementation's imagination weighting | `imagination.py` | terminal-start weight test |

The Brax, Gymnax, and MJX DMC entry points interleave actor collection, replay
insertion, and joint model/actor/critic updates in nested `jax.lax.scan` loops.
Like the author's `Ratio` scheduler, training waits for a full replay batch,
performs one update on the first eligible step, and applies the configured replayed
transitions per environment transition thereafter. The fixed synthetic path remains
a component smoke and is not a DreamerV3 benchmark run.
