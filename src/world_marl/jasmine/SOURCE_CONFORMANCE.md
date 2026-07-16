# Jasmine Source Conformance

## Pinned source

- Repository: `https://github.com/p-doom/jasmine`
- Commit: `420859bc99eecf6b07a7e9edf65d5d145935f1e1`
- License: Apache License 2.0

Every adapted Python module must identify this repository, commit, upstream
path, and its local integration changes in the module docstring.

## Port map

| Upstream path | Local responsibility | Required conformance | Intentional integration changes |
| --- | --- | --- | --- |
| `jasmine/utils/preprocess.py` | patchify/unpatchify | source padding, axes, crop | packaged imports and typed APIs |
| `jasmine/utils/nn.py` | positions, axial blocks/transformer, VQ, cuDNN attention | residual order, two-layer FFN, remat, full-precision norm, bf16 compute, float32 params, cosine VQ, attention padding/masks | NNX state becomes Linen params; CPU semantic path avoids unavailable cuDNN |
| `jasmine/models/tokenizer.py` (`TokenizerMAE`) | continuous MAE tokenizer | per-frame masking, learned mask patch, tanh latent, sigmoid decoder | Linen module/config split; no VQ behavior added |
| `jasmine/models/lam.py` | discrete VQ LAM | action-token layout, 6x32 cosine VQ, straight-through boundary | Linen translation and reusable encode API |
| `jasmine/models/dynamics.py` (`DynamicsDiffusion`) | diffusion-forcing dynamics | per-frame levels, linear noise mixture, x-pred, action/level token layout | pure noising/weight helpers for tests |
| `jasmine/models/genie.py` (`GenieDiffusion`) | frozen tokenizer, optional LAM co-training, sampling | `use_gt_actions=False`, deleted LAM decoder, stop gradients, context corruption, 64-step sampling | nested `jax.lax.scan` replaces NNX scan; no Genie public name |
| `jasmine/utils/train_utils.py` | WSD/cosine schedules | warmup, stable plateau, final linear decay | repository optimizer/config interfaces |
| `jasmine/baselines/diffusion/train_tokenizer_mae.py` | MAE defaults/loss | 300k/48/10k, 30k WSD, `3e-4`, pixel MSE | scan training and repository artifacts |
| `jasmine/baselines/train_lam.py` | LAM defaults/loss/reset | 200k/36/5k, 20k WSD, `3e-5`, reset 50 | functional codebook reset under JIT/scan |
| `jasmine/baselines/diffusion/train_dynamics_diffusion.py` | dynamics defaults/loss/composition | 200k/36/5k, 20k WSD, `1e-4`, ramp-weighted x-pred, co-trained LAM | repository checkpointing and scan phase loop; no Grain/ArrayRecord dependency |
| `jasmine/baselines/diffusion/sample_diffusion.py` | sampling defaults | 64 steps, context corruption 0.1, no CFG | repository media/evaluation and scan orchestration |

## Exact source behavior

The local arm must retain:

- `B,T,N,D` layouts and separate spatial/temporal positions;
- spatial non-causal and temporal causal attention;
- NNX initializers and parameter shapes after Linen translation;
- float32 parameters and normalization, bf16 dense/attention compute;
- block rematerialization and GPU cuDNN dot-product attention semantics;
- per-frame MAE mask probabilities and tanh latent bound;
- source action-token selection, cosine codebook, and straight-through gradient;
- frozen tokenizer and conditional LAM stop-gradient/co-training boundary;
- diffusion noising equation, clean-latent target, ramp loss, context corruption,
  and denoising order;
- source update counts, batch sizes, warmups, WSD decays, and AdamW settings.

No Torch, Procgen, Grain, or ArrayRecord dependency is ported. Those omissions
remove source data/runtime infrastructure, not model behavior.

## Repository-only behavior

The following are local extensions and must not be attributed to Jasmine:

- time-major replay conversion, HWC validation, and boundary filtering;
- reward/continuation heads and DreamerV3 reward distribution reuse;
- expert calibration provenance and latent-code action bridge;
- replay-context simulator reset, CNN PPO, GAE, and real evaluation;
- repository configs, checkpoints, JSONL metrics, media, outcome, and summary;
- the explicit prohibition on Python-loop fallbacks in runtime paths.

## Conformance evidence

Tests lock exact defaults and cover layouts, per-frame masks and diffusion
levels, causal attention, dtype/parameter trees, VQ/straight-through behavior,
code reset, noising/weighting, NNX-to-Linen parameter equivalence, sampler
lowering, freezing/co-training, and deterministic tiny overfit behavior.
Source-sized 64x64 forward/backward/sampler checks are required on GPU.
