# Jafar Source Conformance

## Pinned source

- Repository: `https://github.com/FLAIROx/jafar`
- Commit: `5ff9fc7d5d744c8c2797ba3ad0a095ed7f2e2665`
- License: Apache License 2.0

Every adapted Python module must include a module docstring naming this
repository, commit, upstream path, and the integration changes made in this
repository.

## Port map

| Upstream path | Local responsibility | Required conformance | Intentional integration changes |
| --- | --- | --- | --- |
| `utils/preprocess.py` | patchify/unpatchify | padding, axis layout, crop behavior | imports use packaged `dm_pix`/`einops`; typed APIs |
| `utils/nn.py` | position encoding, ST blocks, transformer, VQ | spatial non-causal attention, temporal causal mask, residual order, remat, cosine VQ, initializers | packaged Linen names and explicit scan-compatible interfaces |
| `models/tokenizer.py` | `TokenizerVQVAE` | patch 4, 512/32 widths, 1,024 codes, 8 blocks/heads, sigmoid decoder | configuration dataclass and artifact-facing methods |
| `models/lam.py` | discrete `LatentActionModel` | action-token layout, future-frame state selection, 6x32 cosine VQ, patch 16, 8 blocks/heads | configuration dataclass; reusable encoder interface |
| `models/dynamics.py` | `DynamicsMaskGIT` | first-frame protection, random mask below 0.5, action alignment, 1,024 logits | pure functions for mask/schedule testing |
| `genie.py` | composition and MaskGIT sampler | frozen tokenizer/LAM boundary, cosine refinement schedule, confidence masking | `jax.lax.scan` replaces Linen/Python loop orchestration; no Genie public name |
| `train_tokenizer.py` | tokenizer loss/defaults | MSE + codebook + 0.25 commitment; 300k/48/10k/3e-4 | repository train state, JSONL metrics, checkpoints |
| `train_lam.py` | LAM loss/defaults/reset | next-frame loss; 200k/36/5k/3e-5; reset at 50 | functional immutable codebook update under JIT/scan |
| `train_dynamics.py` | staged dynamics training | masked CE and frozen tokenizer/LAM | scan-based phase/update loop and repository artifacts |
| `sample.py` | autoregressive sampling | 25 source refinement steps | outer frame generation is `jax.lax.scan`; no per-frame host print |

## Exact source behavior

The local arm must retain:

- source tensor layouts and action-to-next-frame alignment;
- cosine normalization before VQ distance calculation;
- the straight-through estimator and codebook/commitment stop-gradient sides;
- LeCun-uniform learned action/mask tokens and source Linen defaults;
- sigmoid pixel reconstruction;
- source optimizer betas, weight decay, warmup/cosine schedules, update counts,
  and batch sizes;
- the first-frame masking exclusion and masked-only CE denominator;
- source MaskGIT confidence ranking and cosine schedule.

## Repository-only behavior

The following requirements are extensions and must not be described as Jafar
source behavior:

- HWC replay validation and episode-boundary filtering;
- reward and continuation heads;
- expert calibration NPZ validation and the six-code action bridge;
- simulator resets from replay context;
- CNN PPO, GAE, real-environment evaluation, and quality gates;
- repository config, checkpoint, JSONL, provenance, media, outcome, and summary
  formats;
- scan substitution for the source script's Python autoregressive loop.

## Conformance evidence

Tests lock exact defaults and cover shape/layout, causal attention, masks,
cosine VQ, straight-through gradients, code reset, MaskGIT scheduling, sampler
lowering, staged freezing, and deterministic tiny overfit behavior. Source-sized
64x64 forward/backward/sampler checks are required on GPU before acceptance.
