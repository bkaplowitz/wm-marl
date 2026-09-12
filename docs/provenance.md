# Cleanup provenance

The source of truth was the deployed MA-JEPA learner used for the September 12
world-model size / learning-rate sweep. The remote Git branch was older than that
deployed source, so this cleanup imported the working implementation before
removing inactive paths.

- Base branch: `origin/ma-jepa-ppo`, commit `0b98013c23c5199ce0569e57d5ac055900e7c163`.
- Deployed-source fingerprint: `85eb6779590e26a76f226536359ac15327006eef7b55cab20d464d728b343eb6`.
- Fingerprint algorithm: SHA-256 over sorted paths relative to `src/majepa`, each
  path followed by NUL, file bytes, then NUL; include `.py`, `.yaml`, `.yml` files.
- Upstream Embodied/DreamerV3 submodule: `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- JAX/JAXlib 0.4.36, Ninjax 3.6.3, Elements 3.22.0, Granular 0.23.1, Portal 3.8.1,
  Scope 0.7.1, NumPy 1.26.4 and Optax 0.2.5 match the deployed Python runtime.

The CUDA lock retains the pod's cuDNN 9.25.0.15 for reproducibility. PyPI has
[yanked that release](https://pypi.org/project/nvidia-cudnn-cu12/9.25.0.15/);
this cleanup does not claim to validate a replacement GPU runtime.

First-party Python shrank from 46 files / 12,614 lines in the deployed snapshot
to 39 files / 8,440 lines, excluding tests and the pinned upstream runtime.

## Removed

Learned teammate belief and residual adapters; spatial encoders and decoder
paths; the alternate direct joint latent head; factual-value/V-trace and
representation-value treatments; experimental entropy scheduling; obsolete
local-only learner fallbacks; alternate mask calibration; fresh-history and
feedback-gradient treatments; old configuration and launch manifests; unused
backend indirection; one-off experiment/diagnostic launchers and their tests.

The running algorithm's losses, numerical operations, module parameter names,
RNG order, optimizer groups, ordinary training metrics and checkpoint state were
preserved. Empty legacy carry positions remain to preserve that state contract.
The generic value-head size that did not configure the centralized critic was
removed. Sizes and rates for the active sweep remain explicit profiles.

The new final-evaluation launcher uses each checkpoint's saved configuration,
checks checkpoint completion, and refuses to overwrite an existing evaluation.
Operational defaults use local logging, a fresh timestamped output directory,
and 100 episodes for standalone final evaluation. Training-curve evaluations
remain 32 episodes. No running pod job or remote queue was modified.

## Validation

A deterministic CPU/JIT comparison used identical synthetic team replay, deaths,
resets, seeds and small model widths on the immutable deployed snapshot and the
cleaned source. It included dense posterior alignment, the action-margin loss,
self-fed two-step BPTT, imagined PPO and the replay value objective. All compared
arrays were exactly equal (`numpy.array_equal`), not merely within tolerance:

| Comparison | Array leaves |
| --- | ---: |
| Initialization | 525 |
| Online policy result | 34 |
| Reports | 239 |
| Second learner update outputs | 283 |
| State after two learner updates | 525 |

The maintained suite passes **32 tests**, covering training, replay history,
optimizer ownership and nonfinite rollback, PPO frozen-support consistency,
representation gradient boundaries, team rewards/deaths, evaluation quotas and
seeds, saved evaluation architecture, sweep configurations, and the SMAC adapter.
Lint, formatting and command-line help were also checked. A fresh CPU development
environment installs successfully from the lockfile.

These checks validate the cleanup at small dimensions. They are not a new
full-size GPU/StarCraft training experiment. Existing reported win rates belong
to the original deployed source, not newly trained `clean_jepa` checkpoints.
