# World-model gradient experiments

The feature starts from `1d84032`, after fetching all remote branches on
2026-09-19. It retains Osaze's latest model changes and the existing campaign
launcher. Both controls default off.

| Control | Default | Effect |
| --- | --- | --- |
| `agent.world_model_gradients.critic_value_scale` | `0.0` | Positive weight adds factual replay value supervision to the local world-model loss. |
| `agent.world_model_gradients.joint_prediction` | `false` | Lets existing joint prediction losses train their local representation inputs and the categorical encoder/local transitions used by self-fed training. |

The training CLI exposes `--wm-critic-value-scale 0.1` and
`--wm-joint-prediction-gradient`. They are independent and recorded in
`launch.json`, the resolved configuration, and W&B through the existing logger.
Low-level dotted flags and campaign overrides expose the same controls.

## Critic-value supervision

The current local encoder and history model reconstruct factual replay states.
The existing centralized critic predicts values from those features. Its
cross-entropy value loss sends gradients through its input into the local
encoder, categorical encoder, and temporal model. The world-model optimizer
continues to exclude actor and critic parameters. No additional critic or value
head is introduced.

Targets use recorded rewards and detached slow-critic values on recorded
successor states. The bounded V-trace implementation is restored from commit
`47af44e`, with rho and trace coefficients clipped at 1. It uses the configured
replay lambda (0.95) and discount (`1 - 1 / agent.horizon`). The probability ratio
is for the recorded **joint** action: log ratios sum over controllable, present
agents. Collection records each chosen action's log probability after legal
masking, policy mixing, and collection mixing. The action and probability at
index t leave state t; their arrival reward is at t+1. Dead but present agents
retain the team's value target. Episode resets, truncations, terminal arrivals,
and missing roster slots are handled explicitly.

Return targets, importance ratios, and teacher values are detached. Clipping
bounds this replay correction; it does not make finite-sample targets unbiased
or eliminate error from changing latent representations. Metrics under
`ctde/world_model_value/` include clipped-ratio fraction and relative effective
sample size. `loss/world_model_value` is the unscaled auxiliary; the config
records its multiplier.

This is an additional representation objective evaluated once per world-model
update. Ordinary imagined PPO actor/critic optimization and its replay-value
objective remain unchanged. There is no value gradient into the world model
through imagined rewards, and the ordinary PPO batch stays detached. Thus this
experiment tests whether factual value supervision improves the representation,
not full differentiation of PPO through imagination. With the default zero
critic output initialization, feature gradients begin only after the critic
learns nonzero output weights.

A positive scale requires freshly collected replay with `behavior_logprob`;
missing metadata fails instead of assuming an on-policy sample. At zero scale,
collection and replay retain their original schema.

## Joint-prediction gradients

The enabled path opens three existing boundaries:

1. Factual joint prediction takes live local features. Embedding, outcome,
   availability, and multistep losses sharing the joint representation can
   update the local encoder and history model through those features.
2. Dense categorical alignment trains the predicted branch's categorical
   encoder parameters and factual history inputs, as well as the joint predictor.
   The observed categorical distribution remains a stopped teacher.
3. Self-fed training differentiates local transition/categorical parameters and
   their feedback into subsequent joint steps within the existing BPTT window.
   Initial local/joint caches remain detached, and all existing chunk-boundary
   stops remain. This path alone cannot update the observation encoder through
   its detached roots; factual replay provides that encoder path.

EMA embeddings, observed teacher distributions, actions, and target outcomes
remain detached. Sampling and forward calculations are unchanged by the joint
flag; only gradient routing changes. BPTT 1 still cuts cross-step feedback. The
reference uses BPTT 2. Existing loss weights—including the action-margin
objective—are unchanged. Opening their gradients is an experimental choice,
not evidence that all those objectives are beneficial.

## Prepared comparison

`experiments/world-model-gradients.json` reuses the existing campaign launcher:

| Treatment | Value scale | Joint gradients | Seeds |
| --- | ---: | --- | --- |
| baseline | 0 | off | 0, 1, 2 |
| critic-value | 0.1 | off | 0, 1, 2 |
| joint-gradient | 0 | on | 0, 1, 2 |
| both | 0.1 | on | 0, 1, 2 |

The fixed setting is SMAC **2s3z**, five agents, difficulty 7, 50,000 driver
records, 5,000 prefill, single-action imagination horizon 5, and the current
`localmask_reference` profile. It retains WM4096/hidden512/32×64, actor 3×512,
central critic width256/value MLP2×256, fixed entropy0.003, WM LRs1e-4 and
actor/critic LRs3e-5. This latest reference uses independent world and behavior
replay views, **both** sampled with 50% uniform and 50% recent weight. It is not
the earlier uniform-only imagination-root campaign.

The value scale 0.1 is an initial test setting, not a tuned coefficient. Seeds
are matched labels; identical parameter initialization or collection trajectories
across value treatments are not guaranteed, because the auxiliary accesses
policy/critic modules earlier. No strict paired-RNG claim is made.

```sh
uv run --no-sync python -m majepa.campaign init \
  --directory artifacts/world-model-gradients \
  --spec experiments/world-model-gradients.json --dry-run
```

This only resolves configurations. Valid datacenter/volume IDs must replace the
explicit placement placeholders before provisioning. The example requests four
GPU workers, L40 then L40S, and no automatic A100 fallback. No cloud allocation
is performed by preparing or dry-running this experiment. Use the existing
[campaign guide](campaign-launcher.md) for frozen-source staging, monitoring,
checkpoint upload, and final 100-episode greedy evaluation.

Compare each treatment to the new baseline using final evaluation and curves,
and inspect value-loss/trace diagnostics and local/joint gradient norms. The
historical roughly 70% result is context, not a substitute for a matched control.
Three seeds provide an initial comparison; the implementation makes no claim
that either gradient path improves performance.

## Authorized critic-only campaign, September 19

`experiments/critic-value-20260919.json` runs seeds 0, 1, and 2 on one explicitly
requested four-GPU pod, preferring L40 and falling back to L40S. Three independent
workers train; the fourth GPU is unused. The GPU budget is $40, with an eight-hour
pod deadline and a $1.10 per-GPU hourly ceiling. Existing US-KS-2 and US-TX-3
network volumes supply capacity alternatives.
The Australia fallback uses pod-attached storage because OC-AU-1 has no network
volume service; it stages the same verified SC2 assets before workers start.

The reference was recovered from the live `majepa_core_ablations_20260919`
campaign's `resolved.json`, using `nomargin-2s3z-seed0` with its single action-margin
ablation restored to 0.1. This retains BPTT2, horizon5, SIGReg0.05, categorical
32x64, discount horizon333, and **disabled local reward/continuation heads**.
All shared configuration keys match that reference except run paths, retained
curve checkpoints, and the final evaluation episode count. The only additional
model treatment is critic-value scale0.1; joint gradients remain off.

Training is 50,000 steps, followed by 100 greedy evaluation episodes per seed.
Report the arithmetic mean and sample standard deviation (`ddof=1`) across the
three final win rates. The user-recorded comparators in `todo.md` are our roughly
59% baseline (no SD recorded) and DMAWM76.9% with SD13.7 percentage points. Preserve
that distinction when comparing; do not substitute evaluation episode variance
for variation across training seeds. Full source/config/checkpoint/evaluation
artifacts go to `osaze-obahor/majepa-world-model-gradients`, the live W&B entity
used by the existing campaigns.

## Local verification

```sh
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check src tests
```

Replay tests cover the maintained exponential-recency/mixture APIs, including
finite-buffer sampling, FIFO eviction, independent RNG streams and aligned
storage of actions and collection probabilities. Training/configuration/PPO
checks exercise the current single-action API.

The final gradient checks explicitly verify categorical and temporal gradients,
actual mixed collection probabilities, and PPO-only world parameter/optimizer
immutability. A separate pre-edit snapshot comparison matched all 806 output
leaves exactly with both options disabled. Two independent reviewers found no
remaining blockers after evaluation metadata was corrected.

Final verification: the complete suite passed without exclusions, **113 passed
in 215.57 seconds**. Repository-wide Ruff lint and `git diff --check` passed.
Both the source distribution and wheel built successfully; the wheel's model
configuration and gradient implementation were checked against this checkout.
