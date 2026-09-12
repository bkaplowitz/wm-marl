# Evidence and next experiments — September 12, 2026

The clearest current evidence is that world-model optimization materially limits
performance. It does not establish that learning rate is the only remaining
issue, or that larger rates help every map. The cleanup preserves the algorithm;
the suggestions below are proposals, not enabled changes or new queued jobs.

## Completed, comparable evaluations

All entries below use Actor512 / WM4096c64, 50k training records, final-checkpoint
100-episode greedy evaluations, and training seeds 0 and 1. Values are seed 0 /
seed 1, followed by their mean. Both local and joint world-model LRs were changed
together; actor and critic remained at `3e-5`.

| World-model LR | 2s3z | 3s_vs_4z | W&B group |
| --- | --- | --- | --- |
| `4e-5` | 51 / 66 → **58.5%** | 0 / 2 → **1%** | [WM4096 reference](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-rerun-20260911) |
| `8e-5` | 58 / 62 → **60%** | 26 / 0 → **13%** | [8e-5](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-wmlr8e5-20260912) |
| `1e-4` | 63 / 69 → **66%** | 50 / 68 → **59%** | [1e-4](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-wmlr1e4-20260912) |
| `1.5e-4` | 53 / 64 → **58.5%** | 80 / 71 → **75.5%** | [size/LR sweep](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-wm-size-lr-sweep-20260912) |

W&B was checked at approximately 14:17 UTC. The WM2048c64 treatments had not
finished. They must not be conflated with older WM2048c32 runs, which also changed
latent capacity and used different policy sizes/rates.

The `1e-4` reference improves both maps over `4e-5`. At `1.5e-4`, the additional
3s_vs_4z gain comes with lower 2s3z results. There is no single dominant rate yet.
Two seeds and 100 episodes each provide useful screening evidence, not a precise
estimate of seed variance or superiority across the full benchmark.

## What the other numbers indicate

For the `1e-4` runs, the final logged **post-update** actor clip fractions are
1.83% / 1.19% on 2s3z and 4.01% / 4.39% on 3s_vs_4z. KL is about
0.0012–0.0016 and 0.0034–0.0039 respectively. PPO clipping does not currently look
like a constraint blocking most actor updates. These are final logging windows,
not complete histories or independently measured policy drift.

The self-fed H5 recorded-action support error moves from approximately 8–10% to
6% on 2s3z and from 7–9% to 4–5% on 3s_vs_4z between `4e-5` and `1e-4`.
This is a threshold-support diagnostic on replay, not the rate of illegal actions
executed in SMAC. It is consistent with a better learned interface, but the
replay states differ across runs. Small report sample counts further limit
interpretation. See the [reference runs](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-wmlr1e4-20260912).

Critic explained variance is around 0.98 even in some unsuccessful runs. It
measures fit to its training targets, which contain imagined bootstraps; it does
not validate those bootstraps or rank alternative actions against real outcomes.
Similarly, high embedding cosine alone proves neither overfitting nor useful
control information. SIGReg embedding spread remains roughly 0.78–0.83 in the
`1e-4` runs. That is not an obvious collapse signal, although it cannot rule out
loss of particular task-relevant distinctions.

## Priority order

1. **Finish the current size × LR comparison, then validate the selected setting.**
   Add a third seed and check 3s_vs_3z, 3m and 8m before calling a setting general.
   Use final checkpoints and the same 100-episode protocol. Do not pick a different
   best seed or peak checkpoint for each treatment.
2. **Separate the two world-model learning rates.** On WM4096, the next targeted
   pair is local `1e-4` / joint `1.5e-4`, and local `1.5e-4` / joint `1e-4`.
   The existing endpoints provide the comparison. This tests whether joint
   dynamics need faster learning while local features benefit from slower change,
   or vice versa. It directly addresses the observed cross-map trade-off.
3. **Measure executable policy change across a world-model update.** Evaluate a
   fixed set of recorded histories with the actor frozen before/after that update.
   Compare policy KL and action probabilities under true masks. Current PPO KL
   fixes the features and misses this source of policy change. Match histories and
   stochastic draws so latent sampling is not mistaken for parameter drift.
   Add a constraint only if this measurement shows a problem.
4. **Then test actor speed, separately from critic speed.** Compare actor `3e-5`
   and `5e-5` on the selected WM; include `1e-4` only as a bounded extension.
   Earlier actor `1e-4` trials at WM `4e-5` reduced 2s3z finals to 25% / 47% and
   left 3s_vs_4z at zero. They do not establish what happens with a stronger WM,
   but they argue against indiscriminately accelerating the policy. Hold the
   critic at `3e-5` initially. Its high explained variance is not evidence that
   it needs a higher learning rate. [Earlier actor sweep](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-actor512-wm4096-lr-queue-20260911).
5. **Keep clipping, SIGReg and both target-update rates fixed initially.** Clip
   `0.2 → 0.1` is a reasonable later stability ablation. SIGReg weight 0.05 and
   encoder EMA fraction 0.01 lack a clear failure signal today. The target critic
   remains a full copy. Avoid changing all these knobs together.

For a paper, there is also a validation gap: existing reports mostly assess
recorded-action predictions on evolving replay. A fixed held-out-history
comparison should include posterior divergence, legal-action probability quality,
and short-horizon outcome error. Testing whether imagined alternatives rank real
outcomes correctly needs matched simulator branches or repeated matched resets;
a low latent loss cannot substitute for that experiment.

## What the literature supports

[DMAWM's current official configuration](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/configs/trainer_configs/dreamer.yaml)
uses WM LR `1e-4`, actor/critic `3e-5`, five PPO epochs and clipping 0.2. Thus our
actor rate is not unusually low relative to that implementation. Its entropy
coefficient is **0.01**, not the 0.1 discussed earlier. Other differences include
16-step imagination, gamma 0.99, 16 collection environments, and unshared
actors/critics. It uses Adam with norm clipping; our optimizer uses RMS scaling,
momentum and adaptive clipping, so numerical rates are not equivalent steps.
These differences suggest controlled comparisons, not a prescription to copy
all settings.

[DMAWM's world-model implementation](https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/world_model.py)
includes factual value supervision and representation gradients. Their presence
in another algorithm does not establish an isolated benefit in ours. Given the
previous regressions, reintroducing that removed treatment is lower priority
than resolving the demonstrated optimization issue.

[MARIE](https://arxiv.org/abs/2406.15836) combines decentralized predictive models
with centralized aggregation. Our local-history / joint-interaction split already
serves that broad purpose. MARIE uses observation tokenization and a different
policy interface, so its success does not identify a missing teammate-belief
module or justify adding a decoder to MA-JEPA.

[MAPPO](https://arxiv.org/html/2103.01955v4#S5.SS4) finds smaller clipping ranges can
improve stability, sometimes at a cost in learning speed. Its implementation also
clips value updates, whereas ours uses a two-hot critic objective. It supports
trying 0.1 as an ablation, not assuming that our current 0.2 causes the failures.

[LeWorldModel](https://arxiv.org/html/2603.19312v1) demonstrates a smaller, simple
JEPA recipe using latent prediction and SIGReg without an observation decoder.
It supports the case for testing compact models and keeping representation
learning simple. However, its offline pixel-data and planning setting differs
from online multi-agent PPO. Its target-gradient design and regularizer weights
cannot be assumed to transfer directly.

Dense posterior alignment already makes JEPA predictions relevant to the latent
distribution used for control. “More JEPA” should now mean improving and validating
that predictive interface, rather than adding another embedding objective by
name. The current evidence favors controlled optimization work over adding new
modules. It does not yet prove hyperparameter tuning is the only remaining need.
