# JEMA ablation launch order and results

Updated: 22 September 2026. Repository: `wm-marl`. Branch:
`feat/world-model-gradient-experiments`.

This is the reference for future launches and the running comparison table.
The requested order is baseline first, then local-prior removal and the separate
margin-loss ablation, then the smaller latent. The later `3s_vs_4z` request adds
the corresponding controls and ablations on that map. The combined prior/margin
ablation was subsequently inserted after stage 3 of the two-pod schedule below.

## Goal to set

Copy this when starting the execution goal; writing this document does not itself
create or resume an active goal.

```text
Execute docs/jema-ablation-launch-plan-20260922.md in its specified stage order.
Run the 57 remaining training jobs and their 57 separate fixed-100 greedy
evaluations using at most two concurrent four-GPU RunPods and the existing
campaign launcher. Preserve the pinned JEMA baseline except for each listed
ablation and the required map/seed/output identifiers. Use L40, then L40S, with
the approved wider valid-region search, eight-hour pod limits, and the $160
protective spend cap. Do not duplicate completed controls or active jobs.
After every completed three-seed ablation, update this document's results table
with per-seed results, means, sample SDs, differences from the same-map pure
baseline and matching joint-scale control, W&B links, and source/config
provenance. Monitor launch health immediately and again after 120 seconds,
then check progress every 30–60 minutes. Continue through the ordered stages
until all requested training and evaluation jobs finish, or report a concrete
blocker. Stop only these campaigns' exact pods after their work and final
artifact uploads finish. Prior testing is complete: focus on launching,
monitoring, and results, not repeated pytest, lint, compilation, byte-level
equivalence, or similar verification campaigns. Do not push or change the
model implementation as part of this execution goal.
```

## State at handoff

- The earlier `2s3z` baseline and joint-gradient controls have finished. Their
  fixed-100 results appear below; reuse them.
- The local controller for `artifacts/jema-local-prior-off-20260921` was stopped
  at the user's request. Its 60 allocation attempts created no pod; the exact
  campaign pod name was absent from RunPod after the controller stopped.
- No new ablation pod was running at this handoff. The prepared experiments
  below are specifications, not evidence that jobs have been submitted.
- The most recently frozen, unlaunched source was commit
  `3bf906509431765a288c5b273342f47a09e3b86b`, archive SHA256
  `9c63ffe121941f1d386b6e0c1e5262d2eabf996afae30cca75a7627fbdb86190`.
- DreamerV3 is pinned at
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- Reconcile saved manifests and live exact pod names when resuming, because
  this handoff is a dated snapshot. Do not infer current state from this section.

## Experiment definitions

All new experiments use seeds `0, 1, 2`, 50,000 environment steps per run, and
one GPU per independent training job. A completed job includes its separate
100-episode greedy evaluation.

Joint settings mean:

- **Off:** `agent.world_model_gradients.joint_prediction=false`.
- **0.1:** `joint_prediction=true`, `joint_prediction_scale=0.1`.
- **1.0:** `joint_prediction=true`, `joint_prediction_scale=1.0`.

For the new combined ablation, **“otherwise baseline” is interpreted literally**:
joint backpropagation is off, critic feedback remains off, local prior is off,
margin-loss weight is zero, and the latent remains `32×64`. This is three runs
per map, six in total. The 0.1/1.0 joint treatments are not additionally crossed
with this combined condition in the present plan.

| Comparison | Local prior | Margin-loss weight | Latent | Joint settings | Runs per map |
|---|---|---:|---|---|---:|
| Standard controls | On | 0.1 | 32×64 | Off, 0.1, 1.0 | 9 |
| Local-prior removal | Off | 0.1 | 32×64 | 0.1, 1.0 | 6 |
| Margin-loss removal | On | 0.0 | 32×64 | 0.1, 1.0 | 6 |
| Smaller latent | On | 0.1 | 32×32 | Off, 0.1, 1.0 | 9 |
| Combined prior/margin removal | Off | 0.0 | 32×64 | Off | 3 |

The prior-only and margin-only conditions remain separate experiments. The
combined condition is an additional experiment. Disabling the local prior also
removes its associated local-prior losses as implemented; it does not disable
the local action-mask head or turn local outcome heads on.

## Launch order across two four-GPU pods

The columns are two concurrent scheduling lanes, with no more than two active
four-GPU pods at any time. Physical pod IDs may change between stages because
the existing launcher stops a completed campaign's pod and enforces its deadline.

| Stage | Pod A — four GPUs | Pod B — four GPUs | New runs |
|---|---|---|---:|
| 1 | `2s3z`: local prior off; joint 0.1/1.0; 6 runs | `2s3z`: margin weight 0.0, prior on; joint 0.1/1.0; 6 runs | 12 |
| 2 | `2s3z`: 32×32; joint off/0.1/1.0; 9 runs | `3s_vs_4z`: standard 32×64 controls; joint off/0.1/1.0; 9 runs | 18 |
| 3 | `3s_vs_4z`: local prior off; joint 0.1/1.0; 6 runs | `3s_vs_4z`: margin weight 0.0, prior on; joint 0.1/1.0; 6 runs | 12 |
| 4 — added | `2s3z`: prior off + margin weight 0.0, otherwise baseline; 3 runs | `3s_vs_4z`: prior off + margin weight 0.0, otherwise baseline; 3 runs | 6 |
| 5 | `3s_vs_4z`: 32×32; joint off/0.1/1.0; seeds 0 and 2; 6 runs | `3s_vs_4z`: the same 32×32 comparison; seed 1; 3 runs | 9 |
| Total | 30 runs | 27 runs | **57** |

Scheduling instructions:

1. Submit the two independent stage-1 comparisons first. Do not restart the
   completed `2s3z` controls merely to occupy the second pod.
2. Within each comparison use `seed_first`: all listed treatments for seed 0,
   then seed 1, then seed 2. A free worker takes the next eligible queued job;
   there is no completion barrier between seeds.
3. A lane can advance once its preceding assigned comparison finishes. Keep
   the stage order within each lane. Start the `3s_vs_4z` ablations only after
   that map's standard controls finish.
4. Stage 4 follows completion of both stage-3 comparisons. Stage 5 follows
   stage 4. Do not move the combined ablation ahead of the separate ablations.
5. For stage 5, the existing two-pod planner assigns seeds 0 and 2 to one pod
   and seed 1 to the other. That produces the stated 6/3 split. Each combined
   stage-4 comparison has only three jobs, so one GPU on each pod will be idle;
   do not add unrequested treatments to fill those slots.
6. Record each actual campaign directory, pod ID, hourly rate, start time,
   deadline, and outcome in the launch ledger below. Reconcile and finish an
   existing allocation before replacing it.

## Pinned baseline settings

Use the `baseline` profile only. The configuration foundation is JEMA baseline
commit `79de5d2`, integrated on `feat/world-model-gradient-experiments`. Freeze the
chosen source once and retain its provenance; do not silently follow a newer
branch tip or restore overrides from older campaigns.

| Component | Setting |
|---|---|
| Map | `2s3z`, five agents; `3s_vs_4z`, three agents, for the specified map comparisons |
| SMAC difficulty | 7 |
| Training budget | 50,000 environment steps per run |
| Local outcome heads | Disabled throughout |
| Imagination mask | Local, Bernoulli sampled |
| Joint mask | Retained |
| Local/joint world-model learning rates | `1e-4` / `1e-4` |
| Local KL, when local prior is enabled | Dynamics `1.0`, representation `0.1` |
| Posterior alignment | `0.05` |
| SIGReg | `0.05`, per-agent, 256 projections |
| Action-margin loss | `0.1`, except the explicit zero-margin conditions |
| Multi-step cosine loss | `2.0` |
| Self-fed scale / trajectory KL | `0.1` / `0.1` |
| BPTT | 2 |
| JEPA horizons / anchors | `2, 4, 5` / 8 |
| Imagination horizon | 5 |
| Deterministic world model | Width 4096, hidden width 512, two layers, eight heads |
| Categorical latent | `32×64`, except the explicit `32×32` comparisons |
| Encoder | 3×1024, symlog |
| Actor | 3×512 |
| Critic | 3×512 configuration; centralized critic width/value width 256 |
| Actor/critic learning rates | `3e-5` / `3e-5` |
| PPO epochs | Five actor, five critic |
| PPO clipping / GAE lambda | `0.2` / `0.95` |
| Entropy | Fixed `0.003` |
| Latent unimix | `0.01` |
| Policy/collection unimix | `0` / `0` |
| Replay-value loss / lambda | `0.3` / `0.95` |
| Batch / replay context | 16×64 / 192 |
| Replay capacity | 250,000 |
| World-model replay | 50% uniform, 50% recency, decay `0.9998` |
| PPO-root replay | Independently sampled 50/50, decay `0.9998` |
| Replay startup | Fixed4, `snapshot_staggered` |
| Collection environments | 1 |
| World-model/PPO warm-up | 5,000 environment steps |
| Train ratio | 128 |
| Paired/new RNG protocol | Disabled |
| Critic-to-world-model feedback | Off: `critic_value_scale=0.0` |
| Teammate-belief module | Disabled |
| Curve evaluation | Every 5,000 steps, 32 episodes |
| Final evaluation | Separate 100-episode greedy evaluation |

The precise existing ablation keys are:

```text
agent.dyn.parallel_transformer.local_prior
agent.loss_scales.ctde_multistep_jepa_action
agent.dyn.parallel_transformer.stoch
agent.dyn.parallel_transformer.classes
agent.world_model_gradients.joint_prediction
agent.world_model_gradients.joint_prediction_scale
agent.world_model_gradients.critic_value_scale
```

## Existing specifications and launch procedure

Reuse `python -m majepa.campaign` and the frozen controller described in
[campaign-launcher.md](campaign-launcher.md). Do not write another launcher,
queue framework, or results framework.

| Stage/lane | Existing specification or required addition |
|---|---|
| 1A | `experiments/jema-local-prior-off-20260921.json` |
| 1B | `experiments/jema-margin-zero-20260921.json` |
| 2A | `experiments/jema-latent-32x32-20260921.json` |
| 2B | `experiments/jema-joint-controls-3sv4z-20260921.json` |
| 3A | `experiments/jema-local-prior-off-3sv4z-20260921.json` |
| 3B | `experiments/jema-margin-zero-3sv4z-20260921.json` |
| 4A | Add a `2s3z` specification with prior false, margin 0.0, joint false, latent 32×64, seeds 0/1/2 |
| 4B | Add the corresponding `3s_vs_4z` specification |
| 5, both lanes | `experiments/jema-latent-32x32-3sv4z-20260921.json`, with `max_gpus=8` and `gpus_per_pod=4` for the planned two-pod allocation |

The stage-4 specifications and revised placements have not been written at this
documentation handoff. Create them from the existing specs when preparing the
launch; change only the listed experiment and scheduling fields. Preserve old
campaign directories and frozen manifests. Changed placement or allocation
plans require fresh campaign directories, not edits to an immutable manifest.

Operational limits and instructions:

- W&B destination: `osaze-obahor/majepa-ppo-treatments`. Use distinct campaign
  names/groups and exact run IDs for each frozen campaign.
- Two four-GPU pods maximum; L40 first, then L40S. Existing prepared specs have
  `allow_a100=false`. Do not substitute hardware or change GPU counts silently.
- The wider valid-region search is approved. Earlier candidates were US-KS-2,
  OC-AU-1, EU-NL-1, US-NC-1, US-TX-3, plus EUR-IS-2, US-IL-1, US-MO-1, and
  US-TX-4. Availability is time-dependent. Keep network volumes attached only
  to their own datacenter; use the existing attached-storage bootstrap elsewhere.
- Retain the prepared per-GPU hourly ceiling of `$1.10` and maximum eight-hour
  lifetime per physical pod. Record the actual rate before launch.
- Preserve the `$160` protective spend cap. Account for cumulative campaign
  spending and new reservations before advancing; do not interpret it as a new
  allowance for every row or pod. The existing launcher's cap counts GPU spend;
  storage/network charges must also be reported. This plan does not guarantee
  all 57 runs fit into one eight-hour allocation or the remaining budget.
- Save checkpoints locally, upload and verify final checkpoints and evaluation
  artifacts through W&B, and prune superseded local checkpoints only afterward.
- Inspect exact existing campaign pods and sessions before launch. Never kill
  unrelated pods, tmux sessions, W&B processes, or other jobs. Ask before stopping
  anything outside this plan's exact campaign resources.
- Reuse completed testing. Do not spend the next session rerunning pytest,
  lint, compilation, byte comparisons, baseline-equivalence suites, or similar
  broad checks. No model-code changes are needed for this plan.
- Resolve the actual launch spec through the existing planner so unintended
  configuration overrides are visible. Then proceed to launch and operational
  monitoring; do not turn this into another testing project.
- Confirm worker processes and queue entries immediately after launch and again
  after 120 seconds. Check W&B registration, nonempty configuration and source/
  config artifacts, advancing steps, finite losses, and retained pending jobs.
- After startup, check every 30–60 minutes. A live pod alone is not a successful
  launch or a completed experiment. Diagnose a concrete failure before rerunning
  that exact job; preserve its failed record and use a fresh identity for a rerun.

## Update the comparison table after each ablation

Use the existing `results` action to fetch exact manifest-pinned runs:

```sh
.venv/bin/python -m majepa.campaign results \
  --directory artifacts/CAMPAIGN_NAME --format json
```

Its output already checks finished training, matching recorded configuration,
verified source/config/checkpoint/evaluation artifacts, and exactly 100 final
episodes. Use this output rather than handwritten metric-fetching scripts.

After each three-seed condition finishes:

1. Add its row immediately; do not wait for every stage to finish. Partial
   conditions may be listed as `1/3` or `2/3` in the ledger, but must not be
   presented as completed three-seed means.
2. Show seed-0/1/2 wins out of 100, mean win rate, sample SD across seeds, mean
   return, and the mean difference from the same-map pure baseline in percentage
   points. Pair comparisons by seed.
3. Also show the difference from the matching standard joint setting: compare
   a 0.1 ablation with standard 0.1, a 1.0 ablation with standard 1.0, and a joint-
   off ablation with the pure baseline. This distinguishes the ablation effect
   from the joint-gradient treatment effect.
4. Retain train/evaluation W&B links and source commit/archive/config provenance.
   Label comparisons across campaigns, especially when source versions differ.
   Do not describe such results as a same-source controlled comparison without
   supporting evidence.
5. Use only the separate fixed-100 evaluations. The 32-episode curve evaluations
   never substitute for final results. Add the `3s_vs_4z` baseline when stage 2B
   completes; do not use a `2s3z` baseline to compute its treatment differences.

### Verified reference results

Retrieved from the existing results command on 22 September 2026. All rows use
local prior on, margin weight 0.1, and latent 32×64. These are observed results,
not a claim that the baseline and both source archives are identical.

| Map | Condition | Joint | Seed 0 wins | Seed 1 wins | Seed 2 wins | Mean win rate | Sample SD | Mean return | Delta vs pure baseline | Delta vs matching joint control | Campaign |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2s3z | Pure baseline | Off | 57 | 59 | 39 | 51.7% | 11.0 pp | 16.4979 | 0.0 pp | 0.0 pp | Endpoints r1 |
| 2s3z | Standard joint control | 0.1 | 57 | 73 | 67 | 65.7% | 8.1 pp | 17.5770 | +14.0 pp, cross-campaign | 0.0 pp | Joint-scale |
| 2s3z | Standard joint control | 1.0 | 47 | 73 | 80 | 66.7% | 17.4 pp | 17.6293 | +15.0 pp | 0.0 pp | Endpoints r1 |
| 2s3z | Earlier joint result, retained for reference | 0.5 | 56 | 62 | 62 | 60.0% | 3.5 pp | 17.1362 | +8.3 pp, cross-campaign | 0.0 pp | Joint-scale |

Append completed ablation rows to this table. Keep local-prior setting, margin
weight, and latent size explicit in each new condition label.

| Reference campaign | Source commit | Source archive SHA256 | Local manifest |
|---|---|---|---|
| Endpoints r1 | `8acc593fd38b550d86ef0400c4c7e4d47410d7e7` | `3d88618615482524d80b2d2f838f6c391d0059d3984091ec3617e03ed5a64d2c` | `artifacts/jema-joint-endpoints-20260921-r1/manifest.json` |
| Joint-scale | `e63ade5f1c4bbde9d3c740306c53a148a0215be6` | `e6dd90ea9d853a065d65b6bcb1916e2cdb8aa85efe558fcbbf13fa2eea75b5f6` | `artifacts/jema-joint-scale-20260921/manifest.json` |

### Reference W&B runs

| Joint setting | Seed | Training | Fixed-100 evaluation |
|---|---:|---|---|
| Off | 0 | [c55c13d71347](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/c55c13d71347) | [cc9216b2879a](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cc9216b2879a) |
| Off | 1 | [bd2cbdef735a](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/bd2cbdef735a) | [277f218437eb](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/277f218437eb) |
| Off | 2 | [ab1235397a02](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ab1235397a02) | [29bb1ef14687](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/29bb1ef14687) |
| 0.1 | 0 | [5ba9b0c346f8](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/5ba9b0c346f8) | [12f0fd6ad824](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/12f0fd6ad824) |
| 0.1 | 1 | [53a1a9ad5c6c](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/53a1a9ad5c6c) | [9320a8dbb3ee](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9320a8dbb3ee) |
| 0.1 | 2 | [da3842293291](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/da3842293291) | [b72e05a44ab8](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/b72e05a44ab8) |
| 1.0 | 0 | [e2bcd2ab7443](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/e2bcd2ab7443) | [d9f90944735f](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/d9f90944735f) |
| 1.0 | 1 | [28f9113a866d](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/28f9113a866d) | [4d9724298e30](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/4d9724298e30) |
| 1.0 | 2 | [a1137af97ed1](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/a1137af97ed1) | [7b39caaf1309](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/7b39caaf1309) |
| 0.5 | 0 | [0f4cb6492042](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/0f4cb6492042) | [b88f729ddf95](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/b88f729ddf95) |
| 0.5 | 1 | [95455bf81d12](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/95455bf81d12) | [9e0957b42251](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9e0957b42251) |
| 0.5 | 2 | [4fc4e9404889](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/4fc4e9404889) | [9589891d8107](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9589891d8107) |

## Launch ledger

Fill this as stages are actually submitted. A planned specification does not
count as a queued or launched remote job.

| Stage/lane | Campaign directory | Exact pod ID | Rate / start / deadline | Progress and outcome |
|---|---|---|---|---|
| Earlier 1A attempt | `artifacts/jema-local-prior-off-20260921` | None created | No pod allocation | Controller stopped; 60 rejected allocation attempts; zero launched runs |
| 1A | Pending fresh initialization with expanded placements | — | — | Planned, 6 runs |
| 1B | Pending | — | — | Planned, 6 runs |
| 2A | Pending | — | — | Planned, 9 runs |
| 2B | Pending | — | — | Planned, 9 runs |
| 3A | Pending | — | — | Planned, 6 runs |
| 3B | Pending | — | — | Planned, 6 runs |
| 4A | Pending specification | — | — | Planned, 3 runs |
| 4B | Pending specification | — | — | Planned, 3 runs |
| 5A | Pending two-pod initialization | — | — | Planned, 6 runs |
| 5B | Same campaign as 5A | — | — | Planned, 3 runs |
