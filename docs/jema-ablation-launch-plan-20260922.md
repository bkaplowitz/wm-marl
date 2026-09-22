# JEMA ablation launch order and results

Updated: 22 September 2026. Repository: `wm-marl`. Branch:
`feat/world-model-gradient-experiments`.

This is the reference for future launches and the running comparison table.
**Storage instruction (latest):** resume the two existing ablation pods with
Python, source code, SMAC, and StarCraft assets on each pod's local disk;
checkpoints and run outputs may remain on the existing shared network volume.
For every future pod launch, use pod-attached local
storage, sized for assets, dependencies, retained checkpoints, temporary
checkpoint writes, and a free-space reserve. Do not attach a network volume.

On 22 September, the user requested two new ablation pods after the earlier
baseline-reproduction hold. Stage 1 is being launched under that new request. After four-GPU capacity
failed, the user approved mixed L40/L40S allocations and a fallback to four
two-GPU pods (eight GPUs total). The latest instruction authorizes four A100s;
2x H100 was considered and then superseded by that explicit A100 request.
The baseline discrepancy is still unresolved: the direct-shell diagnostic on
`y4leuwgzbs5j1b` failed before training because CUDA initialization failed.
Its records remain in `docs/jema-baseline-shell-reproduction-20260922.md`.

These new campaigns freeze commit `7172773d5a2c1acdbe0486715103b974dd0f7acd`,
which includes the subsequently approved evaluation-profile correction. Final
100-episode evaluation now uses four environments and worker offset 100000,
matching the shell launcher. The older reference results below used one
environment and offset 0. Report comparisons as differing in evaluation
protocol until the old checkpoints have also been evaluated with the corrected
protocol; this change alone has not been proved to explain the baseline gap.

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
the approved wider valid-region search excluding Taiwan (TW), eight-hour pod limits, and the $160
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

| Comparison                    | Local prior | Margin-loss weight | Latent | Joint settings | Runs per map |
| ----------------------------- | ----------- | -----------------: | ------ | -------------- | -----------: |
| Standard controls             | On          |                0.1 | 32×64  | Off, 0.1, 1.0  |            9 |
| Local-prior removal           | Off         |                0.1 | 32×64  | 0.1, 1.0       |            6 |
| Margin-loss removal           | On          |                0.0 | 32×64  | 0.1, 1.0       |            6 |
| Smaller latent                | On          |                0.1 | 32×32  | Off, 0.1, 1.0  |            9 |
| Combined prior/margin removal | Off         |                0.0 | 32×64  | Off            |            3 |

The prior-only and margin-only conditions remain separate experiments. The
combined condition is an additional experiment. Disabling the local prior also
removes its associated local-prior losses as implemented; it does not disable
the local action-mask head or turn local outcome heads on.

## Launch order across two four-GPU pods

The columns are two concurrent scheduling lanes, with no more than two active
four-GPU pods at any time. Physical pod IDs may change between stages because
the existing launcher stops a completed campaign's pod and enforces its deadline.

| Stage     | Pod A — four GPUs                                                 | Pod B — four GPUs                                                     | New runs |
| --------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- | -------: |
| 1         | `2s3z`: local prior off; joint 0.1/1.0; 6 runs                    | `2s3z`: margin weight 0.0, prior on; joint 0.1/1.0; 6 runs            |       12 |
| 2         | `2s3z`: 32×32; joint off/0.1/1.0; 9 runs                          | `3s_vs_4z`: standard 32×64 controls; joint off/0.1/1.0; 9 runs        |       18 |
| 3         | `3s_vs_4z`: local prior off; joint 0.1/1.0; 6 runs                | `3s_vs_4z`: margin weight 0.0, prior on; joint 0.1/1.0; 6 runs        |       12 |
| 4 — added | `2s3z`: prior off + margin weight 0.0, otherwise baseline; 3 runs | `3s_vs_4z`: prior off + margin weight 0.0, otherwise baseline; 3 runs |        6 |
| 5         | `3s_vs_4z`: 32×32; joint off/0.1/1.0; seeds 0 and 2; 6 runs       | `3s_vs_4z`: the same 32×32 comparison; seed 1; 3 runs                 |        9 |
| Total     | 30 runs                                                           | 27 runs                                                               |   **57** |

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

| Component                              | Setting                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------- |
| Map                                    | `2s3z`, five agents; `3s_vs_4z`, three agents, for the specified map comparisons |
| SMAC difficulty                        | 7                                                                                |
| Training budget                        | 50,000 environment steps per run                                                 |
| Local outcome heads                    | Disabled throughout                                                              |
| Imagination mask                       | Local, Bernoulli sampled                                                         |
| Joint mask                             | Retained                                                                         |
| Local/joint world-model learning rates | `1e-4` / `1e-4`                                                                  |
| Local KL, when local prior is enabled  | Dynamics `1.0`, representation `0.1`                                             |
| Posterior alignment                    | `0.05`                                                                           |
| SIGReg                                 | `0.05`, per-agent, 256 projections                                               |
| Action-margin loss                     | `0.1`, except the explicit zero-margin conditions                                |
| Multi-step cosine loss                 | `2.0`                                                                            |
| Self-fed scale / trajectory KL         | `0.1` / `0.1`                                                                    |
| BPTT                                   | 2                                                                                |
| JEPA horizons / anchors                | `2, 4, 5` / 8                                                                    |
| Imagination horizon                    | 5                                                                                |
| Deterministic world model              | Width 4096, hidden width 512, two layers, eight heads                            |
| Categorical latent                     | `32×64`, except the explicit `32×32` comparisons                                 |
| Encoder                                | 3×1024, symlog                                                                   |
| Actor                                  | 3×512                                                                            |
| Critic                                 | 3×512 configuration; centralized critic width/value width 256                    |
| Actor/critic learning rates            | `3e-5` / `3e-5`                                                                  |
| PPO epochs                             | Five actor, five critic                                                          |
| PPO clipping / GAE lambda              | `0.2` / `0.95`                                                                   |
| Entropy                                | Fixed `0.003`                                                                    |
| Latent unimix                          | `0.01`                                                                           |
| Policy/collection unimix               | `0` / `0`                                                                        |
| Replay-value loss / lambda             | `0.3` / `0.95`                                                                   |
| Batch / replay context                 | 16×64 / 192                                                                      |
| Replay capacity                        | 250,000                                                                          |
| World-model replay                     | 50% uniform, 50% recency, decay `0.9998`                                         |
| PPO-root replay                        | Independently sampled 50/50, decay `0.9998`                                      |
| Replay startup                         | Fixed4, `snapshot_staggered`                                                     |
| Collection environments                | 1                                                                                |
| World-model/PPO warm-up                | 5,000 environment steps                                                          |
| Train ratio                            | 128                                                                              |
| Paired/new RNG protocol                | Disabled                                                                         |
| Critic-to-world-model feedback         | Off: `critic_value_scale=0.0`                                                    |
| Teammate-belief module                 | Disabled                                                                         |
| Curve evaluation                       | Every 5,000 steps, 32 episodes                                                   |
| Final evaluation                       | Separate 100-episode greedy evaluation                                           |

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

| Stage/lane    | Existing specification or required addition                                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1A            | `experiments/jema-local-prior-off-20260921.json`                                                                               |
| 1B            | `experiments/jema-margin-zero-20260921.json`                                                                                   |
| 2A            | `experiments/jema-latent-32x32-20260921.json`                                                                                  |
| 2B            | `experiments/jema-joint-controls-3sv4z-20260921.json`                                                                          |
| 3A            | `experiments/jema-local-prior-off-3sv4z-20260921.json`                                                                         |
| 3B            | `experiments/jema-margin-zero-3sv4z-20260921.json`                                                                             |
| 4A            | Add a `2s3z` specification with prior false, margin 0.0, joint false, latent 32×64, seeds 0/1/2                                |
| 4B            | Add the corresponding `3s_vs_4z` specification                                                                                 |
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

| Map  | Condition                                    | Joint | Seed 0 wins | Seed 1 wins | Seed 2 wins | Mean win rate | Sample SD | Mean return |   Delta vs pure baseline | Delta vs matching joint control | Campaign     |
| ---- | -------------------------------------------- | ----- | ----------: | ----------: | ----------: | ------------: | --------: | ----------: | -----------------------: | ------------------------------: | ------------ |
| 2s3z | Pure baseline                                | Off   |          57 |          59 |          39 |         51.7% |   11.0 pp |     16.4979 |                   0.0 pp |                          0.0 pp | Endpoints r1 |
| 2s3z | Standard joint control                       | 0.1   |          57 |          73 |          67 |         65.7% |    8.1 pp |     17.5770 | +14.0 pp, cross-campaign |                          0.0 pp | Joint-scale  |
| 2s3z | Standard joint control                       | 1.0   |          47 |          73 |          80 |         66.7% |   17.4 pp |     17.6293 |                 +15.0 pp |                          0.0 pp | Endpoints r1 |
| 2s3z | Earlier joint result, retained for reference | 0.5   |          56 |          62 |          62 |         60.0% |    3.5 pp |     17.1362 |  +8.3 pp, cross-campaign |                          0.0 pp | Joint-scale  |

Append completed ablation rows to this table. Keep local-prior setting, margin
weight, and latent size explicit in each new condition label.

| Reference campaign | Source commit                              | Source archive SHA256                                              | Local manifest                                             |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------------ | ---------------------------------------------------------- |
| Endpoints r1       | `8acc593fd38b550d86ef0400c4c7e4d47410d7e7` | `3d88618615482524d80b2d2f838f6c391d0059d3984091ec3617e03ed5a64d2c` | `artifacts/jema-joint-endpoints-20260921-r1/manifest.json` |
| Joint-scale        | `e63ade5f1c4bbde9d3c740306c53a148a0215be6` | `e6dd90ea9d853a065d65b6bcb1916e2cdb8aa85efe558fcbbf13fa2eea75b5f6` | `artifacts/jema-joint-scale-20260921/manifest.json`        |

### Reference W&B runs

| Joint setting | Seed | Training                                                                              | Fixed-100 evaluation                                                                  |
| ------------- | ---: | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Off           |    0 | [c55c13d71347](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/c55c13d71347) | [cc9216b2879a](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cc9216b2879a) |
| Off           |    1 | [bd2cbdef735a](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/bd2cbdef735a) | [277f218437eb](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/277f218437eb) |
| Off           |    2 | [ab1235397a02](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ab1235397a02) | [29bb1ef14687](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/29bb1ef14687) |
| 0.1           |    0 | [5ba9b0c346f8](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/5ba9b0c346f8) | [12f0fd6ad824](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/12f0fd6ad824) |
| 0.1           |    1 | [53a1a9ad5c6c](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/53a1a9ad5c6c) | [9320a8dbb3ee](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9320a8dbb3ee) |
| 0.1           |    2 | [da3842293291](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/da3842293291) | [b72e05a44ab8](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/b72e05a44ab8) |
| 1.0           |    0 | [e2bcd2ab7443](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/e2bcd2ab7443) | [d9f90944735f](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/d9f90944735f) |
| 1.0           |    1 | [28f9113a866d](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/28f9113a866d) | [4d9724298e30](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/4d9724298e30) |
| 1.0           |    2 | [a1137af97ed1](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/a1137af97ed1) | [7b39caaf1309](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/7b39caaf1309) |
| 0.5           |    0 | [0f4cb6492042](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/0f4cb6492042) | [b88f729ddf95](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/b88f729ddf95) |
| 0.5           |    1 | [95455bf81d12](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/95455bf81d12) | [9e0957b42251](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9e0957b42251) |
| 0.5           |    2 | [4fc4e9404889](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/4fc4e9404889) | [9589891d8107](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/9589891d8107) |

## Launch ledger

Fill this as stages are actually submitted. A planned specification does not
count as a queued or launched remote job.

| Stage/lane           | Campaign directory                           | Exact pod ID | Rate / start / deadline                       | Progress and outcome                                                    |
| -------------------- | -------------------------------------------- | ------------ | --------------------------------------------- | ----------------------------------------------------------------------- |
| Earlier 1A attempt   | `artifacts/jema-local-prior-off-20260921`    | None created | No pod allocation                             | Controller stopped; 60 rejected allocation attempts; zero launched runs |
| 1A, four-GPU attempt | `artifacts/jema-local-prior-off-20260922-r1` | None created | Controller stopped at a reconciled safe point | 116 rejected attempts; superseded by the authorized two-GPU layout      |
| 1B, four-GPU attempt | `artifacts/jema-margin-zero-20260922-r1`     | None created | Controller stopped at a reconciled safe point | 126 rejected attempts; superseded by the authorized two-GPU layout      |
| 2A                   | Pending                                      | —            | —                                             | Planned, 9 runs                                                         |
| 2B                   | Pending                                      | —            | —                                             | Planned, 9 runs                                                         |
| 3A                   | Pending                                      | —            | —                                             | Planned, 6 runs                                                         |
| 3B                   | Pending                                      | —            | —                                             | Planned, 6 runs                                                         |
| 4A                   | Pending specification                        | —            | —                                             | Planned, 3 runs                                                         |
| 4B                   | Pending specification                        | —            | —                                             | Planned, 3 runs                                                         |
| 5A                   | Pending two-pod initialization               | —            | —                                             | Planned, 6 runs                                                         |
| 5B                   | Same campaign as 5A                          | —            | —                                             | Planned, 3 runs                                                         |

### 22 September stage-1 launch receipts

The existing campaign controller, bootstrap, and per-GPU queue are used without
model or launcher changes. Both specifications use the nine previously approved
regions, exclude Taiwan, and retain four GPUs per pod. W&B remains
`osaze-obahor/majepa-ppo-treatments`. The local PIDs, logs, immutable manifests,
resolved configurations, and frozen source are stored in each campaign directory.
A running local controller waiting for capacity is not a running experiment.

The combined new reservation is $70.40 of GPU runtime plus $1.00 of shutdown
allowances, rather than a new $160 budget for each pod. RunPod billing retrieved
at launch preparation records $34.88465 of GPU charges for the three earlier
JEMA control allocations and $1.68463 for the deleted direct-shell pod. Even
reserving the replacement baseline pod's full eight hours at $2.37/hour keeps
these charges plus the new reservations below the shared $160 protective cap.
Storage charges are additional; stopped pods may still incur storage charges.
Only the two unallocated controllers started for this launch were interrupted
when switching topology; no existing pod or training job was stopped.

### Authorized two-GPU fallback

The user initially rejected A100 and approved mixed L40/L40S hardware and, when four-GPU
pods were unavailable, four two-GPU pods. Each ablation still has six runs and
uses `max_gpus=4`, `gpus_per_pod=2`: pod0 gets seeds 0 and 2 (four jobs), pod1
gets seed 1 (two jobs). No treatment or training setting changed.

The existing launcher now supports Community Cloud country placements. Canada
was the only country reporting two-L40 capacity in the live country checks.
The specs try `{"country":"CA","cloud":"COMMUNITY"}` first, followed by the
nine earlier Secure Cloud regions. At this point Taiwan and A100 were excluded. Each
campaign has a $35.70 cap and each physical pod an eight-hour limit.

| Stage/lane | Campaign                                       | Allocation              | Exact pod        | Rate / created / deadline (UTC) | State at 10:43 UTC                                                                   |
| ---------- | ---------------------------------------------- | ----------------------- | ---------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| 1A         | `artifacts/jema-local-prior-off-20260922-2gpu` | pod0: seeds 0,2; 2 L40  | `13donozgkzjoqk` | $1.38/hour; 10:40:41 / 18:40:41 | Bootstrap running; CUDA initializes successfully on both GPUs; StarCraft downloading |
| 1A         | Same campaign                                  | pod1: seed 1; 2 GPUs    | Pending          | Eight-hour limit once allocated | Controller searching                                                                 |
| 1B         | `artifacts/jema-margin-zero-20260922-2gpu`     | pod0: seeds 0,2; 2 GPUs | Pending          | Eight-hour limit once allocated | Controller searching                                                                 |
| 1B         | Same campaign                                  | pod1: seed 1; 2 GPUs    | Pending          | Eight-hour limit once allocated | Queued allocation                                                                    |

The first margin request failed with an unclassified provider response and its
controller stopped safely. Repeated exact-name API checks confirmed no pod was
created. One instrumented retry returned the known capacity rejection, which
was reconciled using the existing helper, and its controller was resumed.
All receipts and failure records are preserved in the campaign directory.

### Startup verification at 10:55 UTC

On `13donozgkzjoqk`, the automatic bootstrap finished and started both seed-0
local-prior-off treatments at 10:48:48 UTC. GPU 0 runs joint scale 0.1
(child PID 1105, worker 967); GPU 1 runs scale 1.0 (child PID 1103, worker 968).
Both child processes and fresh worker heartbeats were verified more than
120 seconds after launch. Both seed-2 treatments remain pending on the pod.

Both jobs are still compiling JAX training/report functions and have emitted
XLA slow-compilation warnings. They have not yet produced verified advancing
training steps or W&B registration. Do not treat process liveness as a completed
training-startup verification. The saved evidence is
`artifacts/jema-local-prior-off-20260922-2gpu/health-after-120s.json` and
`health-latest.json`; `verify-startup.json` records the still-missing W&B runs.

Only one of the four requested two-GPU pods has been allocated. The local-prior
controller (PID 63607) continues searching for pod1; the margin controller
(PID 64601) continues searching for its two pods. New capacity checks found no
eligible two-L40/L40S availability in Secure Cloud or the non-Taiwan Community
Cloud countries checked. The initial A100 exclusion was superseded by the later four-A100 instruction
below. No later ablation stage has been submitted.

### Four-A100 replacement and reporting fix

The user subsequently authorized A100, briefly requested four H100s, and then
explicitly selected **four A100s**. No H100 was created. The existing launcher's
`allow_a100` support was reused without further launcher changes.

Both seed-0 prior-off jobs on `13donozgkzjoqk` failed during report precompilation:
`ReportingMixin.report()` called a local open-loop rollout even though the local
prior was disabled. The later seed-2 workers used the same faulty frozen source.
RunPod now reports this pod `EXITED`; no successful training result was obtained.
The local capacity controller PID 63607 was interrupted at a reconciled safe point.

The reporting fix returns the already-computed loss and CTDE diagnostic metrics
when the local prior is absent, skipping the unavailable local open-loop report.
It does not restore prior parameters or prior losses, change model updates, or
change any experiment setting. The existing prior-removal test now compiles and
executes `learner.report`, checks finite outputs and retained joint diagnostics,
and confirms that local open-loop metrics are absent. It failed with the exact
remote exception before the fix and passed afterward (1 test, 10.87 seconds).
Targeted Ruff lint and format checks passed. The corrected source is frozen in
`artifacts/jema-local-prior-off-20260922-a100`.

- Corrected prior-off pod: `zql03lhuh40a3i`, four A100-SXM4-80GB GPUs, US-KS-2.
- Rate: $6.36/hour; allocation 11:09:05 UTC; deadline 19:09:05 UTC.
- Queue: all six prior-off runs, joint scales 0.1/1.0 and seeds 0/1/2.
- Source SHA256: `a5845e3fef5ab7db085434db18521503194f6bf6906ba336247e878a6b1708e0`.

During the switch, the existing margin controller allocated `loi5au551sw98b`,
two L40 GPUs in Canada at $1.38/hour, for margin-off seeds 0 and 2. This useful
allocation is retained. The controller PID 64601 continues searching for its
second two-GPU allocation (seed 1). The prepared margin-A100 campaign was never
launched and is superseded; do not launch it because it would duplicate these jobs.

Budget accounting now includes $34.88465 of earlier control GPU charges,
$1.68463 for the first failed shell pod, $1.68182 for its failed Taiwan replacement,
and $27.87366 reserved for the active baseline A100 pod until its original deadline.
The corrected prior-off campaign reserves $51.20; the margin campaign reserves
$35.20. Conservatively allowing a full hour ($1.38) for the stopped prior-off L40
pod and $1.00 shutdown allowances totals **$154.91**, below the shared $160 GPU cap.
Stopped-pod billing had not yet appeared in the billing endpoint; this is a
conservative reservation calculation, not a final invoice. Storage is additional.

### Latest instruction: both ablations on four A100s

The user explicitly requested switching the margin-off pod to four A100s too.
Local controller 64601 was stopped at a reconciled safe point, and only the
old margin pod `loi5au551sw98b` was stopped; its storage was retained. No margin
training result was produced before this switch.

The active replacement is `artifacts/jema-margin-zero-20260922-a100-r1`, containing
all six margin-off runs. Its six-hour limit reserves $38.40 of GPU time plus a
$0.60 allowance. The earlier prepared `jema-margin-zero-20260922-a100` directory
was never launched; do not start it.

The shared worst-case reservation is now $159.59: earlier controls and failed
baseline attempts $38.25110, current baseline reservation $27.87365, corrected
prior-off reservation $51.20, margin-off reservation $38.40, at most one hour on
each stopped L40 pod ($2.76), and $1.10 shutdown allowances. Storage is additional.

The first prior-off A100 bootstrap stopped before training. The same pod was
restarted for diagnosis with its original deadline retained; no replacement
prior-off pod or duplicate job was created. Its saved bootstrap log is in the
campaign artifact directory. Startup is not yet verified.

### Shared-volume bootstrap diagnosis

Both four-A100 pods allocated successfully in US-KS-2 at $6.36/hour each:
prior-off `zql03lhuh40a3i` and margin-off `4490t12q17mvs5`. The shared network
volume did not contain StarCraft II 4.10 and SMAC maps, so workers failed during
asset discovery before claiming any queued training job. Their bootstrap EXIT
traps stopped each pod. These were setup failures, separate from the reporting
bug, and no experiment result or training progress was produced.

Both same pods are being resumed with their original deadlines. The existing
`campaign_assets.py` installs and verifies one shared asset copy under
`/workspace/jema-sc2-assets-20260922`. Pending job identities and frozen sources
are retained; failure logs and outcomes are backed up before queue restart.

### Setup recovery and hardware verification

Episodic Memory recovered the historical setup sequence from the 17 September
asset staging and the 22 September direct-shell replacement: install `uv` and
`unzip`, run `uv sync --locked --python 3.11 --extra dev --extra smac --extra cuda12`,
then run `python -m majepa.campaign_assets` and wait for asset verification.
The gap here was assuming that attaching a network volume implied preinstalled
StarCraft assets. The existing asset installer is now being reused once on the
shared volume. No additional launcher framework was added.

Pinned-JAX compute checks succeeded on CUDA devices 0/1/2/3 of both A100 pods.
RunPod confirms four A100-SXM4-80GB GPUs on each, eight total, $12.72/hour combined.
The original deadlines remain 19:09:05 UTC for prior-off and 17:13:28 UTC for
margin-off. The interrupted recovery was reconciled: prior-off's resume script
was already waiting for assets, and only the missing margin resume was started.
Both queues retained their six pending jobs and original W&B identities.
Controllers 87215 (prior-off) and 87217 (margin-off) monitor the resumed pods.
Failure evidence is retained beside each remote queue and in local artifacts.

The local prior-off stop receipt initially remained after restart because the
job update helper merges fields; that stale `stopped_at` field was explicitly
cleared under the manifest lock. Live RunPod checks confirmed both pods running.
This stale local receipt did not represent an additional pod shutdown.

### Asset verification and first worker launch

The shared installation passed the historical SHA256 checks for StarCraft II
Base75689 and the `2s3z` map. Both pods use
`/workspace/jema-sc2-assets-20260922/StarCraftII`. The SC2 archive expands to
4,938,576,444 bytes across 24,278 entries. Assets remain on network storage;
there is no measured comparison against pod-local SC2 runtime throughput.

At approximately 11:25:49 UTC, both queues started their four seed-0/seed-1
training processes, one per A100. Each pod retains two seed-2 runs queued.
Process startup alone does not establish advancing environment steps or
successful W&B registration; those checks are still pending at this entry.

### Latest storage decision

The user requested stopping and relaunching on pod-local storage, then explicitly
reversed that instruction: **keep using the shared network storage for now**.
Both ablation pods and their local controllers had already been stopped, but no
network volume was deleted and no replacement pod was created. No launcher
storage change was made in this session. Resume these same two four-A100 pods
with their original deadlines and retained, verified shared StarCraft assets.
Do not carry out the superseded network-volume deletion or local-storage migration.
Other users' pods and all shared volumes remain untouched.

### Local runtime with retained network checkpoints

The user then requested using the cleared container disks for Python, SMAC,
StarCraft II, and source code while retaining network-attached checkpoints.
The two existing pods each have a 30 GiB local container disk. The frozen source
is extracted to `/opt/majepa-src`, the locked environment is installed at
`/opt/majepa-venv`, and the installation previously staged by
`campaign_assets.py` is copied to `/opt/StarCraftII` and checked against its
verification receipt. This avoids downloading and retaining another SC2 archive.
The remote campaign directory, logs, checkpoints, and temporary checkpoint
writes remain on the existing network volume. Original source archives,
experiment configurations, W&B identities, and shutdown deadlines are retained.
Future new pods must use pod-attached local storage for the full workload,
including sufficient checkpoint and temporary-file headroom.

### Verified local-runtime restart, 22 September 11:56 UTC

- `zql03lhuh40a3i` (jema-local-prior-off-20260922-a100): four live training workers, 5,360–5,950 environment steps, 46–119 learner updates, all recorded training losses finite. Two seed-2 runs remain queued. Local free space: 14.51 GiB.
- `4490t12q17mvs5` (jema-margin-zero-20260922-a100-r1): four live training workers, 5,010–5,010 environment steps, 2–2 learner updates, all recorded training losses finite. Two seed-2 runs remain queued. Local free space: 14.51 GiB.

Both pods retain four A100-SXM4-80GB GPUs each. Immediate checks and checks
132 seconds later passed. All eight active W&B runs have matching configs and
verified source/config artifacts. The actual SC2 processes execute
`/opt/StarCraftII/Versions/Base75689/SC2_x64` with local `/tmp/sc-*` directories;
Python/SMAC/JAX are in `/opt/majepa-venv`, and runtime code is `/opt/majepa-src`.
The campaign's `repo` path points to that verified local extraction, with the
original shared extraction retained as `repo.network-backup`. Interrupted
pre-training queues and logs were preserved under
`interrupted-before-training-20260922T1135`; no learned checkpoint was resumed.

Local controllers 4049 and 4052 remain active. Original hard deadlines are
19:09:05 UTC (prior-off) and 17:13:28 UTC (margin-off); their combined GPU
rate remains $12.72/hour. Full training and final evaluations are still pending.
Detailed restart, runtime-path, and training receipts are in the two campaign
artifact directories.

At 12:05 UTC the follow-up W&B check confirmed continued learning:

- `jema-local-prior-off-20260922-a100`: W&B environment counters 6,130–9,700; learner updates 142–588; all four runs running with finite optimizer, actor, and critic losses.
- `jema-margin-zero-20260922-a100-r1`: W&B environment counters 7,490–9,920; learner updates 312–616; all four runs running with finite optimizer, actor, and critic losses.

The local monitors subsequently needed restarting after a RunPod API DNS lookup failed; the remote queues and shutdown guards continued independently.

Replacement local monitors 33192 (prior-off) and 33302 (margin-off) passed the follow-up check: recent remote health refreshes, four live workers each, and all eight W&B runs verified with finite optimizer/PPO losses.
