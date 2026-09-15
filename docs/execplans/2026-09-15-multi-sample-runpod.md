# Six MA-JEPA runs with verified RunPod and W&B execution

Status: IMPLEMENTING

## User Request
Implement the approved conversation plan: live dummy A100 smoke test, then six
50k-step SMAC 2s3z runs, seeds 0/1/2 with one and two actions per agent. Two distinct
actions PER AGENT without replacement, both valid samples train, only the first
continues imagination. Prioritize two-sample runs. Four A100s total, one per pod,
maximum four concurrent; $50 aggregate GPU cap including smoke and setup; eight
hours maximum per pod. User explicitly approved stopping ONLY new campaign pods
after completion, failure, or time limit. Preserve all existing jobs and data.

## Binding Architecture
- Base is updated clean_jepa a23ce98, already containing corrected sampling port
  14707a6 and default-one 9316e43. Preserve new configurable replay functionality.
- Explicit shared overrides, not changed defaults: WM deter4096/hidden512/stoch32/
  classes64/unimix.01; actor layers3/units512; central critic width256 and value
  layers2/units256; posterior-alignment weight.05, action-margin loss weight.1;
  BPTT2/horizon5; both WM LRs1e-4; actor/critic LRs3e-5; PPO clip.2 and epochs5/5;
  fixed entropy.003, schedule disabled; policy/collection unimix0; replay250000,
  recent_world_uniform_behavior, world_uniform_mix.5, recency_decay.9998; independent
  uniform behavior roots; WM/PPO start5000; teammate belief and belief_context off.
- Retain previous run settings: task smac_2s3z, five agents, one environment, steps
  50000, train_ratio128, snapshot_staggered replay, startup behavior starts4,
  isolate_report_rng True, self-fed horizons2/4/5/anchors8/scale.1/trajectoryKL.1,
  consumerKL0, fresh_history False, local_feedback_gradient False, both WM warmup0,
  replay value scale.3, slow critic rate1, factual/direct-latent disabled,
  Bernoulli imagination masks, balanced mask loss; local reward/continuation/mask
  heads512 as in previous launch. Explicit one GPU policy/train devices [0].
- W&B entity osaze-obahor belongs to requested org osaze-obahor-org (verified API
  and browser). New project majepa-multi-sample-treatments. Six primary train runs;
  smoke/eval records identified separately. Unique IDs, no collision/resume of old
  runs, full resolved config and provenance, source/checkpoint/eval artifacts.
- Use pinned official image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04,
  locked uv dependencies including dev, smac and cuda12, pinned submodule. Existing
  volume e8ishvsuf7 in US-KS-2 mounted /workspace; unique campaign directory.
  Discover existing StarCraft II installation without modifying it; verify assets
  and free space. Fail clearly if unavailable; do not silently change SC2 version.

## Repository Evidence
- src/majepa/main.py main/_load_configs/_resolve_config_profiles accepts exact
  low-level CLI overrides and saves resolved config; make_logger currently omits
  W&B config/artifacts. elements.logger.WandBOutput accepts wandb.init kwargs.
- src/majepa/config.py MAJEPARunSpec/to_dict currently hardcodes many defaults;
  do not label these experiments with that misleading manifest. Derive recorded
  values from actual resolved config, and use same config for final evaluation.
- Existing pod launcher reference (stdlib SSH/rsync/manifest/detached primitives):
  ../wm-marl-dreamer-v3-parity-port/src/world_marl/runpod.py. Do not import obsolete
  job types or unsupported --stop-after/--terminate-after CLI flags.
- src/majepa/train.py saves final checkpoint then closes logger; evaluation.py
  loads explicit checkpoint. Evaluation command must keep custom architecture.
- Existing tests/test_ppo.py and test_training.py cover legal/repeatable distinct
  per-agent draws, rare-second gradients, and unchanged primary trajectory.

## Acceptance Criteria
- [ ] Minimal launcher with side-effect-free dry-run, exact manifest, safe SSH,
  detached jobs, validated input, restartable monitoring and budget/time backstops.
- [ ] Canonical custom config resolves correctly; seed/sample count are the only
  experimental differences; no default hyperparameter changes.
- [ ] Source/config metadata uploaded; final checkpoint and eval uploaded with
  remote artifact verification; failed uploads visibly fail completion.
- [ ] Focused mocked launcher checks, sampling tests, and independent source review.
- [ ] A100 smoke: JAX GPU compute, SMAC reset/step, W&B metrics, tiny artifact
  upload/download equality, automatic self-stop verified via control-plane API.
- [ ] Six real runs launched in priority order; immediate and >=120s rechecks;
  finite WM/PPO updates and increasing W&B history verified for each.
- [ ] Six final checkpoints at 50000 environment steps, final100 evaluations,
  artifact verification, exact new pods stopped, paired comparison and cost ledger.

## Implementation Steps
1. Reuse existing launch patterns with a minimal first-party RunPod launcher and
   campaign config. Use stdlib subprocess/json/pathlib rather than new frameworks.
   Include dry-run, exact created pod IDs, atomic status manifest, visible logs,
   no broad process killing, no blind duplicate creation after uncertain responses.
2. Initialize W&B with actual config/provenance and upload code/config at startup,
   final checkpoint and evaluation outputs at completion. Reuse W&B APIs and
   existing logger/entrypoint; no monkey patching dependencies.
3. Smoke through the same provision/bootstrap/credential/runner path; test the
   runtime backstop using pod-scoped management credentials, not account secrets.
   Preserve persistent files; stop only exact allocated pod IDs.
4. Run two-sample seed0 until first valid post-prefill PPO update, then two-sample
   seeds1/2 and control seed0; controls1/2 enter as slots free. A100 SXM80 secure
   current price1.59/hr. Do not switch GPU types or exceed cap without user input.
5. Curve eval32 episodes/4 envs/every5000/seed offset50000. Final checkpoint eval100
   episodes/4 envs/worker offset100000/deterministic, matching resolved config.
   Compare final win rates, paired differences, curves, runtime and GPU cost.

## Operational Requirements
Credentials: local RunPod account env key and W&B api.wandb.ai netrc are available;
never print or include secrets in argv, artifacts, source bundles or manifests.
Use explicit public key ~/.ssh/runpod_key.pub at pod creation; SSH with private
key ~/.ssh/runpod_key; save first-use host keys in campaign known_hosts. Existing
pod denies local SSH keys, so access volume through NEW authorized smoke pod.
Start runtime watchdog inside pod independent of local monitor using pod-scoped
management key. Aggregate billing monitor persists starts/rates and reserves
remaining budget, allowing stop before $50. Account for provisioning and all new
pods. Missing credentials, capacity, data or permissions are explicit blockers.
Retain data on persistent network volume after stop. Do not delete pods/volumes.
Use conventional commits and freeze one source revision for all real runs.

## Non-goals
No three-sample implementation or runs yet; assess only after paired results.
No replay redesign, new dependencies/framework, broad refactor, or old job changes.

## Progress
- 2026-09-15: Corrected three launcher review findings with test-first regressions:
  self-stop loads `/etc/rp_environment` on the pod and matches its injected ID to
  the recorded job ID; run metadata receives that recorded ID; short reservation
  and job-merge locks allow monitoring throughout pod creation/provisioning;
  admission and monitoring both charge the greater of reserved and observed cost.
  Five regression cases failed before implementation and now pass. Fresh reviews
  and the live smoke self-stop remain required.
- 2026-09-15: Added src/majepa/campaign.py with shared explicit overrides,
  side-effect-free dry-run, frozen Git/submodule archive, persistent manifest,
  four-pod admission limit, cost reservations, detached bootstrap, boot-time and
  completion watchdogs, and restartable control-plane monitor. CLI SSH/rate schema
  now matches live reads; independent review remains before smoke.
- 2026-09-15: main.py/train.py/evaluation.py now record resolved W&B config,
  campaign source/provenance, final checkpoint, and evaluation artifacts. Uploads
  wait for completion and compare remote artifact manifests/digests; failures
  propagate and finish W&B with failure status. Default runs do not upload artifacts.
- 2026-09-15: User approved conversation plan and implementation. Confirmed
  per-agent distinctness and four total A100s in separate pods. Updated base read.

## Decisions
- 2026-09-15: Reserve 0.5 GPU hours for smoke and 5 GPU hours per real run,
  starting before pod creation: $48.495 at $1.59/hour, with a further $0.50
  shutdown allowance below the $50 cap. Eight hours remains the absolute maximum.
  Full reservations remain charged to the admission budget after early stops;
  unused time is not automatically redistributed. An overrun increases the charge
  to conservative elapsed cost, including provisioning and delayed stop observation.
  When the aggregate charge reaches the budget minus shutdown allowance, monitoring
  requests stops only for exact recorded campaign pod IDs.
- 2026-09-15: Keep network provisioning outside the manifest lock. Reserve names
  atomically before creation and merge only the launched job back into fresh state,
  retaining monitor stop statuses and cost observations. This prevents duplicate
  creation and stale launch snapshots overwriting concurrent monitor updates.
- 2026-09-15: SSH shells on the pinned image omit RunPod environment variables.
  `/etc/rp_environment` contains the pod-scoped key and pod ID, so self-stop sources
  it locally without transferring account credentials. A sourced read-only CLI GET
  returned Unauthorized on the existing pod; successful self-stop is therefore
  unverified until the authorized new smoke pod stops and the control plane confirms.
- 2026-09-15: Reuse runpodctl for account create/read/stop operations after live
  preflight found a Cloudflare 1010 block on direct REST while the CLI works.
  Watchdogs use only the pod-scoped runpodctl credentials and exact own pod ID.
- Directly use resolved low-level training configuration to avoid stale default
  metadata; retain repository public defaults.
- Existing valid singleton action may simulate second transition but has no second
  loss: impossible actions never become valid training samples.

## Deviations
None.

## Verification Evidence
- Corrective implementation: `.venv/bin/python -m pytest tests/test_campaign.py
  tests/test_campaign_artifacts.py tests/test_evaluation.py
  tests/test_configuration.py -q`: 43 passed in 1.99 seconds. New regressions cover
  concurrent monitoring and duplicate rejection during pending creation, preservation
  of monitor updates, overrun admission rejection, aggregate exact-ID stops, a real
  Bash/fake-CLI self-stop with absent SSH environment and mismatching ID rejection,
  and smoke metadata inheritance. `.venv/bin/ruff check src/majepa/campaign.py
  tests/test_campaign.py` and `git diff --check` passed. No paid pods were created
  by the corrective implementer; live credentials/self-stop remain smoke evidence.
- Implementation: `.venv/bin/python -m pytest tests/test_campaign.py
  tests/test_campaign_artifacts.py tests/test_evaluation.py
  tests/test_configuration.py -q`: 38 passed in 1.82 seconds after the bootstrap failure check. Tests cover resolved
  treatment/evaluation parity, budget/concurrency/duplicate rejection, side-effect
  free dry run, persisted uncertain creation, exact-pod stop, and bootstrap failure.
- Artifact worker: 31 focused artifact/evaluation/config cases passed, including
  real local W&B Artifact construction and real Elements checkpoint save with
  mocked network. Ruff and whitespace checks passed. Live upload remains pending.
- Root baseline: `.venv/bin/python -m pytest tests/test_ppo.py
  tests/test_training.py tests/test_replay.py -q`: 29 passed in 166.06 seconds.
- Planning: 43 explicit config overrides parsed in modes1/2, only sample count
  differed. Four focused sampling/trajectory cases passed on CPU (12.78s +2.36s).
- Live reads: A100 SXM80 secure1.59/hr, US-KS-2 availability LOW; existing volume
  e8ishvsuf7 is300GB. Current CLI lacks legacy scheduling flags.
