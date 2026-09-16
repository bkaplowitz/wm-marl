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
- [x] Minimal launcher with side-effect-free dry-run, exact manifest, safe SSH,
  detached jobs, validated input, restartable monitoring and budget/time backstops.
- [x] Canonical custom config resolves correctly; seed/sample count are the only
  experimental differences; no default hyperparameter changes.
- [ ] Source/config metadata uploaded; final checkpoint and eval uploaded with
  remote artifact verification; failed uploads visibly fail completion.
- [x] Focused mocked launcher checks, sampling tests, and independent source review.
- [x] A100 smoke: JAX GPU compute, SMAC reset/step, W&B metrics, tiny artifact
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
- 2026-09-16: The campaign's operational queue and W&B verifier run separately
  from the local budget monitor. Controls1/2 invoke the existing launch CLI only
  after a slot frees. The reused live check revalidates all70 config values,
  finite training scalars, action metrics and source verification; it writes
  all-launches-verified.json only after six jobs have post-prefill receipts and
  are at least120 seconds old. Its one-pass live check passed on the current four
  runs and correctly left the six-run completion receipt absent. All four current
  runs also have complete persistent checkpoints (nine files including done and
  approximately747 MB model state). Campaign training and final evaluation remain
  in progress; queued controls are not represented as already launched.
- 2026-09-16 15:24 UTC: All four active real runs are online in W&B with
  source_verified=true and matching 70-override configs. All four passed finite
  post-prefill PPO verification (342 finite training scalars each, zero illegal
  action fraction, entropy .003); all three two-sample runs have positive second
  sample validity. First-update and config receipts are saved per job. Controls
  seed1/seed2 remain queued pending free slots; full campaign acceptance is open.
- 2026-09-16 15:05 UTC: Pushed/froze `248dd33` into campaign
  `artifacts/majepa-multi-sample-20260916-r2`. Revised smoke `osdui3eoz7i570`
  passed GPU sum 2097152, SC2 4.10 reset/step, W&B metrics and independently
  downloaded byte-matched artifact `smoke-645c10df4379:v0`; persistent outcome
  completed=true and control-plane EXITED confirm automatic stop. Installing
  135 packages on container disk took 632ms after download. Prior slow smoke
  `jmsqswxc2lxa6g` stopped at its deadline and is carried into the new ledger.
  Each real run now reserves 4.8 hours; prior costs, both smoke reservations,
  six real runs and shutdown allowance remain below the $50 aggregate GPU cap.
  Two-sample seed0 (`8ej29mkciqflp6`, W&B `5ae95eed3148`) passed its first PPO
  gate at 5605 steps: 342 finite training scalars, zero illegal-action fraction,
  positive second-sample validity, fixed entropy .003, WM/PPO start5000. All70
  overrides match both resolved config and W&B; source artifact is COMMITTED.
  Then launched two-sample seeds1/2 (`cbkbdfo7d40r4w`, `onqmuhqas8o19a`) and
  control seed0 (`kgxd9ni44yvzp8`). All four actual A100 SXM80 GPUs and immediate/
  >=120s liveness checks verified; all resolved configs match. New three runs
  are compiling, so their W&B/PPO checks remain pending. The final two controls
  are queued through the existing CLI as slots free; the small operational queue
  passed dry-run and a mocked occupied/free-slot check. Exact IDs, evidence, logs,
  and queue state live in the campaign directory. No unrelated pods were changed.
- 2026-09-16: User authorized UK or other available capacity and continued work
  until successful launch. Live catalog has no UK datacenter; US-KS-2 regained
  A100 SXM 80 GB availability. Reverified the 500 GB volume and restored SC2
  hashes before launching smoke jmsqswxc2lxa6g from frozen a52e635. Hardware is
  A100-SXM4-80GB, 81920 MiB. Actual volume use is about 60 GiB. Package copy onto
  the network volume is slow (21 MB in a measured 25-second interval), motivating
  the bounded container-venv correction. Fresh spec, simplicity, and verifier
  reviews passed; verifier independently passed all 51 focused tests in 3.19s,
  Ruff/format/whitespace, generated Bash syntax and interpreter/stop integration.
  The older smoke remains protected by its original deadline; revised-source
  live smoke and the six experiment runs remain outstanding.
- 2026-09-16: Bounded bootstrap correction installs the locked remote environment
  on container disk at `/opt/majepa-venv` using native `UV_PROJECT_ENVIRONMENT`.
  The runner uses that environment's Python; training/evaluation inherit it through
  existing `sys.executable`, and dry-run commands now match. Source, configuration,
  checkpoints, logs, and assets remain on `/workspace`. Two regression assertions
  failed before the change, then all 51 focused tests and Ruff checks passed.
  No remote resources, running bootstrap, commit, or source bundle changed in this
  correction; fresh review and live verification remain with the coordinator.
- 2026-09-15 20:16 UTC: User asked to proceed after checkpoint recovery. Expanded
  volume e8ishvsuf7 from 300 to 500 GB and verified the new size. Revised-source
  smoke j1kxoufxgukkrt launched, passed the >=120-second liveness check, failed
  on missing SC2 assets, and automatically stopped; observed GPU cost $0.140497.
  Restored the missing assets into /workspace/majepa-multi-sample-20260915-r3/assets/StarCraftII:
  13,401 files, 4,938,935,331 bytes, matching executable and 2s3z-map checksums.
  Retry r4 was rejected for unavailable A100 SXM 80 GB capacity in US-KS-2.
  Complete control-plane inventory confirmed no matching pod; recorded a zero-cost
  creation reconciliation. No real training runs started in this retry.
  Aggregate prior observed GPU cost is $1.566685; six 4.9-hour runs plus a
  0.5-hour smoke and shutdown allowance still fit the $50 GPU cap. EUR-IS-1 lists
  the same GPU at $1.59/hour and supports standard network volumes. User choice
  is pending on that region plus a separate 150 GB volume ($10.50/month, billed
  hourly), versus waiting for US-KS-2 capacity. Evidence and the concrete fallback
  proposal are under artifacts/majepa-multi-sample-20260915-r4/. No GPU-type,
  experiment-setting, source, or additional-volume change has been made.
- 2026-09-15: The first two-sample training run passed its live learning gate:
  W&B `96f158e70565` recorded the exact 70 explicit overrides, a committed and
  verified source artifact, and 342 finite training scalars. World-model and PPO
  learning both began at environment step 5000; illegal-action fraction was zero
  and second-sample valid fraction was positive. The shared volume then exhausted
  its quota: seed1 bootstrap and seed2 source upload failed; seed0 became failed
  at 6880 logged steps. The outcome-write failure exposed the shutdown bug below.
  Root stopped only that failed new seed0 pod. All five campaign pods are now
  confirmed stopped, with conservative observed GPU cost $1.4261879621724287.
  A complete 5000-step checkpoint remains on the persistent volume. An attempted
  local checkpoint backup was rejected by automatic approval review; explicit
  user permission for that export is pending and no weights were copied.
- 2026-09-15: Storage expansion approval is pending: shared volume `e8ishvsuf7`
  is 300GB and `du -sk /workspace` reports 312031266 KiB, while filesystem `df`
  misleadingly reports 189TB free. Proposed increase is 500GB, adding $14/month
  at the documented standard storage rate; RunPod does not support shrinking.
  Old ledgers and reservations remain unchanged. The prepared retry allocation
  deducts the closed campaign's conservative observed cost from the $50 cap,
  leaving $48.57381203782757 for a revised-source smoke and six fresh runs.
  Thirty minutes for smoke, 4.9 hours per real run, and the $0.50 shutdown allowance
  total $49.46718796217243 including prior charges. Details are in
  `artifacts/majepa-multi-sample-20260915-r2/retry-allocation-proposal.json`.
- 2026-09-15: Corrected the quota-related failure-stop bug in `campaign.py`.
  Live two-sample seed0 failed at 6880 steps when the 300GB shared volume hit its
  quota; the failed atomic `outcome.json` write left the completion watchdog waiting.
  The coordinator confirmed all five campaign pods stopped and recorded conservative
  observed GPU cost $1.4261879621724287 in the r2 manifest. The completed 5k
  checkpoint remains on the shared volume. `run_job` now requests its own exact pod
  stop in `finally`; bootstrap uses an EXIT trap through the same `own_stop` helper;
  the deadline watchdog still stops if outcome writes fail. Seven failing regression
  cases were observed during test-first development, followed by 50 passing focused
  tests. No pods launched, weights exported, source bundles changed, or commits made
  by this implementer. Fresh review and live revised-source verification remain.
- 2026-09-15: Retry smoke `x2tyvk0gol01xk` passed GPU computation (2097152),
  SMAC reset/step, W&B metrics, and artifact upload/download byte equality.
  W&B run `d0e4f3672ce1` is finished, artifact `smoke-d0e4f3672ce1:v0` was
  independently downloaded and checked locally. Outcome reports completed=true;
  watchdog log and control-plane status confirm automatic stop. Both smoke pods
  together cost at most $0.387 by conservative observed duration; full $1.59 in
  reservations remains in the ledger. Evidence is under
  `artifacts/majepa-multi-sample-20260915-r2/smoke-evidence/`.
- 2026-09-15: Pushed and froze commit `0d099e7`. Fresh second-round spec,
  simplicity, and verification reviews passed; 43 focused campaign/config/artifact/
  evaluation tests passed. The first live A100 smoke pod `oauk0q2coolse1` launched,
  installed locked dependencies, and automatically stopped on SC2 discovery failure.
  Control plane confirmed stopped; conservative observed GPU cost was $0.154.
  Evidence is in `artifacts/majepa-multi-sample-20260915/smoke-evidence/`.
  Root cause: existing `/workspace/StarCraftII` points to container-local
  `/opt/StarCraftII`. Copying the verified installation into a new campaign-owned
  persistent assets directory; no source or hyperparameter changes are needed.
  Retry ledger `artifacts/majepa-multi-sample-20260915-r2/manifest.json` retains
  the entire first reservation under `smoke-attempt1`, its original name and pod ID.
  Two smoke reservations plus six five-hour training allocations total $49.29;
  the $0.50 shutdown allowance remains within the $50 cap.
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
- 2026-09-16: Use uv's native absolute `UV_PROJECT_ENVIRONMENT` setting to avoid
  dependency installation on the slow shared filesystem. `/opt/majepa-venv` is
  private to each one-job pod; environment rebuilding remains part of bootstrap.
  Keep locked extras, Python version, pinned image, persistent artifacts, and all
  experiment settings unchanged. Context7 `/astral-sh/uv` confirms the setting
  overrides the project environment path; no wrapper, symlink, or dependency added.
- 2026-09-15: Treat persistent outcome metadata as independent of the stop request.
  Reuse the existing ID-validated, pod-scoped `own_stop` helper for runner completion
  and bootstrap EXIT; retain the boot-time deadline backstop and watchdog retries.
  Disable only bootstrap's ERR metadata trap before entering training, preserving
  detailed runner outcomes. No storage resize, retry-budget, or experiment-setting
  changes are part of this bounded correction.
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
- Container-environment correction: `.venv/bin/python -m pytest
  tests/test_campaign.py -q -k 'container_environment or dry_run'` failed on both
  intended regressions before implementation (missing environment export and old
  dry-run Python path). After the three production-line changes, `.venv/bin/python
  -m pytest tests/test_campaign.py tests/test_campaign_artifacts.py
  tests/test_evaluation.py tests/test_configuration.py -q` passed 51 tests in 4.01s.
  Existing executable Bash regressions continue to exercise bootstrap/training
  failures with unavailable outcome paths and exact-pod stop behavior. `.venv/bin/ruff
  check src/majepa/campaign.py tests/test_campaign.py`, `.venv/bin/ruff format
  --check src/majepa/campaign.py tests/test_campaign.py`, and `git diff --check`
  passed. Remote installation speed and GPU execution remain unverified by this
  local correction.
- Quota correction: fresh spec and simplicity reviews passed. Independent verifier
  ran the 50 focused tests successfully in 3.13 seconds and passed Ruff/format/
  whitespace checks. Its additional integrated Bash -> runner -> real own-stop
  helper with a fake CLI passed writable, ENOSPC, and EDQUOT cases, preserving
  failure status and restricting all stop calls to the recorded pod ID. Local
  correction verification passes; full-plan verification remains incomplete until
  revised-source live smoke and all six experiments finish. The smoke checkbox is
  reopened for the revised source; earlier successful smoke evidence is retained.
- Quota failure-stop correction: `.venv/bin/python -m pytest tests/test_campaign.py
  tests/test_campaign_artifacts.py tests/test_evaluation.py
  tests/test_configuration.py -q`: 50 passed in 2.97 seconds. Regressions execute
  generated Bash through both bootstrap and runner failure, with writable/unavailable
  outcome paths; simulate ENOSPC and EDQUOT on runner/watchdog metadata writes;
  verify exact recorded stop IDs and preserved training-error metadata. Existing
  real Bash/fake-CLI credential and mismatched-ID tests also pass. `.venv/bin/ruff
  check src/majepa/campaign.py tests/test_campaign.py`, `.venv/bin/ruff format
  --check src/majepa/campaign.py tests/test_campaign.py`, and `git diff --check`
  passed. Live revised-source behavior is not claimed by these local tests.
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
