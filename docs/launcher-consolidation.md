# Reusable RunPod campaign launcher

Extend the working launcher saved in
`artifacts/majepa-deviations-20260917/source/src/majepa/campaign.py` on the
paired-RNG branch. Reuse Osaze's checked-in queue behavior; do not depend on
untracked `/workspace` Python modules. No training algorithm changes, live
launches, commits, or pushes are part of this implementation.

## Design

- `majepa.campaign`: source freeze, explicit JSON experiment specification,
  resolved training/evaluation configs, RunPod provisioning and capacity
  fallback, cost reservations, staging, monitoring, runtime watchdogs.
- `majepa.campaign_queue`: locked persistent per-pod queue; one sequential
  worker per GPU, occupancy and predecessor checks, quota/headroom checks,
  separate job logs, training then final evaluation, exact learner counts,
  verified checkpoint retention. Pod supervisor owns shutdown; jobs never stop
  sibling workers or unrelated processes.
- Existing W&B logger receives resolved config. Campaign runs upload source,
  config, final checkpoint and evaluation artifacts and verify completion.
- Default four GPUs in one pod; `gpus_per_pod=1` preserves independent pods.
  GPUs tried in order L40, L40S. A100 requires explicit opt-in. Budget and
  deadlines apply to the entire pod and include every allocated GPU.
- Explicit maps, treatments, seeds, configuration overrides and W&B destination.
  Frozen source and configs remain immutable. Restarting the controller cannot
  create duplicate pods; uncertain creation fails closed until reconciled.

## Acceptance

- [x] Restore and generalize existing provisioning without hard-coded campaign IDs.
- [x] Add self-contained queue using existing worker/locking/validation patterns.
- [x] Record complete W&B config and verify final artifact upload before pruning.
- [x] Exercise complete mocked multi-GPU lifecycle and failure/resume paths.
- [x] Document dry-run, launch, status, recovery, costs and storage limitations.
- [x] Review integration and complete focused checks without using cloud GPUs.

## Decisions

Multiple pods remain a topology setting, not a second launcher. A job uses one
GPU; multi-GPU pods run independent jobs concurrently, not distributed training.
Predecessors block until successful completion; failed/interrupted jobs require
explicit retry. Storage deletion is restricted to this campaign's completed
intermediate checkpoints after its final checkpoint upload is verified.

## Verification evidence

`python -m pytest tests/test_campaign.py tests/test_campaign_queue.py
tests/test_tracking.py tests/test_evaluation.py -q`: 58 passed.

Ruff check and formatting check passed for the launcher, queue, tracking module
and their tests. `git diff --check` passed.

The documented dry-run and working-tree initialization were executed using the
example specification, followed by archive extraction, the queue's byte-for-byte
source verifier and the frozen controller's status command. This exercised local
packaging and entry points without cloud calls or GPU workloads.

Independent reviews identified and corrected interrupted-creation handling,
observed overrun pricing, derived-name collisions, live PID checks, exact
10-step budget compatibility, replay eligibility bounds, checkpoint retention
and W&B sequence-type normalization. The final fresh integrated review passed;
its independent run of the 50 campaign/queue/tracking tests also passed.

The exact `portpicker==1.6.0` wheel was downloaded and checked against its
`uv.lock` SHA256. Executing its script with `--help` caught and corrected the
wheel's script-versus-module distinction. The installed `bin/portserver.py`
invocation has a regression test and passed a scoped independent review
(14 queue tests).

Final frozen package evidence is in `/tmp/majepa-launcher-verified-20260919`;
source SHA256 is `7b6081b44e4badd05235210bb5dd848e5639bc0424f0620c3f8b876f9867ee1b`.
No live RunPod/SMAC run was performed. Placement identifiers in the example
must be replaced before an actual launch.

The separate `tests/test_configuration.py` suite reports 7 failures and 2 passes:
its failing cases require the removed `imag_action_samples` CLI/dataclass API.
The configuration class, local training CLI and that test file are unchanged by
this work. This launcher does not restore the removed training algorithm feature.
