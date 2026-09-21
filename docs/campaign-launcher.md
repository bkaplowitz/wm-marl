# RunPod experiment campaigns

This consolidates the working September 17 provisioning code and Osaze's
per-GPU queue behavior into tracked modules. No external
`run_truncated_geometric_20260915` module is needed. The ordinary local
`majepa-train` entry point remains available.

Each job trains on **one GPU**. A multi-GPU pod runs independent jobs concurrently;
each worker takes another queued job when it finishes. This does not distribute
a single training run across multiple GPUs.

## Specify and review an experiment

Copy `experiments/campaign.example.json` and edit the W&B destination, maps,
seeds, treatments, budget, and network-volume/datacenter pairs. The example's
placement identifiers are placeholders. A network volume is tied to its own
datacenter; supply additional valid pairs for capacity fallback. The mounted
volume must already contain StarCraft II 4.10 Base75689 and the requested SMAC
maps. An explicit `sc2path` may identify that installation.

A placement may omit `volume` in regions without network storage. The launcher
then allocates pod storage sized to `storage.quota_gb` and runs the saved SC2
download/extraction/hash checks in `majepa.campaign_assets` before starting workers.
The bootstrap installs the image's missing `unzip` prerequisite before downloading.
This storage remains attached to the pod; final artifacts are still verified on
W&B before shutdown. Inspect staging commands without downloads using
`python -m majepa.campaign_assets /tmp/sc2-assets --dry-run`.

```sh
uv run --no-sync python -m majepa.campaign init \
  --directory artifacts/my-campaign \
  --spec experiments/campaign.example.json --dry-run
```

Dry-run resolves all profile defaults and explicit overrides, displays the full
per-job configurations and GPU allocations, and performs no file writes, cloud
requests, or launches. Unknown configuration keys fail before provisioning.
The example uses the sole `baseline` profile, including independently sampled
50/50 uniform/recent world and behavior replay. Change the replay settings
explicitly for a different experiment. No earlier multi-action implementation
is restored by this launcher; treatments must be supported by the selected source.

Default topology: four GPUs in one pod, limited to the number of available jobs.
An explicit `gpus_per_pod` reserves that size even when fewer jobs are queued;
idle GPUs still count toward the budget. Set it to `1` for separate one-GPU pods;
`max_gpus` limits the total.
Treatment/map/seed order specifies priority. Set `run_order` to `seed_first` to
queue one run from each treatment before advancing to the next seed; the default
is `treatment_first`. An optional treatment `depends_on`
list names earlier treatments; matching map/seed jobs stay on the same pod and
wait for successful predecessors. An optional absolute `predecessor` queue-file
path gates the whole pod's queue. It must be available on the mounted volume;
missing files cause a visible wait, and failed predecessor jobs cause failure.

The GPU preference is **L40, then L40S**, trying configured placements for each.
A100 is considered only with `allow_a100: true`. There is no silent change of
pod topology or rate ceiling when capacity is unavailable.

## Freeze, launch, and monitor

```sh
uv run --no-sync python -m majepa.campaign init \
  --directory artifacts/my-campaign \
  --spec experiments/campaign.example.json --working-tree

uv run --no-sync python artifacts/my-campaign/controller.py launch \
  --directory artifacts/my-campaign
```

The initialization freezes the current tracked source, vendored DreamerV3
revision, diff, and the new campaign modules into a SHA256-checked archive.
It saves `spec.json`, `resolved-configs.json`, `manifest.json`, and an exact copy
of the frozen controller. Use that controller for later actions so changes to
your working branch cannot change the campaign's launch logic. Untracked extra
source files require explicit `--include-source path`; secrets and arbitrary
untracked files are not automatically included. Omit `--working-tree` to require
clean committed source. Initializing an existing directory fails.

The local controller requires `runpodctl`, W&B credentials in `~/.netrc`, and
`~/.ssh/runpod_key` plus its public key. Staging sends W&B credentials to the
new pod over SSH. The account RunPod key stays on the controller; the existing
pod-scoped shutdown mechanism is reused. The frozen image/bootstrap installs
the locked CUDA/SMAC environment once per pod, then starts its GPU workers.

```sh
uv run --no-sync python artifacts/my-campaign/controller.py status \
  --directory artifacts/my-campaign
uv run --no-sync python artifacts/my-campaign/controller.py monitor \
  --directory artifacts/my-campaign --once
uv run --no-sync python artifacts/my-campaign/controller.py verify \
  --directory artifacts/my-campaign
```

`status` reads saved local state. `monitor` refreshes pod state and exact process
liveness; it also enforces budget/deadline stops for this campaign's pod IDs.
`verify` checks W&B registration, full configuration, source/config artifacts,
and final checkpoint/evaluation artifact flags. The continuing `launch`
controller runs these checks, including checks after 120 seconds. A live
bootstrap, registered W&B run, and completed experiment are reported separately.
`--once` on `launch` makes one controller iteration; otherwise it keeps trying
capacity and monitoring to completion. Ctrl-C preserves reservations and pod
workers; it does not terminate unrelated jobs. On-pod deadline watchdogs remain
active if the local controller disappears.

## Budgets, completion, and recovery

`max_gpu_hourly_rate` is a **per-GPU ceiling**; RunPod's observed `costPerHr` is
checked against the entire pod's allowance. Every GPU is reserved for the full
`max_hours`, including bootstrap, queueing, and evaluation. The pod deadline
is shared by its jobs, not renewed per seed. The full allocation plus a $0.50
shutdown allowance must fit `budget`. Storage/network charges are additional;
this is a GPU-spend cap. The maximum allowed pod runtime is eight hours.

Completed GPU reservations are retained conservatively in budget accounting.
Observed overruns are counted at the observed pod rate. Capacity retries require
the known provider capacity rejection plus confirmation that no exact-name pod
exists. Ambiguous create responses, including interruption during creation,
block further provisioning. Inspect the exact pod name/ID and reconcile its
state before proceeding; the launcher does not guess whether a billable pod
exists or automatically replace failed allocations.

Each pod has a durable `queue.json`, per-GPU worker logs, heartbeat/PID records,
and `jobs/<name>/train.log` / `final100.log`. Completed jobs are not rerun when a
supervisor restarts. A queue left with a running job requires explicit operator
reconciliation; failed jobs are retained, not silently retried with the same
W&B identity. Snapshot replay does not support transparent mid-training resume.
For a deliberate rerun, initialize a new campaign and new run IDs.

Jobs save and upload their complete resolved config and frozen source at startup.
The worker checks the actual saved config, final step counter, exact learner
update count, and 100 greedy evaluation episodes. It uploads the final checkpoint
and evaluation files, verifies remote artifact commitment and file digests, then
prunes **only that job's completed intermediate checkpoints**. Upload failure
retains its local checkpoints. Training checkpoint retention is forced on;
other training behavior comes from the resolved experiment configuration.

The existing training driver collects in ten-step blocks, so budgets must be
multiples of ten. Exact learner-count validation currently supports one
environment and sufficient replay prefill; unsupported settings fail during
planning. `run.save_every` follows the training loop's clock in **seconds**;
`run.curve_eval_interval` is in environment steps. Curve checkpoints are retained
until final upload verification.

The storage guard checks both filesystem free space and measured volume usage
against `storage.quota_gb`, preserving `storage.min_free_gb` headroom. It checks
before and during jobs; it does not delete another campaign's files. The
supervisor owns pod completion. One failed job is recorded while independent
jobs continue; one finished worker cannot stop the other GPUs.

## Local verification

```sh
uv run --no-sync python -m pytest \
  tests/test_campaign.py tests/test_campaign_queue.py tests/test_tracking.py
```

These tests mock cloud/W&B boundaries and exercise GPU assignment, queue claims,
dependent jobs, source/config integrity, exact counters, artifact failures,
checkpoint retention, startup checks, and budget/recovery behavior. They do not
claim a live multi-GPU RunPod launch has been performed.
