# Running the approved multi-sample campaign

Use the reviewed commit on `feat/majepa-multi-sample-launch`. The campaign uses
one A100 SXM80 per pod, the existing US-KS-2 volume, and a separate persistent
folder for each job. The launcher creates no resources in dry-run mode.

```sh
PYTHONPATH=src python3 -m majepa.campaign init --directory "$PWD/artifacts/majepa-20260915" --dry-run
# After committing the reviewed source, freeze it once:
PYTHONPATH=src python3 -m majepa.campaign init --directory "$PWD/artifacts/majepa-20260915"
PYTHONPATH=src python3 -m majepa.campaign launch --directory "$PWD/artifacts/majepa-20260915" --job smoke
PYTHONPATH=src python3 -m majepa.campaign monitor --directory "$PWD/artifacts/majepa-20260915" --once
```

Run the monitor without `--once` to poll every 30 seconds. It can be restarted
with the same directory. Keep its output in a local log. Each pod also starts an
independent boot-time deadline watchdog, then a completion watchdog before the
bootstrap. Stops preserve the network volume. The monitor records observed
billing duration conservatively from the pre-creation timestamp to first stopped
observation, and reserves the full maximum allocation when admitting each job.

After the smoke's GPU/SMAC/W&B round trip and automatic stop are verified, launch
`samples2-seed0`. Verify a finite post-prefill PPO update before launching
`samples2-seed1`, `samples2-seed2`, and `samples1-seed0`. Launch `samples1-seed1`
and `samples1-seed2` as slots free. Repeat the exact `launch` command with each
job name. Each real job automatically runs the final deterministic 100-episode
evaluation from its final checkpoint with the same custom architecture.

The smoke defaults to 30 minutes; each real job defaults to five hours including
provisioning and setup. `--max-hours` sets one explicit maximum, never above eight;
budget admission may reject it. The total default reservation is $48.495 GPU
cost, plus $0.50 withheld for shutdown latency. Storage cost is outside this GPU
ledger. If the control plane reports a higher GPU rate, launch fails and stops
only that newly created pod.

Local `manifest.json` contains the exact pod and W&B IDs, source revision/hash,
allocation deadlines, and status. Per-pod files are under
`/workspace/<campaign>/<job>/`: `job.log`, `watchdog.log`, `job.json`,
`outcome.json`, `run/`, and `evaluation/`. W&B records live in
`osaze-obahor/majepa-multi-sample-treatments`. Artifact verification receipts live
in each training/evaluation log directory. A stopped pod alone does not establish
successful training; require checkpoint/evaluation artifact receipts and history.

If creation returns an uncertain response, the manifest retains
`creation_uncertain` and rejects recreating that job. Reconcile the exact named
new pod through the control plane before any further action. Never delete a
manifest reservation to bypass admission or manipulate old pods. A failed
bootstrap leaves an outcome file and is stopped by its completion watchdog.
