# BPTT2 workshop matrix — 5 September 2026

The user selected the recurrent reference with **BPTT2** for the full matrix.
The source package remains the screened `f2273e9` snapshot, SHA256
`af6e6d23ab98e60d893071054942889ca73c14d0d9071505c4a9ee886df46b5a`.
Only scheduling, run identity, and map-specific transition budgets changed.
The proposed additional objectives in the earlier workshop recommendation were
not adopted for this matrix.

## Fixed algorithm and evaluation

- One collection environment; 5,000-transition replay prefill, included in budget.
- World-model and PPO learning begin at transition 5,000; world optimizer warmup 0.
- Local/joint world learning rate 4e-5; actor/critic learning rate 3e-5.
- Actor MLP 3 × 1024; critic target full copy (rate 1.0); encoder EMA rate 0.01.
- Recurrent self-fed loss: horizons 2/4/5, eight anchors, scale 0.1, BPTT2.
- Fresh history disabled; imagination horizon 5; factual critic scale 0.3.
- Training ratio 128; all other training settings come from the frozen profile.
- Curves: 32 greedy episodes every 5,000 transitions, four eval environments,
  evaluation worker seed offset 50,000.
- Final: fixed-budget final checkpoint, 100 greedy episodes, four environments,
  evaluation worker seed offset 100,000. No best-checkpoint selection.

## Requested maps

| Map | Training transitions | Agents | Seeds |
| --- | ---: | ---: | --- |
| 3m | 50,000 | 3 | 0, 1, then 2 |
| 8m | 50,000 | 8 | 0, 1, then 2 |
| 2s3z | 50,000 | 5 | 0, 1, then 2 |
| MMM | 50,000 | 10 | 0, 1, then 2 |
| 3s_vs_3z | 50,000 | 3 | 0, 1, then 2 |
| 3s_vs_4z | 50,000 | 3 | 0, 1, then 2 |
| so_many_baneling | 50,000 | 7 | 0, 1, then 2 |
| 2s_vs_1sc | 50,000 | 2 | 0, 1, then 2 |
| 2m_vs_1z | 50,000 | 2 | 0, 1, then 2 |
| 3s_vs_5z | 200,000 | 3 | 0, 1, then 2 |
| 2c_vs_64zg | 200,000 | 2 | 0, 1, then 2 |
| 5m_vs_6m | 200,000 | 5 | 0, 1, then 2 |
| corridor | 200,000 | 6 | 0, 1, then 2 |
| MMM2 | 200,000 | 10 | 0, 1, then 2 |

42 fresh experiments; 2.9M transitions for the first two seeds, then 1.45M
for the third seed. The earlier development screen is not substituted for a
matrix run.

## Queue and preservation

Pod: `root@154.54.102.56:14808`, six A100 GPUs.
Queue root: `/workspace/majepa_bptt2_matrix_20260905`.
Launcher: `/workspace/majepa_bptt2_final100_launcher_20260905/scripts/run_ppo_bptt2_matrix.py`.
Training source: `/workspace/ma_jepa_recurrent_extensions_f2273e9`.

Six detached workers claim jobs under a filesystem lock. Each GPU has a separate
worker lock and port pool. All seed-0 jobs are ordered ahead of seed-1 jobs to
spread initial coverage across maps. All 28 seed-0/1 jobs, including their final
evaluations, must finish before any seed-2 job is claimed. Each worker starts
its next eligible job automatically. A failed job is recorded, other eligible
first-stage jobs continue, and the third-seed stage pauses until that failure
is resolved. Worker and job logs are retained under `workers/`; `queue.json`
contains status, ownership, budgets, and W&B URLs.

Training and final evaluation have distinct, map/seed-specific W&B IDs independent
of GPU assignment. Group:
[ma-jepa-bptt2-matrix-20260905](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-bptt2-matrix-20260905).

Compact results are mirrored locally to
`analysis/ppo_audit/bptt2_matrix_20260905_mirror/`. Successful final checkpoints
are backed up to `/Users/osaze/MARL/remote_archives/majepa_bptt2_matrix_20260905/`.
Both watchers expect 42 outcomes. They preserve source data. These local watchers
require the local machine to remain awake and connected; training and queue
workers continue independently on the pod.

## Launch checks

All 84 training/evaluation configurations were resolved using the pod runtime.
All 14 installed SMAC maps completed a real reset and legal environment step,
with finite observations and valid action masks. This includes the 70-action
`2c_vs_64zg` map and the ten-agent MMM maps.

Focused scheduler checks covered 42 unique identities, distinct claims, the
third-seed barrier, total budgets, and the 200,000-step final-checkpoint guard.
The six live jobs started on separate GPUs at approximately 13:18 UTC.
Deployment, process IDs, source/launcher hashes, and map checks are recorded in
`bptt2_matrix_deployment_20260905.json`. Full training completion remains pending.

The user subsequently selected **100 final episodes only**. At 13:42 UTC the
six workers and their evaluation supervisors were replaced while the six original
training processes and portservers continued. Training PIDs and Linux process
start times were checked before and after this handoff. The frozen training
package and all periodic curve settings remain unchanged. All 42 amended train
and final configurations were resolved again; the adopted completion path was
checked through a successful final100 outcome. Updated ownership and verification
are recorded in `bptt2_matrix_final100_20260905.json`.
