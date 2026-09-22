# Direct-shell baseline reproduction

The ablation plan is on hold while we investigate why our baseline scores are
lower than Osaze's. The requested experiment is to execute the existing
`scripts/run_baseline.sh` unchanged for seeds 0, 1, and 2 and report its final
100-episode evaluation results. Do not replace this experiment with tests or
change the model before collecting the results.

## Source and execution

- Use the archived source from our completed baseline campaign,
  `artifacts/jema-joint-endpoints-20260921-r1/source.tar.gz`.
- Commit: `8acc593fd38b550d86ef0400c4c7e4d47410d7e7`.
- Archive SHA256:
  `3d88618615482524d80b2d2f838f6c391d0059d3984091ec3617e03ed5a64d2c`.
- The archived shell launcher is identical to the current one, SHA256
  `25daf743f086d00e5c403b232552ea7d8c3b1b34fd7c59fcb386f61ec1df94e2`.
- Use the archive's `uv.lock` and pinned DreamerV3 revision
  `e3f02248693a79dc8b0ebd62c93683888ddaccfe`.
- Dedicated pod: `6z28kencavbqk1`, named
  `jema-baseline-shell-20260922T083902Z`; three L40S GPUs, $3.27/hour.
- Eight-hour deadline: Unix time `1790095188.096595`.
- Remote root: `/workspace/jema-baseline-shell-20260922T083902Z`.
- W&B group: `jema-baseline-shell-20260922T083902Z`, under
  `osaze-obahor/majepa-ppo-treatments`.
- Local operational receipt:
  `artifacts/jema-baseline-shell-20260922T083902Z/launch.json`.

Run one seed per GPU by invoking `./scripts/run_baseline.sh` directly from the
extracted source. Set the script's supported environment variables for seed,
Python interpreter, SC2 location, and unique output/W&B identifiers. Use
`CUDA_VISIBLE_DEVICES` to assign one GPU per process. The script itself supplies
the training and final-evaluation commands. Do not supply campaign-generated
configuration files or enable the campaign tracking hook for this experiment.

Training remains `2s3z`, five agents, difficulty 7, 50,000 steps, local outcomes
disabled, local prior enabled, margin weight 0.1, latent 32×64, joint backprop off,
critic feedback off, and all other recorded baseline settings.

## Evidence already found

Resolving the shell launcher's training arguments produces no configuration
differences from our saved baseline runs after accounting for log paths. Osaze's
seed-0 training run also matches every shared recorded configuration field
except its log path. His earlier config does not contain our added local-prior
flag or four world-model-gradient metadata fields.

The **final evaluation protocols differ**:

| Setting | Our completed campaign | Existing shell launcher |
|---|---|---|
| Final episodes | 100 | 100 |
| Policy mode | Greedy (`eval`) | Greedy (`eval`) |
| `run.envs` | 1 | 4 |
| `run.eval_worker_offset` | 0 | 100000 |
| `jax.precompile` | true | false |
| `run.curve_eval_interval` | 5000 | 0 |
| `run.eval_envs` | 4 | 1 |
| `run.world_model_start_step` | 5000 | 0 |

`eval_only()` actually uses `run.envs` and `run.eval_worker_offset` to construct
the evaluation workers. Thus our old result uses one stream of 100 episodes;
the shell result uses four streams of 25 episodes with different worker-derived
environment seeds. This is an observed protocol mismatch, not yet proof that it
explains the score gap. The other listed differences must not automatically be
treated as causes; some are inactive in evaluation-only execution.

## Results

All numbers in this table are wins out of 100 from separate final evaluations.
The Osaze references are the `ke21-control-repro1-*` runs identified by
`docs/provenance.md`, in group `ma-jepa-local-kl-exploration-20260921`.

| Seed | Our original campaign | Osaze reference | Direct shell rerun | Shell minus our original |
|---|---:|---:|---:|---:|
| 0 | [57](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cc9216b2879a) | [70](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ke21-control-repro1-2s3z-s0-621d97-final100) | Pending | Pending |
| 1 | [59](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/277f218437eb) | [75](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ke21-control-repro1-2s3z-s1-621d97-final100) | Pending | Pending |
| 2 | [39](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/29bb1ef14687) | Not found in this reference group | Pending | Pending |
| Three-seed mean | 51.7% | Not available | Pending | Pending |

The matched seeds 0 and 1 average 58.0% in our campaign and 72.5% in Osaze's
reference. Do not compare Osaze's two-seed mean directly with our three-seed mean.

## Follow-up if the results differ

1. Finish the three requested direct-shell training runs and their own final100
   evaluations. Record the exact checkpoints, saved configs, final counters,
   per-episode results, and W&B links.
2. Evaluate each old baseline checkpoint under the shell's four-environment,
   offset-100000 protocol. This holds training fixed while changing evaluation.
3. If necessary, evaluate each new shell-trained checkpoint under the old
   one-environment, offset-0 protocol. The resulting crossed comparison separates
   training-run differences from evaluation-protocol differences.
4. If the gap remains under a matched evaluation protocol, compare the actual
   source, dependency/runtime versions, initialization, replay/collection
   progression, update counts, and recorded training metrics. Do not infer a
   training bug solely from a different win count.
5. Update this document with evidence and the explanation. Resume the ablations
   only after addressing this baseline discrepancy with the user.

## Operational state

The original pod became unreachable after training began. It was stopped and
then deleted, including its pod volume, at the user's request. RunPod confirmed
deletion and a subsequent lookup returned 404; no completed results were recovered.

Replacement pod `y4leuwgzbs5j1b` (`jema-baseline-shell-20260922-r2`) has three L40S
GPUs in Taiwan at $2.37/hour. It uses the same frozen source and original deadline,
with W&B online logging. Current state and connection details are recorded in
`artifacts/jema-baseline-shell-20260922-r2/launch.json`.
Setup was verified complete at 10:01 UTC on 22 September; training had not started.
The user has excluded Taiwan (`TW`) from all future pod placements.

The three unchanged shell scripts were invoked at 10:14:17 UTC for seeds 0/1/2.
All exited before training: CUDA initialization returned error 999, and opening
`/dev/nvidia-uvm` directly returned an input/output error. The same CUDA failure
occurred without a GPU-visibility filter. No result was produced; the pod needs
driver recovery or replacement, not a model/configuration change.
The user authorized replacement; RunPod confirmed deletion of this pod and its
pod-attached storage. All three failed-launch logs are preserved in its local
artifact directory. The replacement must be outside Taiwan.

The user approved an A100 fallback after non-Taiwan L40/L40S capacity rejections.
Replacement `ebmp9z3n12e2vk` has three A100 80GB GPUs in US-MD-1 at $4.77/hour.
Both image CUDA 12.4 and pinned JAX computation passed on all three GPUs before
launch. The unchanged shell scripts for seeds 0/1/2 were invoked at 11:00:30 UTC
on 22 September, with online W&B group `jema-baseline-shell-20260922-r3`.
Current operational state is in `artifacts/jema-baseline-shell-20260922-r3/launch.json`.
