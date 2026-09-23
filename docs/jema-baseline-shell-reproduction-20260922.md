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
| 0 | [57](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/cc9216b2879a) | [70](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ke21-control-repro1-2s3z-s0-621d97-final100) | 36 | -21 |
| 1 | [59](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/277f218437eb) | [75](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/ke21-control-repro1-2s3z-s1-621d97-final100) | 31 | -28 |
| 2 | [39](https://wandb.ai/osaze-obahor/majepa-ppo-treatments/runs/29bb1ef14687) | Not found in this reference group | 50 | +11 |
| Three-seed mean | 51.7% | Not available | 39.0% | -12.7 |

The matched seeds 0 and 1 average 58.0% in our campaign and 72.5% in Osaze's
reference. Do not compare Osaze's two-seed mean directly with our three-seed mean.

Running Osaze's shell launcher unchanged did **not** reproduce his numbers. It
scored below our own campaign, so the launcher is not the missing ingredient.

## The training curves diverge; evaluation protocol is second order

This is the decisive evidence, and it corrects the earlier working hypothesis
that `run.eval_worker_offset` explained the gap.

The held-out curve evaluation during training (`curve_eval_interval 5000`,
`curve_eval_eps 32`, `curve_eval_seed_offset 50000`) is honored identically on
both branches, so it is protocol-independent. `eval/battle_won_mean`:

| Env step | osaze s0 | osaze s1 | campaign s0 | campaign s1 | shell s0 | shell s1 |
|---|---:|---:|---:|---:|---:|---:|
| 15000 | 0.125 | 0.000 | 0.3125 | 0.4375 | 0.0625 | 0.2188 |
| 30000 | 0.031 | 0.469 | 0.375 | 0.4375 | 0.406 | 0.1875 |
| 45000 | 0.8125 | 0.750 | 0.4375 | 0.4375 | 0.656 | 0.2188 |
| 50000 | 0.656 | 0.750 | 0.531 | 0.500 | 0.438 | 0.3125 |

Each run's final100 tracks its own held-out curve: osaze 0.656/0.750 to 70/75,
campaign 0.531/0.500 to 57/59, shell 0.438/0.312 to 36/31. The gap is already
present in training and is merely carried through by evaluation.

The runs start identical and separate only once learning begins:

- At step 5000, `eval/agent_return_mean` is identical across all six runs per
  seed (s0 5.6039, s1 5.9579). Initialization and pre-learning collection match.
- By step 10000 they have separated: s0 osaze 5.5859, campaign 2.1453, shell 0.9174.
- Learning starts at step 5000 (`world_model_start_step` / `ppo_start_step`).

Update ratios are matched, so this is not a replay-ratio difference: osaze
5519/49150, campaign 5534/49270, shell 5583/49660, all about 0.1123.

Osaze's stack is deterministic: his `control` and `control-repro1` runs are
bit-identical (s0 curve 0.65625 and final100 70; s1 curve 0.75 and final100 75).
GPUs differ across runs (osaze s0 L40, osaze s1 L40S, campaign L40S, shell
A100-SXM4-80GB) but that does not explain a gap this large.

### Recovered checkpoint evidence

The three shell checkpoints were pulled off pod `ebmp9z3n12e2vk` before teardown
and verified by SHA-256 against the pod. All three agree exactly:

| Counter | s0 | s1 | s2 |
|---|---:|---:|---:|
| `step` | 50000 | 50000 | 50000 |
| `first_learner_environment_step` | 5000 | 5000 | 5000 |
| `first_ppo_environment_step` | 5000 | 5000 | 5000 |
| `learner_update_calls` | 5626 | 5626 | 5626 |
| `ppo_update_calls` | 5626 | 5626 | 5626 |
| `world_only_update_calls` | 0 | 0 | 0 |

The recorded per-episode metadata confirms the shell evaluation protocol
directly: `worker_index` takes the values 100000, 100001, 100002 and 100003,
that is four workers of 25 episodes at offset 100000. Seed 0 wins split across
those workers as 8, 13, 8 and 7 out of 25. That spread is what independent seed
streams produce by chance at this win rate, so the offset introduces variance
but no systematic bias. A 36-versus-70 gap is roughly seven standard errors and
cannot be attributed to it.

`world_only_update_calls` is 0 and the learner and PPO both start at step 5000,
so this configuration runs no world-model-only warmup. Whether Osaze's runs did
is worth checking against his checkpoint counters.

### Remaining suspects

Configuration diffing found **zero** keys present in all three configs with
differing values. 52 keys exist only in Osaze's config, dropped when the branch
was reduced (`conhead.*`, `rewhead.*`, `marl.ctde.teammate_belief.*`,
`mask_calibration.*`, `ppo.entropy_schedule.*`, `ppo.factual_value.*`,
`agent.paired_rng`, `agent.collection_unimix`,
`agent.simplification.local_outcomes`, `run.paired_dir`, `run.paired_phase`,
and the matching `loss_scales.*`). 13 exist only in ours.

Osaze invokes `--configs smac_vector ma_jepa` plus about 80 explicit CLI
overrides; the reduced branch uses `--configs baseline`. Flags in his argv with
no counterpart among our config keys are the leading suspects:
`--run.isolate_report_rng True`, `--replay.world_uniform_mix 0.5`,
`--jax.prealloc True`, `--run.replay_stream_mode snapshot_staggered`,
`--run.replay_trace_batches 16`, `--replay.recency_decay 0.9998`,
`--run.replay_startup_behavior_min_starts 4`,
`--replay.behavior_recency_decay 0.9998`,
`--replay.behavior_uniform_mix 0.5`, `--run.train_ratio 128`.

`--run.isolate_report_rng` is the most interesting of these: it changes the
training trajectory without changing semantics, which is exactly the signature
of a run that matches at initialization and diverges once learning starts.

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

## Recovered artifacts and pod teardown

Pod `ebmp9z3n12e2vk` was stopped at 14:30 UTC on 22 September after every
artifact was copied locally and verified, then terminated. `get-pod` now
returns 404, so the pod and its 150 GB volume are gone and billing has ended.

Copied and verified:

- `artifacts/jema-offset0-20260922T141004Z/checkpoints/s{0,1,2}/` — the full
  checkpoints in the layout `evaluate.sh` expects, each with `done` and a
  670,018,778-byte `agent.pkl`. SHA-256 verified against the pod:
  - s0 `b6ed7851c47b95a1d368fd99c6b2d54f1a0a664d7edb0f87392ed8c3358a6a7b`
  - s1 `9cde2efc15596118004318a2493ac8bd33a102bca832923bb443fe01c26bbf9a`
  - s2 `1115abaa334c6bed0e9604552a8016b33a13538bd1db58f098b6c4712237c6db`
- `artifacts/jema-baseline-shell-20260922-r3/runs/s{0,1,2}/` — training
  `config.yaml`, `metrics.jsonl`, `scores.jsonl`, `replay_snapshots.jsonl`, the
  checkpoint counters, 50 replay shards per seed, and the complete `final100`
  records including `evaluation_episodes.jsonl`.
- `artifacts/jema-baseline-shell-20260922-r3/{seed0,seed1,seed2,setup}.log`.
- `artifacts/jema-baseline-shell-20260922-r3/source.tar.gz`, SHA-256
  `3d88618615482524d80b2d2f838f6c391d0059d3984091ec3617e03ed5a64d2c`, matching
  the digest recorded at the top of this document.

Only the 12 GB StarCraft II install under `assets/` and the extracted `source/`
tree were left behind; both are reproducible from `source.tar.gz` and the
pinned commit.

Pod SSH access needs `~/.runpod/ssh/runpodctl-ssh-key`, not the keys in
`~/.ssh/`. The pod's injected `PUBLIC_KEY` holds only RSA `runpodctl-ssh-key`
entries, so the ed25519 keys in `~/.ssh/` are rejected, as is the
`ssh.runpod.io` proxy.
