# Targeted L40 diagnostics — 7 September 2026

Requested work: execute action-value and executable-policy-drift diagnostics on the idle fourth L40, and queue the previously proposed controlled factual-value comparisons.

## Host and ownership

- Pod `gw2ctmfarlgbsb`, four NVIDIA L40 48 GB, SSH `root@64.247.206.120 -p 31744`.
- Queue `/workspace/majepa_targeted_l40_20260907/queue.json`; persistent workers own one GPU each.
- GPUs 0–2: restart MMM seeds 0–2 following first-update memory allocation failures. Failed directories and W&B attempts are preserved under the previous queue. Only the allocator settings change: `jax.prealloc=True`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Batch size, architecture, losses, update rates and training RNG are unchanged.
- GPU 3: frozen diagnostic, then the six predeclared control/intervention runs below. Each training run includes 100 final evaluation episodes and its own frozen action-ranking diagnostic. Controls also retain intermediate checkpoints for temporal policy comparisons.
- Learner package remains SHA256 `101751eba8d04bb6f4db7348760703fc15e97d25d6ac00c80bd1ab289c227eb8`; only launcher/diagnostic scripts are added.
- Claim cutoff remains 2026-09-08 07:42:23 UTC; decision deadline remains 2026-09-08 10:42:23 UTC. Pending jobs are not claimed after the cutoff. Queue completion time is not guaranteed.

## Available weights and first diagnostic

The first diagnostic uses the locally preserved `recurrent_bptt2-2s3z-seed0` final checkpoint, timestamp `20260905T123707F944631`. It is explicitly labeled **historical BPTT2, before alignment and mixed replay**, not a current combination checkpoint. Its original full checkpoint remains in the local remote archive. A lossless weights-only gzip copy is staged on the L40 pod; optimizer tensors are omitted because inference forbids writes.

The current alignment + mixed replay completed weights remain on the stopped A100 pod. W&B has their metrics/configurations but no uploaded model artifact. Two API recovery attempts returned the configured six GPUs rather than confirming zero GPUs; guards immediately stopped the source again. A subsequent API read confirms it is `EXITED`. No source data was deleted, and no old six-GPU training was restarted. The diagnostics on new current-combination checkpoints are therefore explicitly queued after the new controls.

W&B diagnostic group: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-targeted-diagnostics-20260907

Initial executable run: `l40d-historical-bptt2-2s3z-a1`. The preceding attempt failed on diagnostic metadata construction before loading a model; its log remains preserved. The corrected diagnostic **completed eight roots, 24 action candidates and 48/48 exactly matched real branches**, with zero prefix rejections. GPU 3 then automatically claimed `control-3s_vs_4z-seed0`.

All three MMM retries passed the previously failing first training update. At the verification snapshot they had reached 5,010 environment transitions, two learner/PPO update calls each, finite losses and 100% GPU utilization. This verifies the memory correction at initial training, not final performance or completion of the full runs.

## Action-value protocol

1. Collect eight roots on real sampled-policy histories. Legal actions use the learned stochastic policy, without collection exploration mixing. The root set is exploratory, not a new win-rate benchmark.
2. Compare up to three legal single-agent action alternatives, holding teammates' root actions fixed. Include movement and attack alternatives where legal.
3. For each candidate, average 16 five-step model rollouts with a frozen target-critic bootstrap. Model randomness is shared across candidates.
4. For each candidate, launch two fresh real games using the original environment seed and replay the exact joint-action prefix. Check exact global SMAC state, local observations and legal masks at every prefix time. Reject mismatches; never silently treat approximately matched histories as counterfactuals.
5. Follow the stochastic decentralized policy to the actual episode end with common continuation randomness. Record actual rewards, full discounted returns, five-step rewards and factual successor-value bootstraps separately.
6. Report within-root action ordering and observed selection regret. This is not the exact GAE(lambda) training target, and finite branch repeats create uncertainty. Prefix equality verifies observable simulator state, not an inaccessible complete SC2 random-state snapshot.

The original model parameters cannot be created or modified by diagnostic forward calls. All tensors are identified by a model hash. Raw histories, per-branch outcomes and summary files persist on the network volume; the final script also uploads compact diagnostic output files to W&B.

## GPU 3 training queue

| Order after initial diagnostic | Map | Seed | Treatment |
| --- | --- | --- | --- |
| 1 | 3s_vs_4z | 0 | Current alignment + mixed replay control |
| 2 | 3s_vs_4z | 0 | Control + factual successor-state value targets |
| 3 | 3s_vs_4z | 0 | Factual targets + representation value loss, scale 0.1 |
| 4 | 2s3z | 0 | Current alignment + mixed replay control |
| 5 | 2s3z | 0 | Control + factual successor-state value targets |
| 6 | 2s3z | 0 | Factual targets + representation value loss, scale 0.1 |

All use 50k transitions including 5k prefill, BPTT2, trajectory alignment 0.1, 50% uniform/50% recency world replay and uniform behavior roots, one environment, the established optimizer settings and full-copy critic target. Factual treatments use the already implemented bounded V-trace correction with recorded collection mixture probabilities and rho/c caps of 1. This is a single-seed mechanism screen, not evidence of improved cross-seed stability.

Each control saves the existing curve-evaluation checkpoints (checkpoint I/O only). After its 100-episode final evaluation and final action-ranking diagnostic, the most recent earlier checkpoint and final checkpoint are compared on identical real histories. The diagnostic holds recorded actions, legal masks and posterior random keys fixed and evaluates old weights, encoder/history replacement only, actor replacement only, and both replacements. It reports legal-policy KL and greedy-action changes. An identical-checkpoint repeat must match exactly. These interval contributions are not additive and do not establish the cause of every individual PPO update.

Workers preserve failures, exit instead of blindly retrying, and do not advance to the next value treatment after a failed diagnostic. Queue state and per-job logs are the authoritative execution records.
