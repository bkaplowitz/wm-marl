# One-environment localmask simplification

The pre-cleanup deployed package is preserved in commit 4d93ec2. No running
experiment uses the changing worktree. All jobs use a frozen, hashed pod copy.

## Reference

Use config profiles `smac_vector ma_jepa localmask_reference`. Set map, number
of agents, seed, step budget and output directory explicitly. The run queue
resolves and records the complete training and final-evaluation configurations.

1 environment; WM4096c64; actor 3x512; local/joint LR 1e-4; actor/critic LR 3e-5;
PPO clip .2, entropy .003; H5; BPTT2; dense alignment .05; direct cosine 2 and
margin .1; independent mixed replay views (.5 uniform, .5 recency .9998);
start4 replay snapshots; 5k prefill; ratio128.

## Cleanup boundary

Removed teammate belief/adapters/planning classes and their unreachable learner
branches, factual successor-value and representation-value treatments, direct
latent-output head, and the obsolete single-stream recent replay. Archived
flags are accepted only when disabled, for compatibility with saved launch
specifications; enabling removed treatments fails explicitly. Removed treatment
profiles. Image support and the retained JEPA objectives remain available.
Diagnostics and evaluation remain separate from the experiment driver; useful
training/report metrics are retained so comparisons do not lose observability.

No live RNG expressions, dual-view selectors, or snapshot timing were changed.
An exact numerical comparison checks initial state, a complete world-model and
PPO update, and training metrics against the preserved implementation. This is
an integration gate, not a claim to have reproduced a full SMAC training run.

## Independent ablations (not cumulative)

- `nojointmask`: remove the joint availability module, its one-step objective,
  and its self-fed objective. PPO continues to use local availability. Self-fed
  mask diagnostics use local predictions, but add no local imagined-mask loss.
- `nolocaloutcomes`: remove local reward/continuation modules and their factual
  auxiliary losses. Joint reward/continuation and local availability stay intact.
- `heads256`: halve all three local prediction-head widths from 512 to 256.
  Other dimensions, objectives and optimizer settings remain the reference.

Each arm: seeds 0/1/2 on 2s3z (50k), 8m (50k), and 3s_vs_4z (100k); final100,
curve32 every5k. Fresh starts only. 27 runs, ordered by arm then seed then map.
Use completed localmask reference scores; no duplicate reference training.
Module deletion changes parameter initialization structure; these are paired
seed architecture tests, not a claim of identical remaining random draws.

## Validation and operation

`smoke.py` runs a small full JIT learner with deaths, resets, separate replay
views, BPTT2, dense alignment, PPO and report paths. It asserts finite outputs,
legal stored PPO actions and absence of removed module parameters.
`test_replay_snapshots.py` checks real replay/stream deterministic ordering.
`run_queue.py` requires a matching validation manifest, fingerprints source,
asserts resolved flags, waits for occupied GPUs, checks storage before each job,
checks learner budgets, and retains final checkpoints while removing completed
intermediate checkpoints. It uses the existing pod runtime/queue engine.

Evaluate final100 seed means and individual seed regressions across all three
maps. Do not automatically combine winning treatments or replace the reference.
