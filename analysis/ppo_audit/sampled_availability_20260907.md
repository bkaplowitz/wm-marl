# Sampled imagined action availability

User authorized this replacement for the eight-environment experiment on
7 September 2026. Stop the remaining eight-env jobs, retain their checkpoints
and logs, and defer their two pending successors. No additional pod is used.

The only learner treatment is
`agent.marl.ctde.imagination_mask_sampling: bernoulli` versus `threshold`.
The same CTDE mask-head probabilities feed independent Bernoulli availability
events instead of a 0.5 threshold. Root masks remain factual, liveness remains
hard and absorbing, and the existing empty-mask no-op fallback remains in place.
The extra random stream is derived from the actor draw key with a fixed fold-in
salt, preserving the existing learner RNG consumption. Real action selection,
recorded-action world supervision, BPTT2 and alignment losses are unchanged.
Every realized mask is retained in the immutable PPO batch for all epochs.

DMAWM reference: pinned commit 2acaaeb82805b55d275e3ce08ac8f713ec9afbb2,
`world_model.py` availability `.sample()` and `utils/output_head.py` Binary
`torch.bernoulli`. This matches that mechanism, not the entire DMAWM algorithm.
The existing mask-calibration option additionally switches prediction heads and
is incompatible with the present recurrence; it is deliberately not enabled.

Four runs: 2s3z seeds 0 and 1, each threshold and Bernoulli. GPUs 0/1 receive
seed 0 control/treatment; GPUs 2/3 receive seed 1 control/treatment. All retain
one collection environment, 50k total transitions including 5k prefill, 5626
learner updates, 50% uniform world replay, trajectory alignment 0.1, BPTT2,
full-copy critic target, encoder EMA 0.01 and the established learning rates.
Curve evaluations use 32 greedy episodes; final benchmark uses 100 greedy
episodes. Sampled-policy final100 and eight-root real action-ranking diagnostics
follow each run and remain separately labeled. Ranking diagnostics now honor
the configured mask sampling, and report real-policy concentration on active
decisions with more than one legal action.

Before GPU 3 starts training it compares 100 greedy versus 100 sampled-policy
episodes on existing one-env 3s_vs_4z seed0 and MMM seed0 final checkpoints.
Evaluation seeds/quotas match within each comparison. Checkpoint file hashes
must remain unchanged. No diagnostic data is imported into training replay.

All complete configurations are compared within seed: only the declared
availability flag may differ. Checkpoint histories are retained for both arms.
The original claim cutoff (2026-09-08 07:42:23 UTC) and decision deadline
(2026-09-08 10:42:23 UTC) remain in effect. This is a two-seed mechanism screen;
it cannot establish general cross-seed stability or baseline competitiveness.

Queue: `/workspace/majepa_sampled_availability_20260907/queue.json`.
W&B: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-sampled-availability-20260907

Readout: paired final win rates, held-out self-fed mask/liveness errors, real
action-ranking accuracy, sampled-versus-greedy difference, and policy
concentration. Mask calibration and imagined illegal behavior can worsen under
sampling; do not promote an arm based solely on greater exploration or one seed.
