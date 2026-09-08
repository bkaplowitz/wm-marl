# Factual-value retest on the sampled-availability foundation

User authorized this bounded retest on all four existing L40 GPUs on 8 September
2026. Two maps (2s3z and 3s_vs_4z), two treatments, seeds 0 and 1: eight new
50k-transition runs. No new pod or additional control training is required.

Reference runs are the completed am1-bernoulli-MAP-sSEED trials, with 100-episode
greedy final evaluations and exactly 5,626 learner updates. The new executable
learner package must have the identical source fingerprint. Full resolved train
and evaluation configurations may differ only in the declared factual-value
flags. This is a same-source/runtime comparison using separately completed
references, not simultaneous reruns or a checkpoint continuation.

- fact: factual successor-state target-critic values, real rewards and boundaries,
  joint V-trace correction using logged actual collection-mixture probabilities;
  rho and c caps 1. Existing replay critic weight stays 0.3.
- factrep: the same treatment plus auxiliary real-value supervision of the
  encoder/history representation, scale 0.1. Standard PPO/central-critic gradient
  boundaries and the JEPA/alignment objectives remain intact.

All retain one environment, 5k replay prefill inside the 50k budget, BPTT2,
eight anchors, trajectory alignment 0.1, half-uniform world replay, Bernoulli
imagined availability, H5 imagination, full-copy critic target, encoder EMA 0.01,
world LR 4e-5, actor/critic LR 3e-5, five PPO epochs and entropy 0.01. No imported
diagnostic trajectory or checkpoint enters training.

GPU 0: fact 2s3z; GPU 1: factrep 2s3z; GPU 2: fact 3s_vs_4z; GPU 3: factrep
3s_vs_4z. Seed 0 first, then seed 1 on each GPU. Each finishes final100 before
the existing frozen real-action ranking diagnostic. Sampled-policy final100 is
omitted in this retest because the intervention targets value learning and the
previous diagnostic did not show a general benefit from sampled execution.

The new launch is bounded to eight jobs and 400k new training transitions.
Workers stop claiming new jobs ten hours after initialization; the review window
is fourteen hours. This is the newly requested retest, not an automatic extension
of the original 48-hour experiment. Failed jobs preserve logs and stop their
worker without automatic training retries. Preserve latest full checkpoints and
exact compressed 5k model weights; compact superseded optimizer state using the
existing retention code. No pod shutdown/deletion is part of this request.

Inspect final wins per seed, factual trace effective sample size/clipping, auxiliary
loss activation, finite updates, and factual-versus-imagined action-value errors.
Do not substitute training-curve peaks or diagnostic wins for final100 scores.

DMAWM implements factual posterior value targets and value gradients into its
world model. We have not verified an ablation attributing a particular win-rate
gain to either component. This retest is not a byte-for-byte DMAWM replication:
its cumulative joint importance weights capped at 4 differ from our V-trace,
and its critic feature-gradient path differs from our separate auxiliary head.
Reference: https://github.com/DiXue98/DMAWM/blob/2acaaeb82805b55d275e3ce08ac8f713ec9afbb2/world_model.py#L408-L493

Launcher: scripts/run_value_grounding_retest.py
Queue: /workspace/majepa_value_grounding_retest_20260908/queue.json
W&B: https://wandb.ai/osaze-obahor/majepa-ppo-treatments/groups/ma-jepa-value-grounding-retest-20260908
