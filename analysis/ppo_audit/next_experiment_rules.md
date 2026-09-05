# Decisions after the correction screen

The target is broad, protocol-matched improvement over DMAWM and MATWM. The
six-run 50k screen is a diagnosis on 2s3z plus a 3m regression check, not that
benchmark claim. Keep final128 results separate from curve32 checkpoint scores.

## Evidence required before the next architecture change

- Compare cooperative returns with and without factual critic grounding on both
  training seeds. Keep the original PPO seed1 control and preserved seed0 control.
  If grounding is harmful, retain its implementation as an option and revise the
  maintained default using the actual results.
- Evaluate frozen representations on episode-held-out factual targets. Poor
  return decodability relative to raw observations supports testing the optional
  local return objective. Linear probe success is not proof that representations
  preserve all decision-relevant information; behavior-return labels also have
  policy and partial-observation noise.
- Measure factual teacher-forced and recurrent self-fed predictions from the
  same raw-history roots. Use coupled categorical samples, matched horizon
  support, and the executable unimix distribution. Reward bias and action-mask
  errors should be measured alongside latent errors. The existing small online
  report is a screening signal, not a clean horizon-comparison experiment.
- Compare centralized values with discounted outcomes on fresh stochastic-policy
  episodes at those same states. Imagined-root value means and full-episode score
  means are not paired calibration data.

## Additional migration change to isolate if value drift persists

Commit c86451e also replaced the prior critic EMA (`slowvalue.rate=0.02`) with
a full copy (`rate=1.0`) and removed the distributional slow-value regularizer
(`imag_loss.slowreg=1.0`, also in replay value loss). The current target is frozen
within the five PPO epochs but copied after each learner call. Thus references
to the current critic as an EMA-stabilized target would be inaccurate.

This is a stability hypothesis, not a demonstrated bug. Rate 0.02 can already be
tested with a configuration-only intervention after the initial screen; restoring
the old regularizer is a distinct change and must not be silently bundled with it.
The old REINFORCE objective bootstrapped from the live critic (`slowtar=False`)
while regularizing toward the EMA, so changing the PPO bootstrap rate alone is
not an exact restoration of the old value objective.

## Promotion and expansion

Only promote changes after fixed-budget final evaluations and their relevant
mechanistic diagnostics agree. A two-seed improvement supports a next screen;
it does not settle cross-task or training-seed robustness. Expand to the complete
SMAC subset and matched seed/budget protocol documented in benchmark_targets.md
before making a claim against either baseline. Preserve unsuccessful results.
