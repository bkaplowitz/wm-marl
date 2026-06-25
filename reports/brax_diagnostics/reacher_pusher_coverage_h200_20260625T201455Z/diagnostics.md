# Brax JEPA Diagnostics

Generated from JEPA summary files and PPO baseline summaries.

| env | status | JEPA return | PPO best | delta | JEPA improvement | online improvement | accept rate | open-loop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hopper | not_launched |  | 346.988 |  |  |  |  |  |
| inverted_double_pendulum | not_launched |  | 6279.939 |  |  |  |  |  |
| inverted_pendulum | not_launched |  | 1000.000 |  |  |  |  |  |
| pusher | done | -293.164 | -303.370 | 10.206 | 517.102 | 0.028 | 0.667 | 0.031 |
| reacher | done | -38.298 | -28.239 | -10.059 | 493.236 | 0.619 | 1.000 | 0.024 |
| walker2d | not_launched |  | 55.749 |  |  |  |  |  |

## Files

- `summary.csv`: machine-readable aggregate table.
- `policy_diagnostics.csv`: best real-env policy selection checkpoints by phase.
- `summary.json`: same aggregate table in JSON.
- `returns_vs_ppo.png`: JEPA final return against PPO best and last.
- `jepa_improvement.png`: offline+online and online-only JEPA gains.
- `model_vs_policy.png`: final model loss against policy improvement.
- `ppo_learning_curves.png`: PPO evaluation curves from baseline runs.
- `jepa_model_loss_curves.png`: JEPA model loss curves from metrics logs.
- `jepa_policy_return_curves.png`: imagined-return policy curves.
- `jepa_policy_selection_returns.png`: real-env selection returns during policy training.
- `jepa_policy_training_metrics.png`: imagined return, value loss, and action saturation during policy training.
- `jepa_model_head_losses.png`: reward and control-value model losses.
