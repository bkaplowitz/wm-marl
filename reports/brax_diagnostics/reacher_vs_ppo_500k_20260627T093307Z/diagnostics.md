# Brax JEPA Diagnostics

Generated from JEPA summary files and PPO baseline summaries.

| env | status | JEPA return | PPO best | delta | JEPA improvement | online improvement | accept rate | open-loop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pusher_seed0 | done | -274.472 |  |  | 392.710 | -0.021 | 1.000 | 0.011 |
| pusher_seed1 | not_launched |  |  |  |  |  |  |  |
| reacher_seed0 | done | -28.856 |  |  | 499.987 | 0.000 | 0.833 | 0.016 |
| reacher_seed1 | done | -36.552 |  |  | 468.924 | 0.000 | 0.667 | 0.016 |

## Files

- `summary.csv`: machine-readable aggregate table.
- `policy_diagnostics.csv`: best real-env policy selection checkpoints by phase.
- `sample_efficiency.csv`: DreamerV3-style return checkpoints against real training replay steps for the selected environment.
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
- `<env>_return_vs_train_steps_500k.png`: DreamerV3-style sample-efficiency curve bounded to 500k real training replay steps by default. JEPA points use actor-replay collection returns, not policy-selection or validation returns. PPO points use in-window evals when present, with a dashed full-run best reference.
