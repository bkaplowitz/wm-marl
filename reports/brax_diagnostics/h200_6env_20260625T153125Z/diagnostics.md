# Brax JEPA Diagnostics

Generated from JEPA summary files and PPO baseline summaries.

| env | status | JEPA return | PPO best | delta | JEPA improvement | online improvement | accept rate | open-loop |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hopper | not_launched |  | 346.988 |  |  |  |  |  |
| inverted_double_pendulum | done | 317.513 | 6279.939 | -5962.426 | 21.932 | -0.000 | 0.000 | 0.047 |
| inverted_double_pendulum.nohup | not_launched |  |  |  |  |  |  |  |
| inverted_pendulum | crashed |  | 1000.000 |  |  |  |  |  |
| inverted_pendulum.nohup | not_launched |  |  |  |  |  |  |  |
| pusher | done | -397.069 | -303.370 | -93.700 | 242.308 | 0.061 | 1.000 | 0.071 |
| pusher.nohup | not_launched |  |  |  |  |  |  |  |
| reacher | done | -39.593 | -28.239 | -11.354 | 638.734 | 0.000 | 1.000 | 0.029 |
| reacher.nohup | not_launched |  |  |  |  |  |  |  |
| walker2d | not_launched |  | 55.749 |  |  |  |  |  |

## Files

- `summary.csv`: machine-readable aggregate table.
- `summary.json`: same aggregate table in JSON.
- `returns_vs_ppo.png`: JEPA final return against PPO best and last.
- `jepa_improvement.png`: offline+online and online-only JEPA gains.
- `model_vs_policy.png`: final model loss against policy improvement.
- `ppo_learning_curves.png`: PPO evaluation curves from baseline runs.
- `jepa_model_loss_curves.png`: JEPA model loss curves from metrics logs.
- `jepa_policy_return_curves.png`: imagined-return policy curves.
