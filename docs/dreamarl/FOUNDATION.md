# DreaMARL Foundation

## Goal

DreaMARL is the working name for a reconstruction-free stochastic visual
multi-agent world model. It will extend the validated M3 causal JEPA
Transformer with coherent joint imagination, decentralized shared policies,
and centralized value learning.

## Locked Data Contract

Every replay sample is time-major and keeps environment and agent axes
explicit:

```text
observations  [time, env, agent, ...]
actions       [time, env, agent, ...]
rewards       [time, env, agent]
team_rewards  [time, env]
agent_alive   [time, env, agent]
is_first      [time, env]
is_last       [time, env]
is_terminal   [time, env]
```

`is_last` cuts temporal context at every reset. `is_terminal` alone disables
value bootstrapping. Agent lifecycle changes use `agent_alive` and do not
silently terminate the joint world.

## First Promotion Sequence

1. Validate the contract on JaxMARL CoinGame and a visual PettingZoo task.
2. Reproduce a maintained CTDE baseline with decentralized action selection.
3. Port the M3 encoder, stochastic state, and causal temporal model per agent.
4. Add one joint-action-conditioned cross-agent block.
5. Advance all agents together in every imagined transition.
6. Compare reconstruction and per-agent JEPA targets under an otherwise fixed
   learner.

No ViT, Mamba, replay prioritization, or task-specific reward logic enters the
first matched DreaMARL implementation.
