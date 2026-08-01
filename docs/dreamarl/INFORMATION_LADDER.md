# DreaMARL One-Step Information Ladder

This is an internal algorithm-development experiment. It is not a proposed
MARL mechanism and is not the intended paper contribution.

## Question

Which additional information, if any, is required to predict an agent's next
stopped representation beyond its truthful local belief and action?

The causal transition contract is:

```text
(belief_t, joint_action_t) -> stopped_target_t_plus_1
```

Replay windows never cross episode boundaries. Complete trajectories are split
between training, validation, and test, stratified by policy-checkpoint label
when several labelled datasets are combined.

## Equal-Capacity Rungs

All active rungs use the same trainable architecture and parameter count.

1. `X0`: focal belief and focal action.
2. `X1`: focal belief and the complete joint action.
3. `X2`: all local beliefs and the complete joint action.
4. `X3`: all local observations and the complete joint action, when stored.
5. `X4`: privileged simulator state and the complete joint action, when exposed.

Each expanded rung is evaluated twice:

- as a complete predictor trained from the same initialization;
- as a zero-output residual over a frozen `X0` predictor.

The residual result prevents a larger information set from appearing useful by
merely relearning focal self dynamics.

## Promotion

An information expansion is promoted only if:

1. its held-out full predictor improves over the preceding rung;
2. its frozen-local residual improves over `X0`;
3. the trajectory-bootstrap lower 95% bound is positive;
4. the result replicates across predictor seeds;
5. the gain occurs on transitions relevant to control failure;
6. a learned architectural approximation later improves recursive imagination
   and decentralized return.

No multi-step or end-to-end architectural run is launched before this `h=1`
gate identifies a useful information expansion.
