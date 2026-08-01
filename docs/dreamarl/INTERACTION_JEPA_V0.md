# Interaction JEPA v0 Implementation Record

Date: 2026-08-01

## Locked Intervention

The v0 intervention adds one causal, self-excluded agent-attention block to the
M3 world model. It supplies zero-initialized residual corrections to the
forward stochastic prior, JEPA predictor, and reward/continuation feature. The
posterior, actor, critic, decoder, local temporal Transformer, replay regime,
optimizer, and imagination length are unchanged.

The representation KL uses the local prior while the dynamics KL uses the
joint prior. This prevents privileged joint context from becoming a training
target for the local posterior.

## Verification Results

### Module and environment tests

- The final DreaMARL suite passed all 37 tests, including the dedicated
  interaction, gradient-isolation, parity, and real-adapter tests.
- Ruff passed for all active first-party DreaMARL files and tests. The frozen
  M3 oracle is excluded from style rewrites because its source hashes are part
  of the reduction contract.

Covered properties include:

- safe all-masked softmax;
- exact zero singleton message after projection;
- agent-permutation equivariance;
- environment-trajectory-preserving shuffled context;
- zero-initialized residuals;
- exact singleton input and parameter gradients;
- nonzero mixer gradients when another agent exists;
- joint imagination start ordering and inverse restoration;
- all seven registered MeltingPot adapters reset and step.

### Full learner singleton reduction

On a deterministic fixed replay batch, canonical aligned DreaMARL at `A=1`
matched the frozen M3 oracle for:

- all 321 shared initial parameter and optimizer-state leaves;
- policy actions and recurrent carry;
- policy outputs;
- every comparable algorithmic loss metric;
- training carry and replay updates;
- all 321 shared post-update parameter and optimizer-state leaves.

All singleton interaction metrics were exactly zero. All 24 direct interaction
parameter leaves had exactly zero update.

### Multi-agent execution

On a deterministic `A=2` learner batch:

- interaction active fraction: `1.0`;
- message RMS after the second update: `0.9728804`;
- 6 of 24 direct interaction parameter leaves changed after two updates;
- direct parameter-update L2 norm: `2.741e-4`;
- the full learner compiled and completed both optimizer updates;
- the gradient suite verified that gradients reach mixer parameters after the
  zero-initialized residual begins moving.

### Real MeltingPot smoke

The aligned model compiled and collected 100 real environment steps on
`externality_mushrooms__dense` with five agents. The resolved model contained
174,151,818 trainable parameters. The run exited normally and the first-party
training loop wrote a final checkpoint. This was an engineering smoke test,
not a performance result.

## Decisive Experiment

The scientific comparison remains:

1. independent frozen M3;
2. aligned Interaction JEPA;
3. equal-capacity shuffled Interaction JEPA.

All arms must use the same task, environment seed, shared initialization,
replay and update budget, optimizer, and evaluation protocol. The aligned
model is promoted only if it improves prediction and return over both controls
across multiple seeds.
