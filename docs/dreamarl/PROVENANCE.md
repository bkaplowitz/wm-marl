# DreaMARL Provenance

DreaMARL is implemented as first-party source in `world_marl.dreamarl`. The
pinned Dreamer-CDP checkout supplies the Embodied runtime only; it does not
supply the DreaMARL agent, world model, policy, value model, replay contract, or
training loop.

The project began from Dreamer-CDP revision
`a851fa3e3d70b624b094ee1810ad4bb602346092` and the repository's causal JEPA
Transformer work. Every launch verifies that infrastructure revision and
records a fingerprint of all executable DreaMARL source files.

The final local-memory/context implementation preceding the joint-world
rewrite is preserved in Git at tag
`dreamarl-local-memory-context-reference-20260803`. Rejected architectures and
their one-off diagnostics are intentionally absent from the maintained source
tree.

The Dreamer-derived foundation remains subject to the license at
`src/world_marl/dreamarl/LICENSE`.
