# Novelty Matrix

This matrix is a dated research snapshot, not a permanent first-claim. It must
be refreshed before submission. Exact source identifiers are recorded in
`configs/jepatransformer/sources.toml`.

| Method | Reconstruction-free embedding prediction | Stochastic temporal model | Transformer temporal state | Visual MARL | Coherent joint imagination | Online policy learning |
|---|---:|---:|---:|---:|---:|---:|
| DreamerV3 | No | Yes | No | No | No | Yes |
| Dreamer-CDP | Yes | Yes | No | No | No | Yes |
| NE-Dreamer | Yes | Yes | Auxiliary only | No | No | Yes |
| STORM | No | Yes | Yes | No | No | Yes |
| JEDI | Yes | Yes | Diffusion backbone | No | No | Yes |
| MARIE | No | Yes | Yes | No visual result | Aggregated local rollouts | Yes |
| MATWM | No | Yes | Yes | Yes | No, focal-agent rollouts | Yes |
| MMSA | Joint learned embeddings with a VAE | Yes | No | No established visual result | Joint value/imagination model | Yes |
| MIRA | Representation autoencoder | Generative | Yes | Multiplayer video | Jointly conditioned generation | No imagined RL |
| Proposed method | Yes | Yes | Yes | Yes | Yes | Yes |

The candidate contribution is not the intersection alone. The causal claim to
test is whether reconstruction-free, joint-action-conditioned stochastic
imagination preserves control while reducing world-model cost or improving
adaptation to changing teammate policies.
