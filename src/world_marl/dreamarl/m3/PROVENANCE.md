# Source Provenance

These files are a mechanically imported copy of the registered M3 algorithm:

- `agent.py`, `rssm.py`, `configs.yaml`, and `main.py` originate from
  Dreamer-CDP commit `a851fa3e3d70b624b094ee1810ad4bb602346092`.
- `m3_rssm.py` is the registered JEPA causal-Transformer overlay from this
  repository.
- `agent.py` and `configs.yaml` include only the three integration edits made
  by `world_marl.jepa_transformer.runtime.prepare_runtime`: registering the
  Transformer RSSM, filtering replay entries to the dynamics entry contract,
  and adding the `jepa_transformer` configuration.

The original MIT license is retained in `LICENSE`. These source files are
frozen during the single-agent parity milestone. MARL changes must be made in
separate modules and prove that this path remains unchanged at one agent.
