# DreamerV3 oracle fixtures

Fixtures in this directory are generated only through `OracleHarness` from an
explicit checkout of Danijar Hafner's DreamerV3 source. Each deterministic NPZ
has a sibling manifest recording the named case, source revision and file
hashes, native profile hash, complete declarative override map, JAX execution
metadata, tensor schema, seed, exact generator command, and canonical JSON stdin
request when the generator uses one. Volatile process metadata such as worker
PIDs is verified out of band and is never written to fixtures.

Each manifest names a registered source spec. A source spec pins the exact
official path-to-SHA256 set independently for both authority revisions, so
fixtures can validate source provenance offline. The `config` source spec covers
`dreamerv3/configs.yaml`. The `distributions` source spec covers the exact
`embodied/jax/outs.py`, `embodied/jax/heads.py`, and `embodied/jax/nets.py`
objects that produce output behavior, bounded-normal construction, scalar
transforms, and two-hot bins. The `networks` source spec covers the exact
`embodied/jax/nets.py`, `embodied/jax/heads.py`, and `dreamerv3/rssm.py` objects
that implement network primitives, encoders, decoders, heads, and BlockGRU.
Later RSSM, loss, optimizer, replay, and train-step cases register the exact
source modules they execute.

Distribution fixtures persist fixed Gumbel tensors together with the exact
official categorical indices, hard one-hot samples, and one-hot
straight-through gradients they produce. Fixture generation and native parity
tests call the public `sample(seed, shape)` methods while temporarily replacing
only the categorical random primitive; every other random primitive delegates
to JAX, including the separately validated Bernoulli compatibility path.

Network fixtures include deterministic initializer, normalization, linear,
block-linear, convolution, transpose-convolution, MLP, BlockGRU, dictionary
encoder/decoder, and head cases. Parameter tensors are stored alongside outputs
under canonical source paths so parity tests must establish a complete bijection
between native and official parameter trees before comparing computations.

The `paper` profile uses source revision
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01` with in-memory overrides for
stride convolutions, LaProp beta2 `0.99`, and the `1,000,000`-transition DMC
budget. The `upstream-current` profile uses revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` with no overrides. The official
checkout is never edited.

Ordinary parity tests load and validate committed fixtures. Provenance tests may
replay the recorded isolated fixture worker, but they never launch the official
training wrapper. Fixture regeneration is an explicit developer operation and
must provide `DREAMERV3_ORACLE_CHECKOUT`.
