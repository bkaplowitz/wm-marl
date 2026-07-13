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
`dreamerv3/configs.yaml`; later distribution, network, RSSM, loss, optimizer,
replay, and train-step cases register the exact source modules they execute.

The `paper` profile uses source revision
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01` with in-memory overrides for
stride convolutions, LaProp beta2 `0.99`, and the `1,000,000`-transition DMC
budget. The `upstream-current` profile uses revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` with no overrides. The official
checkout is never edited.

Ordinary tests load and validate committed fixtures. They do not launch the
official wrapper. Fixture regeneration is an explicit developer operation and
must provide `DREAMERV3_ORACLE_CHECKOUT`.
