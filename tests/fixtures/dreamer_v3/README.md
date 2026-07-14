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

Replay fixtures execute the pinned official `Chunk`, `Replay`, `Uniform`, and
`Consec` class definitions in an isolated worker. Their source spec covers
`embodied/core/chunk.py`, `replay.py`, `selectors.py`, `streams.py`, and
`dreamerv3/configs.yaml`. The recorded request fixes the primary online case,
the capacity/swap-pop case, the complete row schema, NumPy 1.26.4, Elements
3.22.0 helper hashes, debug UUID mode, and deterministic shim hashes. The
fixtures preserve exact valid starts, online FIFO order, PCG64 state and draws,
cross-chunk annotations, consecutive overlap, capacity references, and latent
writeback tensors.

Replay manifests are location-independent recipes. Their command is the
literal descriptor `python:current`,
`module:world_marl.dreamer_v3_baseline.replay_oracle`, `_worker`; their request
contains no checkout, interpreter, Elements package, or distribution path. The
request instead pins CPython/NumPy/Elements semantics and raw SHA256 digests for
`replay_oracle.py`, `oracle.py`, and `config.py`. It also pins
`replay_oracle_contract.py` with a normalized raw-byte self-hash that replaces
exactly its one dedicated 64-hex self-digest literal with 64 ASCII zeros; every
other byte, including comments, remains covered. Ordinary manifest loading compares only persisted data with the
frozen contract and therefore performs no live source inspection.

Executing a replay recipe requires `resolve_generator_invocation()`. Resolution
rechecks all four live generator files, deterministic shims, the exact absolute
current interpreter spelling, NumPy, and pinned Elements helper files, then creates an unsaved
one-shot envelope containing the current absolute checkout, interpreter,
Elements package, and Elements distribution coordinates. The isolated worker
requires that exact envelope and repeats the live checks before reading official
Git objects. A copied source tree therefore resolves its copied worker without
changing the manifest, while any byte change to the generator closure fails
before execution.

Elements locations are never inferred from a machine-specific default. Every
resolution, direct worker test, and regeneration command must provide absolute
coordinates. Tests read the following environment variables, with local test
defaults only when those variables are unset:

```sh
DREAMERV3_ORACLE_CHECKOUT=/absolute/path/to/dreamerv3 \
DREAMERV3_ELEMENTS_PACKAGE_DIR=/absolute/path/to/site-packages/elements \
DREAMERV3_ELEMENTS_DIST_INFO=/absolute/path/to/site-packages/elements-3.22.0.dist-info \
uv run pytest tests/test_dreamer_v3_replay.py -k 'fixture_regeneration or recorded_replay_worker'
```

This provenance seal is a continuing handoff requirement. Every later task
that changes `config.py`, `oracle.py`, `replay_oracle.py`, or
`replay_oracle_contract.py` must recompute the affected frozen contracts,
regenerate both replay manifests through `OracleHarness`, prove the two replay
NPZ files remain byte-identical unless their arrays were intentionally changed,
and rerun the complete 265-test oracle-manifest plus replay gate in that same
task. Committing a closure or contract edit with stale manifests is invalid.

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
The canonical `networks` pairs execute the pinned source with bfloat16 compute;
separate `networks-float32` pairs execute the explicit float32 parity mode. The
network source spec alone authorizes those two execution dtypes; every other
source spec remains canonical-dtype-only. Each request, worker response,
manifest, and `execution.compute_dtype` tensor agree on the executed dtype.
Because NumPy serializes `ml_dtypes.bfloat16` as an opaque `void16`, bfloat16
numeric tensors are losslessly stored as their float32 value views in NPZ while
the replayed worker response retains and tests their true bfloat16 dtype.

Network head cases cover scalar and vector binary, categorical, one-hot, MSE,
symlog-MSE, symexp-twohot, and bounded-normal families, including invalid
family/space pairings, nonuniform discrete classes, and entropy metadata. The
decoder cases include nonlexicographic two-image declaration order with unequal
channel counts and per-key prediction and loss tensors.
Dictionary selection cases execute Encoder with the full Agent observation
mapping, including metadata, and Decoder with the full RSSM feature mapping,
including `logit`; they also record required-only outputs and missing-key
failures to prove the official modules ignore unrelated entries.

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
