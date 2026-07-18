# DreamerV3 numerical fixtures

Each `.npz` file is an immutable numerical parity fixture with a sibling
`.manifest.json`. The compact manifest records only the information needed to
interpret and verify the fixture:

- the conformance profile, observation mode, and exact official revision;
- SHA256 digests of the official Git blobs used for that case;
- the fixture SHA256, tensor names, shapes, and dtypes;
- the fixture dtype, seed, canonical request, and explicit generator command.

The stored command is a location-independent descriptor: it uses
`<reference-checkout>` and `<fixture-dir>` placeholders instead of recording a
developer's absolute paths. The closed request carries the exact case, dtype,
seed, profile, mode, revision, source spec, schema version, fixture stem, and
fixture filename.

Loading a committed fixture does not require an official checkout. It verifies
the pinned source specification, NPZ digest, and complete tensor schema. When a
checkout is supplied, validation additionally reads each official blob with
`git show REVISION:PATH` and recomputes its digest. Official Python source is
never executed by the manifest layer.

The source specifications are intentionally small. Distribution fixtures pin
`embodied/jax/{outs,heads,nets}.py`; network fixtures pin
`embodied/jax/{heads,nets}.py` and `dreamerv3/rssm.py`; RSSM fixtures pin those
files plus `dreamerv3/agent.py` and `dreamerv3/configs.yaml`; replay fixtures pin
`dreamerv3/configs.yaml` and the official chunk, replay, selector, and stream
implementations. There is no callback fingerprint, interpreter contract,
Elements shim, or private official-runtime emulation.

`paper` is the default behavioral profile and uses revision
`bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01`. `upstream-current` must be
selected explicitly and uses revision
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. The reference checkout is
read-only.

Refresh metadata for an existing fixture with:

```sh
uv run python -m world_marl.dreamer_v3_baseline.fixture_generator \
  refresh-manifest \
  --profile paper \
  --observation-mode proprio \
  --reference-checkout /private/tmp/danijar-dreamerv3-20260713 \
  --source-revision bfcdfc183d2c1543a3bf3cdda6edb7fae29b6a01 \
  --output-dir tests/fixtures/dreamer_v3 \
  --fixture-stem paper-proprio-rssm
```

`refresh-manifest` reads the existing NPZ and writes only its manifest. Always
compare every NPZ SHA256 before and after a metadata refresh. Numerical fixture
generation is added case-by-case through the decorated parser registry and
must remain a direct, readable translation of the pinned official code.
