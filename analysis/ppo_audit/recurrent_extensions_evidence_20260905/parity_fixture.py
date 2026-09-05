import hashlib
import json
from pathlib import Path
import sys
import time
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from test_majepa_learner_smoke import _synthetic_replay
from test_majepa_self_fed_training import make_tiny

started = time.monotonic()
source, output = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
files = sorted(path for path in (source / 'src/majepa').rglob('*') if path.suffix in ('.py', '.yaml'))
fingerprint = hashlib.sha256()
for path in files:
    fingerprint.update(str(path.relative_to(source)).encode())
    fingerprint.update(path.read_bytes())
learner, observations, actions = make_tiny(enabled=True)
data = _synthetic_replay(learner, observations, actions)
data = dict(data, _environment_step=jnp.full((2, 8), 10, jnp.int32))
carry = learner.init_train(2)
state = nj.init(learner.train)({}, carry, data, seed=988)
result = jax.jit(nj.pure(learner.train))(state, carry, data, seed=989)

def describe(tree):
    records = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(jax.device_get(value))
        if not np.isfinite(array.astype(np.float32)).all():
            raise ValueError('Nonfinite parity tensor')
        records.append({'path': jax.tree_util.keystr(path), 'shape': list(array.shape), 'dtype': str(array.dtype), 'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
    return records

record = {'source': str(source), 'source_sha256': fingerprint.hexdigest(),
          'fixture_sha256': hashlib.sha256((source / 'tests/test_majepa_learner_smoke.py').read_bytes()).hexdigest(),
          'jax_version': jax.__version__, 'initialization': describe(state), 'update': describe(result),
          'ppo_active': float(result[1][2]['ppo/active']), 'wall_seconds': time.monotonic() - started}
output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: record[key] for key in ['source', 'source_sha256', 'ppo_active', 'wall_seconds']}))
