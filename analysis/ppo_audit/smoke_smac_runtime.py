"""Read-only random-policy gate for a replacement SMAC runtime."""

import json
import numpy as np

from majepa.envs.smac import SMACEnv


def main():
    rng = np.random.default_rng(734)
    results = []
    for name in ("3m", "2s3z"):
        env = SMACEnv(name, seed=734)
        try:
            obs = env.step({"reset": True, "action": np.zeros(env.num_agents, int)})
            total = 0.0
            for step in range(200):
                mask = obs["action_mask"]
                action = np.array([rng.choice(np.flatnonzero(row)) for row in mask])
                obs = env.step({"reset": False, "action": action})
                assert np.isfinite(obs["observation"]).all()
                assert np.all(obs["reward"] == obs["reward"][0])
                total += float(obs["reward"][0])
                if obs["is_last"]:
                    break
            assert obs["is_last"], "Random episode did not finish within map limit"
            results.append({"map": name, "steps": step + 1, "return": total})
        finally:
            env.close()
    print("SMAC_RUNTIME_OK " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
