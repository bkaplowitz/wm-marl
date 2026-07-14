from __future__ import annotations


REPLAY_COMMAND_DESCRIPTOR = (
    "python:current",
    "module:world_marl.dreamer_v3_baseline.replay_oracle",
    "_worker",
)

REPLAY_GENERATOR_FILE_HASHES = {
    "world_marl/dreamer_v3_baseline/config.py": (
        "c2cc207718776bc975f5740d92f8e79c7c3ae103940b199a6abf80b5a21d5615"
    ),
    "world_marl/dreamer_v3_baseline/oracle.py": (
        "dd8a5433119bf73a20bdd646a87d72f842e019b8a16f3f2a719564d984db16d9"
    ),
    "world_marl/dreamer_v3_baseline/replay_oracle.py": (
        "c24a3a09342f16e02c573c37a789193342b8abba8cf4d2e0b467ca1398c42dfb"
    ),
}

REPLAY_RUNTIME_CONTRACT = {
    "elements_helper_hashes": {
        "elements/checkpoint.py": (
            "80f2fe99141d5bd1c96d9c5f32502cfff4fb5e56571595a092c6aace12f42e92"
        ),
        "elements/rwlock.py": (
            "020866b2d6d1216c3d9e8019d641bd5f073ea90d2cf5646361a722b0e156b9be"
        ),
        "elements/uuid.py": (
            "e8829a81c80058e4f130a5821acd9db62de5f72fbb02ad4b10262ccfa3006d6a"
        ),
    },
    "elements_mode": "debug",
    "elements_version": "3.22.0",
    "numpy_version": "1.26.4",
    "python_implementation": "CPython",
    "python_version": "3.11.5",
    "shim_hashes": {
        "Limiters": (
            "fda05288723184af191b75386f6c25cc6ad2109d71348e50be1a242c18e8988a"
        ),
        "RWLock": ("415f94a2d993dd6178e7777452a753bcf6dc83f0e12cb77213d042132d6c9526"),
        "Section": ("bc25e90f9684983cad521d8158f04dda96a9b20a6d66665864a44842fbff30ba"),
        "Timer": ("89203b909ab4d538bf859b592854f13b0dc2030413a3faed72fc32641e4812ef"),
        "UUID": ("10221ae86c5f1aed096c6decb6b5c85098bbd86e9082512bb54dc713d6558cd6"),
        "timestamp": (
            "381bcd691643c9aef4ada2aea17ed3a47c615977c79bf00cf064adf8312675ce"
        ),
    },
    "worker_mode": "isolated-ast-exec",
}


__all__ = [
    "REPLAY_COMMAND_DESCRIPTOR",
    "REPLAY_GENERATOR_FILE_HASHES",
    "REPLAY_RUNTIME_CONTRACT",
]
