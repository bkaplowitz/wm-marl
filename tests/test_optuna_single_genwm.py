"""Search-space flags for the genwm Optuna harness."""

from world_marl.scripts.optuna_single_genwm import parse_args, sample_params


class _RecordingTrial:
    def __init__(self):
        self.categorical_choices = {}

    def suggest_float(self, name, low, high, log=False):
        return low

    def suggest_categorical(self, name, choices):
        self.categorical_choices[name] = list(choices)
        return choices[0]


def test_parse_args_search_space_defaults_and_override():
    args = parse_args(["--task", "brax:reacher"])
    assert args.model_dims == [128, 256]
    assert args.block_sizes == [1, 2, 4]
    assert args.steps_per_blocks == [2, 4, 8]

    restricted = parse_args(
        [
            "--model-dims",
            "128",
            "--block-sizes",
            "1",
            "2",
            "4",
            "--steps-per-blocks",
            "2",
            "4",
        ]
    )
    assert restricted.model_dims == [128]
    assert restricted.block_sizes == [1, 2, 4]
    assert restricted.steps_per_blocks == [2, 4]


def test_sample_params_default_space_unchanged():
    trial = _RecordingTrial()
    params = sample_params(trial, arm="llada2")
    assert trial.categorical_choices["model_dim"] == [128, 256]
    assert trial.categorical_choices["block_size"] == [1, 2, 4]
    assert trial.categorical_choices["steps_per_block"] == [2, 4, 8]
    assert params["model_dim"] == 128


def test_sample_params_uses_restricted_choices():
    trial = _RecordingTrial()
    params = sample_params(
        trial,
        arm="llada2",
        model_dims=[128],
        block_sizes=[1, 2, 4],
        steps_per_blocks=[2, 4],
    )
    assert trial.categorical_choices["model_dim"] == [128]
    assert trial.categorical_choices["block_size"] == [1, 2, 4]
    assert trial.categorical_choices["steps_per_block"] == [2, 4]
    assert params["block_size"] == 1


def test_sample_params_ignores_block_flags_for_non_llada2():
    trial = _RecordingTrial()
    params = sample_params(trial, arm="discrete", model_dims=[128])
    assert "block_size" not in params
    assert "block_size" not in trial.categorical_choices
    assert trial.categorical_choices["model_dim"] == [128]
