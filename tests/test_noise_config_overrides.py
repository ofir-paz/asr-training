"""Tests for the --noise_<key> CLI override mechanism: preprocess/noise_config.yaml stays
the single source of defaults, and each --noise_<key> flag overrides exactly one key when
explicitly passed, leaving everything else untouched.
"""

import sys

import pytest

from training import parser as parser_module

NOISE_CONFIG_PATH = "preprocess/noise_config.yaml"


def _parse(argv):
    old_argv = sys.argv
    sys.argv = ["train-whisper.py"] + argv
    try:
        return parser_module.parse_arguments()
    finally:
        sys.argv = old_argv


class TestResolveNoiseKwargs:
    def test_no_overrides_matches_yaml_exactly(self):
        args = _parse(["--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH, "--output_model_name", "x"])
        resolved = parser_module._resolve_noise_kwargs(args)
        from_yaml = parser_module._load_noise_config(NOISE_CONFIG_PATH)
        for key, value in from_yaml.items():
            assert resolved[key] == value

    def test_scalar_override_changes_only_that_key(self):
        args = _parse(
            [
                "--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH,
                "--noise_apply_prob", "0.3",
                "--output_model_name", "x",
            ]
        )
        resolved = parser_module._resolve_noise_kwargs(args)
        from_yaml = parser_module._load_noise_config(NOISE_CONFIG_PATH)

        assert resolved["noise_apply_prob"] == 0.3
        assert from_yaml["noise_apply_prob"] != 0.3  # override actually changed something
        for key, value in from_yaml.items():
            if key != "noise_apply_prob":
                assert resolved[key] == value

    def test_range_override_becomes_a_tuple(self):
        args = _parse(
            [
                "--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH,
                "--noise_snr_db_range", "2", "8",
                "--output_model_name", "x",
            ]
        )
        resolved = parser_module._resolve_noise_kwargs(args)
        assert resolved["noise_snr_db_range"] == (2.0, 8.0)
        assert isinstance(resolved["noise_snr_db_range"], tuple)

    def test_boolean_override_can_flip_yaml_false_to_true(self):
        args = _parse(
            [
                "--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH,
                "--noise_simulate_radio_channel",
                "--output_model_name", "x",
            ]
        )
        resolved = parser_module._resolve_noise_kwargs(args)
        from_yaml = parser_module._load_noise_config(NOISE_CONFIG_PATH)
        assert from_yaml["noise_simulate_radio_channel"] is False  # shipped default
        assert resolved["noise_simulate_radio_channel"] is True

    def test_noise_dir_override_uses_same_mechanism(self):
        args = _parse(
            [
                "--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH,
                "--noise_dir", "/some/other/dir",
                "--output_model_name", "x",
            ]
        )
        resolved = parser_module._resolve_noise_kwargs(args)
        assert resolved["noise_dir"] == "/some/other/dir"

    def test_augmentation_off_returns_empty_dict(self):
        args = _parse(["--output_model_name", "x"])
        assert parser_module._resolve_noise_kwargs(args) == {}

    def test_missing_noise_config_raises(self):
        args = _parse(["--noise_augmentation", "--output_model_name", "x"])
        with pytest.raises(ValueError, match="requires --noise_config"):
            parser_module._resolve_noise_kwargs(args)

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--noise_apply_prob", ["0.3"]),
            ("--noise_dir", ["/x"]),
            ("--noise_simulate_radio_channel", []),
        ],
    )
    def test_override_without_augmentation_raises(self, flag, value):
        args = _parse([flag, *value, "--output_model_name", "x"])
        with pytest.raises(ValueError, match="require --noise_augmentation"):
            parser_module._resolve_noise_kwargs(args)

    def test_every_resolved_key_is_a_real_datasetpreparator_param(self):
        """The whole point of this mechanism: nothing resolved here should be silently
        dropped when handed to DatasetPreparator via **noise_kwargs."""
        import inspect

        from preprocess.preperator import DatasetPreparator

        args = _parse(
            [
                "--noise_augmentation", "--noise_config", NOISE_CONFIG_PATH,
                "--noise_apply_prob", "0.3", "--noise_snr_db_range", "2", "8",
                "--output_model_name", "x",
            ]
        )
        resolved = parser_module._resolve_noise_kwargs(args)
        accepted = set(inspect.signature(DatasetPreparator.__init__).parameters)
        unknown = [k for k in resolved if k not in accepted]
        assert not unknown, f"keys not accepted by DatasetPreparator: {unknown}"
