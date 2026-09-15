from argparse import Namespace

from training.run_naming import dedupe_name, generate_run_name


def _args(**overrides):
    base = dict(
        model_name="ivrit-ai/whisper-large-v3",
        train_datasets=None,
        use_preprocessed=None,
        noise_augmentation=False,
        resample_augmentation=False,
        use_qlora=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        kd_weight=0.0,
        label_smoothing=0.0,
        mixed_precision=None,
    )
    base.update(overrides)
    return Namespace(**base)


class TestGenerateRunName:
    def test_matches_known_naming_style(self):
        args = _args(
            train_datasets=["a:train", "b:train"],
            noise_augmentation=True,
            kd_weight=0.1,
            label_smoothing=0.1,
            mixed_precision="bf16",
        )
        assert generate_run_name(args) == "lv3-mix-na-b32-kd0.1-sm0.1-bf16"

    def test_model_shortcodes(self):
        assert generate_run_name(_args(model_name="ivrit-ai/whisper-large-v2")).startswith("lv2")
        assert generate_run_name(_args(model_name="ivrit-ai/whisper-large-v1")).startswith("lv1")
        assert generate_run_name(_args(model_name="ivrit-ai/whisper-medium")).startswith("med")
        assert generate_run_name(_args(model_name="ivrit-ai/whisper-small")).startswith("sm")
        assert generate_run_name(_args(model_name="ivrit-ai/whisper-tiny")).startswith("tiny")

    def test_unknown_model_falls_back_to_slug(self):
        name = generate_run_name(_args(model_name="myorg/custom-ckpt-v9"))
        assert name.startswith("customckptv9")

    def test_single_dataset_uses_its_own_name_not_mix(self):
        name = generate_run_name(_args(train_datasets=["ivrit-ai/crowd-transcribe-v5:train"]))
        assert "mix" not in name.split("-")
        assert "crowdtrans" in name  # slugged, then truncated to 10 chars

    def test_multiple_datasets_use_mix(self):
        name = generate_run_name(_args(train_datasets=["a:train", "b:train", "c:train"]))
        assert "mix" in name.split("-")

    def test_flags_only_appear_when_enabled(self):
        off = generate_run_name(_args())
        assert not any(t in ("na", "rs", "qlora") for t in off.split("-"))

        on = generate_run_name(_args(noise_augmentation=True, resample_augmentation=True, use_qlora=True))
        tokens = on.split("-")
        assert "na" in tokens and "rs" in tokens and "qlora" in tokens

    def test_kd_and_smoothing_omitted_when_zero(self):
        name = generate_run_name(_args())
        assert not any(t.startswith("kd") for t in name.split("-"))
        assert not any(t.startswith("sm") for t in name.split("-"))

    def test_effective_batch_size_multiplies_grad_accum(self):
        name = generate_run_name(
            _args(per_device_train_batch_size=8, gradient_accumulation_steps=4)
        )
        assert "b32" in name.split("-")


class TestDedupeName:
    def test_returns_original_when_free(self):
        assert dedupe_name("foo", lambda n: False) == "foo"

    def test_first_collision_appends_2(self):
        assert dedupe_name("foo", lambda n: n == "foo") == "foo_2"

    def test_skips_past_existing_numbered_versions(self):
        existing = {"foo", "foo_2", "foo_3"}
        assert dedupe_name("foo", lambda n: n in existing) == "foo_4"

    def test_unrelated_names_untouched(self):
        assert dedupe_name("bar", lambda n: n == "foo") == "bar"
