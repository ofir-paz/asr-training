"""Tests for preprocess/eval_augmentation.py (test-time noise for evaluate_model.py) and the
NoiseLibrary clip_transform it relies on for band-limiting held-out noise."""

from functools import partial

import numpy as np
import pytest
import torch
import torchaudio

from preprocess.augmentation import resample_augment
from preprocess.eval_augmentation import EvalNoiseAugmentation, audio_digest
from preprocess.noise_augmentation import NoiseAugmenter, NoiseLibrary

SR = 16000


def _write(path, samples, sr=SR):
    torchaudio.save(str(path), torch.from_numpy(samples.astype(np.float32)).unsqueeze(0), sr)


def _entry(seed, duration_s=3.0, sr=SR, text="שלום"):
    rng = np.random.default_rng(seed)
    array = (rng.uniform(-1, 1, int(duration_s * sr)) * 0.3).astype(np.float32)
    return {"audio": {"array": array, "sampling_rate": sr}, "text": text}


@pytest.fixture
def noise_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(4):
        _write(tmp_path / f"noise_{i}.wav", rng.normal(0, 0.2, SR))
    return tmp_path


@pytest.fixture
def augmenter(noise_dir):
    return NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)


class TestSeededAugmentation:
    def test_same_seed_and_audio_give_identical_noise(self, augmenter):
        first = EvalNoiseAugmentation(augmenter, seed=7)(_entry(1))
        second = EvalNoiseAugmentation(augmenter, seed=7)(_entry(1))
        np.testing.assert_array_equal(first["audio"]["array"], second["audio"]["array"])
        assert first["augmentation_noise_clips"] == second["augmentation_noise_clips"]

    def test_independent_of_evaluation_order(self, augmenter):
        augmentation = EvalNoiseAugmentation(augmenter, seed=7)
        a_first = augmentation(_entry(1))
        augmentation(_entry(2))
        augmentation(_entry(3))
        a_again = augmentation(_entry(1))
        np.testing.assert_array_equal(a_first["audio"]["array"], a_again["audio"]["array"])

    def test_different_seeds_give_different_noise(self, augmenter):
        a = EvalNoiseAugmentation(augmenter, seed=7)(_entry(1))
        b = EvalNoiseAugmentation(augmenter, seed=8)(_entry(1))
        assert not np.array_equal(a["audio"]["array"], b["audio"]["array"])

    def test_different_examples_get_different_noise(self, augmenter):
        augmentation = EvalNoiseAugmentation(augmenter, seed=7)
        clean_a, clean_b = _entry(1), _entry(1)
        clean_b["audio"]["array"] = clean_b["audio"]["array"] * 0.5
        noise_a = augmentation(clean_a)["audio"]["array"] - clean_a["audio"]["array"]
        noise_b = augmentation(clean_b)["audio"]["array"] - clean_b["audio"]["array"]
        assert not np.allclose(noise_a / np.std(noise_a), noise_b / np.std(noise_b))

    def test_unseeded_draws_fresh_noise(self, augmenter):
        augmentation = EvalNoiseAugmentation(augmenter, seed=None)
        a = augmentation(_entry(1))
        b = augmentation(_entry(1))
        assert not np.array_equal(a["audio"]["array"], b["audio"]["array"])


class TestEntryHandling:
    def test_resamples_to_augmenter_rate_and_keeps_length(self, augmenter):
        entry = _entry(1, duration_s=2.0, sr=44100)
        out = EvalNoiseAugmentation(augmenter, seed=0)(entry)
        assert out["audio"]["sampling_rate"] == SR
        assert out["audio"]["array"].shape[-1] == 2 * SR

    def test_input_entry_is_not_mutated(self, augmenter):
        entry = _entry(1)
        original = entry["audio"]["array"].copy()
        EvalNoiseAugmentation(augmenter, seed=0)(entry)
        np.testing.assert_array_equal(entry["audio"]["array"], original)
        assert "augmentation_snr_db" not in entry

    def test_records_decision(self, augmenter, noise_dir):
        out = EvalNoiseAugmentation(augmenter, seed=0)(_entry(1))
        low, high = augmenter.snr_db_range
        assert low <= out["augmentation_snr_db"] <= high
        names = {p.name for p in noise_dir.iterdir()}
        assert set(out["augmentation_noise_clips"].split(";")) <= names
        assert out["text"] == "שלום"

    def test_skipped_example_is_clean_and_recorded(self, noise_dir):
        never = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=0.0)
        entry = _entry(1)
        out = EvalNoiseAugmentation(never, seed=0)(entry)
        np.testing.assert_array_equal(out["audio"]["array"], entry["audio"]["array"])
        assert out["augmentation_noise_clips"] == ""
        assert np.isnan(out["augmentation_snr_db"])

    def test_rejects_undecoded_audio(self, augmenter):
        with pytest.raises(ValueError, match="decoded audio"):
            EvalNoiseAugmentation(augmenter, seed=0)({"audio": "/some/file.wav", "text": ""})


def test_audio_digest_depends_on_content_and_rate():
    array = np.linspace(-1, 1, 100, dtype=np.float32)
    assert audio_digest(array, SR) == audio_digest(array.astype(np.float64), SR)
    assert audio_digest(array, SR) != audio_digest(array, 8000)
    assert audio_digest(array, SR) != audio_digest(array * 0.5, SR)


class TestClipTransform:
    def test_band_limit_removes_energy_above_new_nyquist(self, noise_dir):
        lib = NoiseLibrary(str(noise_dir), SR, clip_transform=partial(resample_augment, target_hz=8000))
        spectrum = np.abs(np.fft.rfft(lib.clips[0])) ** 2
        freqs = np.fft.rfftfreq(lib.clips[0].shape[-1], 1 / SR)
        # White noise starts with ~half its energy above 4 kHz. The resampler's anti-aliasing
        # filter rolls off just past the new 4 kHz Nyquist, so measure beyond that transition.
        above = spectrum[freqs > 5000].sum() / spectrum.sum()
        assert above < 1e-4

    def test_no_transform_leaves_clips_unchanged(self, noise_dir):
        plain = NoiseLibrary(str(noise_dir), SR)
        identity = NoiseLibrary(str(noise_dir), SR, clip_transform=lambda clip, sr: clip)
        for a, b in zip(plain.clips, identity.clips):
            np.testing.assert_array_equal(a, b)

    def test_file_paths_stay_aligned_with_clips_when_one_is_skipped(self, noise_dir):
        # "noise_1b" sorts between noise_1 and noise_2, so every later index would shift.
        _write(noise_dir / "noise_1b.wav", np.zeros(SR))
        with pytest.warns(UserWarning, match="silent"):
            lib = NoiseLibrary(str(noise_dir), SR)
        assert len(lib.file_paths) == len(lib.clips) == 4
        assert "noise_1b.wav" not in [p.name for p in lib.file_paths]
        for path, clip in zip(lib.file_paths, lib.clips):
            expected, _ = torchaudio.load(str(path))
            np.testing.assert_allclose(clip, expected.squeeze(0).numpy(), atol=1e-6)
