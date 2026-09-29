"""Tests for preprocess/noise_augmentation.py: the decide/apply core used both by
DatasetPreparator's baked path (preprocess/preperator.py) and train-whisper.py's live
mel-domain collator. Noise clips are small synthetic tones generated on the fly so these
tests are self-contained - no dependency on any machine-specific noise directory.
"""

import numpy as np
import pytest
import torch
import torchaudio

from preprocess.noise_augmentation import (
    NoiseAugmenter,
    NoiseLibrary,
    build_noise_waveform,
    mix_audio_at_snr,
    snr_target_rms,
)

SR = 16000


def _write_tone(path, freq, duration_s=1.0, sr=SR, amplitude=0.5):
    t = np.arange(int(duration_s * sr)) / sr
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(tone).unsqueeze(0), sr)
    return tone


def _speech_like(duration_s=3.0, sr=SR, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, int(duration_s * sr)) * 0.3).astype(np.float32)


@pytest.fixture
def noise_dir(tmp_path):
    for i, freq in enumerate([220, 440, 880]):
        _write_tone(tmp_path / f"tone_{i}.wav", freq)
    return tmp_path


class TestNoiseLibrary:
    def test_loads_all_clips(self, noise_dir):
        lib = NoiseLibrary(str(noise_dir), target_sampling_rate=SR)
        assert len(lib) == 3
        assert all(clip.dtype == np.float32 for clip in lib.clips)

    def test_skips_silent_clips(self, noise_dir):
        torchaudio.save(str(noise_dir / "silent.wav"), torch.zeros(1, SR), SR)
        with pytest.warns(UserWarning, match="silent"):
            lib = NoiseLibrary(str(noise_dir), target_sampling_rate=SR)
        assert len(lib) == 3  # silent clip skipped, not counted

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No noise files"):
            NoiseLibrary(str(tmp_path), target_sampling_rate=SR)


class TestDecideAugmentation:
    def test_apply_prob_respected_statistically(self, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=0.3)
        rng = np.random.default_rng(0)
        decisions = [augmenter.decide_augmentation(rng) for _ in range(2000)]
        applied_rate = sum(d is not None for d in decisions) / len(decisions)
        assert abs(applied_rate - 0.3) < 0.05

    def test_apply_prob_zero_never_applies(self, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=0.0)
        rng = np.random.default_rng(0)
        assert all(augmenter.decide_augmentation(rng) is None for _ in range(50))

    def test_apply_prob_one_always_applies(self, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        rng = np.random.default_rng(0)
        assert all(augmenter.decide_augmentation(rng) is not None for _ in range(50))

    def test_num_noises_range_inclusive_bounds(self, noise_dir):
        augmenter = NoiseAugmenter(
            noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0, num_noises_range=(2, 2)
        )
        rng = np.random.default_rng(0)
        for _ in range(20):
            decision = augmenter.decide_augmentation(rng)
            assert len(decision["noise_indices"]) == 2

    def test_snr_within_configured_range(self, noise_dir):
        augmenter = NoiseAugmenter(
            noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0, snr_db_range=(5.0, 10.0)
        )
        rng = np.random.default_rng(0)
        for _ in range(50):
            decision = augmenter.decide_augmentation(rng)
            assert 5.0 <= decision["snr_db"] <= 10.0


class TestApply:
    def test_returns_input_unchanged_when_no_decision(self, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR)
        audio = _speech_like()
        assert np.array_equal(augmenter.apply(audio, None), audio)

    def test_output_shape_and_dtype_preserved(self, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        rng = np.random.default_rng(1)
        audio = _speech_like()
        out = augmenter.apply(audio, augmenter.decide_augmentation(rng))
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype
        assert np.isfinite(out).all()

    def test_achieves_approximately_target_snr(self, noise_dir):
        augmenter = NoiseAugmenter(
            noise_dir=str(noise_dir),
            target_sampling_rate=SR,
            apply_prob=1.0,
            snr_db_range=(10.0, 10.0),  # fixed, for a precise check
            num_noises_range=(1, 1),
            gain_jitter_db=0.0,  # isolate the SNR-mixing itself from gain jitter
            perturb_prob=0.0,
        )
        rng = np.random.default_rng(2)
        audio = _speech_like()
        mixed = augmenter.apply(audio, augmenter.decide_augmentation(rng))

        added_noise = mixed - audio
        achieved_snr = 20 * np.log10(
            np.sqrt(np.mean(audio**2)) / (np.sqrt(np.mean(added_noise**2)) + 1e-12)
        )
        assert achieved_snr == pytest.approx(10.0, abs=1.0)

    def test_multi_clip_layering_does_not_crash_and_stays_finite(self, noise_dir):
        augmenter = NoiseAugmenter(
            noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0, num_noises_range=(2, 3)
        )
        rng = np.random.default_rng(3)
        audio = _speech_like()
        out = augmenter.apply(audio, augmenter.decide_augmentation(rng))
        assert out.shape == audio.shape
        assert np.isfinite(out).all()


def test_build_noise_waveform_matches_requested_length(noise_dir):
    lib = NoiseLibrary(str(noise_dir), target_sampling_rate=SR)
    decision = {
        "noise_indices": [0, 1],
        "gain_jitters_db": [0.0, 0.0],
        "offset_seeds": [1, 2],
        "time_stretch_factors": [1.0, 1.0],
        "pitch_shift_semitones": [0.0, 0.0],
        "coverage_fracs": [0.7, 0.5],
    }
    out = build_noise_waveform(
        decision, lib, burst_len_frac_range=(0.9, 1.0), target_sampling_rate=SR, length=12345
    )
    assert out.shape == (12345,)
    assert out.dtype == np.float32


def test_snr_target_rms_formula():
    assert snr_target_rms(1.0, 0.0) == pytest.approx(1.0)
    assert snr_target_rms(1.0, 20.0) == pytest.approx(0.1)
    assert snr_target_rms(2.0, 20.0) == pytest.approx(0.2)


def test_mix_audio_at_snr_matches_torchaudio_add_noise():
    """mix_audio_at_snr delegates its scaling to torchaudio.functional.add_noise - verify
    the two agree (this is the swap made after confirming they compute the same formula).
    """
    rng = np.random.default_rng(5)
    audio = (rng.uniform(-1, 1, SR) * 0.3).astype(np.float32)
    noise = (rng.uniform(-1, 1, SR) * 0.1).astype(np.float32)

    mixed = mix_audio_at_snr(audio, noise, snr_db=10.0, prevent_clipping=False)
    expected = torchaudio.functional.add_noise(
        torch.from_numpy(audio), torch.from_numpy(noise), torch.tensor(10.0)
    ).numpy()
    np.testing.assert_allclose(mixed, expected, atol=1e-5)


def test_mix_audio_at_snr_near_silent_inputs_are_noop():
    audio = np.zeros(SR, dtype=np.float32)
    noise = np.ones(SR, dtype=np.float32) * 0.5
    assert np.array_equal(mix_audio_at_snr(audio, noise, snr_db=10.0), audio)


def test_mix_audio_at_snr_prevents_clipping():
    audio = np.ones(SR, dtype=np.float32) * 0.9
    noise = np.ones(SR, dtype=np.float32) * 0.9
    mixed = mix_audio_at_snr(audio, noise, snr_db=0.0, prevent_clipping=True)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6
