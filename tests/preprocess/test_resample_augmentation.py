"""Tests for preprocess/augmentation.py's resample augmentation: the baked, waveform-domain
round trip (resample_augment, used by DatasetPreparator) and its live, mel-domain
approximation (resample_mel_cutoff_bin / apply_resample_mel, used by train-whisper.py's
collator when --use_preprocessed leaves no raw waveform to actually resample).
"""

import numpy as np
import torch

from preprocess.augmentation import apply_resample_mel, resample_augment, resample_mel_cutoff_bin

SR = 16000


def _tone(freq, duration_s=1.0, sr=SR):
    t = np.arange(int(duration_s * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x**2)))


class TestResampleAugmentWaveform:
    def test_output_shape_and_dtype_preserved(self):
        audio = _tone(300)
        out = resample_augment(audio, SR, target_hz=8000)
        assert out.shape == audio.shape
        assert out.dtype == audio.dtype

    def test_noop_when_target_at_or_above_sample_rate(self):
        audio = _tone(300)
        assert np.array_equal(resample_augment(audio, SR, target_hz=SR), audio)
        assert np.array_equal(resample_augment(audio, SR, target_hz=SR * 2), audio)

    def test_suppresses_high_frequency_content_relative_to_in_band(self):
        # A tone above the downsample target's Nyquist should lose most of its energy in
        # the round trip; a tone well within the band should survive largely intact.
        target_hz = 8000
        high_tone = _tone(7000)  # above 8kHz's 4kHz Nyquist
        low_tone = _tone(500)  # well within band

        high_out = resample_augment(high_tone, SR, target_hz=target_hz)
        low_out = resample_augment(low_tone, SR, target_hz=target_hz)

        assert _rms(high_out) < 0.3 * _rms(high_tone)
        assert _rms(low_out) > 0.8 * _rms(low_tone)


class TestResampleMelCutoffBin:
    def test_matches_known_whisper_alignment(self):
        # Cross-checked against WhisperFeatureExtractor's actual mel_filters matrix during
        # development: n_mels=80, sr=16000, target_hz=8000 -> cutoff bin 62.
        assert resample_mel_cutoff_bin(80, SR, 8000) == 62

    def test_no_cutoff_when_target_at_or_above_sample_rate(self):
        assert resample_mel_cutoff_bin(80, SR, SR) == 80

    def test_lower_target_cuts_more_bins(self):
        # A lower target (harsher degradation) should cut more bins, i.e. a smaller cutoff
        # index, not fewer.
        cutoff_harsh = resample_mel_cutoff_bin(80, SR, 4000)
        cutoff_mild = resample_mel_cutoff_bin(80, SR, 12000)
        assert cutoff_harsh < cutoff_mild


class TestApplyResampleMel:
    def test_zeroes_bins_above_cutoff_only(self):
        mel_power = torch.full((80, 50), 3.0)
        out = apply_resample_mel(mel_power, SR, target_hz=8000)
        cutoff = resample_mel_cutoff_bin(80, SR, 8000)
        assert torch.equal(out[:cutoff, :], mel_power[:cutoff, :])
        assert torch.all(out[cutoff:, :] == 0.0)

    def test_noop_when_target_at_or_above_sample_rate(self):
        mel_power = torch.rand(80, 50)
        assert torch.equal(apply_resample_mel(mel_power, SR, target_hz=SR), mel_power)

    def test_does_not_mutate_input(self):
        mel_power = torch.full((80, 10), 5.0)
        original = mel_power.clone()
        apply_resample_mel(mel_power, SR, target_hz=8000)
        assert torch.equal(mel_power, original)
