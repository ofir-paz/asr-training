"""Tests for resample augmentation: the baked, waveform-domain round trip
(preprocess/augmentation.py's resample_augment, used by DatasetPreparator) and its live,
mel-domain approximation (dsp_utils.apply_resample_mel, used by the collator when
--use_preprocessed leaves no raw waveform to actually resample).
"""

import numpy as np
import torch

from training.dsp_utils import apply_resample_mel
from preprocess.augmentation import resample_augment

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
        target_hz = 8000
        high_tone = _tone(7000)  # above 8kHz's 4kHz Nyquist
        low_tone = _tone(500)  # well within band

        high_out = resample_augment(high_tone, SR, target_hz=target_hz)
        low_out = resample_augment(low_tone, SR, target_hz=target_hz)

        assert _rms(high_out) < 0.3 * _rms(high_tone)
        assert _rms(low_out) > 0.8 * _rms(low_tone)


class TestApplyResampleMel:
    def test_noop_when_target_at_or_above_sample_rate(self):
        mel_power = torch.rand(80, 50)
        assert torch.equal(apply_resample_mel(mel_power, SR, target_hz=SR), mel_power)

    def test_does_not_mutate_input(self):
        mel_power = torch.full((80, 10), 5.0)
        original = mel_power.clone()
        apply_resample_mel(mel_power, SR, target_hz=8000)
        assert torch.equal(mel_power, original)

    def test_low_bins_mostly_preserved_high_bins_heavily_attenuated(self):
        mel_power = torch.ones(80, 20)
        out = apply_resample_mel(mel_power, SR, target_hz=8000)
        ratio = (out / mel_power).mean(dim=1)  # per-bin attenuation, (80,)

        assert ratio[5] > 0.9  # well within the passband
        assert ratio[75] < 0.05  # well above the target's Nyquist

    def test_attenuation_is_smooth_not_a_hard_cutoff(self):
        """A real filter's response is monotonic and gradual, not a 1.0/0.0 step - this is
        the actual review point: zeroing bins outright is a brick-wall cutoff, which rings
        in the time domain (Gibbs phenomenon); a proper filter response doesn't."""
        mel_power = torch.ones(80, 20)
        out = apply_resample_mel(mel_power, SR, target_hz=8000)
        ratio = (out / mel_power).mean(dim=1)

        # Monotonically non-increasing as frequency rises.
        assert (ratio[1:] <= ratio[:-1] + 1e-6).all()
        # No single-bin cliff: the transition from mostly-passed to mostly-blocked spans
        # more than one bin (a hard cutoff would jump from ~1.0 to ~0.0 between two bins).
        transition = ((ratio < 0.9) & (ratio > 0.1)).sum().item()
        assert transition >= 2

    def test_partial_attenuation_at_nominal_cutoff(self):
        """At the nominal cutoff frequency itself, a real filter is partially open (unlike
        an ideal brick wall, which would be exactly 1.0 just below and 0.0 just above)."""
        import librosa

        n_mels = 80
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=SR / 2)
        cutoff_bin = int(np.argmin(np.abs(freqs - 4000.0)))  # 8kHz target's Nyquist

        mel_power = torch.ones(n_mels, 20)
        out = apply_resample_mel(mel_power, SR, target_hz=8000)
        ratio = (out / mel_power).mean(dim=1)

        assert 0.05 < ratio[cutoff_bin].item() < 0.95
