"""Mel-spectrogram DSP for the live (--use_preprocessed) augmentation path in dataloader.py,
which only has mel features to work with, never a raw waveform.
"""

from functools import lru_cache

import librosa
import numpy as np
import torch
from scipy.signal import butter, freqz

# Below this, a mel-power array is silence as far as SNR mixing is concerned. Whisper's log
# floor (1e-10) maps back to a power RMS floor of ~1e-5; this sits just above it so the check
# can actually fire (see preprocess/noise_augmentation.py's own, differently-calibrated,
# waveform-domain _SILENCE_RMS_THRESHOLD - the two floors are not the same value).
MEL_SILENCE_RMS_THRESHOLD = 2e-5


def whisper_log_mel_to_power(log_mel: torch.Tensor) -> torch.Tensor:
    """Inverts WhisperFeatureExtractor's (log10(mel)+4)/4 normalization back to linear power."""
    return 10.0 ** (log_mel * 4.0 - 4.0)


def whisper_power_to_log_mel(power: torch.Tensor) -> torch.Tensor:
    """Re-applies Whisper's clamp/log10/dynamic-range/normalize sequence to linear mel power."""
    log10_power = torch.log10(torch.clamp(power, min=1e-10))
    log10_power = torch.maximum(log10_power, log10_power.max() - 8.0)
    return (log10_power + 4.0) / 4.0


def mel_power_rms(power: torch.Tensor) -> torch.Tensor:
    """sqrt(mean(power)) - a loudness proxy, not preprocess.noise_augmentation._rms's
    sqrt(mean(x**2)); squaring an already-squared power value would double it up wrongly."""
    return torch.sqrt(torch.mean(power) + 1e-12)


@lru_cache(maxsize=8)
def _resample_power_response(n_mels: int, sample_rate: int, target_hz: int, order: int = 4) -> np.ndarray:
    """Per-mel-bin power attenuation approximating resample_augment()'s downsample+upsample
    round trip: a real filter's frequency response, not a brick-wall cutoff - zeroing bins
    outright corresponds to convolving with a sinc in the time domain and rings. Each of the
    two resample passes (down, up) contributes one |H(f)|^2 in power, via the same
    convolution-theorem logic as _bandpass_filter/butter elsewhere in this codebase.
    """
    nyquist = sample_rate / 2
    cutoff = min(target_hz / 2, nyquist * 0.999) / nyquist
    b, a = butter(order, cutoff, btype="low")
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=nyquist)
    _, h = freqz(b, a, worN=mel_freqs, fs=sample_rate)
    magnitude = np.abs(h)
    return (magnitude**4).astype(np.float32)  # squared per pass, two passes


def apply_resample_mel(mel_power: torch.Tensor, sample_rate: int, target_hz: int) -> torch.Tensor:
    """Bandwidth-limits mel power the way resample_augment() bandwidth-limits a waveform,
    for when only mel features are available (see _resample_power_response)."""
    if sample_rate <= target_hz:
        return mel_power
    response = torch.from_numpy(_resample_power_response(mel_power.shape[0], sample_rate, target_hz))
    return mel_power * response.unsqueeze(-1)
