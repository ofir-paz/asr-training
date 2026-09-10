import numpy as np
import torch
import torchaudio.functional as ta_f
from numpy.typing import NDArray

from .utils import get_crossfade_mask_pair

# This is the default in:
# https://iver56.github.io/audiomentations/waveform_transforms/shift/
# We assume it's a sane choice for this simple augmentation
fade_duration = 0.005


# Loosely based on:
# https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/shift.py
# With simpler interface and only fwd shift.
# also assumes extending the audio is required
def shift_audio_forward(audio: NDArray[np.float32], shift_sec: float, sample_rate: int) -> NDArray[np.float32]:
    assert shift_sec >= 0, "shift_sec must be non-negative"
    num_places_to_shift = int(round(shift_sec * sample_rate))

    # Assume audio length needs to be extended by the shift amount
    shifted_samples = np.zeros(audio.shape[-1] + num_places_to_shift, dtype=audio.dtype)

    # Copy the original audio to the shifted position
    shifted_samples[num_places_to_shift:] = audio
    # Apply a crossfade to smooth the transition
    fade_length = int(sample_rate * fade_duration)
    fade_in, fade_out = get_crossfade_mask_pair(fade_length)

    fade_in_start = num_places_to_shift
    fade_in_end = min(num_places_to_shift + fade_length, shifted_samples.shape[-1])
    fade_in_length = fade_in_end - fade_in_start

    shifted_samples[
        ...,
        fade_in_start:fade_in_end,
    ] *= fade_in[:fade_in_length]

    return shifted_samples


# Simulates low-quality / telephony audio by downsampling to target_hz and
# resampling back to the original sample rate.  The lossy round-trip
# introduces the kind of bandwidth-limited artefacts that an ASR model
# trained on clean audio may encounter at inference time.
#
# Uses torchaudio.functional.resample (bandlimited sinc + Kaiser window)
# instead of np.interp, which lacks an anti-aliasing filter and would
# introduce aliasing noise on the downsample pass.
def resample_augment(audio: NDArray[np.float32], sample_rate: int, target_hz: int = 8000) -> NDArray[np.float32]:
    if sample_rate <= target_hz:
        # Nothing to do – audio is already at or below the target rate.
        return audio

    t = torch.from_numpy(audio).unsqueeze(0)              # (1, T)
    down = ta_f.resample(t, sample_rate, target_hz)       # (1, T_low)
    up   = ta_f.resample(down, target_hz, sample_rate)    # (1, T)  — same length as input
    return up.squeeze(0).numpy().astype(audio.dtype)
