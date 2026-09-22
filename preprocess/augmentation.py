import numpy as np
import torch
import torchaudio.functional as ta_f
from numpy.typing import NDArray

from .utils import get_crossfade_mask_pair

fade_duration = 0.005  # default from https://iver56.github.io/audiomentations/waveform_transforms/shift/


def shift_audio_forward(audio: NDArray[np.float32], shift_sec: float, sample_rate: int) -> NDArray[np.float32]:
    """Prepends shift_sec of silence, crossfaded in to avoid a click at the seam."""
    assert shift_sec >= 0, "shift_sec must be non-negative"
    num_places_to_shift = int(round(shift_sec * sample_rate))

    shifted_samples = np.zeros(audio.shape[-1] + num_places_to_shift, dtype=audio.dtype)
    shifted_samples[num_places_to_shift:] = audio

    fade_length = int(sample_rate * fade_duration)
    fade_in, fade_out = get_crossfade_mask_pair(fade_length)
    fade_in_start = num_places_to_shift
    fade_in_end = min(num_places_to_shift + fade_length, shifted_samples.shape[-1])
    shifted_samples[..., fade_in_start:fade_in_end] *= fade_in[: fade_in_end - fade_in_start]

    return shifted_samples


def resample_augment(audio: NDArray[np.float32], sample_rate: int, target_hz: int = 8000) -> NDArray[np.float32]:
    """Simulates telephony-quality audio via a downsample/upsample round trip - the lossy
    band-limiting an ASR model trained on clean audio may meet at inference time."""
    if sample_rate <= target_hz:
        return audio

    t = torch.from_numpy(audio).unsqueeze(0)
    down = ta_f.resample(t, sample_rate, target_hz)
    up = ta_f.resample(down, target_hz, sample_rate)
    return up.squeeze(0).numpy().astype(audio.dtype)
