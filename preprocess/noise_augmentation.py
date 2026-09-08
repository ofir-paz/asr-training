"""
Noise augmentation for ASR fine-tuning.

Design goals (see accompanying discussion):
- Noise clips are NOT resized/normalized to a fixed length up front. Instead, a random
  window (or a randomly-rotated loop, if the clip is shorter than the target audio) is
  extracted at apply-time, so the same small noise library works for any utterance length
  and yields free variety across epochs.
- Noise is mixed at a randomized target SNR (dB) rather than a fixed gain, so severity
  varies across examples. `apply_prob` additionally controls what fraction of examples get
  noise at all (so the model still sees plenty of clean audio).
- Everything is driven by a caller-supplied `np.random.Generator` (the same `self.seed`
  DatasetPreparator already uses), split into a `decide_*` step (cheap, deterministic given
  the seed state, safe to run before we've resampled/shifted audio) and an `apply` step
  (does the actual DSP), matching the existing shift-augmentation pattern.
- Optional light in-family perturbation (gain jitter, optional pitch/time stretch) so the
  ~70 raw clips don't get reused completely verbatim every time they're picked.
- Optional radio-channel simulation (bandpass filter, optional soft clipping) since "sounds
  like radio" is a channel effect on top of, not a substitute for, SNR-based noise mixing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

try:
    import librosa

    _HAS_LIBROSA = True
except ImportError:  # pitch/time-stretch perturbation becomes a no-op without it
    _HAS_LIBROSA = False

_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _rms(x: NDArray[np.float32]) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

# not used corrently (because I resample twice.)
def _bandpass_filter(audio: NDArray[np.float32], sample_rate: int, low_hz: float, high_hz: float, order: int = 4) -> NDArray[np.float32]:
    nyq = sample_rate / 2
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)

# simulates people shouting into the wlakie talkie
def _soft_clip(audio: NDArray[np.float32], drive: float) -> NDArray[np.float32]:
    # tanh soft-clipper: cheap stand-in for radio AGC/compression squashing peaks.
    # drive > 1 pushes more of the signal into the nonlinear region.
    if drive <= 1.0:
        return audio
    return np.tanh(audio * drive).astype(np.float32) / np.tanh(drive)

# loops over the noise inside teh window from a random start inside the signal
def _loop_or_window(clip: NDArray[np.float32], length: int, rng: np.random.Generator) -> NDArray[np.float32]:
    """Return a segment of exactly `length` samples from `clip`, without resizing the clip itself.

    - If the clip is longer than `length`: take a random contiguous window.
    - If the clip is shorter: rotate it by a random offset (so the loop seam lands in a
      different place each time it's picked) and tile it out to `length`.
    """
    clip_len = clip.shape[-1]
    if clip_len == 0:
        return np.zeros(length, dtype=np.float32)

    if clip_len >= length:
        max_start = clip_len - length
        start = int(rng.integers(0, max_start + 1))
        return clip[start : start + length]

    start = int(rng.integers(0, clip_len))
    rotated = np.concatenate([clip[start:], clip[:start]])
    reps = int(np.ceil(length / clip_len))
    tiled = np.tile(rotated, reps)
    return tiled[:length]

def _place_bursts(
    clip: NDArray[np.float32],
    length: int,
    coverage_frac: float,
    burst_len_frac_range: tuple[float, float],
    rng: np.random.Generator,
) -> NDArray[np.float32]:
    out = np.zeros(length, dtype=np.float32)
    clip_len = clip.shape[-1]

    max_slots = max(1, int(length // clip_len))          # e.g. 40//4 = 10
    num_bursts = max(1, int(round(max_slots * coverage_frac)))  # e.g. round(10 * 0.8) = 8

    for _ in range(num_bursts):
        frac = float(rng.uniform(*burst_len_frac_range))
        burst_len = max(1, min(int(round(clip_len * frac)), length, clip_len))
        burst = _loop_or_window(clip, burst_len, rng)
        start = int(rng.integers(0, length - burst_len + 1))
        out[start : start + burst_len] += burst
    return out

# scale the noise with the audio signal to hit some specific snr
def mix_audio_at_snr(
    audio: NDArray[np.float32], noise: NDArray[np.float32], snr_db: float, prevent_clipping: bool = True
) -> NDArray[np.float32]:
    """Mix `noise` into `audio` scaled so the result hits `snr_db` (signal RMS vs noise RMS)."""
    audio_rms = _rms(audio)
    noise_rms = _rms(noise)

    if noise_rms < 1e-8 or audio_rms < 1e-8:
        return audio  # nothing sensible to mix (silent clip / silent audio)

    target_noise_rms = audio_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = audio + scaled_noise

    if prevent_clipping:
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak

    return mixed.astype(audio.dtype)


class NoiseLibrary:
    """Loads and caches a directory of noise files, resampled to a single target sample rate.

    Loaded eagerly (not per-worker-lazy) at construction time so behavior is predictable
    when this object gets pickled out to `dataset.map(num_proc=...)` workers.
    """

    def __init__(self, noise_dir: str, target_sampling_rate: int, extensions: tuple[str, ...] = _AUDIO_EXTENSIONS):
        self.noise_dir = Path(noise_dir)
        self.target_sampling_rate = target_sampling_rate # should be 16khz
        self.file_paths = sorted(p for p in self.noise_dir.rglob("*") if p.suffix.lower() in extensions)

        if not self.file_paths:
            raise ValueError(f"No noise files found under {noise_dir} (looked for {extensions})")

        self.clips: list[NDArray[np.float32]] = []
        for path in self.file_paths:
            waveform, sr = torchaudio.load(str(path))  # (channels, samples)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
            if sr != target_sampling_rate:
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sampling_rate)(waveform)
            clip = waveform.squeeze(0).numpy().astype(np.float32)
            if clip.size == 0 or _rms(clip) < 1e-8:
                warnings.warn(f"Noise clip appears silent/empty, skipping: {path}")
                continue
            self.clips.append(clip)

        if not self.clips:
            raise ValueError(f"All noise files under {noise_dir} were empty or silent")

    def __len__(self) -> int:
        return len(self.clips)


@dataclass
class NoiseAugmenter:
    """
    On-the-fly noise augmentation. Usage mirrors the shift-augmentation pattern:

        decision = augmenter.decide_augmentation(rng)   # cheap, called during "decide" step
        audio = augmenter.apply(audio, decision)          # actual DSP, called during "apply" step

    Parameters
    ----------
    noise_dir : directory containing your noise clips (subfolders ok).
    apply_prob : fraction of examples that get any noise at all. Keep this < 1.0 so the
        model still sees clean audio and doesn't learn to expect noise always.
    snr_db_range : (min, max) target SNR in dB, sampled uniformly per example. Lower = more
        noise. For radio-band robustness you generally want a fairly wide range (e.g.
        0-25 dB) so the model handles both clean and heavily-degraded channel conditions,
        rather than one fixed severity.
    num_noises_range : (min, max) how many noise clips to layer together per example
        (inclusive). >1 is useful if you want e.g. hiss + intermittent static together;
        (1, 1) just picks one clip.
    gain_jitter_db : each selected clip additionally gets a random +/- gain jitter (dB)
        before SNR scaling, purely to vary the character of the same clip across picks.
    time_stretch_range / pitch_shift_semitone_range : optional light in-family perturbation
        of the picked noise segment (requires `librosa`; silently skipped if unavailable).
        Keep these ranges small (defaults are already conservative) - the goal is "sounds
        like a slightly different take of the same noise", not a new noise.
    perturb_prob : probability of applying the stretch/pitch perturbation to a given picked
        clip (so not every application is perturbed).
    simulate_radio_channel : if True, bandpass-filters the *noise* (and, if `filter_signal_too`,
        the mixed signal) to the given band, and optionally applies soft clipping - to
        approximate the frequency response / compression of a radio channel rather than just
        adding radio-recorded noise on top of full-bandwidth speech.
    radio_band_hz : bandpass cutcallback frequencies for the above.
    radio_clip_drive : >1.0 enables tanh soft-clipping (crude AGC/compression stand-in);
        1.0 disables it.
    """

    noise_dir: str 
    target_sampling_rate: int
    apply_prob: float = 0.6
    snr_db_range: tuple[float, float] = (4.0, 25.0)
    num_noises_range: tuple[int, int] = (1, 1)
    gain_jitter_db: float = 3.0
    burst_len_frac_range: tuple[float, float] = (0.9, 1.0) # how much to use out of the noise clip
    coverage_frac_range: tuple[float, float] = (0.5, 0.8) # how much of the signal would get pullted (calculted based of 0.5 to 0.8 percent of the time)

    time_stretch_range: Optional[tuple[float, float]] = (0.97, 1.03)
    pitch_shift_semitone_range: Optional[tuple[float, float]] = (-0.5, 0.5)
    perturb_prob: float = 0.3 # the propability for stretch, and pitch should be low
    simulate_radio_channel: bool = True
    radio_band_hz: tuple[float, float] = (50.0, 4000.0)
    filter_signal_too: bool = False
    radio_clip_drive: float = 1.0
    library: NoiseLibrary = field(init=False, repr=False)

    def __post_init__(self):
        self.library = NoiseLibrary(self.noise_dir, self.target_sampling_rate)
        if (self.time_stretch_range is not None or self.pitch_shift_semitone_range is not None) and not _HAS_LIBROSA:
            warnings.warn(
                "librosa not installed; time_stretch_range/pitch_shift_semitone_range will be ignored. "
                "`pip install librosa` to enable light noise-clip perturbation."
            )

    # ---- decide step -----------------------------------------------------
    def decide_augmentation(self, rng: np.random.Generator) -> Optional[dict]:
        """Decide all randomized augmentation params up front. Returns None => no noise this example."""
        if rng.random() > self.apply_prob:
            return None

        num_noises = int(rng.integers(self.num_noises_range[0], self.num_noises_range[1] + 1))
        noise_indices = [int(rng.integers(0, len(self.library))) for _ in range(num_noises)]
        gain_jitters_db = [float(rng.uniform(-self.gain_jitter_db, self.gain_jitter_db)) for _ in range(num_noises)]

        # Independent sub-seeds per noise so windowing offset / perturbation don't interact
        # in ways that couple to unrelated decisions (keeps things reproducible & decorrelated).
        offset_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(num_noises)]

        perturb_flags = [bool(rng.random() < self.perturb_prob) for _ in range(num_noises)]
        time_stretch_factors = [
            float(rng.uniform(*self.time_stretch_range)) if (self.time_stretch_range and perturb_flags[i]) else 1.0
            for i in range(num_noises)
        ]
        pitch_shift_semitones = [
            float(rng.uniform(*self.pitch_shift_semitone_range))
            if (self.pitch_shift_semitone_range and perturb_flags[i])
            else 0.0
            for i in range(num_noises)
        ]

        snr_db = float(rng.uniform(*self.snr_db_range))
        coverage_fracs = [float(rng.uniform(*self.coverage_frac_range)) for _ in range(num_noises)]

        return {
            "noise_indices": noise_indices, # pick the noises
            "gain_jitters_db": gain_jitters_db, # higher db for it 
            "offset_seeds": offset_seeds,
            "time_stretch_factors": time_stretch_factors, #fasten the noise
            "pitch_shift_semitones": pitch_shift_semitones, # get a different pitch for the noise
            "snr_db": snr_db,
            "coverage_fracs": coverage_fracs,
        }

    
    # ---- apply step --------------------------------------------------------
    def apply(self, audio: NDArray[np.float32], decision: Optional[dict]) -> NDArray[np.float32]:
        if decision is None:
            return audio

        length = audio.shape[-1]
        mixed_noise = np.zeros(length, dtype=np.float32)

        for idx, gain_db, offset_seed, stretch, pitch, coverage_frac in zip(
            decision["noise_indices"],
            decision["gain_jitters_db"],
            decision["offset_seeds"],
            decision["time_stretch_factors"],
            decision["pitch_shift_semitones"],
            decision["coverage_fracs"],
        ):
            clip = self.library.clips[idx]
            local_rng = np.random.default_rng(offset_seed)

            if _HAS_LIBROSA and (stretch != 1.0 or pitch != 0.0):
                clip_for_bursts = clip
                if stretch != 1.0:
                    clip_for_bursts = librosa.effects.time_stretch(clip_for_bursts, rate=stretch)
                if pitch != 0.0:
                    clip_for_bursts = librosa.effects.pitch_shift(
                        clip_for_bursts, sr=self.target_sampling_rate, n_steps=pitch
                    )
                clip_for_bursts = clip_for_bursts.astype(np.float32)
            else:
                clip_for_bursts = clip

            segment = _place_bursts(clip_for_bursts, length, coverage_frac, self.burst_len_frac_range, local_rng)            
            segment = segment * (10 ** (gain_db / 20))
            mixed_noise += segment
            
            # might should be dropped in the future, because the noise corpus is resampled twice -> 8khz -> 16khz        
            # if self.simulate_radio_channel:
            #     mixed_noise = _bandpass_filter(mixed_noise, self.target_sampling_rate, *self.radio_band_hz)

        mixed = mix_audio_at_snr(audio, mixed_noise, decision["snr_db"])

        if self.simulate_radio_channel:
            if self.filter_signal_too:
                mixed = _bandpass_filter(mixed, self.target_sampling_rate, *self.radio_band_hz)
            if self.radio_clip_drive > 1.0:
                mixed = _soft_clip(mixed, self.radio_clip_drive)

        return mixed
