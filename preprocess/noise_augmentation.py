"""
Noise augmentation for ASR fine-tuning.

Thin adapter around audiomentations that wires the library's well-tested DSP
(windowing, SNR mixing, perturbation, radio-channel simulation) into the
decide/apply pattern used by DatasetPreparator.

The decide step draws a single integer seed from the caller's np.random.Generator.
The apply step temporarily seeds Python's `random` module and numpy from that
value before calling the audiomentations pipeline, so results are stable and
reproducible for dataset.map caching, even though audiomentations drives its
internal randomisation through the global random state.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from audiomentations import (
    AddBackgroundNoise,
    BandPassFilter,
    Compose,
    Gain,
    PitchShift,
    TanhDistortion,
    TimeStretch,
)
from numpy.typing import NDArray


@dataclass
class NoiseAugmenter:
    """
    On-the-fly noise augmentation backed by audiomentations.

    Usage mirrors the shift-augmentation pattern in DatasetPreparator:

        decision = augmenter.decide_augmentation(rng)   # cheap, called during "decide" step
        audio = augmenter.apply(audio, decision)          # actual DSP, called during "apply" step

    Parameters
    ----------
    noise_dir : directory containing your noise clips (subdirs ok).
    target_sampling_rate : sample rate of the audio being augmented.
    apply_prob : fraction of examples that get any noise at all.
    snr_db_range : (min, max) target SNR in dB. Lower = louder noise.
    gain_jitter_db : ±dB gain jitter applied to each noise clip before mixing,
        to prevent the model from memorising exact clip amplitudes.
    time_stretch_range : (min_rate, max_rate) for light noise-clip perturbation,
        or None to disable.
    pitch_shift_semitone_range : (min, max) semitones of pitch jitter, or None.
    perturb_prob : probability that a given clip gets stretch/pitch/gain applied.
    simulate_radio_channel : if True, apply a bandpass filter + optional tanh
        soft-clip to the mixed signal to approximate radio/walkie-talkie response.
    radio_band_hz : (low_hz, high_hz) bandpass cutoffs for the above.
    radio_clip_drive : tanh distortion drive; >1.0 enables soft-clipping (crude
        AGC/compression stand-in).  1.0 disables it.
    """

    noise_dir: str
    target_sampling_rate: int
    apply_prob: float = 0.6
    snr_db_range: tuple[float, float] = (4.0, 25.0)
    gain_jitter_db: float = 3.0
    time_stretch_range: Optional[tuple[float, float]] = (0.97, 1.03)
    pitch_shift_semitone_range: Optional[tuple[float, float]] = (-0.5, 0.5)
    perturb_prob: float = 0.3
    simulate_radio_channel: bool = False
    radio_band_hz: tuple[float, float] = (50.0, 4000.0)
    radio_clip_drive: float = 1.0

    _compose: Compose = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # -- per-clip perturbation applied to each noise clip before SNR mixing --
        noise_perturbation: list = []
        if self.gain_jitter_db > 0:
            noise_perturbation.append(
                Gain(min_gain_db=-self.gain_jitter_db, max_gain_db=self.gain_jitter_db, p=self.perturb_prob)
            )
        if self.time_stretch_range is not None:
            noise_perturbation.append(
                TimeStretch(
                    min_rate=self.time_stretch_range[0],
                    max_rate=self.time_stretch_range[1],
                    p=self.perturb_prob,
                )
            )
        if self.pitch_shift_semitone_range is not None:
            noise_perturbation.append(
                PitchShift(
                    min_semitones=self.pitch_shift_semitone_range[0],
                    max_semitones=self.pitch_shift_semitone_range[1],
                    p=self.perturb_prob,
                )
            )
        noise_transform = Compose(noise_perturbation) if noise_perturbation else None

        # -- main signal pipeline --
        signal_transforms: list = [
            AddBackgroundNoise(
                sounds_path=self.noise_dir,
                min_snr_db=self.snr_db_range[0],
                max_snr_db=self.snr_db_range[1],
                noise_transform=noise_transform,
                p=1.0,  # probability gate is ours (apply_prob in decide step)
            )
        ]

        if self.simulate_radio_channel:
            # Bandpass: convert (low_hz, high_hz) to audiomentations' center+bandwidth_fraction.
            low_hz, high_hz = self.radio_band_hz
            center_hz = (low_hz + high_hz) / 2.0
            bandwidth_frac = (high_hz - low_hz) / center_hz
            signal_transforms.append(
                BandPassFilter(
                    min_center_freq=center_hz,
                    max_center_freq=center_hz,
                    min_bandwidth_fraction=bandwidth_frac,
                    max_bandwidth_fraction=bandwidth_frac,
                    p=1.0,
                )
            )
            if self.radio_clip_drive > 1.0:
                # audiomentations TanhDistortion expects distortion in [0, 1].
                # Map drive [1, 10] -> distortion [0, 1] linearly.
                distortion = min((self.radio_clip_drive - 1.0) / 9.0, 1.0)
                signal_transforms.append(
                    TanhDistortion(
                        min_distortion=distortion,
                        max_distortion=distortion,
                        p=1.0,
                    )
                )

        self._compose = Compose(signal_transforms)

    # -- decide step --

    def decide_augmentation(self, rng: np.random.Generator) -> Optional[int]:
        """Draw a reproducible seed for the audiomentations pipeline.

        Returns None if this example should be left clean (controlled by apply_prob).
        The seed is drawn here -- before audio is loaded/resampled -- so the decision
        is stable regardless of call order or worker count.
        """
        if rng.random() > self.apply_prob:
            return None
        return int(rng.integers(0, 2**31 - 1))

    # -- apply step --

    def apply(self, audio: NDArray[np.float32], decision: Optional[int]) -> NDArray[np.float32]:
        """Apply the augmentation pipeline with the seed decided earlier.

        Temporarily seeds Python's `random` and numpy's global RNG from `decision`
        so audiomentations' internal randomisation is reproducible, then restores
        the previous state so we don't perturb self.seed or any other caller that
        shares numpy's global random state.
        """
        if decision is None:
            return audio

        py_state = _random.getstate()
        np_state = np.random.get_state()
        try:
            _random.seed(decision)
            np.random.seed(decision)
            return self._compose(audio, sample_rate=self.target_sampling_rate)
        finally:
            _random.setstate(py_state)
            np.random.set_state(np_state)
