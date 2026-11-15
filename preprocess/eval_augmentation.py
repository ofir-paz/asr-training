"""Test-time noise augmentation for evaluate_model.py: mixes noise into each benchmark example
before it reaches the engine, so models trained with and without noise augmentation can be
compared on noisy audio.

Seeding decides whether every model hears the same noise. With a benchmark seed, each
example's generator is seeded from (seed, hash of the example's audio), so the noise an
example gets depends only on that pair - not on dataset order, batching, sharding or which
model is being evaluated - and every model is scored on identical waveforms. Without one,
every example draws fresh entropy and runs are not comparable example-by-example.
"""

import hashlib
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from preprocess.noise_augmentation import NoiseAugmenter


def audio_digest(array, sampling_rate: int) -> int:
    """A 128-bit content hash of an example's audio, stable across runs and processes."""
    digest = hashlib.sha256()
    digest.update(str(int(sampling_rate)).encode())
    digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return int.from_bytes(digest.digest()[:16], "big")


@dataclass
class EvalNoiseAugmentation:
    noise_augmenter: NoiseAugmenter
    # None -> fresh randomness per example (not reproducible, not paired across models).
    seed: Optional[int] = None

    def example_rng(self, array, sampling_rate: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, audio_digest(array, sampling_rate)])

    def __call__(self, entry: dict) -> dict:
        """Returns a copy of `entry` with noisy audio at the augmenter's sample rate, plus
        `augmentation_*` fields recording what was applied (evaluate_model.py writes every
        non-audio field to the results CSV)."""
        audio = entry["audio"]
        if not isinstance(audio, dict):
            raise ValueError("Test-time augmentation needs decoded audio (an array), not a file path")

        rng = self.example_rng(audio["array"], audio["sampling_rate"])
        target_sampling_rate = self.noise_augmenter.target_sampling_rate
        waveform = librosa.resample(
            np.asarray(audio["array"], dtype=np.float32), orig_sr=audio["sampling_rate"], target_sr=target_sampling_rate
        )

        decision = self.noise_augmenter.decide_augmentation(rng)
        file_paths = self.noise_augmenter.library.file_paths

        augmented = dict(entry)
        augmented["audio"] = {
            **audio,
            "array": self.noise_augmenter.apply(waveform, decision),
            "sampling_rate": target_sampling_rate,
        }
        augmented["augmentation_noise_clips"] = (
            ";".join(file_paths[i].name for i in decision["noise_indices"]) if decision else ""
        )
        augmented["augmentation_snr_db"] = decision["snr_db"] if decision else float("nan")
        return augmented
