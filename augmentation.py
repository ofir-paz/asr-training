"""Reproducible audio augmentation for evaluation.

Randomness is keyed by (seed, dataset, sample index): every sample gets its own private
generator, so results are identical across runs, models, worker counts, parallel
modes and batch sizes, and nothing else in the process can affect them.

Adding an augmentation:
    @dataclass(frozen=True)
    class MyAug(Augmentation):
        name: ClassVar[str] = "my_aug"
        some_param: float = 1.0

        def __call__(self, audio, sampling_rate, rng):
            ...  # use rng for ALL randomness, never np.random.* / random.*

It is then available in configs as {"name": "my_aug", "some_param": 2.0}.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Sequence, Type

import numpy as np

# Key under which evaluate_model attaches the dataset index to each entry
SAMPLE_INDEX_KEY = "_sample_index"

AUGMENTATIONS: Dict[str, Type["Augmentation"]] = {}


class Augmentation(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            if cls.name in AUGMENTATIONS:
                raise ValueError(f"Augmentation name {cls.name!r} is already registered")
            AUGMENTATIONS[cls.name] = cls

    @abstractmethod
    def __call__(self, audio: np.ndarray, sampling_rate: int, rng: np.random.Generator) -> np.ndarray:
        """Return the augmented audio. All randomness must come from rng."""

    def to_config(self) -> Dict[str, Any]:
        return {"name": self.name, **asdict(self)}


@dataclass(frozen=True)
class GaussianNoise(Augmentation):
    """Adds white noise at a random SNR drawn uniformly from [min_snr_db, max_snr_db]."""

    name: ClassVar[str] = "gaussian_noise"
    min_snr_db: float = 10.0
    max_snr_db: float = 30.0
    p: float = 1.0

    def __post_init__(self):
        if self.min_snr_db > self.max_snr_db:
            raise ValueError(f"min_snr_db ({self.min_snr_db}) must be <= max_snr_db ({self.max_snr_db})")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")

    def __call__(self, audio: np.ndarray, sampling_rate: int, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.p:
            return audio

        signal_power = float(np.mean(np.square(audio, dtype=np.float64)))
        if signal_power == 0.0:
            return audio  # silence - SNR is undefined

        snr_db = rng.uniform(self.min_snr_db, self.max_snr_db)
        noise_std = np.sqrt(signal_power / 10 ** (snr_db / 10))
        noise = rng.standard_normal(audio.shape) * noise_std
        return (audio + noise).astype(audio.dtype, copy=False)


class AugmentationPipeline:
    """Applies augmentations in order, each with its own generator derived from (seed, dataset, index).

    `dataset` (e.g. "google/fleurs:he_il:test") keeps the same index in different
    datasets from getting the same randomness.

    Each augmentation gets an independent stream by position, so appending a new
    augmentation at the end never changes the randomness of the existing ones.
    """

    def __init__(self, augmentations: Sequence[Augmentation] = (), seed: int = 0, dataset: str = ""):
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.augmentations = list(augmentations)
        self.seed = int(seed)
        self.dataset = dataset
        # Stable across processes and runs (unlike Python's hash(), which is salted per process)
        self._dataset_key = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8], "big")

    @classmethod
    def from_config(
        cls, config: Sequence[Mapping[str, Any]], seed: int = 0, dataset: str = ""
    ) -> "AugmentationPipeline":
        augmentations = []
        for item in config:
            params = dict(item)
            name = params.pop("name", None)
            if name not in AUGMENTATIONS:
                raise ValueError(f"Unknown augmentation {name!r}. Available: {sorted(AUGMENTATIONS)}")
            try:
                augmentations.append(AUGMENTATIONS[name](**params))
            except TypeError as e:
                raise ValueError(f"Invalid parameters for augmentation {name!r}: {e}") from e
        return cls(augmentations, seed, dataset)

    def to_config(self) -> List[Dict[str, Any]]:
        return [aug.to_config() for aug in self.augmentations]

    def __bool__(self) -> bool:
        return bool(self.augmentations)

    def __call__(self, audio: np.ndarray, sampling_rate: int, index: int) -> np.ndarray:
        if int(index) < 0:
            raise ValueError(f"index must be >= 0, got {index}")
        streams = np.random.SeedSequence([self.seed, self._dataset_key, int(index)]).spawn(len(self.augmentations))
        for aug, stream in zip(self.augmentations, streams):
            audio = aug(audio, sampling_rate, np.random.default_rng(stream))
        return audio


def parse_augment_config(value: str) -> List[Dict[str, Any]]:
    """Parse --augment: a path to a JSON file or an inline JSON string (a list, or a single object)."""
    if os.path.isfile(value):
        with open(value) as f:
            config = json.load(f)
    else:
        config = json.loads(value)
    return [config] if isinstance(config, dict) else list(config)
