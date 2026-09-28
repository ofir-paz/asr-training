"""Live augmentations for --use_preprocessed, where only log-mel features exist (no raw waveform).
Each one is a Callable[[features], features] handed to the data collator."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from preprocess.noise_augmentation import NoiseAugmenter, build_noise_waveform, snr_target_rms
from training.dsp_utils import (
    MEL_SILENCE_RMS_THRESHOLD,
    apply_resample_mel,
    mel_power_rms,
    whisper_log_mel_to_power,
    whisper_power_to_log_mel,
)

logger = logging.getLogger(__name__)


@dataclass
class MelResampleAugmentation:
    """Live counterpart of --resample_augmentation, approximated in mel-power space
    since there's no waveform here to actually resample."""

    sample_rate: int
    prob: float
    target_hz: int = 8000

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if np.random.default_rng().random() >= self.prob:
            return features

        signal_power = whisper_log_mel_to_power(features)
        resampled_power = apply_resample_mel(signal_power, self.sample_rate, self.target_hz)
        return whisper_power_to_log_mel(resampled_power)


@dataclass
class MelNoiseAugmentation:
    """Live counterpart of noise augmentation: builds the noise as a waveform (burst/
    pitch/stretch are time-domain ops), then mixes it into speech in mel-power space."""

    noise_augmenter: NoiseAugmenter
    feature_extractor: Any

    def __post_init__(self):
        # Radio-channel effects need a raw waveform to filter/clip; this path only has mel
        # features, so warn instead of silently ignoring a config that looks like it should work.
        augmenter = self.noise_augmenter
        ignored = []
        if augmenter.simulate_radio_channel and augmenter.filter_signal_too:
            ignored.append("filter_signal_too")
        if augmenter.simulate_radio_channel and augmenter.radio_clip_drive > 1.0:
            ignored.append("radio_clip_drive")
        if ignored:
            logger.warning(
                "[MelNoiseAugmentation] noise config sets %s, which the live mel-domain path "
                "cannot apply (needs a raw waveform) - only baked (--train_datasets) noise honors it.",
                " and ".join(ignored),
            )

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        noise_decision = self.noise_augmenter.decide_augmentation(np.random.default_rng())
        if noise_decision is None:
            return features

        extractor = self.feature_extractor
        feat_len = features.shape[-1]
        audio_samples = feat_len * extractor.hop_length
        mixed_noise = build_noise_waveform(
            noise_decision,
            self.noise_augmenter.library,
            self.noise_augmenter.burst_len_frac_range,
            self.noise_augmenter.target_sampling_rate,
            audio_samples,
        )

        noise_feat_result = extractor(
            mixed_noise, sampling_rate=extractor.sampling_rate, return_attention_mask=False
        )
        noise_features = torch.tensor(noise_feat_result["input_features"][0])[:, :feat_len]

        signal_power = whisper_log_mel_to_power(features)
        noise_power = whisper_log_mel_to_power(noise_features)
        signal_rms = mel_power_rms(signal_power)
        noise_rms = mel_power_rms(noise_power)

        # A silent noise realization still sits on Whisper's mel floor, not true zero;
        # scaling that up to hit the target SNR would synthesize hiss out of nothing.
        if noise_rms < MEL_SILENCE_RMS_THRESHOLD or signal_rms < MEL_SILENCE_RMS_THRESHOLD:
            return features

        snr_db = noise_decision["snr_db"]
        target_noise_rms = snr_target_rms(signal_rms, snr_db)
        scaled_noise_power = noise_power * ((target_noise_rms / noise_rms) ** 2)
        mixed_power = signal_power + scaled_noise_power
        mixed_features = whisper_power_to_log_mel(mixed_power)

        noise_names = [self.noise_augmenter.library.file_paths[j].name for j in noise_decision["noise_indices"]]
        logger.debug(
            "[MelNoiseAugmentation] APPLIED MEL-NOISE | clips: %s | SNR: %.2f dB | Mel RMS before: %.5f -> after: %.5f",
            ", ".join(noise_names), snr_db, signal_rms.item(), mel_power_rms(mixed_power).item(),
        )
        return mixed_features
