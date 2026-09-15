"""The data collator: pads/stacks batches, and (for --use_preprocessed, where no raw
waveform survives to bake augmentation into ahead of time) applies noise and resample
augmentation live, in mel-power space, on each batch.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import BatchFeature

from training.dsp_utils import (
    MEL_SILENCE_RMS_THRESHOLD,
    apply_resample_mel,
    mel_power_rms,
    whisper_log_mel_to_power,
    whisper_power_to_log_mel,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    noise_augmenter: Any = None
    # None disables live resample augmentation (default; always the case for eval).
    resample_prob: Optional[float] = None
    resample_target_hz: int = 8000

    def __post_init__(self):
        augmenter = self.noise_augmenter
        if augmenter is None:
            return
        # Radio-channel effects need a raw waveform to filter/clip; this path only has mel
        # features, so warn instead of silently ignoring a config that looks like it should work.
        ignored = []
        if augmenter.simulate_radio_channel and augmenter.filter_signal_too:
            ignored.append("filter_signal_too")
        if augmenter.simulate_radio_channel and augmenter.radio_clip_drive > 1.0:
            ignored.append("radio_clip_drive")
        if ignored:
            logger.warning(
                "[DataCollator] noise config sets %s, which the live mel-domain collator "
                "cannot apply (needs a raw waveform) - only baked (--train_datasets) noise honors it.",
                " and ".join(ignored),
            )

    def _resample_mel(self, base_features: torch.Tensor) -> torch.Tensor:
        """Live counterpart of --resample_augmentation, approximated in mel-power space
        since there's no waveform here to actually resample."""
        if self.resample_prob is None:
            return base_features
        if np.random.default_rng().random() >= self.resample_prob:
            return base_features

        signal_power = whisper_log_mel_to_power(base_features)
        resampled_power = apply_resample_mel(
            signal_power, self.processor.feature_extractor.sampling_rate, self.resample_target_hz
        )
        return whisper_power_to_log_mel(resampled_power)

    def _mix_mel_noise(self, base_features: torch.Tensor, pad_amount: int) -> torch.Tensor:
        """Live counterpart of noise augmentation: builds the noise as a waveform (the
        library is waveforms, and burst/pitch/stretch are time-domain ops), then mixes it
        into the speech in mel-power space since that's all that's available here.

        pad_amount is unused: base_features is already the padding-stripped real content,
        and the caller re-appends the stored pad value after this returns.
        """
        if self.noise_augmenter is None:
            return base_features

        noise_decision = self.noise_augmenter.decide_augmentation(np.random.default_rng())
        if noise_decision is None:
            return base_features

        from preprocess.noise_augmentation import build_noise_waveform, snr_target_rms

        extractor = self.processor.feature_extractor
        feat_len = base_features.shape[-1]
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

        signal_power = whisper_log_mel_to_power(base_features)
        noise_power = whisper_log_mel_to_power(noise_features)
        signal_rms = mel_power_rms(signal_power)
        noise_rms = mel_power_rms(noise_power)

        # A silent noise realization still sits on Whisper's mel floor, not true zero;
        # scaling that up to hit the target SNR would synthesize hiss out of nothing.
        if noise_rms < MEL_SILENCE_RMS_THRESHOLD or signal_rms < MEL_SILENCE_RMS_THRESHOLD:
            return base_features

        snr_db = noise_decision["snr_db"]
        target_noise_rms = snr_target_rms(signal_rms, snr_db)
        scaled_noise_power = noise_power * ((target_noise_rms / noise_rms) ** 2)
        mixed_power = signal_power + scaled_noise_power
        mixed_features = whisper_power_to_log_mel(mixed_power)

        noise_names = [self.noise_augmenter.library.file_paths[j].name for j in noise_decision["noise_indices"]]
        logger.debug(
            "[DataCollator] APPLIED MEL-NOISE | clips: %s | SNR: %.2f dB | Mel RMS before: %.5f -> after: %.5f",
            ", ".join(noise_names), snr_db, signal_rms.item(), mel_power_rms(mixed_power).item(),
        )
        return mixed_features

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for feature in features:
            pad_amount = feature.get("pad_amount", 0)
            base_features = torch.tensor(feature["input_features"])  # (d, feat_len)

            # Order mirrors DatasetPreparator's baked path: bandwidth-limit, then add noise.
            base_features = self._resample_mel(base_features)
            base_features = self._mix_mel_noise(base_features, pad_amount)

            if pad_amount > 0:
                # Broadcast the stored pad column, rather than torch.tensor([pad_value]*n)
                # which converts a list of numpy arrays and is slow.
                pad_value = torch.as_tensor(np.asarray(feature["pad_value"], dtype=np.float32))
                pad_tensor = pad_value.unsqueeze(-1).expand(-1, pad_amount)
                input_features.append(torch.concatenate([base_features, pad_tensor], dim=-1))
            else:
                input_features.append(base_features)

        batch = BatchFeature({"input_features": torch.stack(input_features)})

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"]

        # Labels, represent the input to the decoder
        batch["decoder_input_ids"] = labels[:, :-1]

        # Shift all labels to the left, thus the expected generated label
        # is at the same index of the generated output id from the decoder
        # and the loss function would compare them (cross entropy loss in this case)
        # Note - this means there is no loss calculated for the first "start of transcript" token id
        # since it is not expected to be predicted but always provided.
        # The loss is calculated for the task/lang/notimestamp tokens since the model needs to know
        # to associate them with the proper output
        # **Warning!** the labels are shifted here, and some version of transformers will assume
        # they are not if using the default "ForCausalLMLoss"
        # Once Whisper is updated to use that built-in loss - need to reconsider the collator.
        # Atm the custom loss function expects this shift to be done here.
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # Where we do not need to attend when calculating loss - -100 is the agreed
        # ignored value for the pytorch loss functions
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        bos_index = torch.where(bos_index > 0, bos_index + 1, bos_index)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        return batch
