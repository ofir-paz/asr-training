"""End-to-end (mocked) tests for train-whisper.py's DataCollatorSpeechSeq2SeqWithPadding:
the live noise (_mix_mel_noise) and live resample (_resample_mel) mel-domain augmentation
paths used when --use_preprocessed leaves no raw waveform to augment in the time domain,
plus the full __call__ batch-collation path exercising both together.

Uses a real (default-config, no pretrained download) WhisperFeatureExtractor so the mel
math is exact, and a minimal fake tokenizer (no network/vocab download needed) standing in
for WhisperTokenizer's .pad() - the only tokenizer behavior the collator actually calls.
"""

import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchaudio
from transformers import BatchFeature, Seq2SeqTrainingArguments, WhisperFeatureExtractor

from preprocess.augmentation import resample_mel_cutoff_bin
from preprocess.noise_augmentation import NoiseAugmenter

SR = 16000
N_MELS = 80
DECODER_START_TOKEN_ID = 50258


class FakeTokenizer:
    """Stands in for WhisperTokenizer's .pad() - the only tokenizer call the collator
    makes - without needing a downloaded vocab.
    """

    def __init__(self, pad_token_id=50257):
        self.pad_token_id = pad_token_id

    def pad(self, label_features, return_tensors="pt"):
        max_len = max(len(f["input_ids"]) for f in label_features)
        input_ids, attention_mask = [], []
        for f in label_features:
            ids = list(f["input_ids"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
        return BatchFeature(
            {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}
        )


@pytest.fixture(scope="module")
def feature_extractor():
    return WhisperFeatureExtractor()  # default config: n_mels=80, sr=16000 - no download


@pytest.fixture
def fake_processor(feature_extractor):
    return type(
        "FakeProcessor", (), {"feature_extractor": feature_extractor, "tokenizer": FakeTokenizer()}
    )()


@pytest.fixture
def noise_dir(tmp_path):
    t = np.arange(SR) / SR
    tone = (0.4 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    torchaudio.save(str(tmp_path / "tone.wav"), torch.from_numpy(tone).unsqueeze(0), SR)
    return tmp_path


def _fake_feature(feat_len, pad_amount=0):
    rng = np.random.default_rng(0)
    input_features = rng.uniform(0.2, 0.8, (N_MELS, feat_len)).astype(np.float32)
    # A couple of prev-text-like tokens ahead of the start-of-transcript token, so the
    # collator's prompt-masking (everything before it -> -100) has something to do.
    labels = [999, 998, DECODER_START_TOKEN_ID, 10, 20, 30, 50256]
    return {
        "input_features": input_features,
        "pad_value": np.zeros(N_MELS, dtype=np.float32),
        "pad_amount": pad_amount,
        "labels": labels,
    }


class TestMixMelNoise:
    def test_noop_when_no_augmenter(self, train_whisper_module, fake_processor):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID, noise_augmenter=None
        )
        feats = torch.rand(N_MELS, 500)
        assert torch.equal(collator._mix_mel_noise(feats, pad_amount=0), feats)

    def test_changes_features_when_applied(self, train_whisper_module, fake_processor, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            noise_augmenter=augmenter,
        )
        feats = torch.rand(N_MELS, 500) * 0.6 + 0.2
        out = collator._mix_mel_noise(feats, pad_amount=0)
        assert out.shape == feats.shape
        assert torch.isfinite(out).all()
        assert not torch.allclose(out, feats)

    @pytest.mark.parametrize("feat_len", [100, 587, 1500])
    def test_shape_preserved_across_lengths(self, train_whisper_module, fake_processor, noise_dir, feat_len):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            noise_augmenter=augmenter,
        )
        feats = torch.rand(N_MELS, feat_len) * 0.5 + 0.2
        out = collator._mix_mel_noise(feats, pad_amount=3000 - feat_len)
        assert out.shape == (N_MELS, feat_len)


class TestResampleMel:
    def test_noop_when_disabled(self, train_whisper_module, fake_processor):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID, resample_prob=None
        )
        feats = torch.rand(N_MELS, 400)
        assert torch.equal(collator._resample_mel(feats), feats)

    def test_never_applies_when_prob_zero(self, train_whisper_module, fake_processor):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID, resample_prob=0.0
        )
        feats = torch.rand(N_MELS, 400) * 0.5 + 0.2
        assert torch.equal(collator._resample_mel(feats), feats)

    def test_zeroes_high_mel_bins_when_forced(self, train_whisper_module, fake_processor):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            resample_prob=1.0,
            resample_target_hz=8000,
        )
        feats = torch.full((N_MELS, 400), 0.6)  # plausible mid-range normalized log-mel value
        out = collator._resample_mel(feats)
        assert out.shape == feats.shape
        assert torch.isfinite(out).all()

        cutoff = resample_mel_cutoff_bin(N_MELS, SR, 8000)
        # Above the cutoff, mel power should have collapsed toward the log floor - i.e. be
        # strictly lower than the untouched value, not merely "different".
        assert (out[cutoff:, :] < feats[cutoff:, :]).all()
        # Below the cutoff, left untouched.
        assert torch.equal(out[:cutoff, :], feats[:cutoff, :])


class TestCollatorCallEndToEnd:
    def test_batch_with_noise_and_resample_both_enabled(self, train_whisper_module, fake_processor, noise_dir):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            noise_augmenter=augmenter,
            resample_prob=1.0,
            resample_target_hz=8000,
        )
        # Real examples always pad out to the same fixed total (Whisper's 30s/3000-frame
        # window) regardless of each utterance's real length - match that invariant here
        # so torch.stack has equal-shaped tensors to work with, same as in production.
        features = [_fake_feature(400, pad_amount=250), _fake_feature(600, pad_amount=50)]
        batch = collator(features)

        assert batch["input_features"].shape == (2, N_MELS, 650)
        assert torch.isfinite(batch["input_features"]).all()
        assert "decoder_input_ids" in batch
        assert batch["labels"].shape[0] == 2
        # The prompt/start token position should be masked to -100, not left as raw ids.
        assert (batch["labels"] == -100).any()

    def test_batch_clean_no_augmentation(self, train_whisper_module, fake_processor):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID
        )
        features = [_fake_feature(300)]
        batch = collator(features)
        assert batch["input_features"].shape == (1, N_MELS, 300)
        assert torch.isfinite(batch["input_features"]).all()

    def test_augmented_and_clean_batches_differ(self, train_whisper_module, fake_processor, noise_dir):
        """Sanity check that enabling augmentation actually changes what the model sees,
        for an identical input batch."""
        features = [_fake_feature(400)]

        clean_collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID
        )
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        noisy_collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            noise_augmenter=augmenter,
        )

        clean_batch = clean_collator(features)
        noisy_batch = noisy_collator(features)
        assert not torch.allclose(clean_batch["input_features"], noisy_batch["input_features"])


class _TinyModel(nn.Module):
    """Stand-in for WhisperForConditionalGeneration: just enough surface (a forward
    signature) for Trainer construction. get_train_dataloader() never actually calls it -
    this class exists purely so WhisperDistillationTrainer.__init__ has a model to hold.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, **kwargs):
        raise NotImplementedError("not meant to be called - this test only exercises get_train_dataloader()")


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TestTrainerWiring:
    """Confirms the augmentation collator isn't just correct in isolation, but is actually
    what runs when the real Seq2SeqTrainer machinery (WhisperDistillationTrainer, which
    train-whisper.py's main() constructs and hands the collator to) builds a training
    DataLoader and pulls a batch through it - the same path a real `trainer.train()` call
    would use. No model forward pass, GPU, or dataset download involved: get_train_dataloader()
    only needs a model object to exist, not to be run.
    """

    def test_train_dataloader_invokes_noise_and_resample_augmentation(
        self, train_whisper_module, fake_processor, noise_dir
    ):
        augmenter = NoiseAugmenter(noise_dir=str(noise_dir), target_sampling_rate=SR, apply_prob=1.0)
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor,
            decoder_start_token_id=DECODER_START_TOKEN_ID,
            noise_augmenter=augmenter,
            resample_prob=1.0,
            resample_target_hz=8000,
        )
        train_dataset = _ListDataset([_fake_feature(300) for _ in range(4)])

        with tempfile.TemporaryDirectory() as out_dir:
            training_args = Seq2SeqTrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=2,
                report_to=[],
                remove_unused_columns=False,
                logging_steps=1,
                max_steps=1,
            )
            trainer = train_whisper_module.WhisperDistillationTrainer(
                args=training_args,
                model=_TinyModel(),
                train_dataset=train_dataset,
                data_collator=collator,
            )

            # Patching the instance's plain (non-dunder) methods, not collator.__call__
            # itself - __call__ is looked up on the type for `collator(...)` syntax, so an
            # instance-level patch of it is silently never hit; regular named methods don't
            # have that special-method lookup quirk.
            with patch.object(
                collator, "_mix_mel_noise", wraps=collator._mix_mel_noise
            ) as noise_spy, patch.object(
                collator, "_resample_mel", wraps=collator._resample_mel
            ) as resample_spy:
                dataloader = trainer.get_train_dataloader()
                batch = next(iter(dataloader))

            assert noise_spy.call_count >= 1
            assert resample_spy.call_count >= 1
            assert batch["input_features"].shape == (2, N_MELS, 300)
            assert torch.isfinite(batch["input_features"]).all()

    def test_train_dataloader_skips_augmentation_when_disabled(
        self, train_whisper_module, fake_processor
    ):
        collator = train_whisper_module.DataCollatorSpeechSeq2SeqWithPadding(
            processor=fake_processor, decoder_start_token_id=DECODER_START_TOKEN_ID
        )
        train_dataset = _ListDataset([_fake_feature(300) for _ in range(4)])

        with tempfile.TemporaryDirectory() as out_dir:
            training_args = Seq2SeqTrainingArguments(
                output_dir=out_dir,
                per_device_train_batch_size=2,
                report_to=[],
                remove_unused_columns=False,
                logging_steps=1,
                max_steps=1,
            )
            trainer = train_whisper_module.WhisperDistillationTrainer(
                args=training_args,
                model=_TinyModel(),
                train_dataset=train_dataset,
                data_collator=collator,
            )
            dataloader = trainer.get_train_dataloader()
            batch = next(iter(dataloader))

        assert batch["input_features"].shape == (2, N_MELS, 300)
        assert torch.isfinite(batch["input_features"]).all()
