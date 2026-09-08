#!/usr/bin/env python3
# coding: utf-8

import argparse
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union

import evaluate
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk, ReadInstruction
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BatchFeature,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from preprocess.preperator import (
    DatasetPreparator,
    process_datasets,
    whisper_max_target_positions,
)

# Split on : but allow : inside [] for the HF split slicing syntax
# https://huggingface.co/docs/datasets/loading#slice-splits
dataset_spec_split_pattern = r":(?=(?:[^\[\]]|\[[^\[\]]*\])*$)"

# The preparator requires the transcript to live in a "transcript" column, but datasets
# in the wild name it differently - crowd-transcribe-v5 uses "sentence", saspeech/eval-d1
# use "text", fleurs uses "transcription". Renaming is metadata-only - no data is copied.
transcript_column_aliases = ["sentence", "text", "transcription"]


def normalize_transcript_column(dataset, dataset_name):
    if "transcript" in dataset.features:
        return dataset

    for alias in transcript_column_aliases:
        if alias in dataset.features:
            print(f"{dataset_name}: using column '{alias}' as 'transcript'")
            return dataset.rename_column(alias, "transcript")

    raise ValueError(
        f"{dataset_name}: no transcript column found "
        f"(tried 'transcript', {transcript_column_aliases})"
    )


@dataclass
class DatasetRowFilter:
    """A quality rule expressed over a dataset's own columns.

    `columns` scopes the predicate to the columns it reads, which is what keeps the audio
    column undecoded - filtering whole rows would decode every example in the dataset.
    """

    columns: List[str]
    keep: Callable[..., bool]


# Some datasets carry quality signals of their own that are not part of the training
# schema. Those rules are declared per dataset here rather than teaching the shared
# preparator about columns only one dataset has. Length is deliberately not filtered on -
# the preparator already drops examples whose labels exceed the model's target positions.
dataset_row_filters = {
    "ivrit-ai/crowd-transcribe-v5": DatasetRowFilter(
        columns=["extra_data"],
        keep=lambda extra_data: not any(
            extra_data[flag]
            for flag in ("skipped", "unintelligible", "foreign_language", "noisy", "multiple_speakers")
        ),
    ),
}


def apply_dataset_row_filter(dataset, dataset_name):
    row_filter = dataset_row_filters.get(dataset_name)
    if row_filter is None:
        return dataset

    filtered = dataset.filter(row_filter.keep, input_columns=row_filter.columns)
    print(
        f"{dataset_name}: quality filter dropped "
        f"{dataset.num_rows - filtered.num_rows} of {dataset.num_rows} rows"
    )
    return filtered


def load_datasets(dataset_specs):
    datasets = []
    for spec in dataset_specs:
        parts = re.split(dataset_spec_split_pattern, spec)

        dataset_name = parts[0]
        split = parts[1] if len(parts) == 2 else "train"

        
        try:
            dataset = load_dataset(dataset_name, split=split)
            if dataset.builder_name == "json" and not "transcript" in dataset.features:
                print(f"Assumed dataset format mis-detection. Attempting to load. using `load_from_disk` instead. (See comments in code)")
                raise ValueError("Dataset format mis-detection.")
        
        # Local datasets, could suffer from a bug where there are more ".json" files
        # than ".arrow" files which leads to a mis-detection of the dataset format.
        # The "load_from_disk" API can get around this problem since it's designed to load
        # such locally stored dataset generated using "save_to_disk"
        except:
            dataset = load_from_disk(dataset_name)
            
            # But, we want to support the flexible "split instruction" syntax like load_dataset provides.
            # Hf made this extremely hard, by hiding the parsing and results inside a wrapped internal class.
            # Why? why HF ?!
            read_instruction = ReadInstruction.from_spec(split)
            actual_ri_data = read_instruction._relative_instructions[0]
            slice_units = actual_ri_data.unit
            # We won't go that crazy - only support "abs" units (not pct syntax)
            if slice_units != 'abs':
                # This is such shame - HF please fix this.
                raise ValueError(f'Unable to support the split definition: ${split} - please read the code for more details.')
            
            split_name = actual_ri_data.splitname
            from_entry = actual_ri_data.from_
            to_entry = actual_ri_data.to
            dataset = dataset[split_name]
            if from_entry is not None:
                dataset = dataset.skip(from_entry)
            else:
                from_entry = 0
            if to_entry is not None:
                dataset = dataset.take(to_entry - from_entry)

        dataset = normalize_transcript_column(dataset, dataset_name)
        datasets.append(apply_dataset_row_filter(dataset, dataset_name))
    return datasets


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Ensure input_features are decompressed if needed:
        input_features = []
        for feature in features:
            pad_amount = feature.get("pad_amount", 0)
            if pad_amount > 0:
                pad_value = feature["pad_value"]  # (d)
                pad_tensor = torch.tensor([pad_value] * pad_amount).T  # (d, pad_amount)
                base_features = torch.tensor(feature["input_features"])  # (d, feat_len)
                final_features = torch.concatenate([base_features, pad_tensor], dim=-1)  # (d, feat_len + pad_amount)
                input_features.append(final_features)
            else:
                input_features.append(torch.tensor(feature["input_features"]))

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


def compute_metrics(pred, processor, metric, normalizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace the loss-ignored value with the padding token for this model
    # which would be decoded to an empty string
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0]
    label_str_norm = [label_str_norm[i] for i in range(len(label_str_norm)) if len(label_str_norm[i]) > 0]

    wer = metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


def prepare_model_for_qlora(model):
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=64,
        lora_alpha=1,
        use_rslora=True,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"],
        # modules_to_save=["embed_tokens"],
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


class WhisperDistillationTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with label smoothing and an optional frozen-teacher KL penalty.

    The loss lives here rather than in a `compute_loss_func` hook because that hook only
    receives the outputs and labels - running a teacher needs the model inputs, which
    only `compute_loss` sees.

    The teacher is fed the same encoder features and decoder prefix as the student, so
    the penalty is a per-token KL between two aligned next-token distributions. Pointing
    it at the checkpoint being fine-tuned makes the term a trust region: it starts at
    exactly zero and grows only as the student drifts.

    Label smoothing and the KL penalty are both applied during training only - eval loss
    stays plain cross entropy so it can be compared across runs that configure them
    differently. Track the regularized objective through `train/loss` and `train/kd_kl`.
    """

    def __init__(
        self,
        *args,
        teacher_model=None,
        kd_weight: float = 0.0,
        kd_temperature: float = 1.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.label_smoothing = label_smoothing
        # Transformers halves-again the loss: training_step divides by
        # gradient_accumulation_steps unless the model takes loss kwargs or a
        # compute_loss_func was supplied. Our compute_loss already normalizes by
        # num_items_in_batch, which counts the tokens of the WHOLE accumulation window, so
        # the micro-batch losses already sum to the correct full-batch loss. Dividing again
        # scales every gradient by 1/accum - silently training at lr/accum. The original
        # code escaped this by passing compute_loss_func; a Trainer subclass has to suppress
        # it here. The flag's only other uses are inside Trainer.compute_loss, which this
        # class overrides in full, so nothing else changes.
        self.model_accepts_loss_kwargs = True

        self._ce_total = 0.0
        self._kd_kl_total = 0.0
        self._loss_steps = 0

        self.teacher_dtype = None
        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()
            self.teacher_model.requires_grad_(False)
            self.teacher_dtype = next(self.teacher_model.parameters()).dtype

    def _transcription_loss(self, logits, labels, num_items_in_batch, label_smoothing):
        # Until the Whisper model loss is updated to use the new Transfomers loss infrastruture,
        # it suffers from  bug in how grad acc steps loss is calculated. This is a workaround.
        # See https://huggingface.co/blog/gradient_accumulation
        vocab_size = logits.shape[2]
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=label_smoothing
        )
        # move labels to correct device to enable PP
        labels = labels.to(logits.device)

        loss = loss_fct(logits.view(-1, vocab_size), labels.reshape(-1))
        if reduction == "sum":
            loss = loss / num_items_in_batch

        return loss

    def _teacher_kl(self, student_logits, teacher_logits, labels):
        """Mean per-token KL(teacher || student) over the tokens that carry loss.

        Masking to the scored tokens keeps the padding and the prompt prefix - which are
        already excluded from the cross entropy - from dominating the penalty.
        """
        scored = labels.to(student_logits.device) != -100
        temperature = self.kd_temperature

        student_log_probs = torch.nn.functional.log_softmax(
            student_logits[scored].float() / temperature, dim=-1
        )
        teacher_log_probs = torch.nn.functional.log_softmax(
            teacher_logits[scored].float() / temperature, dim=-1
        )
        kl = torch.nn.functional.kl_div(
            student_log_probs, teacher_log_probs, log_target=True, reduction="batchmean"
        )

        # Hinton's T^2 keeps the distillation gradient scale comparable across temperatures
        return (temperature**2) * kl

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        try:
            outputs = model(**inputs)
            # Both regularizers are training-only, so eval loss stays plain cross entropy
            # and remains comparable across runs with different smoothing/KD settings.
            loss = self._transcription_loss(
                outputs.logits,
                labels,
                num_items_in_batch,
                label_smoothing=self.label_smoothing if model.training else 0.0,
            )
            if model.training:
                self._ce_total += loss.item()
                self._loss_steps += 1

            if self.teacher_model is not None and self.kd_weight > 0 and model.training:
                # A reduced-precision teacher will not accept the full precision features the
                # collator produces, so match them to its weights explicitly rather than rely
                # on an ambient autocast region being active. Integer inputs - the decoder
                # prefix - must be left alone.
                teacher_inputs = {
                    name: value.to(self.teacher_dtype) if torch.is_floating_point(value) else value
                    for name, value in inputs.items()
                }
                with torch.no_grad():
                    teacher_logits = self.teacher_model(**teacher_inputs).logits
                kl = self._teacher_kl(outputs.logits, teacher_logits, labels)
                self._kd_kl_total += kl.item()
                loss = loss + self.kd_weight * kl
        finally:
            inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # train/loss folds together the smoothed cross entropy and the KL penalty, which
        # leaves a long unattended run undiagnosable - a rising total could be either the
        # model fitting worse or the teacher penalty taking over. Report both terms.
        # compute_loss normalizes the cross entropy by the token count of the WHOLE
        # accumulation window, so its per-micro-batch values sum to one true per-token loss
        # per optimizer step. Averaging them over micro-batches would report that value
        # divided by the accumulation setting, so two runs that fit identically would print
        # different numbers purely because their accumulation differs. Scale back to per
        # optimizer step. The KL is already a per-token mean over its own micro-batch and
        # needs no such correction - which is also why kd_share has to be formed from the
        # corrected cross entropy, not the raw running total.
        if self._loss_steps:
            optimizer_steps = self._loss_steps / self.args.gradient_accumulation_steps
            cross_entropy = self._ce_total / optimizer_steps
            logs["ce"] = cross_entropy
            if self.teacher_model is not None and self.kd_weight > 0:
                kl = self._kd_kl_total / self._loss_steps
                logs["kd_kl"] = kl
                logs["kd_share"] = (self.kd_weight * kl) / max(
                    cross_entropy + self.kd_weight * kl, 1e-12
                )
            self._ce_total = 0.0
            self._kd_kl_total = 0.0
            self._loss_steps = 0
        super().log(logs, start_time)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a Whisper model with custom datasets.")
    parser.add_argument(
        "--train_datasets",
        nargs="*",
        help="Dataset(s) to train on. Format: dataset_name[:split_name]",
    )
    parser.add_argument(
        "--target_language", type=str, default="hebrew", help="The target training language (Only a single language training is currently supported)"
    )
    parser.add_argument("--save_processed", help="Dataset name to save processed data (will save both train and eval)")
    parser.add_argument(
        "--include_timestamps_prob",
        type=float,
        default=0.5,
        help="Probability to include timestamps with a sample (This might be a synthetic augmentation or an existing transcription timestamps)",
    )
    parser.add_argument(
        "--include_prev_text_prob",
        type=float,
        default=0.5,
        help="Probability to include previous text with a sample only when prev transcript is present on the sample",
    )
    parser.add_argument(
        "--inject_synthetic_timestamps",
        help="If timestamps are to be included with a sample but not provided, a start+end timestamp token will be injected",
        action="store_true",
    )
    parser.add_argument(
        "--audio_shift_augmentation",
        help="When timestamps are injected, also randomize shift augmentation on it",
        action="store_true",
    )
    parser.add_argument(
        "--use_preprocessed",
        nargs="+",
        help="Dataset name to load preprocessed data from (either local path or remote dataset)",
    )
    parser.add_argument(
        "--use_preprocessed_probs", nargs="+", type=float, help="Probability of using preprocessed data"
    )
    parser.add_argument(
        "--ds_processor_proc_num", type=int, default=1, help="Number of parallel processors for datasets preparation"
    )
    parser.add_argument("--model_name", default="openai/whisper-large-v2", help="Name of the model to train")
    parser.add_argument("--output_model_name", required=True, help="Name of the fine-tuned model to generate")
    parser.add_argument("--hf_org_name", default="ivrit-ai", help="Name of HF Org to push the model to")
    parser.add_argument("--skip_push_to_hub", action="store_true", help="Don't push result model to hub")
    parser.add_argument(
        "--eval_datasets",
        nargs="*",
        help="Reference dataset(s) for evaluation. Format: dataset_name[:split_name]",
    )
    parser.add_argument(
        "--save_only_model", action="store_true", default=False, help="Save only the model without optimizer state"
    )
    parser.add_argument(
        "--max_checkpoints_to_keep",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep during training",
    )
    parser.add_argument(
        "--resume_from_checkpoint", action="store_true", help="Try and resuming for last saved checkpoint"
    )
    parser.add_argument(
        "--resume_from_checkpoint_path", type=str, help="Path to checkpoint to resume from", default=None
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between each model save/upload.")
    parser.add_argument(
        "--ignore_data_skip", action="store_true", help="Ignore data skip when resuming from checkpoint"
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["bf16", "fp16", "tf32", None],
        default=None,
        help="Mixed precision mode for training",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        choices=["sdpa"],
        help="Attention implementation to use (only 'sdpa' available)",
    )
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for training")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Recompute activations during the backward pass instead of storing them. "
        "Trades roughly 30%% more compute for a large drop in activation memory, which is "
        "what caps the per-device batch size on a single accelerator",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    # Transformers gives warmup_steps precedence whenever it is > 0, which silently
    # discarded warmup_ratio. Accept exactly one and translate it in main().
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup_ratio", type=float, help="Fraction of total training steps spent warming up (default: 0.1)"
    )
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="How many steps to train for - overrides num_train_epochs"
    )
    warmup_group.add_argument(
        "--warmup_steps", type=int, help="Absolute number of warmup steps, instead of --warmup_ratio"
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="constant_with_warmup", help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay (AdamW L2 regularization)")
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for the transcription cross entropy",
    )
    parser.add_argument(
        "--kd_weight",
        type=float,
        default=0.0,
        help="Weight of the KL penalty against a frozen teacher (0 disables distillation)",
    )
    parser.add_argument(
        "--kd_teacher_model",
        type=str,
        default=None,
        help="Teacher to regularize towards. Defaults to --model_name, which makes the "
        "penalty a trust region around the checkpoint being fine-tuned",
    )
    parser.add_argument(
        "--kd_temperature", type=float, default=1.0, help="Softmax temperature for the KL penalty"
    )
    parser.add_argument(
        "--eval_steps", type=int, help="Number of steps between two evals, if not specified defaults to logging_steps."
    )
    parser.add_argument(
        "--predict_wer", action="store_true", help="Use WER as the metric for best model instead of loss"
    )
    parser.add_argument("--max_eval_set_size", type=int, help="Maximum number of entries to fetch from eval dataset.")

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Per-device train batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Per-device eval batch size.")

    parser.add_argument("--run_name", help="Run name to report to the run tracker")
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of step between each log")

    pg = parser.add_argument_group("noise augmentation")
    pg.add_argument(
        "--noise_config",
        type=str,
        default=None,
        metavar="YAML_PATH",
        help="Path to a YAML file with noise augmentation settings. "
             "Enables live noise augmentation during training (only works with --train_datasets, not --use_preprocessed). "
             "See preprocess/noise_config.yaml for an annotated example.",
    )
    pg.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the noise_dir from --noise_config without editing the YAML.",
    )
    pg.add_argument(
        "--noise_augmentation",
        action="store_true",
        default=False,
        help="Enable live noise augmentation during training. Requires --noise_config.",
    )
    return parser.parse_args()


def _load_noise_config(yaml_path: str) -> dict:
    """Load noise augmentation kwargs from a YAML file.

    Keys can be written with or without the 'noise_' prefix — both are accepted.
    Lists are converted to tuples. Defaults live in noise_config.yaml itself.
    """
    import yaml

    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = {}
    for key, val in raw.items():
        canonical = key if key.startswith("noise_") else f"noise_{key}"
        cfg[canonical] = tuple(val) if isinstance(val, list) else val

    if not cfg.get("noise_dir"):
        raise ValueError("noise_config YAML must include 'noise_dir'")

    return cfg


def main():
    args = parse_arguments()

    if args.use_preprocessed and (args.train_datasets or args.eval_datasets):
        raise ValueError("Cannot use both preprocessed data and specify train/eval datasets. Choose one method.")

    if args.use_preprocessed and args.save_processed:
        raise ValueError("Cannot use preprocessed data and save preprocessed data at the same time.")

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.target_language, task="transcribe")

    noise_kwargs = {}
    if args.noise_augmentation:
        if not args.noise_config:
            raise ValueError("--noise_augmentation requires --noise_config to also be set")
        noise_kwargs = _load_noise_config(args.noise_config)
        if args.noise_dir:
            noise_kwargs["noise_dir"] = args.noise_dir
        noise_kwargs["noise_augmentation"] = True
        print(f"Noise augmentation enabled (config: {args.noise_config})")
        print(f"  noise_dir: {noise_kwargs['noise_dir']}")
    elif args.noise_dir:
        raise ValueError("--noise_dir requires --noise_augmentation to also be set")

    preparator = DatasetPreparator(
        processor,
        proc_num=args.ds_processor_proc_num,
        timestamp_sample_prob=args.include_timestamps_prob,
        condition_on_prev_sample_prob=args.include_prev_text_prob,
        inject_synthetic_timestamps=args.inject_synthetic_timestamps,
        audio_shift_augmentation=args.audio_shift_augmentation,
        **noise_kwargs,
    )

    dataset_shuffle_seed = 745
    if args.use_preprocessed:
        preprocessed_dataset_dicts = []
        for preprocessed in args.use_preprocessed:
            try:
                # Try to load from disk first
                dataset_dict = load_from_disk(preprocessed)
            except FileNotFoundError:
                # If not found on disk, try to load as a remote dataset
                dataset_dict = load_dataset(preprocessed)
            preprocessed_dataset_dicts.append(dataset_dict)

        if len(preprocessed_dataset_dicts) == 1:
            train_set = dataset_dict["train"]
            eval_set = dataset_dict["eval"]
        else:
            probs = None
            if args.use_preprocessed_probs is not None:
                assert len(args.use_preprocessed_probs) == len(preprocessed_dataset_dicts)
                probs = args.use_preprocessed_probs
            train_set = interleave_datasets(
                [d["train"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # We set the seed so each distributed process will interleave in the same way
                # otherwise - the dataloader across each process ends up with different lengths
                # which screws up the collective synchronization
                # See https://huggingface.co/docs/accelerate/en/concept_guides/internal_mechanism
                seed=dataset_shuffle_seed,
            )
            eval_set = interleave_datasets(
                [d["eval"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # We set the seed so each distributed process will interleave in the same way
                # See above.
                seed=dataset_shuffle_seed,
            )

    elif args.save_processed:

        if not args.train_datasets or not args.eval_datasets:
            raise ValueError("Both --train_datasets and --eval_datasets must be provided when using --save_processed")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets(args.eval_datasets)

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

        dataset_dict = DatasetDict({"train": train_set, "eval": eval_set})
        dataset_dict.save_to_disk(args.save_processed)
        print(f"Preprocessed datasets saved to {args.save_processed}")
        return  # Exit after saving preprocessed data
    else:
        if not args.train_datasets or not args.eval_datasets:
            raise ValueError("Both --train_datasets and --eval_datasets must be provided for training")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets(args.eval_datasets)

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

    if args.max_eval_set_size:
        eval_set = eval_set.shuffle(seed=dataset_shuffle_seed).select(range(args.max_eval_set_size))

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    )

    metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()

    if args.use_qlora:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            args.model_name, attn_implementation=args.attn_implementation
        )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    assert (
        model.config.max_target_positions == whisper_max_target_positions
    ), f"Model max_target_positions {model.config.max_target_positions} != {whisper_max_target_positions}"

    if args.use_qlora:
        model = prepare_model_for_qlora(model)

    model.config.use_cache = False

    model.generate = partial(model.generate, language=args.target_language, task="transcribe", use_cache=True)

    # Exactly one warmup form reaches Transformers, so the other cannot silently win
    if args.warmup_steps is not None:
        warmup_steps, warmup_ratio = args.warmup_steps, 0.0
    else:
        warmup_steps, warmup_ratio = 0, 0.1 if args.warmup_ratio is None else args.warmup_ratio
    print(f"Warmup: {f'{warmup_steps} steps' if warmup_steps else f'{warmup_ratio:.1%} of training'}")

    teacher_model = None
    if args.kd_weight > 0:
        teacher_name = args.kd_teacher_model or args.model_name
        # The teacher is frozen and only ever runs forward, so it can be held at whatever
        # reduced precision the autocast region already computes in - that halves its
        # footprint and keeps its dtype matching the activations flowing through it even
        # when autocast is not active. Deriving this from --mixed_precision rather than
        # exposing a separate flag keeps the two from contradicting each other. tf32 is an
        # fp32 compute mode, so it correctly maps to full precision here.
        teacher_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision)
        print(
            f"Loading frozen KD teacher: {teacher_name} "
            f"(weight={args.kd_weight}, T={args.kd_temperature}, "
            f"dtype={teacher_dtype or torch.get_default_dtype()})"
        )
        teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_name, attn_implementation=args.attn_implementation, torch_dtype=teacher_dtype
        )
        # No generation happens on the teacher - skip the cache it would otherwise build
        teacher_model.config.use_cache = False
    elif args.kd_teacher_model:
        print("Ignoring --kd_teacher_model since --kd_weight is 0")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=model.config.max_target_positions,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="all" if args.run_name else "none",
        load_best_model_at_end=False,
        metric_for_best_model="wer" if args.predict_wer else "loss",
        greater_is_better=False,
        push_to_hub=(not args.skip_push_to_hub),
        run_name=args.run_name,
        hub_model_id=f"{args.hf_org_name}/{args.output_model_name}" if not args.skip_push_to_hub else None,
        remove_unused_columns=False,
        # Configure mixed precision based on the argument
        bf16=True if args.mixed_precision == "bf16" else None,
        fp16=True if args.mixed_precision == "fp16" else None,
        tf32=True if args.mixed_precision == "tf32" else None,
        # Configure prediction loss and metric based on predict_wer
        prediction_loss_only=False if args.predict_wer else True,
        # Configure save_total_limit if max_checkpoints_to_keep is provided
        save_total_limit=args.max_checkpoints_to_keep,
        # Configure save_only_model
        save_only_model=True if args.save_only_model else None,
        gradient_checkpointing=args.gradient_checkpointing,
        # Whisper keeps no cache during training, so the non-reentrant implementation is safe
        # and is the one that composes with the rest of the Trainer machinery.
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        # There is not branching in training the Whisper model
        ddp_find_unused_parameters=False,
        # This would take longer, but will calculate the loss
        # with proper averaging across GPUs.
        # this is important when the dataset samples vary
        # wildly in the amount of tokens contributing to the loss and
        # the distribution of those samples is very unbalanced
        average_tokens_across_devices=True,
    )

    trainer = WhisperDistillationTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric, normalizer),
        processing_class=processor,
        teacher_model=teacher_model,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        label_smoothing=args.label_smoothing,
    )

    resume_from_checkpoint = False
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        resume_from_checkpoint = True
        if args.resume_from_checkpoint_path is not None:
            resume_from_checkpoint = args.resume_from_checkpoint_path
            print(f"Resuming checkpoint {resume_from_checkpoint}")
        else:
            print("No checkpoint path provided, resuming from latest")

    print("Start training!")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the model
    trainer.save_model(args.output_model_name)


if __name__ == "__main__":
    main()
