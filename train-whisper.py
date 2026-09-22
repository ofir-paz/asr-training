#!/usr/bin/env python3
# coding: utf-8

import logging
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("preprocess.preperator").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

import evaluate
import torch
from datasets import DatasetDict, interleave_datasets, load_dataset, load_from_disk, ReadInstruction
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from training.dataloader import DataCollatorSpeechSeq2SeqWithPadding
from training.parser import _resolve_noise_kwargs, parse_arguments
from preprocess.preperator import (
    DatasetPreparator,
    process_datasets,
    whisper_max_target_positions,
)
from training.run_naming import dedupe_name, generate_run_name, local_dir_exists

# Splits on ":" but not inside "[...]" - HF's split-slicing syntax.
dataset_spec_split_pattern = r":(?=(?:[^\[\]]|\[[^\[\]]*\])*$)"

# Column names some datasets use instead of "transcript" (renaming is metadata-only).
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
    """A quality rule; `columns` scopes it so filtering doesn't decode audio."""

    columns: List[str]
    keep: Callable[..., bool]


# Per-dataset quality rules not part of the shared schema.
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
                print("Assumed dataset format mis-detection. Attempting to load using `load_from_disk` instead.")
                raise ValueError("Dataset format mis-detection.")
        except Exception:  # local dataset misdetected as remote; scoped so Ctrl-C isn't swallowed
            dataset = load_from_disk(dataset_name)

            # Support load_dataset's split-slicing syntax here too.
            read_instruction = ReadInstruction.from_spec(split)
            actual_ri_data = read_instruction._relative_instructions[0]
            slice_units = actual_ri_data.unit
            if slice_units != 'abs':  # only "abs" units supported, not pct syntax
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


def compute_metrics(pred, processor, metric, normalizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # else decodes as garbage

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
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


class WhisperDistillationTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with label smoothing and an optional frozen-teacher KL penalty
    (a trust region when the teacher is the checkpoint being fine-tuned). Both are
    training-only; eval loss stays plain cross entropy so runs stay comparable."""

    def __init__(
        self,
        *args,
        teacher_model=None,
        kd_weight: float = 0.0,
        kd_temperature: float = 1.0,
        label_smoothing: float = 0.0,
        eval_data_collator=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Optional clean collator for eval, so on-the-fly augmentation doesn't make eval noisy.
        self.eval_data_collator = eval_data_collator
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.label_smoothing = label_smoothing
        # Suppresses Transformers' second division by grad_accum_steps - compute_loss
        # already normalizes by num_items_in_batch (the whole accumulation window).
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

    def get_eval_dataloader(self, eval_dataset=None):
        """Swaps in eval_data_collator for construction only - DataLoader captures
        collate_fn once, so self.data_collator can be restored right after."""
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)

        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def _transcription_loss(self, logits, labels, num_items_in_batch, label_smoothing):
        # Workaround for a grad-accumulation loss bug; see https://huggingface.co/blog/gradient_accumulation
        vocab_size = logits.shape[2]
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
        labels = labels.to(logits.device)

        loss = loss_fct(logits.view(-1, vocab_size), labels.reshape(-1))
        if reduction == "sum":
            loss = loss / num_items_in_batch

        return loss

    def _teacher_kl(self, student_logits, teacher_logits, labels):
        """Mean per-token KL(teacher || student), masked to the tokens that carry loss."""
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
        return (temperature**2) * kl  # Hinton's T^2, keeps gradient scale comparable across T

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        try:
            outputs = model(**inputs)
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
                # Match teacher dtype explicitly rather than rely on ambient autocast.
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
        # Logs ce/kd_kl separately (a rising train/loss alone can't say which grew), scaled
        # to per-optimizer-step so it's comparable across different grad_accum settings.
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


def _init_run_tracking(args, noise_kwargs: dict) -> None:
    """Pre-inits wandb with the full config (incl. resolved noise YAML) before the Trainer
    exists, so its own WandbCallback layers TrainingArguments onto it instead of replacing it."""
    if args.no_report:
        return

    try:
        import wandb
    except ImportError:
        print("wandb not installed - skipping config pre-logging (pass --no_report to silence this).")
        return

    config = {**vars(args), "noise": noise_kwargs}  # noise_kwargs nested: not on the command line at all

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "huggingface"),
        name=args.run_name,
        config=config,
    )
    if args.noise_config:
        wandb.save(args.noise_config, policy="now")  # exact YAML, recoverable byte-for-byte


def main():
    args = parse_arguments()

    if args.use_preprocessed and (args.train_datasets or args.eval_datasets):
        raise ValueError("Cannot use both preprocessed data and specify train/eval datasets. Choose one method.")

    if args.use_preprocessed and args.save_processed:
        raise ValueError("Cannot use preprocessed data and save preprocessed data at the same time.")

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.target_language, task="transcribe")

    noise_kwargs = _resolve_noise_kwargs(args)

    preparator = DatasetPreparator(
        processor,
        proc_num=args.ds_processor_proc_num,
        timestamp_sample_prob=args.include_timestamps_prob,
        condition_on_prev_sample_prob=args.include_prev_text_prob,
        inject_synthetic_timestamps=args.inject_synthetic_timestamps,
        audio_shift_augmentation=args.audio_shift_augmentation,
        resample_augmentation=args.resample_augmentation,
        resample_target_hz=args.resample_target_hz,
        resample_prob=args.resample_prob,
        **noise_kwargs,
    )

    dataset_shuffle_seed = 745
    if args.use_preprocessed:
        preprocessed_dataset_dicts = []
        for preprocessed in args.use_preprocessed:
            try:
                dataset_dict = load_from_disk(preprocessed)
            except FileNotFoundError:
                dataset_dict = load_dataset(preprocessed)
            preprocessed_dataset_dicts.append(dataset_dict)

        if len(preprocessed_dataset_dicts) == 1:
            train_set = preprocessed_dataset_dicts[0]["train"]
            eval_set = preprocessed_dataset_dicts[0]["eval"]
        else:
            probs = None
            if args.use_preprocessed_probs is not None:
                assert len(args.use_preprocessed_probs) == len(preprocessed_dataset_dicts)
                probs = args.use_preprocessed_probs
            train_set = interleave_datasets(
                [d["train"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
                # Fixed seed so every distributed rank interleaves identically.
                seed=dataset_shuffle_seed,
            )
            eval_set = interleave_datasets(
                [d["eval"] for d in preprocessed_dataset_dicts],
                probabilities=probs,
                stopping_strategy="all_exhausted",
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
        return
    else:
        if not args.train_datasets or not args.eval_datasets:
            raise ValueError("Both --train_datasets and --eval_datasets must be provided for training")

        train_datasets = load_datasets(args.train_datasets)
        eval_datasets = load_datasets(args.eval_datasets)

        train_set = process_datasets(train_datasets, preparator)
        eval_set = process_datasets(eval_datasets, preparator)

    if args.max_eval_set_size:
        eval_set = eval_set.shuffle(seed=dataset_shuffle_seed).select(range(args.max_eval_set_size))

    resuming = bool(args.resume_from_checkpoint or args.resume_from_checkpoint_path)
    if args.output_model_name is None:
        args.output_model_name = generate_run_name(args)
        print(f"--output_model_name not given, auto-generated: {args.output_model_name}")
    if resuming:
        if local_dir_exists(args.output_model_name):
            print(f"Resuming into existing output dir: {args.output_model_name}")
    else:
        deduped = dedupe_name(args.output_model_name, local_dir_exists)
        if deduped != args.output_model_name:
            print(f"Output dir '{args.output_model_name}' already exists, using '{deduped}' instead")
            args.output_model_name = deduped
    # run_name always mirrors output_model_name (--run_name is a deprecated no-op).
    if args.run_name is not None and args.run_name != args.output_model_name:
        print(
            f"--run_name '{args.run_name}' is ignored (deprecated, no-op) - using "
            f"'{args.output_model_name}' for both the output dir and the wandb run name."
        )
    args.run_name = args.output_model_name

    _init_run_tracking(args, noise_kwargs)

    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    train_noise_augmenter = preparator.noise_augmenter if args.use_preprocessed else None
    # None when --train_datasets already baked it in; set for the live (--use_preprocessed) path.
    train_resample_prob = (
        args.resample_prob if (args.use_preprocessed and args.resample_augmentation) else None
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
        noise_augmenter=train_noise_augmenter,
        resample_prob=train_resample_prob,
        resample_target_hz=args.resample_target_hz,
    )
    # Eval always gets the clean collator, so WER/loss stay comparable across evals.
    eval_data_collator = (
        DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=decoder_start_token_id,
            noise_augmenter=None,
            resample_prob=None,
        )
        if (train_noise_augmenter is not None or train_resample_prob is not None)
        else None
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

    # Exactly one warmup form reaches Transformers, so the other can't silently win.
    if args.warmup_steps is not None:
        warmup_steps, warmup_ratio = args.warmup_steps, 0.0
    else:
        warmup_steps, warmup_ratio = 0, 0.1 if args.warmup_ratio is None else args.warmup_ratio
    print(f"Warmup: {f'{warmup_steps} steps' if warmup_steps else f'{warmup_ratio:.1%} of training'}")

    teacher_model = None
    if args.kd_weight > 0:
        teacher_name = args.kd_teacher_model or args.model_name
        # Teacher dtype follows --mixed_precision (halves memory; tf32 maps to full precision).
        teacher_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision)
        print(
            f"Loading frozen KD teacher: {teacher_name} "
            f"(weight={args.kd_weight}, T={args.kd_temperature}, "
            f"dtype={teacher_dtype or torch.get_default_dtype()})"
        )
        teacher_model = WhisperForConditionalGeneration.from_pretrained(
            teacher_name, attn_implementation=args.attn_implementation, torch_dtype=teacher_dtype
        )
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
        report_to="none" if args.no_report else "all",
        load_best_model_at_end=False,
        metric_for_best_model="wer" if args.predict_wer else "loss",
        greater_is_better=False,
        push_to_hub=(not args.skip_push_to_hub),
        run_name=args.run_name,
        # Only the final path segment is valid in a Hub repo id (output_model_name can be a full path).
        hub_model_id=(
            f"{args.hf_org_name}/{Path(args.output_model_name).name}"
            if not args.skip_push_to_hub
            else None
        ),
        remove_unused_columns=False,
        bf16=True if args.mixed_precision == "bf16" else None,
        fp16=True if args.mixed_precision == "fp16" else None,
        tf32=True if args.mixed_precision == "tf32" else None,
        prediction_loss_only=False if args.predict_wer else True,
        save_total_limit=args.max_checkpoints_to_keep,
        save_only_model=True if args.save_only_model else None,
        gradient_checkpointing=args.gradient_checkpointing,
        # Whisper keeps no cache during training, so the non-reentrant implementation is safe.
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ignore_data_skip=args.ignore_data_skip,
        ddp_find_unused_parameters=False,  # no branching in the Whisper forward pass
        average_tokens_across_devices=True,  # correct loss averaging when token counts are unbalanced
    )

    trainer = WhisperDistillationTrainer(
        args=training_args,
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=data_collator,
        eval_data_collator=eval_data_collator,
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

    trainer.save_model(args.output_model_name)


if __name__ == "__main__":
    main()
