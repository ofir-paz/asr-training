"""CLI argument parsing, plus resolving --noise_config's YAML (the default source for every
noise_* setting) against the --noise_<key> CLI overrides (the "real" per-run values).
"""

import argparse


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
        "--resample_augmentation",
        help="Simulate low-quality audio by downsampling to --resample_target_hz and back",
        action="store_true",
    )
    parser.add_argument(
        "--resample_target_hz",
        type=int,
        default=8000,
        help="Target sample rate for the resample augmentation round-trip (default: 8000)",
    )
    parser.add_argument(
        "--resample_prob",
        type=float,
        default=0.6,
        help="Probability of applying resample augmentation to each sample (default: 0.6)",
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
    parser.add_argument(
        "--output_model_name",
        default=None,
        help="Name of the fine-tuned model to generate. Also used as the run tracker (e.g. "
        "wandb) run name (--run_name is a deprecated no-op, see its help), so a run is "
        "always trivially matched to its local output dir/checkpoints. If omitted, a name "
        "is generated from the run's hyperparameters (e.g. 'lv3-mix-na-b32-kd0.1'). Either "
        "way, if that name already exists as a local output dir, '_2', '_3', etc. is "
        "appended rather than overwriting it - unless --resume_from_checkpoint is set, in "
        "which case the name is used as-is so resuming finds the existing checkpoints.",
    )
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
    # warmup_steps > 0 otherwise silently beats warmup_ratio in Transformers; force one choice.
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

    parser.add_argument(
        "--no_report", action="store_true", help="Disable reporting to run trackers (e.g. wandb)."
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Deprecated, accepted for backward compatibility only: run_name always mirrors "
        "--output_model_name now (see its help), so this has no effect. Kept as a no-op flag "
        "so older commands that pass it (typically set equal to --output_model_name) don't "
        "fail to parse.",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Number of step between each log")

    pg = parser.add_argument_group("noise augmentation")
    pg.add_argument(
        "--noise_config",
        type=str,
        default=None,
        metavar="YAML_PATH",
        help="Path to a YAML file with noise augmentation settings - the single source of "
             "defaults for every noise_* setting below; --noise_<key> flags (e.g. "
             "--noise_apply_prob) override one key at a time without editing the file. "
             "With --use_preprocessed, noise is mixed live in the mel domain by the data "
             "collator (fresh per batch, so it varies across epochs). With --train_datasets, "
             "noise is instead baked once into the mel features at dataset-prep time "
             "(same realization every epoch). Eval always uses a noise-free collator "
             "regardless of this setting. See preprocess/noise_config.yaml for an annotated example.",
    )
    pg.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the noise_dir from --noise_config without editing the YAML.",
    )
    # One override flag per --noise_config key, all defaulting to None (use the YAML).
    pg.add_argument("--noise_apply_prob", type=float, default=None, help="Override apply_prob.")
    pg.add_argument(
        "--noise_snr_db_range", type=float, nargs=2, default=None, metavar=("MIN_DB", "MAX_DB"),
        help="Override snr_db_range.",
    )
    pg.add_argument(
        "--noise_num_noises_range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override num_noises_range.",
    )
    pg.add_argument("--noise_gain_jitter_db", type=float, default=None, help="Override gain_jitter_db.")
    pg.add_argument(
        "--noise_coverage_frac_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override coverage_frac_range.",
    )
    pg.add_argument(
        "--noise_burst_len_frac_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override burst_len_frac_range.",
    )
    pg.add_argument("--noise_perturb_prob", type=float, default=None, help="Override perturb_prob.")
    pg.add_argument(
        "--noise_time_stretch_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override time_stretch_range.",
    )
    pg.add_argument(
        "--noise_pitch_shift_semitone_range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override pitch_shift_semitone_range.",
    )
    pg.add_argument(
        "--noise_simulate_radio_channel", action=argparse.BooleanOptionalAction, default=None,
        help="Override simulate_radio_channel.",
    )
    pg.add_argument(
        "--noise_radio_band_hz", type=float, nargs=2, default=None, metavar=("LOW_HZ", "HIGH_HZ"),
        help="Override radio_band_hz.",
    )
    pg.add_argument(
        "--noise_filter_signal_too", action=argparse.BooleanOptionalAction, default=None,
        help="Override filter_signal_too.",
    )
    pg.add_argument("--noise_radio_clip_drive", type=float, default=None, help="Override radio_clip_drive.")
    pg.add_argument(
        "--noise_augmentation",
        action="store_true",
        default=False,
        help="Enable live noise augmentation during training. Requires --noise_config.",
    )
    return parser.parse_args()


def _load_noise_config(yaml_path: str) -> dict:
    """Loads --noise_config's YAML. Keys accepted with or without the 'noise_' prefix; lists
    become tuples."""
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


# Every key --noise_config accepts; each has a matching --noise_<key> CLI flag with the same
# name, so the merge in _resolve_noise_kwargs is a plain getattr/dict-set.
_NOISE_CONFIG_OVERRIDE_KEYS = [
    "noise_dir",
    "noise_apply_prob",
    "noise_snr_db_range",
    "noise_num_noises_range",
    "noise_gain_jitter_db",
    "noise_coverage_frac_range",
    "noise_burst_len_frac_range",
    "noise_perturb_prob",
    "noise_time_stretch_range",
    "noise_pitch_shift_semitone_range",
    "noise_simulate_radio_channel",
    "noise_radio_band_hz",
    "noise_filter_signal_too",
    "noise_radio_clip_drive",
]


def _resolve_noise_kwargs(args) -> dict:
    """YAML defaults, then per-key CLI overrides layered on top. {} when --noise_augmentation
    is off; raises if a --noise_<key> override was passed without it (silently ignoring would
    be worse - it looks like it should have done something)."""
    if not args.noise_augmentation:
        set_without_augmentation = [
            key for key in _NOISE_CONFIG_OVERRIDE_KEYS if getattr(args, key) is not None
        ]
        if set_without_augmentation:
            flags = ", ".join(f"--{key}" for key in set_without_augmentation)
            raise ValueError(f"{flags} require --noise_augmentation to also be set")
        return {}

    if not args.noise_config:
        raise ValueError("--noise_augmentation requires --noise_config to also be set")
    noise_kwargs = _load_noise_config(args.noise_config)

    overridden = []
    for key in _NOISE_CONFIG_OVERRIDE_KEYS:
        value = getattr(args, key)
        if value is not None:
            # nargs=2 flags arrive as lists; match _load_noise_config's own tuple convention.
            noise_kwargs[key] = tuple(value) if isinstance(value, list) else value
            overridden.append(key)

    noise_kwargs["noise_augmentation"] = True
    print(f"Noise augmentation enabled (config: {args.noise_config})")
    if overridden:
        print(f"  CLI overrides: {', '.join(f'{k}={noise_kwargs[k]}' for k in overridden)}")
    print(f"  noise_dir: {noise_kwargs['noise_dir']}")
    return noise_kwargs
