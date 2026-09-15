"""Derives a short, descriptive run/output name from train-whisper.py's own CLI args.

There's no off-the-shelf library for this: W&B/MLflow/etc. only generate random
adjective-noun names (e.g. "silly-golden-otter") when you don't supply one - they have
no notion of *this project's* hyperparameters. The naming convention actually used for
past runs (e.g. "lv3-mix-na-b32-kd-low-smooth") is a hand-picked project-specific
convention, so this reproduces it as code: one short token per setting that meaningfully
changes what the run *is* (model, dataset composition, which augmentations/regularizers
are on). Where the original convention used a vague qualitative bucket ("low"), this
uses the actual value instead (e.g. "kd0.1") so two differently-tuned runs can't collide
on the same generated name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

_MODEL_SHORTCODES = [
    (re.compile(r"large-v3", re.I), "lv3"),
    (re.compile(r"large-v2", re.I), "lv2"),
    (re.compile(r"large-v1", re.I), "lv1"),
    (re.compile(r"large(?!-v)", re.I), "lg"),
    (re.compile(r"medium", re.I), "med"),
    (re.compile(r"small", re.I), "sm"),
    (re.compile(r"base", re.I), "base"),
    (re.compile(r"tiny", re.I), "tiny"),
]


def _slug(text: str, max_len: int = 12) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:max_len]


def _model_shortcode(model_name: str) -> str:
    for pattern, code in _MODEL_SHORTCODES:
        if pattern.search(model_name):
            return code
    # Unknown model family (e.g. a custom fine-tuned checkpoint) - fall back to a slug
    # of the last path segment ("my-org/custom-ckpt-v2" -> "customckptv2").
    return _slug(model_name.rsplit("/", 1)[-1]) or "model"


def _fmt_num(x: float) -> str:
    """Compact number formatting for name tokens: 1.0 -> '1', 0.10 -> '0.1'."""
    return f"{x:g}"


def generate_run_name(args) -> str:
    """Build a name like 'lv3-mix-na-b32-kd0.1-sm0.1' from the args that define what
    this run actually is. Called when --output_model_name is omitted; also used as the
    wandb run name (train-whisper.py keeps the two in sync, no separate --run_name).
    """
    tokens = [_model_shortcode(args.model_name)]

    dataset_specs = list(args.train_datasets or args.use_preprocessed or [])
    if len(dataset_specs) > 1:
        tokens.append("mix")
    elif len(dataset_specs) == 1:
        # Strip any ":split" suffix before slugging, e.g. "org/dataset:train[:1000]"
        base = dataset_specs[0].split(":", 1)[0].rsplit("/", 1)[-1]
        tokens.append(_slug(base, max_len=10) or "ds")

    if getattr(args, "noise_augmentation", False):
        tokens.append("na")
    if getattr(args, "resample_augmentation", False):
        tokens.append("rs")
    if getattr(args, "use_qlora", False):
        tokens.append("qlora")

    effective_batch = args.per_device_train_batch_size * max(args.gradient_accumulation_steps, 1)
    tokens.append(f"b{effective_batch}")

    if getattr(args, "kd_weight", 0.0):
        tokens.append(f"kd{_fmt_num(args.kd_weight)}")

    if getattr(args, "label_smoothing", 0.0):
        tokens.append(f"sm{_fmt_num(args.label_smoothing)}")

    if getattr(args, "mixed_precision", None):
        tokens.append(args.mixed_precision)

    return "-".join(tokens)


def dedupe_name(name: str, exists_fn: Callable[[str], bool]) -> str:
    """Append _2, _3, ... until `exists_fn(candidate)` is False.

    Applied both to auto-generated names and to explicitly-given ones - an explicit
    --output_model_name that collides with an existing output dir gets versioned too
    rather than silently overwriting it.
    """
    if not exists_fn(name):
        return name
    i = 2
    while exists_fn(f"{name}_{i}"):
        i += 1
    return f"{name}_{i}"


def local_dir_exists(name: str) -> bool:
    return Path(name).exists()
