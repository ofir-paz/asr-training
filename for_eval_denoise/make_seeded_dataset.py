#!/usr/bin/env python3
"""Write the seeded-noise ("-s") version of benchmark datasets to disk, so it can be denoised and re-evaluated.

Reproduces the test-time noise that origin/feature/benchmark-with-augs (commit 70737f5) mixes in inside
evaluate_model.py, with the settings recorded in results_2/seeded/*/*.csv:

    --noise-config preprocess/noise_config.yaml --noise-dir audio-set/test_augs
    --noise-apply-prob 1.0 --noise-band-limit-hz 8000 --augmentation-seed 1234

Each example's noise is seeded from (seed, hash of its decoded audio), so writing it once here gives the
same waveform evaluate_model.py would have fed the engine. Output is 16 kHz mono (what the augmenter
returns), stored as float WAV so the samples are kept exactly. Run from the asr-training root with the
main .venv (Python 3.10):

    python for_eval_denoise/make_seeded_dataset.py --output for_eval_denoise/seeded --only ivrit_ai_eval_whatsapp
"""

import argparse
import dataclasses
import io
import json
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(HERE)]

from denoise_dataset import BENCHMARK  # noqa: E402  same dataset list and labels as run_bench.py

WHISPER_SAMPLING_RATE = 16000


def build_augmentation(noise_config, noise_dir, apply_prob, band_limit_hz, seed):
    """Same construction as build_eval_augmentation() in the branch's evaluate_model.py."""
    from preprocess.augmentation import resample_augment
    from seeded_aug.eval_augmentation import EvalNoiseAugmentation
    from seeded_aug.noise_augmentation import NoiseAugmenter
    from training.parser import _load_noise_config

    field_names = {field.name for field in dataclasses.fields(NoiseAugmenter)}
    noise_kwargs = {
        key if key in field_names else key.removeprefix("noise_"): value
        for key, value in _load_noise_config(noise_config).items()
    }
    noise_kwargs["noise_dir"] = noise_dir
    noise_kwargs["apply_prob"] = apply_prob
    noise_kwargs["clip_transform"] = partial(resample_augment, target_hz=band_limit_hz)
    augmenter = NoiseAugmenter(target_sampling_rate=WHISPER_SAMPLING_RATE, **noise_kwargs)
    print(f"noise: {len(augmenter.library)} clips from {augmenter.noise_dir}, apply_prob={augmenter.apply_prob}, "
          f"snr_db_range={augmenter.snr_db_range}, band_limit_hz={band_limit_hz}, seed={seed}")
    return EvalNoiseAugmentation(augmenter, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Folder that receives one sub-folder per dataset label")
    parser.add_argument("--only", nargs="+", metavar="LABEL", help="Only these benchmark labels (default: all)")
    parser.add_argument("--noise-config", default="preprocess/noise_config.yaml")
    parser.add_argument("--noise-dir", default="audio-set/test_augs")
    parser.add_argument("--noise-apply-prob", type=float, default=1.0)
    parser.add_argument("--noise-band-limit-hz", type=int, default=8000)
    parser.add_argument("--augmentation-seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import datasets
    import soundfile as sf

    augmentation = build_augmentation(args.noise_config, args.noise_dir, args.noise_apply_prob,
                                      args.noise_band_limit_hz, args.augmentation_seed)
    output = Path(args.output).resolve()
    jobs = [job for job in BENCHMARK if not args.only or job[2] in args.only]

    for spec, config, label in jobs:
        name, split, text_column = spec.split(":")
        parquet = output / label / "data" / f"{split}-00000-of-00001.parquet"
        if parquet.exists() and not args.overwrite:
            print(f"[{label}] already exists, skipping (use --overwrite to redo)")
            continue
        kwargs = {"name": config} if config else {}
        # Loaded and decoded exactly as evaluate_model.py does (it also passes trust_remote_code=True)
        ds = datasets.load_dataset(name, trust_remote_code=True, **kwargs)[split]
        applied = []

        def add_noise(example, index):
            noisy = augmentation(example)
            buffer = io.BytesIO()
            sf.write(buffer, noisy["audio"]["array"], noisy["audio"]["sampling_rate"], format="WAV", subtype="FLOAT")
            applied.append({"index": index, "noise_clips": noisy["augmentation_noise_clips"],
                            "snr_db": noisy["augmentation_snr_db"]})
            return {"audio": {"bytes": buffer.getvalue(), "path": None}}

        ds = ds.map(add_noise, with_indices=True, load_from_cache_file=False,
                    writer_batch_size=16, desc=f"[{label}] adding seeded noise")
        ds = ds.cast_column("audio", datasets.Audio())
        parquet.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(str(parquet))
        (output / label / "augmentation.json").write_text(json.dumps({
            "source": spec + (f" (config {config})" if config else ""),
            "noise_config": args.noise_config, "noise_dir": args.noise_dir, "apply_prob": args.noise_apply_prob,
            "band_limit_hz": args.noise_band_limit_hz, "seed": args.augmentation_seed,
            "code": "origin/feature/benchmark-with-augs @ 70737f5 (vendored in for_eval_denoise/seeded_aug)",
            "per_example": applied,
        }, indent=2, ensure_ascii=False))
        print(f"[{label}] saved {len(ds)} examples to {parquet.parent.parent}")


if __name__ == "__main__":
    main()
