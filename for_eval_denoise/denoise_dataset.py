#!/usr/bin/env python3
"""Denoise the asr-training benchmark datasets with NVIDIA RE-USE, on Apple Silicon (MLX).

For every dataset it writes a local copy with exactly the same split, columns, row order and
audio sampling rate as the original; only the audio samples are replaced by the denoised ones.
asr-training's evaluate_model.py can score the copy with no code changes (see README.md).

    python denoise_dataset.py --output denoised                     # the whole benchmark
    python denoise_dataset.py --output denoised --only fleurs --limit 20
"""

import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same list as asr-training/run_bench.py: (dataset:split:text_column, config name, label).
# The label names the output folder and matches the CSV name run_bench.py gives its results.
BENCHMARK = [
    ("ivrit-ai/eval-d1:test:text", None, "ivrit_ai_eval_d1"),
    ("ivrit-ai/eval-whatsapp:test:text", None, "ivrit_ai_eval_whatsapp"),
    ("upai-inc/saspeech:test:text", None, "saspeech"),
    ("google/fleurs:test:transcription", "he_il", "fleurs"),
    ("imvladikon/hebrew_speech_kan:validation:sentence", None, "hebrew_speech_kan"),
]

VARIANTS = {
    "reuse": "NVIDIA-NemotronLabs-RE-USE",
    "laughter": "NVIDIA-NemotronLabs-RE-USE-laughter",
}


def load_enhancer(variant, mlx_dir):
    """Returns an mlx-speech REUSEEnhancer; converts NVIDIA's PyTorch weights to MLX on first use."""
    from mlx_speech.generation.reuse import REUSEEnhancer

    mlx_dir = Path(mlx_dir) / variant
    if not (mlx_dir / "model.safetensors").exists():
        # Same as running convert_weights.py: downloads nvidia/<repo> if it isn't next to this script.
        import convert_weights

        source = HERE / VARIANTS[variant]
        if not (source / "model.safetensors").exists():
            from huggingface_hub import snapshot_download

            snapshot_download(f"nvidia/{VARIANTS[variant]}", local_dir=str(source),
                              allow_patterns=["model.safetensors", "config.json"])
        print(f"Converting {source} -> {mlx_dir}")
        convert_weights.convert(source, mlx_dir)
    return REUSEEnhancer.from_dir(mlx_dir)


def eager_chunked_hann_ola():
    """mlx_speech's chunked_hann_ola, with the result evaluated after every chunk.

    The library loop is lazy: for a 5-min 44.1 kHz clip it queues all ~120 chunks before computing, keeping every chunk's
    intermediates alive at once (77 GB and climbing on ivrit_ai_eval_d1). Same operations in the same order, so the output
    is unchanged; only peak memory drops to one chunk.
    """
    import math

    import mlx.core as mx
    from mlx_speech.models.reuse import stft

    def chunked_hann_ola(process, wave, chunk_size, *, hop_portion=0.5, pad_value=1e-8):
        squeeze = wave.ndim == 1
        if squeeze:
            wave = wave[None, :]
        wave = wave.astype(mx.float32)
        total = wave.shape[-1]
        hop_length = int(hop_portion * chunk_size)
        if hop_length < 1:
            raise ValueError("hop_portion * chunk_size must be >= 1")
        window = stft.hann_window(chunk_size)
        n_chunks = max(1, math.ceil((total - chunk_size) / hop_length) + 1) if total > chunk_size else 1

        enhanced = mx.zeros_like(wave)
        window_sum = mx.zeros_like(wave)
        for i in range(n_chunks):
            start = i * hop_length
            end = min(start + chunk_size, total)
            chunk = wave[:, start:end]
            if chunk.shape[-1] < 2:  # skip degenerate tail
                continue
            out = stft._pad_or_trim_to_match(chunk, process(chunk), pad_value)
            seg = out.shape[-1]
            w_slice = window[:seg]
            cols = mx.arange(seg)
            enhanced[:, start + cols] += out * w_slice[None, :]
            window_sum[:, start + cols] += w_slice[None, :]
            mx.eval(enhanced, window_sum)  # the only change vs. the library

        mask = window_sum > 1e-8
        enhanced = mx.where(mask, enhanced / mx.maximum(window_sum, stft._EPS), enhanced)
        return enhanced[0] if squeeze else enhanced

    return chunked_hann_ola


def make_denoiser(enhancer, chunk_seconds):
    import mlx.core as mx
    import mlx_speech.generation.reuse as reuse_module
    import numpy as np

    reuse_module.chunked_hann_ola = eager_chunked_hann_ola()  # enhance() looks it up in this module

    def denoise(array, sampling_rate):
        """float mono numpy array -> denoised float32 numpy array of the same length."""
        out = enhancer.enhance(mx.array(np.asarray(array, dtype=np.float32)), sampling_rate, chunk_size_s=chunk_seconds)
        mx.eval(out)
        cleaned = np.array(out)
        # Every clip ends in a different-sized chunk, so MLX's buffer cache keeps growing until macOS kills the
        # process (happened at clip 54/54 after 95 min). Free it after each clip.
        del out
        mx.clear_cache()
        return cleaned

    return denoise


def encode_flac(array, sampling_rate):
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, array, sampling_rate, format="FLAC", subtype="PCM_24")
    return buffer.getvalue()


def denoise_dataset(spec, config, label, denoise, output, limit, overwrite):
    import datasets
    import soundfile as sf

    name, split, text_column = (spec.split(":") + [None, None])[:3]
    split = split or "test"
    parquet = output / label / "data" / f"{split}-00000-of-00001.parquet"
    if parquet.exists() and not overwrite:
        print(f"[{label}] already exists, skipping (use --overwrite to redo)")
        return

    print(f"[{label}] loading {name}" + (f" ({config})" if config else ""))
    kwargs = {"name": config} if config else {}
    ds = datasets.load_dataset(name, trust_remote_code=True, **kwargs)[split]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # Decode the audio ourselves (soundfile) so the original feature type can be restored on save.
    audio_feature = ds.features["audio"]
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    import mlx.core as mx

    # Each denoised clip is also written here as soon as it is done, so an interrupted run resumes where it stopped.
    # Kept outside the dataset folder so load_dataset(<folder>) only sees the parquet; --overwrite starts it fresh.
    clip_cache = output / ".clips" / label
    if overwrite:
        shutil.rmtree(clip_cache, ignore_errors=True)
    clip_cache.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def process(example, index):
        cached = clip_cache / f"{index:06d}.flac"
        if cached.exists():
            return {"audio": {"bytes": cached.read_bytes(), "path": None}}
        audio = example["audio"]
        source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
        array, sampling_rate = sf.read(source, dtype="float32", always_2d=True)
        mono = array.mean(axis=1)
        clip_start = time.time()
        cleaned = denoise(mono, sampling_rate)
        flac = encode_flac(cleaned, sampling_rate)
        cached.with_suffix(".tmp").write_bytes(flac)
        cached.with_suffix(".tmp").rename(cached)  # never leave a half-written clip behind
        # One line per clip, so progress, speed and memory are visible in a log file
        done = index + 1
        elapsed = time.time() - started
        eta = elapsed / done * (len(ds) - done)
        print(f"[{label}] {done}/{len(ds)} · {len(mono) / sampling_rate:.1f}s audio @{sampling_rate}Hz · "
              f"took {time.time() - clip_start:.0f}s · elapsed {elapsed / 60:.1f}m · ETA {eta / 60:.0f}m · "
              f"mlx peak {mx.get_peak_memory() / 2**30:.1f}GB", flush=True)
        mx.reset_peak_memory()
        return {"audio": {"bytes": flac, "path": None}}

    ds = ds.map(process, with_indices=True, load_from_cache_file=False, writer_batch_size=16,
                desc=f"[{label}] denoising")
    ds = ds.cast_column("audio", audio_feature)

    parquet.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(parquet))
    print(f"[{label}] saved {len(ds)} examples to {parquet.parent.parent}")
    return {
        "label": label,
        "source": f"{spec}" + (f" (config {config})" if config else ""),
        "path": str(parquet.parent.parent),
        "evaluate_model_dataset_arg": f"{parquet.parent.parent}:{split}:{text_column}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Folder that receives one sub-folder per dataset label")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="reuse",
                        help="reuse = general speech enhancement, laughter = better laughter preservation")
    parser.add_argument("--only", nargs="+", metavar="LABEL", help="Only these benchmark labels (default: all)")
    parser.add_argument("--dataset", action="append", metavar="SPEC[@CONFIG]=LABEL",
                        help="Extra dataset instead of the benchmark list, e.g. org/ds:test:text=my_label "
                             "(add @config before '=' for a dataset config name). Repeatable.")
    parser.add_argument("--chunk-seconds", type=float, default=5.0,
                        help="Denoise in overlapping chunks of this length (NVIDIA's chunked default is 5)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N examples of each dataset (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mlx-weights", default=str(HERE / "mlx_weights"), help="Where converted MLX weights are cached")
    args = parser.parse_args()

    if args.dataset:
        jobs = []
        for item in args.dataset:
            left, label = item.rsplit("=", 1)
            spec, _, config = left.partition("@")
            jobs.append((spec, config or None, label))
    else:
        jobs = [job for job in BENCHMARK if not args.only or job[2] in args.only]
    if not jobs:
        sys.exit(f"nothing to do; known labels: {[job[2] for job in BENCHMARK]}")

    enhancer = load_enhancer(args.variant, args.mlx_weights)
    denoise = make_denoiser(enhancer, args.chunk_seconds)

    output = Path(args.output).resolve()
    manifest_path = output / "denoised_datasets.json"
    manifest = {}
    if manifest_path.exists():
        manifest = {entry["label"]: entry for entry in json.loads(manifest_path.read_text())}

    failed = []
    for spec, config, label in jobs:
        try:
            entry = denoise_dataset(spec, config, label, denoise, output, args.limit, args.overwrite)
        except Exception as exc:  # keep going like run_bench.py does
            print(f"[{label}] FAILED: {exc!r}")
            failed.append(label)
            continue
        if entry:
            manifest[label] = {**entry, "variant": args.variant, "chunk_seconds": args.chunk_seconds}

    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(list(manifest.values()), indent=2))
    print(f"Manifest: {manifest_path}")
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
