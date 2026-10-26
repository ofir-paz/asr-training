"""Benchmark: mlx_speech enhance() (one chunk at a time) vs batched chunks, on an eval-d1 excerpt.

    python bench_batch.py [SECONDS] [BATCH,BATCH,...]      e.g. python bench_batch.py 30 2,4,8
"""
import io, sys, time, math
import numpy as np, soundfile as sf, datasets, mlx.core as mx
from mlx_speech.generation.reuse import REUSEEnhancer, _COMPRESS_FACTOR
from mlx_speech.models.reuse.stft import (hann_window, mag_phase_stft, mag_phase_istft,
                                          stft_params_for, sweep_artifact_filter)

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30
BATCHES = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4, 8]
ARROW = "/Users/Shared/hf-home/datasets/ivrit-ai___eval-d1/default/0.0.0/7899b0427df2e1d722c043ac41211ac7da15c737/eval-d1-test.arrow"

ds = datasets.Dataset.from_file(ARROW).cast_column("audio", datasets.Audio(decode=False))
arr, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]), dtype="float32", always_2d=True)
wave = arr.mean(axis=1)[60 * sr: 60 * sr + int(SECONDS * sr)]  # skip the intro
enh = REUSEEnhancer.from_dir("/Users/Roi/Desktop/asr-training/for_eval_denoise/mlx_weights/reuse")
CHUNK_S = 5.0


def batched(wave, sr, batch):
    n_fft, hop, win = stft_params_for(sr)
    chunk = int(CHUNK_S * sr); step = int(0.5 * chunk); total = len(wave)
    n = max(1, math.ceil((total - chunk) / step) + 1) if total > chunk else 1
    spans = [(i * step, min(i * step + chunk, total)) for i in range(n)]
    spans = [s for s in spans if s[1] - s[0] >= 2]
    window = np.array(hann_window(chunk))
    out = np.zeros(total, np.float32); wsum = np.zeros(total, np.float32)
    by_len = {}
    for s in spans:
        by_len.setdefault(s[1] - s[0], []).append(s)
    for length, group in by_len.items():
        for k in range(0, len(group), batch):
            g = group[k:k + batch]
            x = mx.array(np.stack([wave[a:b] for a, b in g]))
            mag, pha = mag_phase_stft(x, n_fft, hop, win, compress_factor=_COMPRESS_FACTOR)
            amp, ph, _ = enh.model(mag, pha)
            y = mag_phase_istft(sweep_artifact_filter(amp), ph, n_fft, hop, win, compress_factor=_COMPRESS_FACTOR)
            y = np.array(y)
            if y.shape[1] >= length:
                y = y[:, :length]
            else:
                y = np.concatenate([y, np.full((len(g), length - y.shape[1]), 1e-8, np.float32)], 1)
            for (a, b), row in zip(g, y):
                out[a:b] += row * window[:length]; wsum[a:b] += window[:length]
    mask = wsum > 1e-8
    out[mask] /= np.maximum(wsum[mask], 1e-8)
    return np.clip(out, -1, 1)


t = time.time(); ref = enh.enhance(mx.array(wave), sr, chunk_size_s=CHUNK_S); mx.eval(ref); ref = np.array(ref)
base = time.time() - t
print(f"{SECONDS:.0f}s audio @ {sr} Hz | original: {base:.1f}s ({base / SECONDS:.2f}s per audio-s)", flush=True)
for b in BATCHES:
    t = time.time(); got = batched(wave, sr, b); el = time.time() - t
    print(f"batch {b:>2}: {el:.1f}s ({el / SECONDS:.2f}s per audio-s, x{base / el:.2f}) "
          f"max|diff|={np.abs(got - ref).max():.2e} corr={np.corrcoef(got, ref)[0, 1]:.6f} "
          f"peak_mem={mx.get_peak_memory() / 2**30:.1f}GB", flush=True)
    mx.reset_peak_memory()
