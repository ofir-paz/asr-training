"""Find what grows in memory while denoising clip after clip (the full runs grew ~3 GB per clip until macOS killed them).

    for_eval_denoise/.venv/bin/python for_eval_denoise/debug_memory.py
Prints, per clip: process footprint (macOS `footprint`), MLX active / cache memory.
"""
import os
import re
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_speech.generation.reuse import REUSEEnhancer

HERE = Path(__file__).resolve().parent


def footprint_gb():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"Footprint: ([0-9.]+) ([GMK]B)", out)
    return float(m.group(1)) / {"GB": 1, "MB": 1024, "KB": 1024**2}[m.group(2)] if m else float("nan")


enhancer = REUSEEnhancer.from_dir(HERE / "mlx_weights" / "reuse")
rng = np.random.default_rng(0)
for i, seconds in enumerate([12.3, 17.9, 9.4, 21.1, 14.6, 11.2, 19.8, 8.7]):
    wave = (0.05 * rng.standard_normal(int(seconds * 16000))).astype(np.float32)
    out = enhancer.enhance(mx.array(wave), 16000, chunk_size_s=5.0)
    mx.eval(out)
    cleaned = np.array(out)
    del out
    mx.clear_cache()
    print(f"clip {i} ({seconds}s): footprint {footprint_gb():.1f} GB | mlx active {mx.get_active_memory() / 2**30:.2f} GB, "
          f"cache {mx.get_cache_memory() / 2**30:.2f} GB", flush=True)
