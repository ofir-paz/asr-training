"""Check that denoise_dataset.py's eager chunk loop gives the same output as mlx_speech's library loop, with less memory.

    for_eval_denoise/.venv/bin/python for_eval_denoise/verify_eager_ola.py [SECONDS]
Uses a 44.1 kHz stereo eval-d1 excerpt (mixed to mono, like the pipeline).
"""
import io
import sys
import time
from pathlib import Path

import datasets
import mlx.core as mx
import mlx_speech.generation.reuse as reuse_module
import numpy as np
import soundfile as sf
from mlx_speech.generation.reuse import REUSEEnhancer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from denoise_dataset import eager_chunked_hann_ola  # noqa: E402

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 60
ds = datasets.load_dataset("ivrit-ai/eval-d1")["test"].cast_column("audio", datasets.Audio(decode=False))
array, sr = sf.read(io.BytesIO(ds[0]["audio"]["bytes"]), dtype="float32", always_2d=True)
wave = mx.array(array.mean(axis=1)[60 * sr: 60 * sr + int(SECONDS * sr)])
enhancer = REUSEEnhancer.from_dir(HERE / "mlx_weights" / "reuse")

results = {}
library_loop = reuse_module.chunked_hann_ola
for name, loop in [("library", library_loop), ("eager", eager_chunked_hann_ola())]:
    reuse_module.chunked_hann_ola = loop
    mx.clear_cache()
    mx.reset_peak_memory()
    t = time.time()
    out = enhancer.enhance(wave, sr, chunk_size_s=5.0)
    mx.eval(out)
    results[name] = np.array(out)
    print(f"{name:>7}: {time.time() - t:.0f}s, mlx peak {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
reuse_module.chunked_hann_ola = library_loop

a, b = results["library"], results["eager"]
print(f"{SECONDS:.0f}s @ {sr} Hz: bit-identical={np.array_equal(a, b)}, max|diff|={np.abs(a - b).max():.2e}")
