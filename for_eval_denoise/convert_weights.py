#!/usr/bin/env python3
"""Convert NVIDIA RE-USE PyTorch weights (model.safetensors) to the MLX layout used by mlx-speech.

    python convert_weights.py                      # reuse   -> mlx_weights/reuse/model.safetensors
    python convert_weights.py --variant laughter   # laughter -> mlx_weights/laughter/model.safetensors

Only renames keys and transposes conv weights (torch channel-first -> MLX channel-last); the learned
values are untouched. Mapping follows mlx-speech's models/reuse/loader.py (MIT).
Needs: mlx, numpy, safetensors.
"""

import argparse
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
VARIANTS = {"reuse": "NVIDIA-NemotronLabs-RE-USE", "laughter": "NVIDIA-NemotronLabs-RE-USE-laughter"}

SEQ_SUFFIX = {"0": "conv", "1": "norm", "2": "act"}
CONVNORMACT = re.compile(
    r"^(dense_encoder\.dense_conv_[12]"
    r"|(?:dense_encoder|mask_decoder|phase_decoder)\.dense_block\.dense_block\.\d+)"
    r"\.([012])\.(weight|bias)$"
)
UPCONV = re.compile(r"^((?:mask_decoder|phase_decoder)\.up_conv[12])\.([012])\.(.+)$")


def torch_key_to_mlx(key):
    m = CONVNORMACT.match(key)
    if m:
        prefix, idx, leaf = m.groups()
        return f"{prefix}.{SEQ_SUFFIX[idx]}.{leaf}"
    m = UPCONV.match(key)
    if m:
        prefix, idx, rest = m.groups()
        return f"{prefix}.{SEQ_SUFFIX[idx]}.{rest}"
    return key


def convert_tensor(mlx_key, value):
    arr = np.asarray(value)
    if mlx_key.endswith("conv1d.weight"):  # depthwise Conv1d [out, in/groups, k] -> [out, k, in/groups]
        arr = np.transpose(arr, (0, 2, 1))
    elif arr.ndim == 4 and mlx_key.endswith(".weight"):  # Conv2d [out, in, kh, kw] -> [out, kh, kw, in]
        arr = np.transpose(arr, (0, 2, 3, 1))
    return mx.array(arr)


def convert(source_dir, output_dir):
    state = {}
    with safe_open(str(source_dir / "model.safetensors"), "numpy") as f:
        for key in f.keys():
            mlx_key = torch_key_to_mlx(key)
            state[mlx_key] = convert_tensor(mlx_key, f.get_tensor(key))
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_dir / "model.safetensors"), state, metadata={"format": "mlx"})
    if (source_dir / "config.json").is_file():
        shutil.copy2(source_dir / "config.json", output_dir / "config.json")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="reuse")
    parser.add_argument("--mlx-weights", default=str(HERE / "mlx_weights"))
    args = parser.parse_args()

    source = HERE / VARIANTS[args.variant]
    if not (source / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(f"nvidia/{VARIANTS[args.variant]}", local_dir=str(source),
                          allow_patterns=["model.safetensors", "config.json"])
    output = Path(args.mlx_weights) / args.variant
    state = convert(source, output)
    print(f"Converted {len(state)} tensors -> {output / 'model.safetensors'}")


if __name__ == "__main__":
    main()
