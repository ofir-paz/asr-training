#!/usr/bin/env python3
"""Convert a Hugging Face (transformers) Whisper checkpoint to mlx-whisper format.

The source directory is only read; output goes to a new directory.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_whisper.whisper import ModelDimensions, Whisper

KEY_RENAMES = [
    ("model.", ""),
    (".layers", ".blocks"),
    (".self_attn", ".attn"),
    (".attn_layer_norm", ".attn_ln"),
    (".encoder_attn.", ".cross_attn."),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".final_layer_norm", ".mlp_ln"),
    (".q_proj", ".query"),
    (".k_proj", ".key"),
    (".v_proj", ".value"),
    (".out_proj", ".out"),
    (".fc1", ".mlp1"),
    (".fc2", ".mlp2"),
    ("embed_positions.weight", "positional_embedding"),
    ("decoder.embed_tokens", "decoder.token_embedding"),
    ("encoder.layer_norm", "encoder.ln_post"),
    ("decoder.layer_norm", "decoder.ln"),
]


def remap_key(key: str) -> str:
    for old, new in KEY_RENAMES:
        key = key.replace(old, new)
    return key


def load_hf_weights(src: Path) -> dict:
    index_file = src / "model.safetensors.index.json"
    if index_file.exists():
        shards = sorted(set(json.load(open(index_file))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    weights = {}
    for shard in shards:
        weights.update(mx.load(str(src / shard)))
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, help="Source HF checkpoint directory (read only)")
    parser.add_argument("--mlx-path", required=True, help="New output directory for the MLX model")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    src = Path(args.hf_model).resolve()
    dst = Path(args.mlx_path).resolve()
    if dst == src or src in dst.parents:
        sys.exit("Refusing: --mlx-path must not be the source directory or inside it")
    if dst.exists():
        sys.exit(f"Refusing: {dst} already exists; choose a new directory")
    dtype = getattr(mx, args.dtype)

    hf = json.load(open(src / "config.json"))
    dims = ModelDimensions(
        n_mels=hf["num_mel_bins"],
        n_audio_ctx=hf["max_source_positions"],
        n_audio_state=hf["d_model"],
        n_audio_head=hf["encoder_attention_heads"],
        n_audio_layer=hf["encoder_layers"],
        n_vocab=hf["vocab_size"],
        n_text_ctx=hf["max_target_positions"],
        n_text_state=hf["d_model"],
        n_text_head=hf["decoder_attention_heads"],
        n_text_layer=hf["decoder_layers"],
    )

    raw = load_hf_weights(src)
    print(f"Loaded {len(raw)} tensors from {src}")

    # proj_out is tied to the token embedding; MLX computes encoder positions as sinusoids
    raw.pop("proj_out.weight", None)
    weights = {}
    for key, value in raw.items():
        key = remap_key(key)
        if key == "encoder.positional_embedding":
            continue
        if "conv" in key and value.ndim == 3:
            value = value.swapaxes(1, 2)
        weights[key] = value.astype(dtype)

    model = Whisper(dims, dtype)
    expected = {k for k, _ in tree_flatten(model.parameters())} - {"alignment_heads"}
    missing, extra = expected - weights.keys(), weights.keys() - expected
    if missing or extra:
        sys.exit(f"Key mismatch\nmissing: {sorted(missing)[:20]}\nextra: {sorted(extra)[:20]}")

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    dst.mkdir(parents=True)
    mx.save_safetensors(str(dst / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    config = asdict(dims)
    config["model_type"] = "whisper"
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved MLX model to {dst}")


if __name__ == "__main__":
    main()
