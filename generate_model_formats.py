#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
import shutil
import tempfile
import git

def show_help(parser):
    parser.print_help()
    sys.exit(1)

def check_command(command):
    """Check if a command is available in the system."""
    return shutil.which(command) is not None

def check_git():
    """Check if git is installed."""
    if not check_command("git"):
        print("Error: git is not installed")
        print("Please install git before proceeding")
        sys.exit(1)

def clone_repo(repo_url, target_dir):
    """Clone a git repository."""
    try:
        if not os.path.exists(target_dir):
            print(f"Cloning {repo_url} to {target_dir}...")
            git.Repo.clone_from(repo_url, target_dir)
        else:
            print(f"Repository directory {target_dir} already exists, skipping clone.")
        return True
    except Exception as e:
        print(f"Error cloning repository {repo_url}: {str(e)}")
        return False

def convert_to_ct2(model_name, output_dir, quantization):
    """Convert a model to CTranslate2 format.
    
    Uses the --force flag to overwrite existing output directory if it exists.
    """
    if not check_command("ct2-transformers-converter"):
        print("Error: ct2-transformers-converter is not installed")
        print("Please install it using: pip install ctranslate2")
        sys.exit(1)
    
    print(f"Converting model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {quantization}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        "ct2-transformers-converter",
        "--model", model_name,
        "--output_dir", output_dir,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
        "--quantization", quantization,
        "--force"
    ]
    
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print("Conversion to CT2 completed successfully!")
        print(f"Converted model is available at: {output_dir}")
        return True
    else:
        print("Error: CT2 conversion failed")
        return False

def convert_to_onnx(model_name, output_dir):
    """Convert a model to ONNX format using optimum-cli."""
    if not check_command("optimum-cli"):
        print("Error: optimum-cli is not installed")
        print("Please install it using: pip install optimum[exporters]")
        return False
    
    print(f"Converting model: {model_name}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    command = [
        "optimum-cli", "export", "onnx",
        "--task", "automatic-speech-recognition",
        "--device", "cpu",
        "--model", model_name,
        output_dir
    ]
    
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command)
    
    if result.returncode == 0:
        print("Conversion to ONNX completed successfully!")
        print(f"Converted model is available at: {output_dir}")
        return True
    else:
        print("Error: ONNX conversion failed")
        return False

def convert_to_ggml(model_name, output_dir):
    """Convert a model to GGML format."""
    check_git()
    
    # Create temporary directory for cloning repositories
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    try:
        # Clone required repositories
        whisper_repo = os.path.join(temp_dir, "whisper")
        whisper_cpp_repo = os.path.join(temp_dir, "whisper.cpp")
        model_dir = os.path.join(temp_dir, "whisper-model")
        
        repos_cloned = True
        repos_cloned &= clone_repo("https://github.com/openai/whisper", whisper_repo)
        repos_cloned &= clone_repo("https://github.com/ggml-org/whisper.cpp", whisper_cpp_repo)
        repos_cloned &= clone_repo(f"https://huggingface.co/{model_name}", model_dir)
        
        if not repos_cloned:
            print("Error: Failed to clone one or more repositories")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the conversion script
        print(f"Converting model to GGML format...")
        convert_script = os.path.join(whisper_cpp_repo, "models", "convert-h5-to-ggml.py")
        
        command = [
            "python3", convert_script, model_dir, whisper_repo, "."
        ]
        
        # Change to the output directory to run the conversion
        original_dir = os.getcwd()
        os.chdir(output_dir)
        
        result = subprocess.run(command)
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("Conversion to GGML completed successfully!")
            print(f"Converted model is available at: {output_dir}")
            return True
        else:
            print("Error: GGML conversion failed")
            return False
    
    finally:
        # Clean up temp directory
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

# HF Whisper parameter-name fragments -> the OpenAI-style names mlx_whisper loads, applied in order.
_MLX_NAME_REPLACEMENTS = [
    ("model.", ""),
    (".layers.", ".blocks."),
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("decoder.layer_norm.", "decoder.ln."),
    ("decoder.embed_tokens.", "decoder.token_embedding."),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
    ("self_attn_layer_norm", "attn_ln"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("final_layer_norm", "mlp_ln"),
    ("self_attn.", "attn."),
    ("encoder_attn.", "cross_attn."),
    ("q_proj", "query"),
    ("k_proj", "key"),
    ("v_proj", "value"),
    ("out_proj", "out"),
    ("fc1", "mlp1"),
    ("fc2", "mlp2"),
]


def hf_to_mlx_name(name):
    for old, new in _MLX_NAME_REPLACEMENTS:
        name = name.replace(old, new)
    return name


def convert_to_mlx(model_name, output_dir):
    """Convert a Hugging Face Whisper checkpoint to the mlx_whisper format (config.json +
    fp16 weights.safetensors), which mlx_whisper needs - it cannot load HF checkpoints."""
    import numpy as np
    import torch
    from safetensors.numpy import save_file
    from transformers import WhisperForConditionalGeneration

    print(f"Converting model: {model_name}")
    print(f"Output directory: {output_dir}")

    model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
    config = model.config

    weights = {}
    for name, tensor in model.state_dict().items():
        # proj_out is tied to the token embedding; the encoder's positional embedding is a fixed
        # sinusoid that mlx_whisper recomputes, so neither is stored.
        if name == "proj_out.weight" or name == "model.encoder.embed_positions.weight":
            continue
        array = tensor.numpy()
        if name.startswith("model.encoder.conv"):
            if array.ndim == 3:  # torch Conv1d (out, in, kernel) -> MLX (out, kernel, in)
                array = array.transpose(0, 2, 1)
        weights[hf_to_mlx_name(name)] = np.ascontiguousarray(array.astype(np.float16))

    # mlx_whisper's default alignment heads: every head of the last half of the decoder layers.
    first_layer = config.decoder_layers // 2
    weights["alignment_heads"] = np.array(
        [[layer, head] for layer in range(first_layer, config.decoder_layers)
         for head in range(config.decoder_attention_heads)],
        dtype=np.int64,
    )

    mlx_config = {
        "n_mels": config.num_mel_bins,
        "n_audio_ctx": config.max_source_positions,
        "n_audio_state": config.d_model,
        "n_audio_head": config.encoder_attention_heads,
        "n_audio_layer": config.encoder_layers,
        "n_vocab": config.vocab_size,
        "n_text_ctx": config.max_target_positions,
        "n_text_state": config.d_model,
        "n_text_head": config.decoder_attention_heads,
        "n_text_layer": config.decoder_layers,
        "model_type": "whisper",
    }

    os.makedirs(output_dir, exist_ok=True)
    save_file(weights, os.path.join(output_dir, "weights.safetensors"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(mlx_config, f, indent=4)

    print("Conversion to MLX completed successfully!")
    print(f"Converted model is available at: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert a model to different formats")
    parser.add_argument("-m", "--model", default="openai/whisper-large-v3", 
                        help="Model name or path (default: openai/whisper-large-v3)")
    parser.add_argument("-o", "--output", help="Base output directory (default: model-name)")
    parser.add_argument("-f", "--formats", default=None, 
                        help="Comma-separated list of output formats (default: all, options: ct2,onnx,ggml,mlx)")
    parser.add_argument("-q", "--quant", default="float16",
                        help="Quantization type for CT2 format (default: float16)")
    
    args = parser.parse_args()
    
    model_name = args.model
    
    # Define valid formats
    valid_formats = ["ct2", "onnx", "ggml", "mlx"]
    
    # If no formats specified, use all formats
    if args.formats is None:
        format_list = valid_formats
        print(f"No formats specified, generating all formats: {', '.join(format_list)}")
    else:
        format_list = [fmt.strip() for fmt in args.formats.split(",")]
    
    # Validate formats
    for fmt in format_list:
        if fmt not in valid_formats:
            print(f"Error: Unsupported format {fmt}")
            print(f"Supported formats: {', '.join(valid_formats)}")
            show_help(parser)
    
    # Set base output directory if not specified
    if args.output:
        base_output_dir = args.output
    else:
        model_basename = os.path.basename(model_name)
        base_output_dir = f"{model_basename}"
    
    # Process each format
    results = {}
    
    for fmt in format_list:
        # Create format-specific subdirectory
        output_dir = os.path.join(base_output_dir, fmt)
        
        print(f"\n=== Converting to {fmt.upper()} format ===")
        
        # Convert to the specified format
        if fmt == "ct2":
            results[fmt] = convert_to_ct2(model_name, output_dir, args.quant)
        elif fmt == "onnx":
            results[fmt] = convert_to_onnx(model_name, output_dir)
        elif fmt == "ggml":
            results[fmt] = convert_to_ggml(model_name, output_dir)
        elif fmt == "mlx":
            results[fmt] = convert_to_mlx(model_name, output_dir)
    
    # Print summary
    print("\n=== Conversion Summary ===")
    for fmt, success in results.items():
        status = "Success" if success else "Failed"
        print(f"{fmt.upper()}: {status}")
    
    # Check if any conversion failed
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()