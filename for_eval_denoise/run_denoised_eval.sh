#!/usr/bin/env bash
# Evaluate the models on the denoised benchmark copies (only datasets present in denoised/ are run).
# Results land next to the clean runs, one folder per model with the same name as in results_2/clean,
# so model_comparison.ipynb can pair clean vs denoised per model.
#
#   ./for_eval_denoise/run_denoised_eval.sh            (from the asr-training root, main .venv active)
set -euo pipefail

cd "$(dirname "$0")/.."
DENOISED_DIR=for_eval_denoise/denoised
OUT_ROOT=/Users/Roi/Desktop/results_2/clean_denoised
# Byte-identical to /Users/Shared/asr/mlx/lv3-mix-na-b32-kd-low-smooth-mlx used by the results_2/clean run
MODELS=(
  /Users/Shared/asr/training/lv3-mix-na-b32-kd-low-smooth-mlx
)

for model in "${MODELS[@]}"; do
  name="$(basename "$model")-c"
  echo "=== $name ==="
  python run_bench.py --engine engines/mlx_engine.py --model "$model" \
    --output-dir "$OUT_ROOT/$name" --denoised-dir "$DENOISED_DIR" \
    --parallel-mode process --devices gpu,gpu,gpu,gpu --batch-size 1
done
