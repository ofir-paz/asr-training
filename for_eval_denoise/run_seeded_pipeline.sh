#!/usr/bin/env bash
# Seeded-noise ("-s") audio -> RE-USE denoised -> evaluation, for one model on the WhatsApp benchmark.
# Every step skips work that already exists, so the script can be re-run after an interruption.
#
#   ./for_eval_denoise/run_seeded_pipeline.sh          (from anywhere; uses both venvs by full path)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAIN_PY="$ROOT/.venv/bin/python"                     # Python 3.10: evaluation + noise augmentation
export PATH="$ROOT/.venv/bin:$PATH"                  # run_bench.py starts ./evaluate_model.py via "env python3"
DENOISE_PY="$ROOT/for_eval_denoise/.venv/bin/python"  # Python 3.14: mlx-speech denoiser
LABEL=ivrit_ai_eval_whatsapp
MODEL=/Users/Shared/asr/training/lv3-mix-na-b32-kd-low-smooth-mlx
NAME="$(basename "$MODEL")-s"
RESULTS=/Users/Roi/Desktop/results_2

echo "=== 1/4 seeded-noise dataset -> for_eval_denoise/seeded/$LABEL"
"$MAIN_PY" for_eval_denoise/make_seeded_dataset.py --output for_eval_denoise/seeded --only "$LABEL"

echo "=== 2/4 sanity check: evaluate on the saved seeded audio -> $RESULTS/seeded_rerun/$NAME"
"$MAIN_PY" run_bench.py --engine engines/mlx_engine.py --model "$MODEL" \
  --output-dir "$RESULTS/seeded_rerun/$NAME" --denoised-dir for_eval_denoise/seeded \
  --parallel-mode process --devices gpu,gpu,gpu,gpu --batch-size 1

echo "=== 3/4 denoise the seeded audio -> for_eval_denoise/denoised_seeded/$LABEL"
(cd for_eval_denoise && "$DENOISE_PY" -u denoise_dataset.py --output denoised_seeded \
  --dataset "$ROOT/for_eval_denoise/seeded/$LABEL:test:text=$LABEL")

echo "=== 4/4 evaluate on the denoised seeded audio -> $RESULTS/seeded_denoised/$NAME"
"$MAIN_PY" run_bench.py --engine engines/mlx_engine.py --model "$MODEL" \
  --output-dir "$RESULTS/seeded_denoised/$NAME" --denoised-dir for_eval_denoise/denoised_seeded \
  --parallel-mode process --devices gpu,gpu,gpu,gpu --batch-size 1

echo "=== done"
