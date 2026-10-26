#!/usr/bin/env bash
# Full denoising experiment for ONE benchmark dataset and ONE model:
#   clean:  denoise original audio            -> evaluate  (results_2/clean_denoised/<model>-c/<label>.csv)
#   seeded: re-create the "-s" noisy audio    -> verify it matches the original -s runs
#           -> evaluate it as-is (same-session reference, results_2/seeded_rerun/<model>-s/<label>.csv)
#           -> denoise it -> evaluate         (results_2/seeded_denoised/<model>-s/<label>.csv)
# Every step skips work whose output already exists (denoising also resumes per clip), so re-running after a crash is safe.
#
#   ./for_eval_denoise/run_dataset_pipeline.sh <label> [model_path]
#   labels: ivrit_ai_eval_whatsapp ivrit_ai_eval_d1 hebrew_speech_kan fleurs saspeech
set -euo pipefail

LABEL="$1"
MODEL="${2:-/Users/Shared/asr/training/lv3-mix-na-b32-kd-low-smooth-mlx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MAIN_PY="$ROOT/.venv/bin/python"                     # Python 3.10: evaluation + noise augmentation
export PATH="$ROOT/.venv/bin:$PATH"                  # run_bench.py starts ./evaluate_model.py via "env python3"
DENOISE_PY="$ROOT/for_eval_denoise/.venv/bin/python"  # Python 3.14: mlx-speech denoiser
RESULTS=/Users/Roi/Desktop/results_2
BASE="$(basename "$MODEL")"
case "$LABEL" in
  fleurs)            SPLIT=test;       TEXT=transcription ;;
  hebrew_speech_kan) SPLIT=validation; TEXT=sentence ;;
  *)                 SPLIT=test;       TEXT=text ;;
esac

evaluate() {  # evaluate() <folder with one sub-folder per label> <output dir>
  "$MAIN_PY" run_bench.py --engine engines/mlx_engine.py --model "$MODEL" \
    --output-dir "$2" --denoised-dir "$1" \
    --parallel-mode process --devices gpu,gpu,gpu,gpu --batch-size 1
}

echo "=== [$LABEL] 1/6 $(date '+%F %T') denoise clean audio -> for_eval_denoise/denoised/$LABEL"
(cd for_eval_denoise && "$DENOISE_PY" -u denoise_dataset.py --output denoised --only "$LABEL")

echo "=== [$LABEL] 2/6 $(date '+%F %T') evaluate clean denoised -> $RESULTS/clean_denoised/$BASE-c/$LABEL.csv"
evaluate for_eval_denoise/denoised "$RESULTS/clean_denoised/$BASE-c"

echo "=== [$LABEL] 3/6 $(date '+%F %T') seeded-noise audio -> for_eval_denoise/seeded/$LABEL (+ verify vs original -s runs)"
"$MAIN_PY" for_eval_denoise/make_seeded_dataset.py --output for_eval_denoise/seeded --only "$LABEL"
if ! "$MAIN_PY" for_eval_denoise/verify_seeded.py "$LABEL"; then
  echo "=== [$LABEL] seeded audio does NOT match the original -s runs; skipping the seeded steps for this dataset"
  exit 0
fi

echo "=== [$LABEL] 4/6 $(date '+%F %T') evaluate seeded as-is -> $RESULTS/seeded_rerun/$BASE-s/$LABEL.csv"
evaluate for_eval_denoise/seeded "$RESULTS/seeded_rerun/$BASE-s"

echo "=== [$LABEL] 5/6 $(date '+%F %T') denoise seeded audio -> for_eval_denoise/denoised_seeded/$LABEL"
(cd for_eval_denoise && "$DENOISE_PY" -u denoise_dataset.py --output denoised_seeded \
  --dataset "$ROOT/for_eval_denoise/seeded/$LABEL:$SPLIT:$TEXT=$LABEL")

echo "=== [$LABEL] 6/6 $(date '+%F %T') evaluate seeded denoised -> $RESULTS/seeded_denoised/$BASE-s/$LABEL.csv"
evaluate for_eval_denoise/denoised_seeded "$RESULTS/seeded_denoised/$BASE-s"

echo "=== [$LABEL] done $(date '+%F %T')"
