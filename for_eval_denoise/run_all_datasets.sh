#!/usr/bin/env bash
# Queue: run_dataset_pipeline.sh for every benchmark dataset, fastest to denoise first (see EVAL_DENOISE_GUIDE.md §3D).
# One log per dataset in for_eval_denoise/logs/pipeline_<label>.log (appended across re-runs). A failing dataset is reported and the queue continues.
#
#   nohup ./for_eval_denoise/run_all_datasets.sh [wait_for_pid] > for_eval_denoise/logs/run_all_datasets.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/for_eval_denoise/logs"
# whatsapp first: everything for it already exists or is in progress, so it only fills in what's missing
ORDER=(ivrit_ai_eval_whatsapp hebrew_speech_kan fleurs ivrit_ai_eval_d1 saspeech)

if [ -n "${1:-}" ]; then  # don't share the GPU with a pipeline that is still running
  echo "$(date '+%F %T') waiting for pid $1 to finish"
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi

for label in "${ORDER[@]}"; do
  echo "$(date '+%F %T') START $label"
  # append (>>): a re-run must not erase the logs of earlier runs
  if "$ROOT/for_eval_denoise/run_dataset_pipeline.sh" "$label" >> "$LOGS/pipeline_$label.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $label"
  else
    echo "$(date '+%F %T') FAIL  $label (exit $?, see $LOGS/pipeline_$label.log)"
  fi
done
echo "$(date '+%F %T') queue finished"
