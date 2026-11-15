#!/usr/bin/env python3
"""Select the highest-quality rows of a whisper-training dataset up to an audio-hours budget,
and save them as a DatasetDict that train-whisper.py can load (`--train_datasets <output>:train`).

Rows are ranked by `metadata[<score_field>]` (e.g. knesset's alignment `quality_score`) and
taken best-first until the budget is spent. The selected rows keep the dataset's original
order, so the subset is not sorted by quality. A selection manifest is written beside the
data so the subset can be reproduced or audited.

    python preprocess/select_top_quality.py \
        --dataset ivrit-ai/knesset-plenums-whisper-training --split train \
        --max_hours 50 --output /Users/Shared/asr/raw/knesset-hq-50h
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset


def select_top_quality(scores: np.ndarray, durations: np.ndarray, max_seconds: float) -> np.ndarray:
    """Indices of the best-scoring rows whose durations sum to at most `max_seconds`, in
    ascending index order. Ties keep dataset order, so the selection is deterministic."""
    ranked = np.argsort(-scores, kind="stable")
    within_budget = np.cumsum(durations[ranked]) <= max_seconds
    return np.sort(ranked[within_budget])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max_hours", type=float, required=True)
    parser.add_argument("--score_field", default="quality_score", help="Field of the `metadata` column to rank by")
    parser.add_argument("--duration_field", default="duration", help="Field of the `metadata` column, in seconds")
    parser.add_argument("--output", required=True, help="Directory to save the selected DatasetDict to")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)
    metadata = dataset.select_columns(["metadata"])["metadata"]
    scores = np.array([row[args.score_field] for row in metadata], dtype=np.float64)
    durations = np.array([row[args.duration_field] for row in metadata], dtype=np.float64)

    selected = select_top_quality(scores, durations, args.max_hours * 3600)
    if len(selected) == 0:
        raise ValueError(f"No rows fit within {args.max_hours} hours")

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "score_field": args.score_field,
        "max_hours": args.max_hours,
        "rows": int(len(selected)),
        "of_rows": int(len(dataset)),
        "hours": float(durations[selected].sum() / 3600),
        "min_score": float(scores[selected].min()),
        "median_score": float(np.median(scores[selected])),
        "median_score_overall": float(np.median(scores)),
    }
    if "entry_id" in metadata[0]:
        sessions = Counter(metadata[i]["entry_id"] for i in selected)
        manifest["distinct_entry_ids"] = len(sessions)
        manifest["max_rows_per_entry_id"] = max(sessions.values())
    print(json.dumps(manifest, indent=2))

    output = Path(args.output)
    DatasetDict({args.split: dataset.select(selected)}).save_to_disk(str(output))
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(output / "selected_indices.npy", selected)
    print(f"Saved {len(selected):,} rows to {output}")


if __name__ == "__main__":
    main()
