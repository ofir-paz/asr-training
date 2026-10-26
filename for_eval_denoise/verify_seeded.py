#!/usr/bin/env python3
"""Check that the seeded-noise dataset written by make_seeded_dataset.py got exactly the noise of the original "-s" runs.

Compares, per example, the noise clip and SNR in <seeded>/<label>/augmentation.json with the
metadata_augmentation_noise_clips / metadata_augmentation_snr_db columns of every results_2/seeded/*-s/<label>.csv.
Because each example's noise is seeded from a hash of its decoded audio, a full match also proves the input audio is the same.
Exits non-zero on any mismatch (the pipeline then skips that dataset's seeded steps).

    python for_eval_denoise/verify_seeded.py ivrit_ai_eval_whatsapp
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label")
    parser.add_argument("--seeded-dir", default=str(HERE / "seeded"))
    parser.add_argument("--runs", default="/Users/Roi/Desktop/results_2/seeded", help="Folder with the original -s run folders")
    args = parser.parse_args()

    mine = pd.DataFrame(json.loads((Path(args.seeded_dir) / args.label / "augmentation.json").read_text())["per_example"])
    mine = mine.set_index("index")
    runs = sorted(Path(args.runs).glob(f"*/{args.label}.csv"))
    if not runs:
        sys.exit(f"no original seeded runs with {args.label}.csv under {args.runs}")

    ok = True
    for csv in runs:
        o = pd.read_csv(csv).set_index("id")
        if set(o.index) != set(mine.index):
            print(f"{csv.parent.name}: different ids ({len(o)} vs {len(mine)})"); ok = False; continue
        o = o.loc[mine.index]
        clips = int((o["metadata_augmentation_noise_clips"].fillna("") == mine["noise_clips"].fillna("")).sum())
        snr = int(np.isclose(o["metadata_augmentation_snr_db"].fillna(-1), mine["snr_db"].fillna(-1), atol=1e-6).sum())
        n = len(mine)
        print(f"{csv.parent.name}: same noise clip {clips}/{n}, same SNR {snr}/{n}")
        ok &= clips == n and snr == n
    print("MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
