import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("MPLBACKEND", "Agg")

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from asr_eval.bench.datasets._registry import _add_custom_annotations_from_csv
from asr_eval.bench.evaluator import DatasetData, get_dataset_data
from asr_eval.bench.loader import PredictionLoader
from asr_eval.utils.storage import make_storage

BENCH_DIR = Path(__file__).resolve().parent
CACHE_DIR = BENCH_DIR / ".cache"
OUTPUT_DIR = BENCH_DIR / "outputs"

# Only score the first N samples of a raw evaluation CSV, to keep
# alignment fast while iterating.
EVAL_FIRST = 1000000

METRICS = ('wer', 'n_replacements', 'n_insertions', 'n_deletions')


def benchmark_id(input_csv: Path) -> str:
    """A unique, human-readable id for one benchmark run, derived from
    the source evaluation CSV name and the current time.
    """
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'{Path(input_csv).stem}__{timestamp}'


def prepare_annotations_and_predictions(input_csv: Path) -> tuple[Path, Path]:
    """Builds asr_eval's annotations.csv / predictions.csv storage
    format out of a raw per-run evaluation CSV (expected columns:
    model, engine, dataset, id, norm_reference_text,
    norm_predicted_text, transcription_time).
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    annotations_csv = CACHE_DIR / "annotations.csv"
    predictions_csv = CACHE_DIR / "predictions.csv"

    df = pd.read_csv(input_csv).sort_values('id')
    if EVAL_FIRST < df.shape[0]:
        print(f"Limiting to first {EVAL_FIRST} samples (out of {df.shape[0]})")
        df = df.iloc[:EVAL_FIRST]
    else:
        print(f"Using all {df.shape[0]} samples")

    # Required header: dataset_name, sample_id, text
    annotations_df = (
        df[['dataset', 'id', 'norm_reference_text']]
        .rename(columns={
            'dataset': 'dataset_name',
            'id': 'sample_id',
            'norm_reference_text': 'text',
        })
        .drop_duplicates(subset=['dataset_name', 'sample_id'])
    )
    annotations_df.to_csv(annotations_csv, index=False)

    preds = df[[
        'model', 'engine', 'dataset', 'id',
        'norm_predicted_text', 'transcription_time',
    ]].copy()
    preds['pipeline_name'] = (
        preds['model'].apply(lambda x: x.split('/')[-1])
        + '_' + preds['engine'].apply(lambda x: x.split('/')[-1])
    )
    preds = (
        preds
        .drop(columns=['model', 'engine'])
        .rename(columns={'dataset': 'dataset_name', 'id': 'sample_id'})
    )
    preds.insert(2, 'augmentor', 'none')

    preds_text = (
        preds.drop(columns=['transcription_time'])
        .rename(columns={'norm_predicted_text': 'value'})
    )
    preds_text['artifact_type'] = 'text'

    preds_time = (
        preds.drop(columns=['norm_predicted_text'])
        .rename(columns={'transcription_time': 'value'})
    )
    preds_time['artifact_type'] = 'elapsed_time'

    predictions_df = pd.concat(
        [preds_text, preds_time], ignore_index=True
    ).sort_values(by=['dataset_name', 'sample_id', 'pipeline_name', 'artifact_type'])
    predictions_df.to_csv(predictions_csv, index=False)

    return annotations_csv, predictions_csv


def build_loader(input_csv: Path) -> PredictionLoader:
    """Prepares annotations/predictions for `input_csv` and loads them
    into a `PredictionLoader`, ready for alignment and scoring.
    """

    annotations_csv, predictions_csv = prepare_annotations_and_predictions(input_csv)
    _add_custom_annotations_from_csv(annotations_csv)

    return PredictionLoader(
        storage=make_storage(predictions_csv),
        cache=make_storage(CACHE_DIR / "alignments"),
        pipelines=('*',),
        dataset_specs=('*',),
    )


def iter_dataset_data(
    loader: PredictionLoader,
) -> Iterator[tuple[str, str, str, DatasetData]]:
    """Yields (dataset_name, augmentor, parser, dataset_data) for every
    combination found in the loaded predictions.
    """

    combos = sorted({
        (key.dataset_name, key.augmentor, key.parser)
        for key in loader.grouped_loaded_predictions
    })
    for dataset_name, augmentor, parser in combos:
        multiple_alignments = loader.get_multiple_alignments(
            dataset_name=dataset_name,
            augmentor_name=augmentor,
            parser_name=parser,
        )
        yield dataset_name, augmentor, parser, get_dataset_data(multiple_alignments)
