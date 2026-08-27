import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import plotext as plt

from asr_eval.bench.loader import PredictionLoader
from asr_eval.bench.evaluator import get_dataset_data
from asr_eval.bench.datasets._registry import _add_custom_annotations_from_csv
from asr_eval.align.metrics import dataset_metric_to_dataframe
from asr_eval.utils.storage import make_storage

ANNOTATIONS_CSV = "/Users/ofir/Projects/asr-training/temp-wildcard/annotations.csv"
PREDICTIONS_CSV = "/Users/ofir/Projects/asr-training/temp-wildcard/predictions.csv"
CACHE_DIR = "/Users/ofir/Projects/asr-training/temp-wildcard/.cache"
OUTPUT_CSV = "/Users/ofir/Projects/asr-training/temp-wildcard/metrics_report.csv"

METRICS = ('wer', 'n_replacements', 'n_insertions', 'n_deletions')


def build_metrics_dataframe(loader: PredictionLoader) -> pd.DataFrame:
    combos = sorted(set(
        (key.dataset_name, key.augmentor, key.parser)
        for key in loader.grouped_loaded_predictions
    ))

    per_combo_rows: list[pd.DataFrame] = []
    for dataset_name, augmentor, parser in combos:
        multiple_alignments = loader.get_multiple_alignments(
            dataset_name=dataset_name,
            augmentor_name=augmentor,
            parser_name=parser,
        )
        dataset_data = get_dataset_data(multiple_alignments)

        merged: pd.DataFrame | None = None
        for metric in METRICS:
            metric_df = dataset_metric_to_dataframe(
                dataset_data.dataset_metric, what=metric
            ).rename(columns={
                'value': metric,
                'lower': f'{metric}_lower',
                'upper': f'{metric}_upper',
            })
            merged = (
                metric_df if merged is None
                else merged.merge(metric_df, on='pipeline')
            )

        assert merged is not None
        merged.insert(0, 'dataset_name', dataset_name)
        merged.insert(1, 'augmentor', augmentor)
        merged.insert(2, 'parser', parser)
        merged['n_samples'] = len(dataset_data.full_samples)
        per_combo_rows.append(merged)

    return pd.concat(per_combo_rows, ignore_index=True).sort_values(
        ['dataset_name', 'augmentor', 'parser', 'wer']
    )


def print_and_plot(metrics_df: pd.DataFrame):
    plt.theme('clear')

    for (dataset_name, augmentor, parser), group in metrics_df.groupby(
        ['dataset_name', 'augmentor', 'parser']
    ):
        group = group.sort_values('wer')
        title = f'{dataset_name} (augmentor={augmentor}, parser={parser})'

        print(f'\n=== {title} ===')
        print(group.drop(columns=['dataset_name', 'augmentor', 'parser'])
              .round(4).to_string(index=False))

        plt.clear_figure()
        plt.bar(
            group['pipeline'].tolist(),
            (group['wer'] * 100).tolist(),
            orientation='horizontal',
        )
        plt.title(f'WER % — {title}')
        plt.plot_size(90, 4 + 2 * len(group))
        plt.show()


def main():
    _add_custom_annotations_from_csv(ANNOTATIONS_CSV)

    loader = PredictionLoader(
        storage=make_storage(PREDICTIONS_CSV),
        cache=make_storage(CACHE_DIR),
        pipelines=('*',),
        dataset_specs=('*',),
    )

    metrics_df = build_metrics_dataframe(loader)
    metrics_df.to_csv(OUTPUT_CSV, index=False)
    print(f'Wrote {len(metrics_df)} rows to {OUTPUT_CSV}')

    print_and_plot(metrics_df)


if __name__ == "__main__":
    main()
