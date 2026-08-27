from pathlib import Path

import pandas as pd
import plotext as plt

from asr_eval.align.metrics import dataset_metric_to_dataframe
from asr_eval.bench.loader import PredictionLoader

from _common import BENCH_DIR, METRICS, OUTPUT_DIR, benchmark_id, build_loader, iter_dataset_data

SASPEECH_CSV = BENCH_DIR / "saspeech.csv"


def build_metrics_dataframe(loader: PredictionLoader) -> pd.DataFrame:
    per_combo_rows: list[pd.DataFrame] = []
    for dataset_name, augmentor, parser, dataset_data in iter_dataset_data(loader):
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


def generate_metrics_report(input_csv: Path) -> Path:
    loader = build_loader(input_csv)
    metrics_df = build_metrics_dataframe(loader)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"metrics_{benchmark_id(input_csv)}.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f'Wrote {len(metrics_df)} rows to {output_path}')

    print_and_plot(metrics_df)
    return output_path


if __name__ == "__main__":
    generate_metrics_report(SASPEECH_CSV)
