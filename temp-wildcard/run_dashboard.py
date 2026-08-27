import os
os.environ["MPLBACKEND"] = "Agg"

import html as html_lib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset  # type: ignore

from asr_eval.align.metrics import dataset_metric_to_dataframe, plot_dataset_metric
from asr_eval.bench.dashboard.utils import render_sample_as_html_lines
from asr_eval.bench.datasets import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids
from asr_eval.bench.evaluator import DatasetData, get_dataset_data
from asr_eval.bench.loader import PredictionLoader
from asr_eval.utils.storage import make_storage

EVAL_FIRST = 1000
METRICS = ('wer', 'n_replacements', 'n_insertions', 'n_deletions')
METRIC_DESCRIPTIONS = {
    'wer': (
        'Word Error Rate: the share of reference words that were'
        ' substituted, inserted, or deleted in the prediction.'
    ),
    'n_replacements': (
        'The number of reference words that were substituted with a'
        ' different, incorrect word in the prediction.'
    ),
    'n_insertions': (
        'The number of extra words the prediction added that do not'
        ' appear in the reference.'
    ),
    'n_deletions': (
        'The number of reference words that are missing entirely from'
        ' the prediction.'
    ),
}
COLUMN_DESCRIPTION = (
    '"value" is the metric computed over all samples; "lower" and'
    ' "upper" are the 10th and 90th percentile bounds from bootstrap'
    ' resampling.'
)

TEMP_DIR = Path("/Users/ofir/Projects/asr-training/temp-wildcard")
SASPEECH_CSV = TEMP_DIR / "saspeech.csv"
ANNOTATIONS_CSV = TEMP_DIR / "annotations.csv"
PREDICTIONS_CSV = TEMP_DIR / "predictions.csv"
CACHE_DIR = TEMP_DIR / ".cache"
OUTPUT_HTML = TEMP_DIR / "dashboard.html"


@register_dataset('upai-inc/saspeech', splits=('test',))
def load_saspeech(split: str = 'test') -> Dataset:
    dataset = (
        load_dataset("csv", data_files={split: str(SASPEECH_CSV)}, split=split)
        .sort('id')
    )
    dataset = dataset.select(range(min(EVAL_FIRST, len(dataset))))
    return (
        dataset
        .rename_column('norm_reference_text', 'transcription')
        .map(assign_sample_ids, with_indices=True)
    )


def prepare_annotations_and_predictions(input_csv: Path) -> None:
    """Builds annotations.csv and predictions.csv (asr_eval storage
    format) out of the raw per-run evaluation CSV.
    """

    df = pd.read_csv(input_csv).sort_values('id').iloc[:EVAL_FIRST]

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
    annotations_df.to_csv(ANNOTATIONS_CSV, index=False)

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
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)


def render_metrics_section(dataset_data: DatasetData) -> str:
    tables_html = [
        '<div class="metric-table">'
        f'<h3>{metric}</h3>'
        f'<p class="metric-descr">{METRIC_DESCRIPTIONS[metric]}</p>'
        + dataset_metric_to_dataframe(dataset_data.dataset_metric, what=metric)
            .sort_values('value').round(3).to_html(index=False)
        + '</div>'
        for metric in METRICS
    ]
    plots_html = [
        '<img style="max-width: 24%; display: inline-block;"'
        ' src="data:image/svg+xml;base64,'
        + plot_dataset_metric(dataset_data.dataset_metric, what=metric, show=False)
        + '">'
        for metric in METRICS
    ]
    plt.close('all')

    return (
        f'<div>{"".join(plots_html)}</div>'
        f'<p class="metric-descr">{COLUMN_DESCRIPTION}</p>'
        f'<div class="metric-tables">{"".join(tables_html)}</div>'
    )


def render_samples_section(dataset_data: DatasetData) -> str:
    blocks = [
        '<br/>'.join(lines)
        for sample in dataset_data.samples
        if (lines := render_sample_as_html_lines(sample)) is not None
    ]
    return (
        '<div style="font-family: Consolas, \'Ubuntu Mono\', Monaco, monospace;'
        f' white-space: pre;">{"<hr/>".join(blocks)}</div>'
    )


def build_dashboard_html(loader: PredictionLoader) -> str:
    combos = sorted({
        (key.dataset_name, key.augmentor, key.parser)
        for key in loader.grouped_loaded_predictions
    })

    sections: list[str] = []
    for dataset_name, augmentor, parser in combos:
        multiple_alignments = loader.get_multiple_alignments(
            dataset_name=dataset_name,
            augmentor_name=augmentor,
            parser_name=parser,
        )
        dataset_data = get_dataset_data(multiple_alignments)

        sections.append(
            '<section>'
            f'<h2>{html_lib.escape(dataset_name)}'
            f' (augmentor={augmentor}, parser={parser})</h2>'
            f'<p>{len(dataset_data.full_samples)} samples used for averaging</p>'
            f'{render_metrics_section(dataset_data)}'
            f'{render_samples_section(dataset_data)}'
            '</section>'
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ASR Dashboard</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; margin-bottom: 10px; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
section {{ margin-bottom: 40px; }}
.metric-descr {{ color: #555; font-size: 0.9em; margin: 0 0 8px; max-width: 600px; }}
.metric-tables {{ display: flex; flex-direction: column; gap: 20px; }}
.metric-table table {{ width: 100%; max-width: 700px; }}
</style>
</head>
<body>
<h1>ASR Dashboard</h1>
{"".join(sections)}
</body>
</html>"""


def generate_dashboard_html(input_csv: Path) -> None:
    prepare_annotations_and_predictions(input_csv)

    loader = PredictionLoader(
        storage=make_storage(str(PREDICTIONS_CSV)),
        cache=make_storage(str(CACHE_DIR)),
        pipelines=('*',),
        dataset_specs=('*',),
    )

    OUTPUT_HTML.write_text(build_dashboard_html(loader), encoding='utf-8')
    print(f"Saved portable dashboard to {OUTPUT_HTML}")


if __name__ == "__main__":
    generate_dashboard_html(SASPEECH_CSV)
