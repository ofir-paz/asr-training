import html as html_lib
from pathlib import Path

import matplotlib.pyplot as plt

from asr_eval.align.metrics import dataset_metric_to_dataframe, plot_dataset_metric
from asr_eval.bench.dashboard.utils import render_sample_as_html_lines
from asr_eval.bench.evaluator import DatasetData
from asr_eval.bench.loader import PredictionLoader

from _common import BENCH_DIR, METRICS, OUTPUT_DIR, benchmark_id, build_loader, iter_dataset_data

SASPEECH_CSV = BENCH_DIR / "saspeech.csv"

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
    sections = [
        (
            '<section>'
            f'<h2>{html_lib.escape(dataset_name)}'
            f' (augmentor={augmentor}, parser={parser})</h2>'
            f'<p>{len(dataset_data.full_samples)} samples used for averaging</p>'
            f'{render_metrics_section(dataset_data)}'
            f'{render_samples_section(dataset_data)}'
            '</section>'
        )
        for dataset_name, augmentor, parser, dataset_data in iter_dataset_data(loader)
    ]

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


def generate_dashboard_html(input_csv: Path) -> Path:
    loader = build_loader(input_csv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"dashboard_{benchmark_id(input_csv)}.html"
    output_path.write_text(build_dashboard_html(loader), encoding='utf-8')
    print(f"Saved portable dashboard to {output_path}")
    return output_path


if __name__ == "__main__":
    generate_dashboard_html(SASPEECH_CSV)
