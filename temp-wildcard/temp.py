import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")

import random
import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from asr_eval.bench.dashboard.run import run_dashboard, PredictionLoader, make_storage
from asr_eval.bench.datasets import register_dataset

from datasets import Audio, load_dataset, Dataset # type: ignore

from asr_eval.bench.datasets._registry import register_dataset
from asr_eval.bench.datasets.mappers import assign_sample_ids

# EVAL_FIRST = 11

@register_dataset('upai-inc/saspeech', splits=('test',))
def load_saspeech(split: str = 'test') -> Dataset:
    df = pd.read_csv("/Users/ofir/Projects/asr-training/temp-wildcard/saspeech.csv")#.iloc[:EVAL_FIRST]

    return (
        load_dataset(
            "upai-inc/saspeech",
            split=split,
            trust_remote_code=True,
        )
        .rename_column('text', 'transcription')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
        .map(assign_sample_ids, with_indices=True)
        # Change labels to offline ones
        #.map(lambda x: {'transcription': df.loc[df['id'] == x['id'], 'norm_reference_text'].values[0]})
    )

def launch_asr_dashboard(input_csv: str):
    # Load your existing dataframe
    df = pd.read_csv(input_csv)#.iloc[:EVAL_FIRST]
    
    # Create a temporary directory to house the intermediate files
    # This automatically cleans up when the python script exits
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        ann_path = temp_path / "annotations.csv"
        pred_path = temp_path / "predictions.csv"
        
        print(f"Creating temporary files in {temp_dir}...")
        
        # --- 1. Prepare annotations.csv ---
        # Required header: dataset_name, sample_id, text
        annotations_df = df[['dataset', 'id', 'norm_reference_text']].copy()
        annotations_df.rename(columns={
            'dataset': 'dataset_name',
            'id': 'sample_id',
            'norm_reference_text': 'text'
        }, inplace=True)
        
        # Drop duplicates in case multiple engines evaluated the same sample
        annotations_df = annotations_df.drop_duplicates(subset=['dataset_name', 'sample_id'])
        annotations_df.to_csv(ann_path, index=False)
        
        # --- 2. Prepare predictions.csv ---
        preds = df[['model', 'engine', 'dataset', 'id', 'norm_predicted_text', 'transcription_time']].copy()
        preds["pipeline_name"] = preds['model'].apply(lambda x: x.split('/')[-1]) + "_" + preds['engine'].apply(lambda x: x.split('/')[-1])
        preds.drop(columns=['model', 'engine'], inplace=True)
        preds.insert(2, 'augmentor', 'none')
        preds.insert(4, 'artifact_type', None)
        preds.rename(columns={'dataset': 'dataset_name', 'id': 'sample_id'}, inplace=True)

        # Extract text predictions
        preds_text = preds.drop(columns=['transcription_time'])
        preds_text['artifact_type'] = 'text'
        preds_text.rename(columns={'norm_predicted_text': 'value'}, inplace=True)
        
        # Extract elapsed_time predictions
        preds_time = preds.drop(columns=['norm_predicted_text'])
        preds_time['artifact_type'] = 'elapsed_time'
        preds_time.rename(columns={'transcription_time': 'value'}, inplace=True)
        
        # Concatenate both metric types into the final predictions format
        predictions_df = pd.concat([preds_text, preds_time], ignore_index=True).sort_values(by=['dataset_name', 'sample_id', 'pipeline_name', 'artifact_type'])
        predictions_df.to_csv(pred_path, index=False)
        
        # --- 3. Launch Dashboard Subprocess ---
        print("Launching asr_eval dashboard...\n")
        
        run_dashboard(
            loader=PredictionLoader(
                storage=make_storage(pred_path),
                cache=make_storage("/Users/ofir/Projects/asr-training/temp-wildcard/.cache"),
                pipelines=('*',),
                dataset_specs=('*',),
            ),
            assets_dir="tmp/dashboard_assets",
            pre_export_audio=False,
            host="0.0.0.0",
            port=random.choice(list(range(8050, 8900))),
        )

if __name__ == "__main__":
    # Point this to your original CSV file
    launch_asr_dashboard("/Users/ofir/Projects/asr-training/temp-wildcard/saspeech.csv")