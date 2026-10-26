# Denoise ASR benchmarks with NVIDIA RE-USE (Apple Silicon / MLX)

Goal: take the Hebrew ASR benchmark datasets used in `../asr-training`, remove noise and reverb from
their audio with NVIDIA's **RE-USE** speech-enhancement model, and save them in the **same format**, so you
can run the normal evaluation on the denoised audio and compare WER with the original.

This folder only produces the denoised datasets. Running ASR / comparing WER is done with `asr-training`.

## What's here

| Path | What it is |
|---|---|
| `denoise_dataset.py` | **The pipeline.** Dataset in -> denoised dataset out |
| `requirements.txt` | Python packages |
| `NVIDIA-NemotronLabs-RE-USE/` | Original NVIDIA model (PyTorch weights + code, needs CUDA, so not used on the Mac). Its weights are the *source* that gets converted to MLX |
| `Real-time_RE-USE/` | Streaming model (research license only). CUDA-only, **no MLX port, not used by the pipeline** |
| `convert_weights.py` | PyTorch weights -> MLX weights (already run; output is in `mlx_weights/`) |
| `mlx_weights/reuse/` | **The converted MLX weights** (39 MB), ready to use. Verified: 1416 tensors, names and shapes identical to a community MLX conversion of the same architecture |

## Which model, and why MLX

NVIDIA's code imports `mamba_ssm`, which only runs on NVIDIA GPUs. On a Mac we use
[`mlx-speech`](https://github.com/appautomaton/mlx-speech) (MIT), a pure-MLX port of the same RE-USE architecture
(its output matches the PyTorch model at 0.9997 correlation, per its docs). `convert_weights.py` converts NVIDIA's
`model.safetensors` into the MLX layout (key renames + conv transposes, no change to the learned weights) into
`mlx_weights/<variant>/`. `denoise_dataset.py` does the same automatically if that folder is missing.

- `--variant reuse` (default): `nvidia/NVIDIA-NemotronLabs-RE-USE`, general enhancement, commercial use OK (OpenMDW-1.1).
- `--variant laughter`: `nvidia/NVIDIA-NemotronLabs-RE-USE-laughter`, better at preserving laughter. Run `python convert_weights.py --variant laughter` (downloads it) or just use the flag.

## Setup (once)

Needs Python 3.13+ (mlx-speech's requirement).

```bash
cd for_eval_denoise
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Moving to another machine

Copy this whole folder **except `.venv/`** (keep `mlx_weights/`; `NVIDIA-NemotronLabs-RE-USE/` is only needed to
re-convert, and `Real-time_RE-USE/` isn't used). Then redo the Setup step there. It must be an Apple Silicon Mac: MLX does
not run on Windows/Linux/Intel. On a CUDA machine use NVIDIA's own scripts in `NVIDIA-NemotronLabs-RE-USE/` instead.

## Run

Whole benchmark (the same five datasets as `asr-training/run_bench.py`):

```bash
python denoise_dataset.py --output denoised
```

A couple of datasets, or a quick test first:

```bash
python denoise_dataset.py --output denoised --only fleurs saspeech
python denoise_dataset.py --output denoised --only ivrit_ai_eval_d1 --limit 20     # 20 examples smoke test
```

Labels: `ivrit_ai_eval_d1`, `ivrit_ai_eval_whatsapp`, `saspeech`, `fleurs`, `hebrew_speech_kan`.
Datasets already denoised are skipped (add `--overwrite` to redo). A failing dataset is reported and the rest continue.

Any other dataset (`SPEC[@CONFIG]=LABEL`, repeatable):

```bash
python denoise_dataset.py --output denoised --dataset "my-org/my-ds:test:text=my_ds"
python denoise_dataset.py --output denoised --dataset "google/fleurs@he_il:test:transcription=fleurs"
```

Other options: `--variant laughter`, `--chunk-seconds 5` (audio is denoised in overlapping chunks, 5 s like NVIDIA's chunked script).

## Output: same names and structure as the input

```
denoised/
├── denoised_datasets.json          # label -> path and the exact --dataset argument to use
├── ivrit_ai_eval_d1/data/test-00000-of-00001.parquet
├── ivrit_ai_eval_whatsapp/data/test-00000-of-00001.parquet
├── saspeech/data/test-00000-of-00001.parquet
├── fleurs/data/test-00000-of-00001.parquet
└── hebrew_speech_kan/data/validation-00000-of-00001.parquet
```

Each folder behaves like the original Hugging Face dataset for `datasets.load_dataset(<folder>)`:

- same **split name** (`test` / `validation`), same **columns** (`audio`, `text` / `transcription` / `sentence`, and every metadata column), same **row order**;
- `audio` keeps the original **sampling rate** and stays an `Audio` feature (`{"array", "sampling_rate"}`); stereo is mixed down to mono (the model is mono);
- only the audio samples change. Folder names equal the result-CSV names `run_bench.py` writes, so original and denoised results line up.

## Evaluate (in `asr-training`, no code changes)

Use the original evaluation command and only change the `--dataset` value to the denoised folder (it is the
`evaluate_model_dataset_arg` field in `denoised_datasets.json`). Do **not** pass `--name` (the config is already baked in):

```bash
cd ../asr-training
./evaluate_model.py --engine engines/<your_engine>.py --model <your_model> \
    --dataset /Users/sibony/Desktop/for_eval_denoise/denoised/fleurs:test:transcription \
    --output results_denoised/fleurs.csv
```

Run it the same way on the originals, then compare `results_denoised/*.csv` against the original results.

## Notes and limits

- **Speed**: untested on large runs; the MLX Mamba scan is sequential. Use `--limit` first to time it.
- **Real-time RE-USE** isn't supported here (no MLX port; NVIDIA's code is CUDA-only).
- **Domain mismatch**: RE-USE was trained on noisy speech; on already-clean recordings it can slightly alter the audio, which is part of what you are measuring.
