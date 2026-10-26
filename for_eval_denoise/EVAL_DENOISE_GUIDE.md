---
name: eval-denoise
description: Evaluate ASR models on denoised versions of the asr-training benchmarks (NVIDIA RE-USE on Apple Silicon/MLX), for clean audio and for the seeded test-time-noise ("-s") audio, and compare against the original runs in results_2 with model_comparison.ipynb section 14. Use when asked to denoise a benchmark dataset, evaluate a model on denoised or seeded-noise audio, check progress of the denoise queue, or extend this experiment to other datasets/models.
---

# Denoising experiment: what was done, where everything is, how to rerun it

**Question:** does running a speech denoiser (NVIDIA RE-USE) on the audio before Whisper lower WER?

**Setup:** model `lv3-mix-na-b32-kd-low-smooth-mlx`, all 5 benchmark datasets of `run_bench.py`, two conditions:

| condition | audio the model hears | reference it is compared with |
|---|---|---|
| clean | original audio → RE-USE | original clean run `results_2/<model>-c` |
| seeded | audio + seeded test-time noise (identical to the `-s` runs) → RE-USE | original seeded run `results_2/<model>-s`, and a same-session rerun `results_2/seeded_rerun/<model>-s` |

Only the audio changes between paired runs: same utterances, transcripts, engine (`engines/mlx_engine.py`), model weights.

---

## 0. Quick start

```bash
cd /Users/Roi/Desktop/asr-training

# whole experiment for one dataset (resumable; every finished step is reused on re-run)
./for_eval_denoise/run_dataset_pipeline.sh <label> > for_eval_denoise/logs/pipeline_<label>.log 2>&1

# all datasets in a queue (what is running now); optional arg = a PID to wait for first
nohup ./for_eval_denoise/run_all_datasets.sh > for_eval_denoise/logs/run_all_datasets.log 2>&1 &

# progress
cat for_eval_denoise/logs/run_all_datasets.log             # START / DONE / FAIL per dataset
grep -a "^===" for_eval_denoise/logs/pipeline_<label>.log    # which of the 6 steps it is on
tail -c 400 for_eval_denoise/logs/pipeline_<label>.log       # per-clip denoise line: i/N, time, ETA, memory

# results: open benchmark_analysis/model_comparison.ipynb, run all, read section 14
```
Labels: `ivrit_ai_eval_whatsapp`, `hebrew_speech_kan`, `fleurs`, `ivrit_ai_eval_d1`, `saspeech`.

---

## 1. Environments (two venvs, never mixed)

| venv | Python | used for | how it is called |
|---|---|---|---|
| `asr-training/.venv` | 3.10 | `run_bench.py`, `evaluate_model.py`, noise augmentation, notebook | `source .venv/bin/activate` or `.venv/bin/python` |
| `for_eval_denoise/.venv` | 3.14 | denoiser only (`mlx-speech` needs Python ≥ 3.13) | always by full path: `for_eval_denoise/.venv/bin/python` |

Rebuild the denoise venv (it was rebuilt on 2026-10-25 because the copied one pointed at a missing Python 3.15):
```bash
cd for_eval_denoise && rm -rf .venv && /opt/homebrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
`run_bench.py` starts `./evaluate_model.py` through its `#!/usr/bin/env python3` shebang, so the **main venv must be first on
PATH**. Activate it, or `export PATH=$ROOT/.venv/bin:$PATH` as the scripts do.

---

## 2. The pipeline per dataset (`run_dataset_pipeline.sh <label> [model]`)

| step | what | command it runs | output |
|---|---|---|---|
| 1 | denoise clean audio | `for_eval_denoise/.venv/bin/python denoise_dataset.py --output denoised --only <label>` | `for_eval_denoise/denoised/<label>/` |
| 2 | evaluate it | `run_bench.py ... --denoised-dir for_eval_denoise/denoised --output-dir results_2/clean_denoised/<model>-c` | `results_2/clean_denoised/<model>-c/<label>.csv` |
| 3 | re-create the seeded noisy audio, then **verify** it equals the original `-s` noise | `make_seeded_dataset.py --output for_eval_denoise/seeded --only <label>` then `verify_seeded.py <label>` | `for_eval_denoise/seeded/<label>/` (+ `augmentation.json`) |
| 4 | evaluate the seeded audio as-is (same-session reference) | `run_bench.py ... --denoised-dir for_eval_denoise/seeded --output-dir results_2/seeded_rerun/<model>-s` | `results_2/seeded_rerun/<model>-s/<label>.csv` |
| 5 | denoise the seeded audio | `denoise_dataset.py --output denoised_seeded --dataset "<root>/for_eval_denoise/seeded/<label>:<split>:<text>=<label>"` | `for_eval_denoise/denoised_seeded/<label>/` |
| 6 | evaluate it | `run_bench.py ... --denoised-dir for_eval_denoise/denoised_seeded --output-dir results_2/seeded_denoised/<model>-s` | `results_2/seeded_denoised/<model>-s/<label>.csv` |

`run_bench.py` arguments in all eval steps: `--engine engines/mlx_engine.py --model <model> --parallel-mode process --devices gpu,gpu,gpu,gpu --batch-size 1`
(the same as the original runs). If step 3's verification fails, the seeded steps of that dataset are skipped and the log says so.

**Eval steps cover every label present.** `run_bench.py --denoised-dir X` evaluates every dataset folder that exists in `X`, not
only `<label>`. All seeded datasets were built up front (to verify them), so the WhatsApp pipeline's step 4 evaluated the seeded
rerun of all five datasets in one go. Later pipelines reuse those CSVs. This is harmless: each CSV is written once and reused after.

**Resuming.** Every step reuses existing output: `denoise_dataset.py` skips a finished dataset and resumes an unfinished one clip
by clip from `<output>/.clips/<label>/`. `evaluate_model.py` reuses an existing CSV, and `make_seeded_dataset.py` skips an existing
folder. After a crash or reboot, just run the same command again.

### The queue (`run_all_datasets.sh`), started 2026-10-25 18:34
Order is fastest-to-denoise first, so results come back early (the 44.1 kHz datasets are 2–4× slower per minute of audio):

| order | dataset | clips | audio | format | est. clean denoise | est. seeded denoise (16 kHz) |
|---|---|---|---|---|---|---|
| 0 | ivrit_ai_eval_whatsapp | 54 | 71 min | 16 kHz (1 clip 48 kHz) | done (97 min) | running when queued (≈95 min) |
| 1 | hebrew_speech_kan | 2000 | 104 min | 16 kHz stereo | ≈2.4 h | ≈2.4 h |
| 2 | fleurs (he_il) | 792 | 123 min | 16 kHz mono | ≈2.8 h | ≈2.8 h |
| 3 | ivrit_ai_eval_d1 | 17 | 86 min | 44.1 kHz stereo | ≈4–9 h | ≈2 h |
| 4 | saspeech | 2986 | 236 min | 44.1 kHz mono | ≈12–25 h | ≈5.3 h |

**Status (stopped 2026-10-26 10:35 by decision: denoising does not help).** Done: WhatsApp, kan and fleurs (all 6 steps),
eval_d1 steps 1–4 (clean denoised + evaluated; seeded audio built, verified and evaluated as-is). Not run: eval_d1 steps 5–6
(seeded denoise: 6 of 17 clips are saved in `denoised_seeded/.clips/ivrit_ai_eval_d1/`, so it resumes from there) and all of
saspeech except its seeded rerun. To finish later, run `./for_eval_denoise/run_dataset_pipeline.sh ivrit_ai_eval_d1` and then
`saspeech`; finished steps are skipped.

Total for all 5 ≈ 1.5–2 days. Evaluations take minutes per dataset. Logs: `for_eval_denoise/logs/run_all_datasets.log` (queue) and
`for_eval_denoise/logs/pipeline_<label>.log` (one per dataset).

---

## 3. Every file created or changed

### Code (in the repo)
| path | status | what it is |
|---|---|---|
| `run_bench.py` | **modified** | new flag `--denoised-dir DIR`: read each benchmark from `DIR/<label>` (same split and text column, `--name` dropped, HF access check skipped). Labels with no folder there are skipped with a warning. Without the flag nothing changes. |
| `.gitignore` | modified | ignores `for_eval_denoise/denoised*/`, `for_eval_denoise/logs/`, `for_eval_denoise/seeded/` |
| `for_eval_denoise/denoise_dataset.py` | modified | per-clip progress line (`[label] i/N · Xs audio @SR · took Ys · elapsed · ETA · mlx peak`); MLX cache freed after each clip; each finished clip saved to `<output>/.clips/<label>/` (resume); `eager_chunked_hann_ola()` replaces the library's chunk loop with one that evaluates per chunk (bit-identical output, bounded memory on long clips) |
| `for_eval_denoise/verify_eager_ola.py` | new | checks that the eager chunk loop is bit-identical to the library loop (60 s eval-d1 excerpt: max diff 0) |
| `for_eval_denoise/make_seeded_dataset.py` | new | writes the seeded-noise audio to disk, bit-identical to what the `-s` runs fed the model (16 kHz, float WAV) |
| `for_eval_denoise/verify_seeded.py` | new | checks per example that the noise clip and SNR equal the original `-s` CSVs; exits 1 on mismatch |
| `for_eval_denoise/seeded_aug/` | new | `eval_augmentation.py` + `noise_augmentation.py` vendored from `origin/feature/benchmark-with-augs` @ `70737f5`. Only change: re-indented `NoiseLibrary.__init__` (the pushed branch has an `IndentationError` there) |
| `for_eval_denoise/run_dataset_pipeline.sh` | new | the 6-step pipeline for one dataset (§2) |
| `for_eval_denoise/run_all_datasets.sh` | new | the queue over all datasets |
| `for_eval_denoise/run_denoised_eval.sh` | new | first version, clean WhatsApp eval only (superseded by `run_dataset_pipeline.sh`) |
| `for_eval_denoise/run_seeded_pipeline.sh` | new | first version, seeded WhatsApp only (superseded by `run_dataset_pipeline.sh`) |
| `for_eval_denoise/debug_memory.py` | new | diagnostic: footprint + MLX active/cache memory per clip (showed the denoiser itself does not leak once the cache is cleared) |
| `for_eval_denoise/bench_batch.py` | new | benchmark: one chunk at a time vs batched chunks (batching does **not** help, §6) |
| `benchmark_analysis/model_comparison.ipynb` | modified | new **section 14** (2 cells at the end) |
| `for_eval_denoise/EVAL_DENOISE_GUIDE.md` | new | this file |

### Data (not in git), under `for_eval_denoise/`
| path | contents |
|---|---|
| `denoised/<label>/data/<split>-00000-of-00001.parquet` | clean audio denoised. Original sample rate, mono, FLAC-24; same split, columns and row order as the HF dataset |
| `denoised_seeded/<label>/data/...` | seeded audio denoised (16 kHz) |
| `seeded/<label>/data/...` + `seeded/<label>/augmentation.json` | seeded noisy audio (16 kHz, float WAV) + settings and per-example noise clip / SNR |
| `denoised*/denoised_datasets.json` | manifest: label → path and the `--dataset` argument for `evaluate_model.py` |
| `denoised*/.clips/<label>/*.flac` | per-clip resume cache (safe to delete once the parquet exists) |
| `logs/` | `run_all_datasets.log`, `pipeline_<label>.log`, plus the first WhatsApp runs: `denoise_whatsapp.log`, `eval_denoised.log`, `seeded_pipeline.log`, `seeded_pipeline_killed_run1.log` |

Each folder loads with `datasets.load_dataset("<folder>")`, and `evaluate_model.py --dataset <folder>:<split>:<text>` scores it unchanged.

### Results, in `/Users/Roi/Desktop/results_2`, next to the originals (nothing original is overwritten)
| folder | what |
|---|---|
| `clean_denoised/lv3-mix-na-b32-kd-low-smooth-mlx-c/<label>.csv` | clean → denoised |
| `seeded_rerun/lv3-mix-na-b32-kd-low-smooth-mlx-s/<label>.csv` | seeded audio as-is, re-run in this session (reference) |
| `seeded_denoised/lv3-mix-na-b32-kd-low-smooth-mlx-s/<label>.csv` | seeded → denoised |

Folder names keep the `-c` / `-s` suffix of the run they pair with. They sit one level below `results_2`, so the notebook's main
loader (`RESULTS_DIR.glob("*/*.csv")`) does **not** pick them up; only section 14 reads them.

---

## 4. How the seeded noise matches the original `-s` runs
The `-s` runs used Ofir's branch `origin/feature/benchmark-with-augs` (commit `70737f5`, test-time augmentation inside
`evaluate_model.py`). The settings are recorded in every `-s` CSV:
```
--noise-config preprocess/noise_config.yaml --noise-dir audio-set/test_augs
--noise-apply-prob 1.0 --noise-band-limit-hz 8000 --augmentation-seed 1234
```
Each example's RNG is seeded from `(1234, sha256 of its decoded audio)`, so the noise doesn't depend on order or model.
`make_seeded_dataset.py` uses the same code (vendored) and the same construction as the branch's `build_eval_augmentation()`, and
loads the data like `evaluate_model.py` does (`trust_remote_code=True`, decoded by `datasets` in the main venv).
**Verified on all 5 datasets (5,849 examples):** the noise clip and SNR equal the original `-s` CSVs for every example, for every
model that has a `-s` run. That also proves the decoded input audio is identical, since the seed hashes it.
The `na` model has **no original `-s` run on `hebrew_speech_kan`**; for that dataset the reference is `seeded_rerun` only.

---

## 5. Reading the results (notebook section 14)
Open `benchmark_analysis/model_comparison.ipynb` and run all cells. Section 14 reuses `paired_bootstrap` (section 6) and the
loop rule (section 11). `DENOISED_DIRS` at the top of the cell maps each condition to its reference and denoised folders:
- `clean`: original `-c` run vs. `clean_denoised`
- `seeded`: original `-s` run vs. `seeded_denoised`
- `seeded (rerun)`: `seeded_rerun` vs. `seeded_denoised`. The cleanest pairing, because both sides were transcribed in the same
  session with the same code.

Missing folders or missing reference runs are skipped with a message. Per model × dataset × condition it shows: WER without and
with denoising, ΔWER (negative = denoising helped), paired-bootstrap 95% CI, `p(denoised better)`, utterances better/same/worse
(as-is and with the loop rule), a side-by-side bar chart, and the utterances denoising helped or hurt most.

### Conclusion (2026-10-26)
| dataset | clean: WER → denoised (ΔWER) | seeded vs same-session rerun: WER → denoised (ΔWER) |
|---|---|---|
| fleurs | 17.44% → 17.43% (−0.01%) | 20.84% → 21.19% (+0.35%) |
| hebrew_speech_kan | 8.08% → 8.66% (+0.58%) **sig.** | 9.59% → 10.92% (+1.33%) **sig.** |
| ivrit_ai_eval_d1 | 5.05% → 5.32% (+0.27%) **sig.** | not run |
| ivrit_ai_eval_whatsapp | 6.93% → 7.12% (+0.19%) | 9.93% → 10.12% (+0.19%) |

(Loop rule applied; **sig.** = paired-bootstrap 95% CI excludes 0.) **RE-USE denoising never improved
`lv3-mix-na-b32-kd-low-smooth-mlx`.** It is neutral at best and significantly worse on kan and eval-d1, more so on noisy audio.
The model was trained with noise augmentation, and the denoiser's artifacts seem to cost more than the noise it removes. The
experiment was stopped here. The overview table and ΔWER chart at the top of notebook section 14 show this.

### Results so far (WhatsApp, clean)
| | clean | denoised |
|---|---|---|
| as-is: WER, ΔWER, 95% CI | 7.74% | 7.12%, −0.62%, [−2.55%, +0.58%], not significant |
| loop rule: WER, ΔWER, 95% CI | 6.93% | 7.12%, +0.19%, [−0.21%, +0.64%], not significant |
| clips better / same / worse | | 14 / 25 / 15 |

Most of the as-is gain is clip 10, a decoding loop on clean audio (98 errors) that didn't happen on the denoised audio (21 errors).

### Results so far (WhatsApp, seeded noise), finished 2026-10-25 20:06
| reference | scoring | WER | WER denoised | ΔWER | 95% CI | significant | clips better / same / worse |
|---|---|---|---|---|---|---|---|
| original `-s` | as-is | 11.81% | 10.12% | −1.69% | [−7.27%, +1.26%] | no | 12 / 11 / 31 |
| original `-s` | loop rule | 9.42% | 10.12% | +0.71% | [+0.10%, +1.42%] | **yes (worse)** | 12 / 11 / 31 |
| `seeded_rerun` | as-is / loop rule | 9.93% | 10.12% | +0.19% | [−0.86%, +1.13%] | no | 13 / 11 / 30 |

### Results so far (hebrew_speech_kan, 2000 clips), finished 2026-10-25 23:39
| condition | WER | WER denoised | ΔWER | 95% CI | significant | clips better / same / worse |
|---|---|---|---|---|---|---|
| clean (vs original `-c`) | 8.08% | 8.66% | +0.58% (+7% rel.) | [+0.26%, +0.88%] | **yes (worse)** | 123 / 1694 / 183 |
| seeded (vs `seeded_rerun`; no original `-s` run for na on kan) | 9.59% | 10.92% | +1.33% (+14% rel.) | [+0.90%, +1.76%] | **yes (worse)** | 182 / 1503 / 315 |

On the larger set the effect is clear: denoising makes this model significantly worse, and more so on noisy audio.

### Results so far (fleurs he_il, 792 clips), finished 2026-10-26 04:34 (loop-rule rows; as-is is the same here)
| condition | WER | WER denoised | ΔWER | 95% CI | significant | clips better / same / worse |
|---|---|---|---|---|---|---|
| clean (vs original `-c`) | 17.44% | 17.43% | −0.01% | [−0.22%, +0.20%] | no | 58 / 679 / 55 |
| seeded (vs original `-s`) | 20.82% | 21.19% | +0.37% | [−0.07%, +0.81%] | no | 140 / 490 / 162 |
| seeded (vs `seeded_rerun`) | 20.84% | 21.19% | +0.35% | [−0.10%, +0.78%] | no | 141 / 490 / 161 |

Fleurs is clean read speech: no effect on clean audio, and a small non-significant worsening on noisy audio.

### WhatsApp interpretation
Even on noisy audio, denoising **does not help** this model on WhatsApp. Per clip it is worse more than twice as often as it is
better (30–31 worse vs 12–13 better). The as-is "gain" vs the original `-s` run is a decoding loop in that run (clip 10), not
the denoiser. The `na` model was trained with noise augmentation, which may be why it doesn't need the denoiser.

### Run-to-run noise in the evaluation itself (read before trusting any ΔWER)
The seeded rerun fed the model **bit-identical** audio to the original `-s` run. 51 of 54 WhatsApp transcripts came out identical;
3 differed (clip 10: 239 vs 24 errors, clip 13: 39 vs 87), and WER moved from 11.81% to 9.93%. The cause is
`mlx_whisper.transcribe`'s default temperature fallback (`temperature=(0.0, 0.2, …, 1.0)`, `compression_ratio_threshold=2.4`):
when a greedy decode looks like a loop or scores badly, it retries with unseeded random sampling. `engines/mlx_engine.py` uses
those defaults. Consequences:
- About ±1–2% absolute WER on small sets can come from decoding luck alone, mostly from a few hard clips. Prefer the loop-rule rows,
  the per-utterance tables and the `seeded (rerun)` pairing.
- To remove this noise in future experiments, run the engine with `temperature=0.0` (no fallback). Do it on both sides of a
  comparison, since it changes absolute WER.

---

## 6. Things learned the hard way (read before extending)
- **Speed.** RE-USE in MLX runs ≈1.36 s per second of audio at 16 kHz and ≈3 s/s at 44.1 kHz (M5 Max). It is a Python loop over
  time steps in the Mamba scan (`mlx_speech/models/reuse/mamba/scan.py`, 30 blocks × 4 directional scans), and memory-bandwidth
  bound (~1.8 GB tensors per 5 s chunk at 44.1 kHz). **Batching chunks does not help** (`bench_batch.py`: ×1.00 at batch 2, slower
  at 4 and 8, identical outputs). The real fix would be rewriting the scan as a chunked parallel scan. Not done.
- **Memory / crash.** The first seeded WhatsApp denoise was killed by macOS (`Killed: 9`, swap 46.5/48 GB) on clip 54/54 after
  95 min, with nothing saved. MLX's buffer cache grew with every differently-sized final chunk. Fixed: `mx.clear_cache()` after each
  clip, `mlx peak` in the log line (≈3.5 GB per 16 kHz clip), and per-clip saving for resume. The resume cache is tied to the
  dataset and settings: if you change `--variant`, `--chunk-seconds` or `--limit`, use a new `--output` or `--overwrite`.
- **Memory on long clips (lazy graph).** MLX is lazy, and the library's `chunked_hann_ola` queues every chunk of a clip
  before computing. On a 5-min 44.1 kHz eval-d1 clip that is ~120 chunks × 30 blocks × hundreds of scan steps of graph nodes:
  the footprint hit 77 GB within minutes, mostly CPU-side `MALLOC_SMALL`, which MLX's own peak counter does not show. Since
  2026-10-26 04:40, `denoise_dataset.py` swaps in `eager_chunked_hann_ola()` (an `mx.eval` after each chunk). Its output is
  bit-identical (`verify_eager_ola.py`). WhatsApp, kan and fleurs were denoised before the swap. That doesn't matter, because
  the outputs are identical; it only changes memory use.
- **Logs of the first queue pass were overwritten.** The queue was restarted at 04:40 (for the fix above) and the old
  `run_all_datasets.sh` wrote `>` to the per-dataset logs. So the original per-clip lines of WhatsApp/kan/fleurs are gone; their
  timings are recorded in this guide. The script now appends (`>>`). Earlier queue log: `logs/run_all_datasets_part1.log`.
  First eval-d1 attempt: `logs/pipeline_ivrit_ai_eval_d1_stopped_lazy_graph.log`.
- **GPU.** Both the denoiser and `mlx_whisper` run on the GPU (MLX default `Device(gpu, 0)`). Low "Device Utilization" readings
  come from small sequential steps, not CPU fallback. `--devices gpu,gpu,gpu,gpu` only means 4 worker processes on the same GPU
  (`mlx_engine.py` ignores the device string). Per-clip `transcription_time` with 4 workers is ≈4× a single-process run, because
  they share the GPU.
- **Sample rate.** The denoiser runs at, and writes, the input's own rate (no resampling). Stereo is mixed to mono. Whisper
  resamples to 16 kHz in `mlx_engine.py` anyway. Seeded audio is always 16 kHz, because the augmenter outputs 16 kHz.
- **`--limit N` is not a small test** when a dataset has few long clips (eval-d1 has 17 × 5 min). Check durations first.
- `datasets` 5.x in the denoise venv prints `trust_remote_code is not supported`. It is harmless; all 5 datasets load.
- `origin/feature/benchmark-with-augs` can't be imported as pushed (indentation bug), hence the vendored `seeded_aug/`.
- Logs: `datasets.map` saves every 16 examples, and pipes through `grep`/`tail` buffer output. Write straight to a log file
  (`> log 2>&1`, `python -u`) and use the per-clip lines.
- Peek inside a running Python process without stopping it: `sudo /opt/homebrew/bin/uvx py-spy dump --pid <PID> --locals`
  (needs a real Terminal for the sudo password).

---

## 7. Which model, and extending it
- Model: `/Users/Shared/asr/training/lv3-mix-na-b32-kd-low-smooth-mlx`. It is byte-identical (same sha1) to
  `/Users/Shared/asr/mlx/lv3-mix-na-b32-kd-low-smooth-mlx`, which the original `results_2` runs used. Trained with (abridged)
  `train-whisper.py --noise_augmentation --noise_config preprocess/noise_config.yaml --model_name ivrit-ai/whisper-large-v3 --kd_weight 0.3 --label_smoothing 0.005 --max_steps 7000 ...`.
- **Another model:** `./for_eval_denoise/run_dataset_pipeline.sh <label> /Users/Shared/asr/mlx/<model>`. The denoised and seeded
  audio are reused (they don't depend on the model); only the evaluations run. Results go to `<basename>-c` / `<basename>-s`
  folders, which pair with the same-named folders in `results_2`.
- **Another HF dataset:** `denoise_dataset.py --output denoised --dataset "org/name[@config]:split:text_col=label"`, then
  `evaluate_model.py --dataset <folder>:split:text_col` (run_bench only knows its 5 labels).
- **Laughter-preserving denoiser:** `--variant laughter` (downloads and converts weights the first time). Use a different
  `--output`, e.g. `denoised_laughter`.
