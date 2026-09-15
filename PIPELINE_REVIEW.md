# Pipeline review: `train-whisper.py` and everything it calls

A full trace of the training pipeline, written so you can check whether I understood it
correctly. Scope: `train-whisper.py` and every **local** module it reaches —
`preprocess/preperator.py`, `preprocess/augmentation.py`, `preprocess/noise_augmentation.py`,
`preprocess/utils.py`, `run_naming.py`. Third-party packages in `.venv` (transformers,
datasets, torchaudio, librosa, …) are treated as trusted and not reviewed.

Part 1 is my understanding of how it works. Part 2 is bugs found and fixed. Part 3 is
things that look wrong or risky but I deliberately did **not** change.

---

## Part 1 — How the pipeline works

### 1.1 The one thing that determines everything: baked vs. live

Almost every design decision downstream follows from **which of the two modes you're in**.

| | `--train_datasets` (baked) | `--use_preprocessed` (live) |
|---|---|---|
| Where audio augmentation happens | `DatasetPreparator._prepare_example_audio`, once, during `dataset.map()` | `DataCollatorSpeechSeq2SeqWithPadding.__call__`, every time a batch is built |
| Domain it operates in | raw waveform (real DSP) | mel power (approximation — no waveform exists anymore) |
| Realization | frozen into the dataset; identical every epoch | fresh every epoch, every time the example is seen |
| What's on disk | raw audio + transcripts | `input_features` (mel), `labels`, `pad_value`, `pad_amount`, `input_length` |

The reason the live path can't just reuse the baked path: once a dataset has been
preprocessed, **the waveform is gone** — only mel features were saved. You cannot
downsample or SNR-mix a spectrogram the way you would audio, so the live path re-derives
the same operations in mel-power space, accepting some approximation error.

### 1.2 Execution order in `main()`

1. `parse_arguments()`.
2. Two mutual-exclusion checks: `--use_preprocessed` can't combine with
   `--train_datasets`/`--eval_datasets`, nor with `--save_processed`.
3. `WhisperProcessor.from_pretrained(model_name, language, task="transcribe")`.
4. If `--noise_augmentation`: `_load_noise_config()` reads the YAML, `--noise_dir`
   overrides its `noise_dir`, and `noise_augmentation=True` is injected. Guard: noise
   augmentation requires `--noise_config`; `--noise_dir` alone is an error.
5. **`DatasetPreparator` is constructed unconditionally** — even in `--use_preprocessed`
   mode, where it never prepares anything. It's built anyway because constructing it is
   what loads the `NoiseLibrary` (all noise clips decoded into RAM), and the live collator
   borrows that object. Slightly surprising, but intentional.
6. Dataset branch:
   - `--use_preprocessed`: `load_from_disk` each path (falling back to `load_dataset`).
     One dataset → used directly. Several → `interleave_datasets` with
     `--use_preprocessed_probs`, `stopping_strategy="all_exhausted"`, fixed `seed=745`
     (the seed matters: every distributed rank must interleave identically or the
     per-rank dataloader lengths diverge and collective sync breaks).
   - `--save_processed`: prepare train+eval, `save_to_disk`, **`return`** — exits before
     any model is loaded. This is the mode that produces `--use_preprocessed` inputs.
   - otherwise: prepare train+eval in memory and keep going.
7. `--max_eval_set_size` → `shuffle(seed=745).select(range(n))`.
8. **Name resolution** (`run_naming.py`): if `--output_model_name` is omitted, generate one
   from the hyperparameters (`lv3-mix-na-b32-kd0.1-sm0.1-bf16`). Then, unless resuming,
   `dedupe_name` appends `_2`/`_3`/… if that directory already exists. Resuming skips
   dedup deliberately — versioning the name would point the trainer at an empty directory
   and silently start from scratch. Finally `run_name` is forced equal to
   `output_model_name` (`--run_name` is an accepted no-op).
9. `_init_run_tracking()` — `wandb.init()` with `{**vars(args), "noise": noise_kwargs}` and
   `wandb.save(noise_config)`. Done *before* the Trainer exists, which is the supported way
   to inject custom config: `WandbCallback.setup()` checks `if wandb.run is None` and, finding
   one, only layers `TrainingArguments` on top rather than replacing it.
10. Collators: the train one gets the augmenters; the eval one is always clean, and is only
    built at all if some live augmentation is active.
11. Model load → `forced_decoder_ids=None`, `suppress_tokens=[]`, assert
    `max_target_positions == 448`, optional QLoRA, `use_cache=False`, and `model.generate`
    is `partial`'d with language/task.
12. Warmup: exactly one of steps/ratio reaches Transformers (they're mutually exclusive in
    argparse, and `warmup_steps > 0` would otherwise silently beat `warmup_ratio`).
13. Optional frozen KD teacher, held at the same reduced precision as `--mixed_precision`.
14. `Seq2SeqTrainingArguments` → `WhisperDistillationTrainer` → `train()` → `save_model()`.

### 1.3 The baked path in detail (`DatasetPreparator.prepare_dataset`)

1. `cast_column("audio", Audio(sampling_rate=16000))` — lazy; nothing decodes yet.
2. `_validate_dataset_features` — requires `audio` (an `Audio` feature) and `transcript`.
3. `_select_transcribable_entries` — drops blank transcripts (they'd teach the model to
   emit nothing for audible speech). Scoped to the transcript column so audio stays undecoded.
4. If either sampling probability < 1.0, `estimate_attribute_ratios` samples the dataset
   (up to 4000 examples, stopping early once a Beta confidence interval is tight enough) to
   measure how many examples actually *have* timestamps/prev-text. `_relative_sampling_ratio`
   then converts your *dataset-wide target* into a *per-eligible-example* rate — because you
   can only sample the attribute where the choice exists, and cross-over segments force
   timestamps on regardless, putting a floor under the achievable share.
5. `dataset.map(_prepare_example_fn)` — per example:
   - **`_decide_example_augmentation`** (all randomness decided up front, from the shared
     `self.seed`, so decisions are reproducible and independent of when audio gets loaded):
     shift amount (`beta(2,3) * max_shift`, skewed toward zero), the noise decision dict, and
     the resample coin flip.
   - **`_prepare_example_audio`** — the real DSP, in this order:
     resample to 16 kHz → `shift_audio_forward` → `resample_augment` → `noise_augmenter.apply`
     → feature extraction → **padding compression**: Whisper always emits 3000 frames (30 s),
     so rather than storing thousands of identical silence frames, it stores the real region
     plus one `pad_value` column and a `pad_amount` count.
   - **`_prepare_example_text`** — tokenize; strip timestamps if sampled off *and* strippable;
     optionally prepend truncated prev-text; optionally inject synthetic timestamps; emit
     `labels = prev_ids + prefix_tokens + token_ids + [eot]`.
6. `_select_legal_entries` — drop anything whose labels exceed 448.

### 1.4 The live path in detail (`DataCollatorSpeechSeq2SeqWithPadding.__call__`)

Per example in the batch:

1. `base_features = tensor(input_features)` — shape `(n_mels, feat_len)`, padding already stripped.
2. `_resample_mel` — with probability `resample_prob`: convert log-mel → linear power, zero
   every mel bin above the target rate's Nyquist (`resample_mel_cutoff_bin` via
   `librosa.mel_frequencies`), convert back. This stands in for the downsample/upsample round
   trip, whose dominant audible effect is exactly that bandwidth limiting.
3. `_mix_mel_noise` — roll `decide_augmentation`; if it fires, `build_noise_waveform` realizes
   the decision into an actual noise **waveform** (the noise library *is* waveforms, and
   burst placement / pitch / stretch are inherently time-domain), feature-extract that noise,
   then mix in linear mel power at the target SNR.
4. Re-append `pad_amount` copies of `pad_value` → back to `(n_mels, 3000)`.
5. `torch.stack` — **this is why `feat_len + pad_amount` must be identical across the batch.**
   It always is in real data, because Whisper pads everything to the same 3000 frames.
6. Labels: pad → `decoder_input_ids = labels[:, :-1]` → shift labels left by one → mask
   padding to `-100` → mask everything up to and including the start-of-transcript token to
   `-100` (so prev-text prompt tokens don't contribute loss).

### 1.5 The log-mel ⇄ power conversion

Whisper stores `(log10(mel) + 4) / 4` after clamping the dynamic range to `max - 8`. Both
constants are a matched pair: the `8` fixes the window width, and `4 = 8/2` is the half-width
that maps that window onto roughly `[-1, 1]`. `_whisper_log_mel_to_power` /
`_whisper_power_to_log_mel` are exact inverses of each other (pure algebra on a log curve —
**no waveform reconstruction, nothing lossy**). Any mel-domain augmentation converts down to
linear power, operates, and converts back.

### 1.6 The training loop (`WhisperDistillationTrainer`)

- `compute_loss` — cross entropy with label smoothing (training only), plus an optional
  KL penalty against a frozen teacher (training only, so eval loss stays plain CE and stays
  comparable across runs).
- `_transcription_loss` uses `reduction="sum"` normalized by `num_items_in_batch` — the token
  count of the *whole* accumulation window — which is the documented fix for HF's gradient-
  accumulation loss bug. `model_accepts_loss_kwargs = True` then stops Transformers from
  dividing by `gradient_accumulation_steps` a second time (which would silently train at
  `lr / accum`).
- `get_eval_dataloader` temporarily swaps in the clean collator. This works because the
  `DataLoader` captures `collate_fn` at construction, so restoring `self.data_collator`
  immediately afterward is safe.
- `log()` reports `ce` and `kd_kl` separately, so a rising total loss can be attributed.

---

## Part 2 — Bugs found and fixed

### 2.1 Live mel noise could synthesize hiss out of digital silence

**Severity: real, silent corruption of training data (though currently unreachable).**

`mix_audio_at_snr` (waveform path) refuses to mix when either side is near-silent. The
mel path had no such guard. Digital silence does **not** round-trip to zero mel power — it
lands on Whisper's `1e-10` log floor — so scaling it to hit the target SNR multiplies it by
a huge factor. Measured: **×28,111,882**, turning pure silence into flat broadband noise at
exactly the requested SNR.

```python
# before — no guard at all
snr_db = noise_decision["snr_db"]
target_noise_rms = snr_target_rms(signal_rms, snr_db)

# after
if noise_rms < _MEL_SILENCE_RMS_THRESHOLD or signal_rms < _MEL_SILENCE_RMS_THRESHOLD:
    return base_features
```

### 2.2 …and the obvious fix for 2.1 was itself the same bug again

Worth calling out separately, because it's the *exact* class of bug found earlier in
`_rms` (the dead `1e-8` threshold). My first attempt reused
`preprocess.noise_augmentation._SILENCE_RMS_THRESHOLD = 2e-6` for the mel guard. **That can
never fire in mel space**: the waveform floor is `1e-6`, but the mel-power floor is
`sqrt(1e-10) ≈ 1e-5` — *ten times above the threshold*. The test I wrote caught it.

The fix is a separate, correctly-derived constant with the reasoning written down:

```python
_MEL_SILENCE_RMS_THRESHOLD = 2e-5   # sits just above the ~1e-5 mel floor
```

**Lesson worth keeping:** a threshold is only meaningful relative to the floor of the
function it's compared against. Two different domains need two different constants, and
sharing one across them is silently broken, not DRY.

### 2.3 `--ignore_data_skip` was parsed but never used

The flag existed, was documented in `--help`, and did **nothing** — it was never forwarded
to `Seq2SeqTrainingArguments`. Anyone resuming a long run and passing it to skip the
expensive data-replay got no effect and no warning.

```python
# after (added to Seq2SeqTrainingArguments)
ignore_data_skip=args.ignore_data_skip,
```

### 2.4 `hub_model_id` is malformed when the output name is a path

Your own runs use an absolute path (`/Users/Shared/asr/training/lv3-...`). Interpolating
that into `f"{org}/{name}"` produced `ivrit-ai//Users/Shared/asr/training/lv3-...`, which is
not a valid Hub repo id. **Currently masked because you always pass `--skip_push_to_hub`**
(which sets it to `None`) — it would break the moment you dropped that flag.

```python
# before
hub_model_id=f"{args.hf_org_name}/{args.output_model_name}" if not args.skip_push_to_hub else None
# after — only the final path segment is a legal repo name
hub_model_id=f"{args.hf_org_name}/{Path(args.output_model_name).name}" if ... else None
```

### 2.5 Hardcoded 16 kHz / hop 160 in the live noise path

`sampling_rate=16000` and `feat_len * 160` were literals. Correct only by coincidence of
Whisper's fixed config, and the same smell already fixed once in the `pitch_shift` call.

```python
# after
extractor = self.processor.feature_extractor
audio_samples = feat_len * extractor.hop_length
noise_feat_result = extractor(mixed_noise, sampling_rate=extractor.sampling_rate, ...)
```

### 2.6 Padding tensor rebuilt the slow way on every example

`torch.tensor([pad_value] * pad_amount)` asks torch to convert a Python list of N numpy
arrays — which it warns about and does slowly, **once per example per batch, for the entire
run**. (This is the `UserWarning` that appeared in every test run.)

```python
# before
pad_tensor = torch.tensor([pad_value] * pad_amount).T
# after — one tensor, broadcast into shape
pad_value = torch.as_tensor(np.asarray(feature["pad_value"], dtype=np.float32))
pad_tensor = pad_value.unsqueeze(-1).expand(-1, pad_amount)
```
Verified bit-identical output; the warning is gone.

### 2.7 Loop variable reused instead of the list it was appended to

```python
for preprocessed in args.use_preprocessed:
    dataset_dict = load_from_disk(preprocessed)
    preprocessed_dataset_dicts.append(dataset_dict)

if len(preprocessed_dataset_dicts) == 1:
    train_set = dataset_dict["train"]   # ← the loop variable, not the list
```
Correct today *only* because the loop ran exactly once in that branch. Changed to index the
list explicitly. Latent trap, not a live bug.

### 2.8 Bare `except:` swallowed `KeyboardInterrupt`

`load_datasets` used a bare `except:`, so Ctrl-C during a slow remote dataset load was
caught and silently retried as a local load. Narrowed to `except Exception:`.

---

## Part 3 — Things I did **not** change (and why)

### 3.1 Shared RNG across `dataset.map(num_proc>1)` workers — still open

`DatasetPreparator.seed` is a single `np.random.Generator` driving every per-example decision.
`dataset.map` pickles the whole preparator to each worker **before** any example is processed,
so every worker starts from an identical RNG state and makes correlated decisions. Only
matters if you set `--ds_processor_proc_num > 1` (default is 1). Not fixed because changing
the seeding invalidates `map`'s on-disk cache. Full write-up in
`preprocess/NOISE_AUGMENTATION_AUDIT.md`.

### 3.2 Mel-domain SNR is an approximation of waveform SNR

`_mel_power_rms` is `sqrt(mean(power))` on an already-squared quantity — a loudness proxy,
not true amplitude RMS. So "10 dB SNR" is not numerically the same thing in the baked and
live paths. Inherent to the design; documented, not fixable without reconstructing audio.

### 3.3 Padding is frozen at the pre-augmentation level

`pad_value` was computed during preprocessing and is re-appended **unchanged** after live
augmentation. But `_whisper_power_to_log_mel` recomputes the `max - 8` floor from the
*augmented* real region. Add noise and the real region's floor rises, while the stored pad
columns stay at the old level — a mild discontinuity at the pad boundary. Harmless in
practice (the pad region is masked silence), and fixing it means recomputing the pad value
per example, which costs more than the artifact is worth. Flagging so it isn't a surprise.

### 3.4 Double augmentation is possible and nothing detects it

If you preprocess a dataset **with** `--noise_augmentation` and then train on it with
`--use_preprocessed --noise_augmentation`, you get noise baked in *and* mixed live on top —
roughly double the intended degradation, at an SNR neither setting asked for. Nothing in the
data records that noise was already applied, so this can't be detected automatically.
**Worth checking that your preprocessed datasets under `/Users/Shared/asr/processed/` were
built clean.**

### 3.5 `_prepare_example_fn` catches every exception and returns `None`

A failing example prints `Exception: ...` and returns `None`, which then fails inside
`datasets.map` with a confusing "should return a dict" error rather than the real cause.
It converts a clear error into an unclear one instead of skipping the example. Left alone
because changing error-handling behavior mid-project is riskier than the confusion it saves.

### 3.6 `input_length` is written but never read

`_prepare_example_audio` computes and stores it in every preprocessed dataset; nothing in
the codebase consumes it. Harmless (it's tiny), and it's genuinely useful for debugging or
length-based batching later, so I left it.

### 3.7 `bf16=True if ... else None` passes `None` for boolean arguments

`bf16`, `fp16`, `tf32`, `save_only_model` are all passed `None` rather than `False` when
inactive. Works because `None` is falsy, but it's relying on that rather than stating it.
Cosmetic, pre-existing, left alone.

---

## Part 4 — Verification

- `python -m pytest tests/ -q` → **98 passed** (~6 s, CPU only, no model download).
- One new test added: `test_silent_noise_realization_is_skipped_not_amplified` — it's the
  test that caught bug 2.2.
- `pad_tensor` fix verified bit-identical to the old construction.
- Confirmed every one of the 14 keys in `noise_config.yaml` maps to a real
  `DatasetPreparator` parameter (nothing silently ignored).
- Confirmed `ignore_data_skip` is a genuine `Seq2SeqTrainingArguments` field.
- Confirmed `ReadInstruction.from_spec("train")` handles the plain (unsliced) case, so the
  local-dataset fallback in `load_datasets` isn't broken for the common path.

**Not verified:** no real end-to-end training run was executed — no model weights were
loaded and no GPU was used. Everything above is static analysis plus targeted tests of
individual components and the Trainer's dataloader wiring.
