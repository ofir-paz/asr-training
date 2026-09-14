# Noise augmentation: end-to-end audit

Trace of `--noise_augmentation` from CLI flag to the mel features the model actually
trains on, bugs found and fixed, and what's left as a documented limitation rather than
a silent surprise. Written after the recent migration off `audiomentations` to the
custom `decide_augmentation()` / `apply()` implementation in `noise_augmentation.py`
(burst placement + per-example coverage isn't something `audiomentations`'
`AddBackgroundNoise` supports, which is why that migration happened).

## The two paths, and which flags pick each one

There are **two completely different places** noise gets mixed in, controlled by
whether `--use_preprocessed` is set:

| | `--train_datasets` (no `--use_preprocessed`) | `--use_preprocessed` |
|---|---|---|
| Where mixing happens | `DatasetPreparator._prepare_example_audio` (`preperator.py:278`), once per example, during `dataset.map(...)` | `DataCollatorSpeechSeq2SeqWithPadding._mix_mel_noise` (`train-whisper.py`), once per example, **every time the example is batched** |
| Domain | raw waveform, before feature extraction | linear mel power, reconstructed from already-extracted log-mel features |
| Realization | baked once, same noise every epoch (cached to disk if you re-run) | fresh draw every epoch/step — same example gets different noise each time it's seen |
| Why | the raw dataset still has audio available to mix into | a preprocessed/saved dataset only stores `input_features` (mel), not raw audio, so there's nothing to mix noise into in the time domain anymore |

`train-whisper.py:840` picks the collator's `noise_augmenter` based on exactly this:

```python
train_noise_augmenter = preparator.noise_augmenter if args.use_preprocessed else None
```

I traced this end to end and **it is wired correctly**: `preparator.noise_augmenter`
is only non-`None` when `--noise_augmentation` was passed (which requires
`--noise_config`), and it's only handed to the live collator when `--use_preprocessed`
is also set. When you're preparing from `--train_datasets` instead, the same
`preparator.noise_augmenter` is used by `prepare_dataset()` to bake noise in once, and
the collator correctly gets `None` (baking noise twice — once at prep time, once again
live — would double-apply it). Eval always gets a clean collator
(`noise_augmenter=None`) regardless, so WER/eval-loss stay comparable across
checkpoints. All of that logic in `main()` is correct as-is; I did not need to change it.

**If you're running with `--use_preprocessed`**, double check the dataset you point it
at was *not* itself prepared with `--noise_augmentation` (i.e. was saved via
`--save_processed` without `--noise_augmentation`, or came from raw
`--train_datasets` without noise) — otherwise you'd get noise baked in **and** noise
mixed live on top, which is a real way to end up with way more/harsher noise than the
`snr_db_range`/`apply_prob` in your config imply.

## Config plumbing: verified fully wired

`_load_noise_config()` (`train-whisper.py:713`) turns every YAML key into
`noise_<key>` and `DatasetPreparator.__init__` accepts `noise_<key>` for every one of
them, which then gets passed straight into the `NoiseAugmenter(...)` constructor. I
checked every key in `noise_config.yaml` against both ends of that chain
(`apply_prob`, `snr_db_range`, `num_noises_range`, `gain_jitter_db`,
`coverage_frac_range`, `burst_len_frac_range`, `perturb_prob`, `time_stretch_range`,
`pitch_shift_semitone_range`, `simulate_radio_channel`, `radio_band_hz`,
`filter_signal_too`, `radio_clip_drive`) — all present and correctly named on both
sides. This was mid-migration in your uncommitted changes (`preperator.py` was missing
`noise_num_noises_range`, `noise_filter_signal_too`, `noise_coverage_frac_range`,
`noise_burst_len_frac_range` as recently as the version before your current edit) but
your working tree already has all of them threaded through correctly — nothing left to
do there.

## Probabilities: verified correct

`decide_augmentation()` (`noise_augmentation.py:230`):
- `apply_prob` — `rng.random() > apply_prob` → `None`. `P(noise applied) = apply_prob`. Correct.
- `perturb_prob` — per-clip `rng.random() < perturb_prob`, independent per clip. Correct.
- `num_noises_range` — `rng.integers(lo, hi + 1)`, correctly inclusive on both ends.
- `snr_db_range`, `coverage_frac_range`, `burst_len_frac_range`, `gain_jitter_db` — all
  plain `rng.uniform(*range)`. Correct.

I ran `preprocess/test/test.py` (a standalone script already in the repo, CPU-only, no
model/dataset load — safe to run alongside a training job) against your real
`noise_dir` (56 clips) with synthetic speech and `apply_prob=0.6`: 4/5 draws got noise,
consistent with the target rate at that sample size, and the logged SNR/gain/coverage
values were all correctly inside their configured ranges. Note its default
`--naug-path` points at a copy of `noise_augmentation.py` that doesn't exist next to
the script — pass `--naug-path preprocess/noise_augmentation.py` explicitly (or copy
the file next to the test script) or it'll `FileNotFoundError` before running anything.

## Bugs found and fixed

1. **Live collator ignored `librosa`-missing fallback** (`train-whisper.py`,
   `_mix_mel_noise`). `NoiseAugmenter.apply()` (the baked path) gates time-stretch/pitch-shift
   behind `_HAS_LIBROSA` and degrades gracefully with a warning if librosa isn't
   installed. The collator's mel-domain reimplementation called
   `librosa.effects.time_stretch`/`pitch_shift` unconditionally — if librosa were ever
   missing from the environment, the first noise decision that rolled a
   stretch/pitch perturbation would hard-crash training mid-run instead of just
   skipping perturbation like the baked path does. Fixed to use the same
   `_HAS_LIBROSA` guard (librosa **is** installed in your `.venv`, so this wasn't
   currently firing, but it's a landmine for any environment change).

2. **Hardcoded `sr=16000`** in the same `pitch_shift` call, instead of
   `self.noise_augmenter.target_sampling_rate`. Only correct by coincidence (Whisper is
   always 16kHz); fixed to derive from the augmenter so it can't silently drift out of
   sync if that ever changes.

3. **`simulate_radio_channel` / `filter_signal_too` / `radio_clip_drive` are silently
   no-ops in the live collator path.** These are waveform-domain effects (a nonlinear
   `tanh` soft-clip, a bandpass filter) applied to the *mixed* signal in
   `NoiseAugmenter.apply()`. The live collator only ever has mel features for the
   speech, never a mixed waveform, so there is no way to actually apply them there —
   this isn't a bug you can "just fix", it's a real capability gap between the two
   paths. What *was* a bug: nothing told you this was happening. Your current
   `noise_config.yaml` has `simulate_radio_channel: false`, so this wasn't biting you
   today — but if you ever flip it on while training with `--use_preprocessed`, you'd
   get radio-channel simulation on the baked/preprocessing path and **silently not**
   on the live path, with no indication anything was skipped. Added a one-time warning
   in `DataCollatorSpeechSeq2SeqWithPadding.__post_init__` that fires if the live
   collator is constructed with a `noise_augmenter` that has these settings enabled.

4. **`NoiseAugmenter.simulate_radio_channel` defaulted to `True`** in the dataclass
   field (`noise_augmentation.py`), contradicting its own docstring ("Set to true to
   apply...", implying off-by-default) and the shipped `noise_config.yaml`
   (`simulate_radio_channel: false`). It happened to be harmless today because both
   real call sites (`DatasetPreparator`, `preprocess/test/test.py`) always pass the
   argument explicitly — but it's a footgun for any future direct construction of
   `NoiseAugmenter(...)` that omits it. Flipped the default to `False` to match the
   documented/shipped behavior.

None of these four change behavior for your current `noise_config.yaml`
(`simulate_radio_channel: false` already) — they only change what happens if that
setting, or a missing-librosa environment, is ever hit. Safe to pick up whenever you
next restart training.

## Known limitation, not fixed (needs a decision from you)

**RNG state is shared and pickled across `dataset.map(num_proc=N)` workers.**
`DatasetPreparator.seed` (a single `np.random.Generator`) drives *every* per-example
decision: shift augmentation, timestamp/prev-text sampling, resample augmentation, and
now noise. When `--ds_processor_proc_num > 1`, `dataset.map` pickles the whole
`DatasetPreparator` (RNG state included) to each worker process *before* any example is
processed, so every worker starts from the **identical** RNG state. That means the
first decision each worker makes (e.g. the noise decision for the first example in each
shard) is byte-for-byte identical across shards — same clip picked, same SNR, same
coverage — and decisions stay correlated for a while after that too, until enough
conditional branches (whether noise applied, how many clips, etc.) cause the workers'
consumption of the RNG stream to drift apart. Net effect: less real randomness/variety
across the dataset than `apply_prob`/`snr_db_range` etc. imply, scaling with
`--ds_processor_proc_num`.

This is a pre-existing pattern (not introduced by the noise work) and your current
default is `proc_num=1`, where it's a non-issue. I did not change it because:
- it touches the seeding that `dataset.map`'s on-disk cache is keyed against — changing
  it would silently invalidate any cached preprocessed runs, and
- I don't know whether you rely on the current seed for reproducibility elsewhere.

If you ever run with `--ds_processor_proc_num > 1` and want proper decorrelation, the
fix is to seed each worker independently — e.g. `dataset.map(..., with_rank=True)` and
build a per-worker `np.random.default_rng` from `numpy.random.SeedSequence(base_seed).spawn(num_proc)[rank]`
instead of sharing one `Generator` object across all of them.

## Known approximation, documented not fixed

**The live collator's SNR is not the same quantity as the baked path's SNR.**
`NoiseAugmenter.apply()` computes SNR from raw-waveform RMS. `_mix_mel_noise` can't do
that (no waveform available), so it approximates SNR using the mean RMS of the
*mel-power* spectrogram instead — mathematically a different quantity (mel filterbank +
log compression change the relationship between waveform loudness and mean mel power,
and speech with more silence will have a different waveform-RMS-to-mel-power-RMS ratio
than continuous speech). I verified the mel-domain math itself is correct (it correctly
inverts Whisper's `(log10(mel)+4)/4` normalization and mixes power-domain, which is the
right way to combine two independent signals' spectra), but a `snr_db` of, say, 10dB
will not sound identical between the baked and live paths. If you're tuning
`snr_db_range` by ear/eval against one path, expect to retune slightly if you switch to
the other.

## Unrelated stray change

`git diff` also shows `audio-set/process.ipynb` with a one-line kernelspec metadata
change (`.venv (3.10.21)` → `.venv (3.10.21.final.0)`). That's not something this audit
touched — looks like an IDE/Jupyter kernel-picker refresh. Flagging it since it showed
up in the same working tree; revert or commit it separately as you see fit.

## Files touched by this audit

- `train-whisper.py` — items 1-3 above (librosa guard, sample-rate fix, radio-channel warning).
- `preprocess/noise_augmentation.py` — item 4 (default flip).
- `preprocess/noise_config.yaml` — trailing newline only, no behavior change.
- `preprocess/preperator.py` — **not modified by me**; your existing uncommitted changes
  there (threading through `num_noises_range`, `filter_signal_too`,
  `coverage_frac_range`, `burst_len_frac_range`) were already correct and complete.
