"""
test_noise_augmentation.py

Quick listening test for NoiseAugmenter before running a full training job.

For N examples it will:
  - pick a speech clip (from --speech-dir, or synthesize a speech-like tone if you
    don't have any handy)
  - run decide_augmentation() + apply() on it, exactly like DatasetPreparator does
  - save {i}_orig.wav and {i}_aug.wav side by side in --out-dir
  - log the exact random decision for that file to decisions.csv (which noise clip(s)
    were picked, SNR dB, gain jitter, stretch/pitch, coverage) so you can correlate what
    you hear with what was chosen
  - build an index.html with <audio> players so you can click through them in a browser
    instead of digging through a folder

Usage:
    python test_noise_augmentation.py \
        --noise-dir /path/to/noise/clips \
        --speech-dir /path/to/clean/speech/wavs \
        --out-dir ./noise_test_output \
        --n 30

Speech source (pick one, or omit both for synthetic clips):
  --speech-dir      a folder of wav/flac/etc files
  --speech-dataset  a HF `datasets` Arrow dataset dir (saved via `dataset.save_to_disk(...)`).
                    Two dataset shapes are supported automatically:
                      (a) Raw-audio dataset: must have an 'audio' column (HF Audio feature).
                      (b) Pre-processed Whisper dataset: has 'input_features' (128-bin log-mel,
                          shape [128, 3000]) and optionally 'pad_amount'. The mel spectrogram is
                          inverted back to a waveform via Griffin-Lim so the same augmentation
                          code path that runs during training can be exercised with real data.
                          The reconstructed audio is approximate but has the right duration,
                          energy, and broad spectral shape - good enough to verify mixing/SNR.
                    If it is a DatasetDict, pass --split to pick a split.

If neither is given, synthetic speech-like clips (varying-pitch tone bursts with silence gaps,
so there's a real envelope for the mixer to react to) are generated instead - useful if you just
want to sanity check the mixing/DSP mechanics before you have real data on hand.

By default apply_prob is forced to 1.0 (every example gets noise) since the point here is
to listen to the augmentation, not to reproduce the 60% application rate. Pass
--apply-prob to override, e.g. --apply-prob 0.6 to also see how often you'd get clean audio
at your real training setting.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import DatasetDict, load_from_disk

_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

# Whisper feature constants (same values used by WhisperFeatureExtractor)
_WHISPER_N_MELS = 128
_WHISPER_HOP_LENGTH = 160      # samples at 16 kHz  →  10 ms per frame
# For Griffin-Lim *inversion* we use a larger n_fft so that n_stft > n_mels
# (Whisper's n_fft=400 gives only 201 STFT bins for 128 mel bands, which makes
# the InverseMelScale least-squares system rank-deficient).
# n_fft=1024 → 513 STFT bins, well-determined. Hop/win kept at Whisper values
# so frame timing matches the original.
_GRIFFIN_LIM_N_FFT = 1024
_GRIFFIN_LIM_WIN_LENGTH = 1024


def _mel_to_audio(features: np.ndarray, pad_amount: int, sr: int) -> np.ndarray:
    """Invert a Whisper log-mel spectrogram back to an approximate waveform.

    Parameters
    ----------
    features : np.ndarray, shape (n_mels, n_frames)  – stored log-mel values
        (Whisper normalises to roughly [-1, 1.4]; we undo that before inverting).
    pad_amount : int
        Number of padding frames that were appended by DatasetPreparator.
        These are stripped before inversion so the output duration is correct.
    sr : int
        Target sample rate (should be 16 000 for Whisper).

    Returns
    -------
    np.ndarray, shape (n_samples,), dtype float32
    """
    feats = np.asarray(features, dtype=np.float32)  # (n_mels, n_frames)

    # Strip padding frames from the right
    if pad_amount and pad_amount > 0:
        # pad_amount is in Whisper feature frames (each frame = hop_length samples)
        # DatasetPreparator stores it as the number of padded frames
        n_real_frames = feats.shape[1] - int(pad_amount)
        n_real_frames = max(n_real_frames, 1)
        feats = feats[:, :n_real_frames]

    # Undo Whisper's log-mel normalisation:
    #   stored = (log_mel - log_mel_max) / 4 + 1
    #   log_mel = (stored - 1) * 4 + log_mel_max
    # WhisperFeatureExtractor uses log_mel_max = log(clamp(mel, 1e-10)).max()
    # We approximate by reversing the affine part; the exact max cancels in
    # Griffin-Lim because only relative magnitudes matter.
    log_mel = (feats - 1.0) * 4.0          # un-normalise (offset absorbed by GL)
    mel_power = np.exp(log_mel).astype(np.float32)  # linear mel power

    # Griffin-Lim inversion via torchaudio
    mel_tensor = torch.from_numpy(mel_power).unsqueeze(0)  # (1, n_mels, n_frames)

    # Build an InverseMelScale + GriffinLim chain.
    # We use a larger n_fft for inversion (_GRIFFIN_LIM_N_FFT) so the system
    # n_stft (513) >> n_mels (128) — giving a well-determined least-squares problem.
    inv_mel = torchaudio.transforms.InverseMelScale(
        n_stft=_GRIFFIN_LIM_N_FFT // 2 + 1,
        n_mels=feats.shape[0],
        sample_rate=sr,
        f_min=0.0,
        f_max=sr / 2,
    )
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=_GRIFFIN_LIM_N_FFT,
        win_length=_GRIFFIN_LIM_WIN_LENGTH,
        hop_length=_WHISPER_HOP_LENGTH,
        power=1.0,      # InverseMelScale outputs amplitude, not power
        n_iter=32,
    )

    with torch.no_grad():
        linear_spec = inv_mel(mel_tensor)         # (1, n_fft/2+1, n_frames)
        waveform = griffin_lim(linear_spec)       # (1, n_samples)

    audio = waveform.squeeze(0).numpy().astype(np.float32)
    # Normalise to a sensible peak level
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * 0.7).astype(np.float32)


def _load_noise_augmenter_class(naug_path: Path):
    """Import NoiseAugmenter from a given file path, so this script doesn't depend on
    your project's package layout (preprocess/, src/, etc.)."""
    spec = importlib.util.spec_from_file_location("noise_augmentation", naug_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["noise_augmentation"] = module
    spec.loader.exec_module(module)
    return module.NoiseAugmenter


def _load_speech_files(speech_dir: Path) -> list[Path]:
    files = sorted(p for p in speech_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTENSIONS)
    if not files:
        raise ValueError(f"No audio files found under {speech_dir}")
    return files


def _synth_speech(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Fake 'speech': a handful of tone bursts at varying pitch with silence gaps, so the
    envelope/pauses at least loosely resemble speech instead of a flat continuous tone."""
    duration = float(rng.uniform(3.0, 7.0))
    n = int(duration * sr)
    audio = np.zeros(n, dtype=np.float32)
    t_cursor = 0
    while t_cursor < n:
        burst_len = int(rng.uniform(0.15, 0.6) * sr)
        gap_len = int(rng.uniform(0.05, 0.25) * sr)
        burst_len = min(burst_len, n - t_cursor)
        if burst_len <= 0:
            break
        f0 = float(rng.uniform(120, 400))
        tt = np.arange(burst_len) / sr
        # a couple of harmonics + slow vibrato so it's not a pure sine
        tone = (
            0.6 * np.sin(2 * np.pi * f0 * tt)
            + 0.25 * np.sin(2 * np.pi * f0 * 2 * tt)
            + 0.15 * np.sin(2 * np.pi * (f0 * 3 + 5 * np.sin(2 * np.pi * 4 * tt)) * tt)
        )
        envelope = np.hanning(burst_len)
        audio[t_cursor : t_cursor + burst_len] += (tone * envelope).astype(np.float32)
        t_cursor += burst_len + gap_len
    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak * 0.7).astype(np.float32)
    return audio


def _load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32)


def _save_wav(path: Path, audio: np.ndarray, sr: int):
    tensor = torch.from_numpy(np.clip(audio, -1.0, 1.0)).unsqueeze(0)
    torchaudio.save(str(path), tensor, sr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-dir", required=True, type=Path)
    ap.add_argument("--speech-dir", type=Path, default=None, help="Folder of wav/flac/etc speech files")
    ap.add_argument(
        "--speech-dataset",
        type=Path,
        default=None,
        help="Path to a HF `datasets` Arrow dataset (dir saved via save_to_disk) with an 'audio' column, "
        "instead of --speech-dir",
    )
    ap.add_argument("--split", type=str, default=None, help="Split to use if --speech-dataset is a DatasetDict")
    ap.add_argument("--audio-column", type=str, default="audio", help="Column name for the audio feature")
    ap.add_argument("--out-dir", type=Path, default=Path("./noise_test_output"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--naug-path",
        type=Path,
        default=Path(__file__).parent / "noise_augmentation.py",
        help="Path to noise_augmentation.py (defaults to a file next to this script)",
    )
    # Mirrors of DatasetPreparator's noise_* kwargs, so you can reproduce your real config
    ap.add_argument("--apply-prob", type=float, default=1.0, help="Forced to 1.0 by default so every example is audibly augmented")
    ap.add_argument("--snr-db-range", type=float, nargs=2, default=(0.0, 25.0))
    ap.add_argument("--num-noises-range", type=int, nargs=2, default=(1, 1))
    ap.add_argument("--gain-jitter-db", type=float, default=3.0)
    ap.add_argument("--time-stretch-range", type=float, nargs=2, default=(0.97, 1.03))
    ap.add_argument("--pitch-shift-range", type=float, nargs=2, default=(-0.5, 0.5))
    ap.add_argument("--perturb-prob", type=float, default=0.3)
    ap.add_argument("--simulate-radio-channel", action="store_true")
    ap.add_argument("--radio-clip-drive", type=float, default=1.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    NoiseAugmenter = _load_noise_augmenter_class(args.naug_path)

    augmenter = NoiseAugmenter(
        noise_dir=str(args.noise_dir),
        target_sampling_rate=args.sr,
        apply_prob=args.apply_prob,
        snr_db_range=tuple(args.snr_db_range),
        num_noises_range=tuple(args.num_noises_range),
        gain_jitter_db=args.gain_jitter_db,
        time_stretch_range=tuple(args.time_stretch_range),
        pitch_shift_semitone_range=tuple(args.pitch_shift_range),
        perturb_prob=args.perturb_prob,
        simulate_radio_channel=args.simulate_radio_channel,
        radio_clip_drive=args.radio_clip_drive,
    )
    print(f"Loaded {len(augmenter.library)} noise clips from {args.noise_dir}")

    rng = np.random.default_rng(args.seed)

    if args.speech_dataset is not None and args.speech_dir is not None:
        raise ValueError("Pass only one of --speech-dataset or --speech-dir")

    speech_files = None
    speech_ds = None
    if args.speech_dataset is not None:
        speech_ds = load_from_disk(str(args.speech_dataset))
        if isinstance(speech_ds, DatasetDict):
            split = args.split or next(iter(speech_ds.keys()))
            print(f"'{args.speech_dataset}' is a DatasetDict, using split '{split}' (pass --split to change)")
            speech_ds = speech_ds[split]

        # Detect dataset shape: pre-processed Whisper (input_features) vs raw audio
        ds_columns = speech_ds.column_names
        if "input_features" in ds_columns and args.audio_column not in ds_columns:
            speech_ds_mode = "features"
            print(
                f"Detected pre-processed Whisper dataset (has 'input_features', no '{args.audio_column}' column).\n"
                f"  Will invert log-mel spectrogram → audio via Griffin-Lim for each example."
            )
        else:
            speech_ds_mode = "audio"
            from datasets import Audio as HFAudio  # local import to keep top-level clean
            speech_ds = speech_ds.cast_column(args.audio_column, HFAudio(sampling_rate=args.sr))
            print(f"Detected raw-audio dataset ('{args.audio_column}' column present).")

        print(f"Loaded {len(speech_ds)} examples from arrow dataset {args.speech_dataset}")
    elif args.speech_dir is not None:
        speech_files = _load_speech_files(args.speech_dir)
        print(f"Loaded {len(speech_files)} speech files from {args.speech_dir}")
    else:
        print("No --speech-dataset or --speech-dir given, generating synthetic speech-like clips instead.")

    rows = []
    for i in range(args.n):
        if speech_ds is not None:
            pick = int(rng.integers(0, len(speech_ds)))
            example = speech_ds[pick]
            if speech_ds_mode == "features":
                pad_amount = example.get("pad_amount", 0) or 0
                audio = _mel_to_audio(example["input_features"], pad_amount, args.sr)
                src_name = f"arrow[{pick}] (griffin-lim)"
            else:
                audio = np.asarray(example[args.audio_column]["array"], dtype=np.float32)
                src_name = example[args.audio_column].get("path") or f"arrow[{pick}]"
        elif speech_files is not None:
            pick = int(rng.integers(0, len(speech_files)))
            src_path = speech_files[pick]
            audio = _load_audio_mono(src_path, args.sr)
            src_name = src_path.name
        else:
            audio = _synth_speech(rng, args.sr)
            src_name = "synthetic"

        decision = augmenter.decide_augmentation(rng)
        augmented = augmenter.apply(audio, decision)

        orig_name = f"{i:03d}_orig.wav"
        aug_name = f"{i:03d}_aug.wav"
        _save_wav(args.out_dir / orig_name, audio, args.sr)
        _save_wav(args.out_dir / aug_name, augmented, args.sr)

        if decision is None:
            rows.append({"idx": i, "source": src_name, "noise_applied": False})
        else:
            noise_names = [augmenter.library.file_paths[j].name for j in decision["noise_indices"]]
            rows.append(
                {
                    "idx": i,
                    "source": src_name,
                    "noise_applied": True,
                    "snr_db": round(decision["snr_db"], 2),
                    "noise_clips": ";".join(noise_names),
                    "gain_jitters_db": ";".join(f"{g:.1f}" for g in decision["gain_jitters_db"]),
                    "time_stretch_factors": ";".join(f"{s:.3f}" for s in decision["time_stretch_factors"]),
                    "pitch_shift_semitones": ";".join(f"{p:.2f}" for p in decision["pitch_shift_semitones"]),
                    "coverage_fracs": ";".join(f"{c:.2f}" for c in decision["coverage_fracs"]),
                }
            )

    # CSV log
    fieldnames = ["idx", "source", "noise_applied", "snr_db", "noise_clips", "gain_jitters_db",
                  "time_stretch_factors", "pitch_shift_semitones", "coverage_fracs"]
    with open(args.out_dir / "decisions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Browser-playable index
    html_rows = []
    for row in rows:
        detail = "no noise applied" if not row["noise_applied"] else (
            f"SNR {row['snr_db']} dB | {row['noise_clips']} | gain jitter {row['gain_jitters_db']} dB | "
            f"stretch {row['time_stretch_factors']} | pitch {row['pitch_shift_semitones']} st | "
            f"coverage {row['coverage_fracs']}"
        )
        html_rows.append(f"""
        <tr>
          <td>{row['idx']:03d}</td>
          <td>{row['source']}</td>
          <td>{detail}</td>
          <td><audio controls src="{row['idx']:03d}_orig.wav"></audio></td>
          <td><audio controls src="{row['idx']:03d}_aug.wav"></audio></td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NoiseAugmenter listening test</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; vertical-align: middle; }}
th {{ background: #f4f4f4; text-align: left; }}
audio {{ width: 220px; }}
</style></head>
<body>
<h2>NoiseAugmenter listening test ({len(rows)} examples)</h2>
<p>apply_prob used: {args.apply_prob} — noise dir: {args.noise_dir}</p>
<table>
<tr><th>#</th><th>source</th><th>decision</th><th>original</th><th>augmented</th></tr>
{''.join(html_rows)}
</table>
</body></html>"""
    (args.out_dir / "index.html").write_text(html)

    n_noisy = sum(1 for r in rows if r["noise_applied"])
    print(f"\nWrote {len(rows)} pairs to {args.out_dir}")
    print(f"{n_noisy}/{len(rows)} had noise applied")
    print(f"Open {args.out_dir / 'index.html'} in a browser to listen, or check decisions.csv for the raw params.")


if __name__ == "__main__":
    main()