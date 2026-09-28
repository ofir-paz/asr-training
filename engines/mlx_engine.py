import time
from typing import Callable
import librosa
import mlx_whisper

from augmentation import SAMPLE_INDEX_KEY, AugmentationPipeline

# Tells evaluate_model this engine applies the --augment config
SUPPORTS_AUGMENTATION = True

def create_app(**kwargs) -> Callable:
    # MLX can load directly from Hugging Face or a local path
    model_path = kwargs.get("model_path", "ivrit-ai/whisper-large-v3")
    augment = AugmentationPipeline.from_config(
        kwargs.get("augment") or [], seed=kwargs.get("aug_seed", 0), dataset=kwargs.get("aug_dataset", "")
    )

    def transcribe_fn(entries):
        # Unpack the list from the benchmark suite
        if not isinstance(entries, list):
            entries = [entries]
        
        results = []
        for entry in entries:
            audio = entry["audio"]
            if isinstance(audio, str):
                # Load audio from a file path
                arr, sr = librosa.load(audio, sr=None, mono=True)
            elif isinstance(audio, dict):
                arr = audio["array"]
                sr = audio["sampling_rate"]
            else:
                arr = audio
                sr = 16000

            # Ensure the audio is at 16kHz for Whisper
            resampled = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            if augment:
                # The per-sample generator is created inside augment() from (aug_seed, index)
                resampled = augment(resampled, 16000, index=entry[SAMPLE_INDEX_KEY])
            t0 = time.perf_counter()
            
            # Run inference natively on Apple Silicon GPU
            out = mlx_whisper.transcribe(
                resampled,
                path_or_hf_repo=model_path,
                language="he"
                # , temperature=0.0  # disables the sampling fallback, which uses MLX's global random state
            )
            
            dt = time.perf_counter() - t0
            
            text = out.get("text", "")
            results.append((text, dt))
            
        return results

    return transcribe_fn