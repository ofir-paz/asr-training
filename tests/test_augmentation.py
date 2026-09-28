import json
import multiprocessing as mp
import random

import numpy as np
import pytest

from augmentation import (
    AUGMENTATIONS,
    SAMPLE_INDEX_KEY,
    AugmentationPipeline,
    GaussianNoise,
    parse_augment_config,
)

SR = 16000
NOISE_CONFIG = [{"name": "gaussian_noise", "min_snr_db": 5, "max_snr_db": 20}]


def make_audio(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def make_pipeline(seed=0, config=NOISE_CONFIG):
    return AugmentationPipeline.from_config(config, seed=seed)


def _augment_shard(shard):
    # Simulates a process-mode worker; the global seed must not matter
    np.random.seed(42)
    pipeline = make_pipeline()
    audio = make_audio()
    return [(i, pipeline(audio, SR, index=i)) for i in shard]


def test_same_index_same_output():
    audio = make_audio()
    np.testing.assert_array_equal(make_pipeline()(audio, SR, index=3), make_pipeline()(audio, SR, index=3))


def test_different_index_different_output():
    audio = make_audio()
    assert not np.array_equal(make_pipeline()(audio, SR, index=0), make_pipeline()(audio, SR, index=1))


def test_different_seed_different_output():
    audio = make_audio()
    assert not np.array_equal(make_pipeline(seed=0)(audio, SR, index=0), make_pipeline(seed=1)(audio, SR, index=0))


def test_same_index_in_different_datasets_differs():
    audio = make_audio()
    fleurs = AugmentationPipeline.from_config(NOISE_CONFIG, dataset="google/fleurs:he_il:test")
    d1 = AugmentationPipeline.from_config(NOISE_CONFIG, dataset="ivrit-ai/eval-d1::test")
    assert not np.array_equal(fleurs(audio, SR, index=0), d1(audio, SR, index=0))


def test_same_dataset_same_output():
    audio = make_audio()
    a = AugmentationPipeline.from_config(NOISE_CONFIG, dataset="google/fleurs:he_il:test")
    b = AugmentationPipeline.from_config(NOISE_CONFIG, dataset="google/fleurs:he_il:test")
    np.testing.assert_array_equal(a(audio, SR, index=0), b(audio, SR, index=0))


def test_call_order_does_not_matter():
    pipeline, audio = make_pipeline(), make_audio()
    forward = {i: pipeline(audio, SR, index=i) for i in range(5)}
    backward = {i: pipeline(audio, SR, index=i) for i in reversed(range(5))}
    for i in range(5):
        np.testing.assert_array_equal(forward[i], backward[i])


def test_immune_to_global_rng_consumption():
    pipeline, audio = make_pipeline(), make_audio()
    np.random.seed(0)
    expected = pipeline(audio, SR, index=0)
    np.random.rand(100)
    random.random()
    np.testing.assert_array_equal(pipeline(audio, SR, index=0), expected)


@pytest.mark.parametrize("n_workers", [2, 3])
def test_independent_of_process_workers(n_workers):
    pipeline, audio = make_pipeline(), make_audio()
    indices = list(range(6))
    expected = {i: pipeline(audio, SR, index=i) for i in indices}

    shards = [indices[w::n_workers] for w in range(n_workers)]
    with mp.get_context("spawn").Pool(n_workers) as pool:
        parts = pool.map(_augment_shard, shards, chunksize=1)

    for i, out in (pair for part in parts for pair in part):
        np.testing.assert_array_equal(out, expected[i])


def test_appending_augmentation_keeps_existing_randomness():
    audio = make_audio()
    one = make_pipeline(config=NOISE_CONFIG)
    two = make_pipeline(config=NOISE_CONFIG + [{"name": "gaussian_noise", "p": 0.0}])
    np.testing.assert_array_equal(one(audio, SR, index=7), two(audio, SR, index=7))


def test_gaussian_noise_hits_requested_snr():
    audio = make_audio(seconds=5)
    aug = GaussianNoise(min_snr_db=10, max_snr_db=10)
    out = aug(audio, SR, np.random.default_rng(0))
    noise = out.astype(np.float64) - audio
    snr_db = 10 * np.log10(np.mean(audio.astype(np.float64) ** 2) / np.mean(noise**2))
    assert snr_db == pytest.approx(10, abs=0.2)
    assert out.dtype == audio.dtype
    assert out.shape == audio.shape


def test_gaussian_noise_p_zero_and_silence_unchanged():
    rng = np.random.default_rng(0)
    audio = make_audio()
    np.testing.assert_array_equal(GaussianNoise(p=0.0)(audio, SR, rng), audio)
    silence = np.zeros(SR, dtype=np.float32)
    np.testing.assert_array_equal(GaussianNoise()(silence, SR, rng), silence)


def test_gaussian_noise_validation():
    with pytest.raises(ValueError):
        GaussianNoise(min_snr_db=30, max_snr_db=10)
    with pytest.raises(ValueError):
        GaussianNoise(p=1.5)


def test_config_round_trip():
    pipeline = make_pipeline()
    assert pipeline.to_config() == [{"name": "gaussian_noise", "min_snr_db": 5, "max_snr_db": 20, "p": 1.0}]
    rebuilt = AugmentationPipeline.from_config(pipeline.to_config(), seed=pipeline.seed)
    audio = make_audio()
    np.testing.assert_array_equal(rebuilt(audio, SR, index=0), pipeline(audio, SR, index=0))


def test_config_errors():
    with pytest.raises(ValueError, match="Unknown augmentation"):
        AugmentationPipeline.from_config([{"name": "nope"}])
    with pytest.raises(ValueError, match="Invalid parameters"):
        AugmentationPipeline.from_config([{"name": "gaussian_noise", "snr": 5}])
    with pytest.raises(ValueError):
        AugmentationPipeline(seed=-1)


def test_empty_pipeline_is_falsy():
    assert not AugmentationPipeline()
    assert make_pipeline()


def test_registry_contains_gaussian_noise():
    assert AUGMENTATIONS["gaussian_noise"] is GaussianNoise


def test_parse_augment_config(tmp_path):
    assert parse_augment_config(json.dumps(NOISE_CONFIG)) == NOISE_CONFIG
    assert parse_augment_config(json.dumps(NOISE_CONFIG[0])) == NOISE_CONFIG
    path = tmp_path / "aug.json"
    path.write_text(json.dumps(NOISE_CONFIG))
    assert parse_augment_config(str(path)) == NOISE_CONFIG


def test_process_entry_passes_dataset_index():
    evaluate_model = pytest.importorskip("evaluate_model")
    seen = []

    def fake_transcribe(entries):
        seen.extend(e[SAMPLE_INDEX_KEY] for e in entries)
        return [("שלום", 0.0) for _ in entries]

    entries = [{"audio": {"array": make_audio(), "sampling_rate": SR}, "text": "שלום"} for _ in range(3)]
    rows = evaluate_model.process_entry(
        (10, entries, fake_transcribe, "text", evaluate_model.HebrewTextNormalizer(), None)
    )

    assert seen == [10, 11, 12]
    assert [r["id"] for r in rows] == [10, 11, 12]
    assert all(f"metadata_{SAMPLE_INDEX_KEY}" not in r for r in rows)
    assert all(SAMPLE_INDEX_KEY not in e for e in entries)  # caller's entries are not mutated
