"""Covers: FR-06 (CNN + cumulant ensemble), SR-02 (SHA-256 verified ONNX load,
tampered-model rejection)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigscope.classify.ensemble import cumulant_classify, ensemble_classify
from sigscope.classify.infer import ModelIntegrityError, load_classifier
from sigscope.classify.train import train
from sigscope.synth.generator import generate_signal

pytest.importorskip("torch", reason="classifier training requires the 'train' optional dependency group")


@pytest.fixture(scope="module")
def tiny_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a throwaway, fast (not accurate) model just to exercise the plumbing:
    export/checksum/calibration/ensemble wiring. Real accuracy numbers come from
    scripts/benchmark_modulation.py against the properly-trained models/ artifact.
    """
    out_dir = tmp_path_factory.mktemp("tiny_model")
    train(out_dir, examples_per_class_snr=3, window_length=256, epochs=1, seed=0)
    return out_dir / "modulation_cnn.manifest.json"


def test_load_classifier_succeeds_with_correct_checksum(tiny_manifest: Path) -> None:
    bundle = load_classifier(tiny_manifest)
    assert len(bundle.class_names) == 11
    assert bundle.window_length == 256


def test_tampered_model_is_rejected(tiny_manifest: Path) -> None:
    import json

    manifest = json.loads(tiny_manifest.read_text())
    onnx_path = tiny_manifest.parent / manifest["onnx_file"]
    original = onnx_path.read_bytes()
    tampered = bytearray(original)
    tampered[-1] ^= 0xFF  # flip a bit near the end of the file
    onnx_path.write_bytes(bytes(tampered))
    try:
        with pytest.raises(ModelIntegrityError):
            load_classifier(tiny_manifest)
    finally:
        onnx_path.write_bytes(original)  # restore for other tests in this module


def test_ensemble_classify_returns_valid_probability_distribution(tiny_manifest: Path) -> None:
    bundle = load_classifier(tiny_manifest)
    sig, _ = generate_signal("qpsk", num_symbols=200, snr_db=20.0, seed=1)
    label, confidence, all_scores = ensemble_classify(
        sig.samples, bundle, sample_rate=sig.sample_rate, symbol_rate_hz=100_000.0
    )
    assert label in bundle.class_names
    assert 0.0 <= confidence <= 1.0
    assert abs(sum(all_scores.values()) - 1.0) < 1e-3


def test_cumulant_classify_returns_known_label() -> None:
    sig, _ = generate_signal("bpsk", num_symbols=3000, snr_db=30.0, seed=2)
    label, scores = cumulant_classify(sig.samples)
    assert label in scores
    assert abs(sum(scores.values()) - 1.0) < 1e-3
