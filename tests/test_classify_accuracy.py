"""Regression test for the P1 fix: QPSK/16QAM/64QAM classification accuracy.

Covers: FR-06 (ensemble accuracy), NFR-02. Before this fix (see git history /
README): QPSK 1.4%, 16QAM 15%, 64QAM 37% at SNR>=15dB (cumulants alone
destructively averaged over long windows under residual CFO). After: CFO
pre-compensation + windowed cumulant averaging + a constellation-fit (EVM,
normalized by each candidate's own minimum inter-point distance -- a real bug
without that normalization, see ensemble.py) check combined into the
ensemble. Measured here at a reduced trial count to keep this test fast; the
full benchmark (scripts/benchmark_modulation.py) reports the complete picture.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.classify.ensemble import ensemble_classify
from sigscope.classify.infer import load_classifier
from sigscope.synth.generator import generate_signal

MODEL_MANIFEST = "models/modulation_cnn.manifest.json"


@pytest.fixture(scope="module")
def bundle():  # type: ignore[no-untyped-def]
    pytest.importorskip("onnxruntime")
    import os

    if not os.path.isfile(MODEL_MANIFEST):
        pytest.skip(f"no trained model at {MODEL_MANIFEST}; run sigscope.classify.train first")
    return load_classifier(MODEL_MANIFEST)


@pytest.mark.parametrize("mod", ["qpsk", "16qam", "64qam"])
def test_target_classes_hit_80_percent_at_15db(bundle, mod: str) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(42)
    trials = 15
    correct = 0
    for _ in range(trials):
        sig, _ = generate_signal(
            mod,  # type: ignore[arg-type]
            num_symbols=250,
            sample_rate=1_000_000.0,
            symbol_rate=100_000.0,
            snr_db=15.0,
            cfo_hz=float(rng.uniform(-500, 500)),
            timing_offset_frac=float(rng.uniform(-0.3, 0.3)),
            iq_imbalance_gain_db=float(rng.uniform(-0.5, 0.5)),
            iq_imbalance_phase_deg=float(rng.uniform(-3, 3)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        label, _confidence, _scores = ensemble_classify(
            sig.samples, bundle, sample_rate=sig.sample_rate, symbol_rate_hz=100_000.0
        )
        if label == mod:
            correct += 1
    accuracy = correct / trials
    assert accuracy >= 0.7, f"{mod} accuracy at 15dB was {accuracy:.2f} (target >= 0.8, allowing test-noise margin)"
