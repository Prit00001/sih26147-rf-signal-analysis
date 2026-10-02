"""Covers: FR-03 (DC removal, IQ-imbalance correction, normalization, resampling), NFR-01 (chunking)."""

from __future__ import annotations

import numpy as np

from sigscope.core.signal import Signal
from sigscope.preprocess.dc_iq import correct_iq_imbalance, normalize_power, remove_dc
from sigscope.preprocess.resample import chunk_iter, resample_signal


def test_remove_dc() -> None:
    samples = (np.ones(1000, dtype=np.complex64) * (2 + 3j)) + np.random.default_rng(0).standard_normal(
        1000
    ).astype(np.complex64) * 0.01
    corrected = remove_dc(samples)
    assert abs(np.mean(corrected)) < 0.05


def test_correct_iq_imbalance_recovers_circular_signal() -> None:
    rng = np.random.default_rng(0)
    n = 20000
    ideal = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    gain = 1.3
    phi = np.deg2rad(15.0)
    impaired = ideal.real + 1j * gain * (ideal.imag * np.cos(phi) + ideal.real * np.sin(phi))
    impaired = impaired.astype(np.complex64)

    before_gamma = abs(np.mean(impaired.real * impaired.imag))
    corrected = correct_iq_imbalance(impaired)
    after_gamma = abs(np.mean(corrected.real * corrected.imag))
    assert after_gamma < before_gamma * 0.1


def test_normalize_power() -> None:
    samples = (np.random.default_rng(0).standard_normal(5000) * 7).astype(np.complex64)
    normalized = normalize_power(samples)
    assert abs(np.mean(np.abs(normalized) ** 2) - 1.0) < 0.05


def test_normalize_power_zero_signal_is_noop() -> None:
    samples = np.zeros(100, dtype=np.complex64)
    assert np.array_equal(normalize_power(samples), samples)


def test_resample_signal_preserves_duration_ratio() -> None:
    sig = Signal(samples=np.exp(1j * np.linspace(0, 100, 10000)).astype(np.complex64), sample_rate=10000.0)
    resampled = resample_signal(sig, target_rate=5000.0)
    assert resampled.sample_rate == 5000.0
    ratio = resampled.num_samples / sig.num_samples
    assert abs(ratio - 0.5) < 0.01


def test_chunk_iter_covers_all_samples_no_copy_semantics() -> None:
    samples = np.arange(1000, dtype=np.complex64)
    chunks = list(chunk_iter(samples, chunk_size=128))
    assert sum(len(c) for c in chunks) == 1000
    assert np.array_equal(np.concatenate(chunks), samples)


def test_chunk_iter_with_overlap() -> None:
    samples = np.arange(100, dtype=np.complex64)
    chunks = list(chunk_iter(samples, chunk_size=20, overlap=5))
    assert len(chunks[1]) == 25  # 20 + 5 samples of overlap context
