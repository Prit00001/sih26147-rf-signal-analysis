"""Covers: FR-10, FR-15 (auto_estimate/apply dual-mode contract), NFR-01 (chunked demod)."""

from __future__ import annotations

import numpy as np
from demod_test_utils import best_aligned_ser, tx_symbol_indices

from sigscope.core.signal import Signal, SourceFormat
from sigscope.demod.pipeline import DemodParams, DemodStage, chunked_demodulate
from sigscope.synth.generator import generate_signal


def _as_signal(samples: np.ndarray, sample_rate: float) -> Signal:
    return Signal(samples=samples.astype(np.complex64), sample_rate=sample_rate, source_format=SourceFormat.SYNTHETIC)


def test_auto_estimate_then_apply_round_trip() -> None:
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=3000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        seed=1,
    )
    stage = DemodStage(_as_signal(sig.samples, sig.sample_rate))
    params, confidence = stage.auto_estimate(modulation="qpsk")
    assert confidence > 0.0
    result = stage.apply(params)
    tx_symbols = tx_symbol_indices(gt.bits, "qpsk")
    ser, *_ = best_aligned_ser(result.symbols, tx_symbols, "qpsk")
    assert ser < 0.02


def test_apply_with_analyst_override_symbol_rate() -> None:
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=3000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        seed=2,
    )
    stage = DemodStage(_as_signal(sig.samples, sig.sample_rate))
    override = DemodParams(modulation="qpsk", symbol_rate_hz=gt.symbol_rate)
    result = stage.apply(override)
    tx_symbols = tx_symbol_indices(gt.bits, "qpsk")
    ser, *_ = best_aligned_ser(result.symbols, tx_symbols, "qpsk")
    assert ser < 0.02


def test_apply_rejects_unknown_modulation() -> None:
    sig, _ = generate_signal("qpsk", num_symbols=500, seed=3)
    stage = DemodStage(_as_signal(sig.samples, sig.sample_rate))
    try:
        stage.apply(DemodParams(modulation="not-a-real-modulation", symbol_rate_hz=100_000.0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_chunked_demodulate_covers_a_long_signal() -> None:
    sig, gt = generate_signal(
        "qpsk", num_symbols=40_000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=4
    )
    signal = _as_signal(sig.samples, sig.sample_rate)
    params = DemodParams(modulation="qpsk", symbol_rate_hz=gt.symbol_rate)
    result = chunked_demodulate(signal, params, chunk_seconds=0.02)
    # Each 0.02s chunk re-acquires sync independently, so some symbols near
    # each chunk boundary are lost to transient/acquisition -- fewer total
    # recovered symbols than a single unchunked pass, by design (NFR-01
    # trades some sample efficiency for guaranteed bounded per-chunk memory).
    assert len(result.decided_indices) > 30_000
    assert len(result.hard_bits) == len(result.decided_indices) * 2
