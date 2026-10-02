"""Covers: FR-10 (FSK demodulation)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.demod.fsk import demodulate_fsk
from sigscope.synth.generator import FSK_BITS_PER_SYMBOL, generate_signal


def _tx_symbols(bits: list[int], bps: int) -> np.ndarray:
    tx_bits = np.array(bits, dtype=np.int64)
    n_full = len(tx_bits) // bps
    weights = 1 << np.arange(bps - 1, -1, -1)
    return (tx_bits[: n_full * bps].reshape(-1, bps) @ weights).astype(np.int64)


@pytest.mark.parametrize("mod", ["2fsk", "4fsk", "8fsk"])
def test_fsk_ber_low_at_high_snr(mod: str) -> None:
    sig, gt = generate_signal(
        mod,
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        seed=1,
    )
    result = demodulate_fsk(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
    tx_symbols = _tx_symbols(gt.bits, FSK_BITS_PER_SYMBOL[mod])
    a, b = result.decided_indices, tx_symbols[: len(result.decided_indices)]
    length = min(len(a), len(b))
    ser = float(np.mean(a[:length] != b[:length]))
    assert ser < 0.01


@pytest.mark.parametrize("mod", ["2fsk", "4fsk", "8fsk"])
def test_fsk_degrades_gracefully_with_snr(mod: str) -> None:
    def ser_at(snr_db: float) -> float:
        sig, gt = generate_signal(
            mod,
            num_symbols=3000,
            sample_rate=1_000_000.0,
            symbol_rate=100_000.0,
            snr_db=snr_db,
            seed=1,
        )
        result = demodulate_fsk(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
        tx_symbols = _tx_symbols(gt.bits, FSK_BITS_PER_SYMBOL[mod])
        a, b = result.decided_indices, tx_symbols[: len(result.decided_indices)]
        length = min(len(a), len(b))
        return float(np.mean(a[:length] != b[:length]))

    ser_high = ser_at(20.0)
    ser_low = ser_at(0.0)
    assert ser_high <= ser_low  # monotonic (or equal) degradation as SNR drops


def test_fsk_timing_offset_recovers() -> None:
    sig, gt = generate_signal(
        "4fsk",
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        timing_offset_frac=0.4,
        seed=2,
    )
    result = demodulate_fsk(sig.samples, "4fsk", sig.sample_rate, gt.symbol_rate)
    tx_symbols = _tx_symbols(gt.bits, 2)
    a, b = result.decided_indices, tx_symbols[: len(result.decided_indices)]
    length = min(len(a), len(b))
    ser = float(np.mean(a[:length] != b[:length]))
    assert ser < 0.02
