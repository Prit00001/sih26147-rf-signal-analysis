"""Covers: FR-10 (CMA/LMS equalizer). Confirms the equalizer stays close to
identity (does not damage BER) on this project's no-multipath channel model
-- see psk_qam.py's use_equalizer docstring for the measured finding that
motivated defaulting it off."""

from __future__ import annotations

import numpy as np
from demod_test_utils import best_aligned_ser, tx_symbol_indices

from sigscope.demod.psk_qam import demodulate_psk_qam
from sigscope.synth.generator import generate_signal


def test_equalizer_does_not_regress_ber_at_high_snr() -> None:
    sig, gt = generate_signal(
        "qpsk", num_symbols=4000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=1
    )
    tx_symbols = tx_symbol_indices(gt.bits, "qpsk")

    without_eq = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, gt.symbol_rate, use_equalizer=False)
    with_eq = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, gt.symbol_rate, use_equalizer=True)

    ser_without, *_ = best_aligned_ser(without_eq.symbols, tx_symbols, "qpsk")
    ser_with, *_ = best_aligned_ser(with_eq.symbols, tx_symbols, "qpsk")
    assert ser_without < 0.02
    assert ser_with < 0.05  # allow some equalizer-induced noise, but not a breakdown


def test_equalizer_stays_near_identity_on_flat_channel() -> None:
    from sigscope.demod.equalizer import cma_lms_equalize

    rng = np.random.default_rng(0)
    symbols = (rng.standard_normal(500) + 1j * rng.standard_normal(500)).astype(np.complex64)
    symbols /= np.sqrt(np.mean(np.abs(symbols) ** 2))
    out = cma_lms_equalize(symbols, modulus=1.0)
    # No actual channel distortion to correct: output should stay close to input.
    error = np.mean(np.abs(out - symbols) ** 2)
    assert error < 0.5
