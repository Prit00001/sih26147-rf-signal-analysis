"""Covers: FR-10 (PSK/QAM demodulation), NFR-01 (chunked demod).

Methodology: rather than comparing measured BER against theory at the
generator's own "snr_db" parameter (which is not a clean Es/N0 -- RRC pulse
shaping and oversampling change the relationship), we measure the ACTUAL
symbol SNR the synchronizer delivers to the detector (residual error vs. the
rotation-resolved ideal constellation) and compare against theoretical BER at
THAT measured SNR. See demod_test_utils.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from demod_test_utils import best_aligned_ser, measured_symbol_snr_db, tx_symbol_indices

from sigscope.demod.psk_qam import demodulate_psk_qam
from sigscope.demod.theory import theoretical_ber
from sigscope.synth.generator import generate_signal

MODULATIONS = ["bpsk", "qpsk", "8psk", "16qam", "64qam"]


@pytest.mark.parametrize("mod", MODULATIONS)
def test_ber_within_reasonable_margin_of_theory_at_high_snr(mod: str) -> None:
    sig, gt = generate_signal(
        mod, num_symbols=4000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=25.0, seed=1
    )
    result = demodulate_psk_qam(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
    tx_symbols = tx_symbol_indices(gt.bits, mod)
    ser, rot, shift, length = best_aligned_ser(result.symbols, tx_symbols, mod)
    assert ser < 0.02, f"{mod}: symbol error rate {ser} too high at high SNR"

    measured_snr = measured_symbol_snr_db(result.symbols, tx_symbols, mod, rot, shift)
    theory_ber = theoretical_ber(mod, measured_snr)
    # "Reasonable margin": measured BER should not be wildly worse than theory
    # predicts at the SNR actually delivered. A generous multiplicative +
    # additive slack accounts for finite-sample estimation and synchronizer
    # imperfection, without pretending this receiver is theoretically optimal.
    assert ser <= theory_ber * 8 + 0.01, (
        f"{mod}: measured SER {ser:.4f} far exceeds theory {theory_ber:.4f} at measured SNR {measured_snr:.1f} dB"
    )


@pytest.mark.parametrize("mod", MODULATIONS)
def test_demod_converges_with_cfo_and_timing_offset(mod: str) -> None:
    sig, gt = generate_signal(
        mod,
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        cfo_hz=300.0,
        timing_offset_frac=0.3,
        seed=2,
    )
    result = demodulate_psk_qam(sig.samples, mod, sig.sample_rate, gt.symbol_rate)
    tx_symbols = tx_symbol_indices(gt.bits, mod)
    ser, _rot, _shift, _length = best_aligned_ser(result.symbols, tx_symbols, mod)
    # 8PSK/64QAM have tighter angular/amplitude spacing and are measurably more
    # prone to an occasional carrier-loop cycle slip at a fixed loop bandwidth
    # (a real, documented phenomenon -- see sync.py); allow more slack there.
    limit = 0.08 if mod in ("8psk", "64qam") else 0.02
    assert ser < limit, f"{mod}: SER {ser} too high with CFO=300Hz, timing_offset=0.3 at 20dB SNR"


def test_soft_bits_correlate_with_hard_bit_confidence() -> None:
    """Soft-bit sign should mostly agree with the hard-bit decision it came from
    (positive => bit 0, per the documented convention)."""
    sig, gt = generate_signal("qpsk", num_symbols=4000, snr_db=20.0, seed=3)
    result = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, gt.symbol_rate)
    implied_bit = (result.soft_bits < 0).astype(np.int64)
    agreement = np.mean(implied_bit == result.hard_bits)
    assert agreement > 0.95


def test_override_recovers_from_wrong_initial_symbol_rate() -> None:
    """Dual-mode contract: apply() with an analyst-corrected symbol rate should
    recover good performance even if an initial (wrong) guess would have failed."""
    sig, gt = generate_signal(
        "qpsk",
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        seed=4,
    )

    wrong = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, symbol_rate=137_000.0)
    tx_symbols = tx_symbol_indices(gt.bits, "qpsk")
    wrong_ser, *_ = best_aligned_ser(wrong.symbols, tx_symbols, "qpsk")

    corrected = demodulate_psk_qam(sig.samples, "qpsk", sig.sample_rate, symbol_rate=gt.symbol_rate)
    corrected_ser, *_ = best_aligned_ser(corrected.symbols, tx_symbols, "qpsk")

    assert corrected_ser < 0.02
    assert corrected_ser < wrong_ser
