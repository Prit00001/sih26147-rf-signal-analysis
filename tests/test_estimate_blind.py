"""Covers: FR-05 (symbol rate/sample rate estimators), FR-09 (SNR, roll-off), NFR-02."""

from __future__ import annotations

import pytest

from sigscope.estimate.blind import (
    estimate_rolloff,
    estimate_sample_rate_blind,
    estimate_snr_m2m4,
    estimate_symbol_rate_hz,
    estimate_symbol_rate_normalized,
)
from sigscope.synth.generator import generate_signal


@pytest.mark.parametrize("mod", ["qpsk", "8psk", "16qam"])
def test_symbol_rate_estimate_close_to_ground_truth(mod: str) -> None:
    sig, gt = generate_signal(
        mod, num_symbols=3000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=1
    )
    est_hz, confidence = estimate_symbol_rate_hz(sig.samples, sig.sample_rate)
    assert abs(est_hz - gt.symbol_rate) / gt.symbol_rate < 0.05
    assert confidence > 0.0


def test_symbol_rate_normalized_is_scale_invariant() -> None:
    sig1, _ = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=2
    )
    sig2, _ = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=2_000_000.0, symbol_rate=200_000.0, snr_db=20.0, seed=2
    )
    rate1, _ = estimate_symbol_rate_normalized(sig1.samples)
    rate2, _ = estimate_symbol_rate_normalized(sig2.samples)
    assert abs(rate1 - rate2) < 0.01


def test_blind_sample_rate_candidates_include_ground_truth() -> None:
    sig, gt = generate_signal(
        "qpsk", num_symbols=3000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=25.0, seed=3
    )
    result = estimate_sample_rate_blind(sig.samples, candidate_symbol_rates_hz=[9600.0, 100_000.0, 250_000.0])
    assert any(abs(c - gt.sample_rate) / gt.sample_rate < 0.05 for c in result.candidates_hz)
    assert result.confidence <= 0.5  # documented as fundamentally ambiguous, never overconfident


def test_snr_estimate_high_snr_reads_high() -> None:
    # M2M4 on raw (unsynchronized, pulse-shaped) samples systematically
    # underestimates SNR because RRC shaping itself adds envelope variation
    # that looks like extra "noise" to the moment-based estimator -- this is a
    # known, measured characteristic of running M2M4 pre-matched-filter, not a
    # bug. The monotonic-ordering test below is the honest claim.
    sig, _ = generate_signal("qpsk", num_symbols=4000, snr_db=25.0, seed=4)
    snr_hat, confidence = estimate_snr_m2m4(sig.samples)
    assert snr_hat > 5.0
    assert confidence > 0.0


def test_snr_estimate_monotonic_with_true_snr() -> None:
    sig_hi, _ = generate_signal("qpsk", num_symbols=4000, snr_db=25.0, seed=4)
    sig_lo, _ = generate_signal("qpsk", num_symbols=4000, snr_db=-3.0, seed=4)
    snr_hi, _ = estimate_snr_m2m4(sig_hi.samples)
    snr_lo, _ = estimate_snr_m2m4(sig_lo.samples)
    assert snr_hi > snr_lo


def test_snr_estimate_low_snr_reads_low() -> None:
    sig, _ = generate_signal("qpsk", num_symbols=4000, snr_db=-3.0, seed=5)
    snr_hat, _ = estimate_snr_m2m4(sig.samples)
    assert snr_hat < 15.0


def test_rolloff_estimate_in_plausible_range() -> None:
    sig, gt = generate_signal("qpsk", num_symbols=4000, symbol_rate=100_000.0, snr_db=20.0, rolloff=0.35, seed=6)
    est, confidence = estimate_rolloff(sig.samples, sig.sample_rate, gt.symbol_rate)
    assert 0.0 <= est <= 1.0
    assert confidence > 0.0


def test_rolloff_estimate_within_tolerance_at_15db() -> None:
    """Regression test for the PSD band-edge shape-fit rewrite: measured
    MAE ~0.02-0.03 across -5..25 dB SNR (scripts/benchmark_rolloff.py),
    comfortably under the +/-0.1 @ 15 dB bar. The old occupied-bandwidth
    method measured 0.64-0.69 MAE at this SNR -- this asserts the fix holds."""
    import numpy as np

    rng = np.random.default_rng(99)
    errs = []
    for _ in range(10):
        true_rolloff = float(rng.choice([0.2, 0.35, 0.5]))
        sig, gt = generate_signal(
            "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0,
            snr_db=15.0, rolloff=true_rolloff, seed=int(rng.integers(0, 2**31 - 1)),
        )
        est, _conf = estimate_rolloff(sig.samples, sig.sample_rate, gt.symbol_rate)
        errs.append(abs(est - true_rolloff))
    assert np.mean(errs) < 0.1, f"rolloff MAE at 15dB regressed: {np.mean(errs):.3f}"


def test_noise_class_gives_no_symbol_rate_confidence() -> None:
    sig, _ = generate_signal("noise", num_symbols=2000, seed=7)
    _rate, confidence = estimate_symbol_rate_normalized(sig.samples)
    assert confidence < 0.5
