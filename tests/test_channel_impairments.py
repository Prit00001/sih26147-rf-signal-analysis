"""Covers P6: multipath, flat fading, and timing drift added to the
synthetic channel model (synth/generator.py). All three default to off
(empty taps / zero rate / zero drift), so every pre-existing test's
behavior is unchanged -- these tests exercise only the new, opt-in paths.
"""

from __future__ import annotations

import numpy as np

from sigscope.synth.generator import generate_signal


def test_multipath_tap_changes_the_signal_and_raises_average_power() -> None:
    rng_seed = 11
    clean, _ = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=30.0, seed=rng_seed
    )
    echoed, _ = generate_signal(
        "qpsk",
        num_symbols=2000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=30.0,
        seed=rng_seed,
        multipath_taps=((5, 0.0, 0.0),),  # one same-strength, in-phase echo
    )
    assert not np.array_equal(clean.samples, echoed.samples)
    # An extra in-phase, same-strength echo must raise average power (direct
    # + echo adds constructively on average for a random-phase-ish signal).
    assert np.mean(np.abs(echoed.samples) ** 2) > np.mean(np.abs(clean.samples) ** 2)


def test_fading_introduces_time_varying_power_envelope() -> None:
    sig, _ = generate_signal(
        "qpsk",
        num_symbols=8000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=30.0,
        fading_rate_hz=20.0,
        seed=12,
    )
    no_fade, _ = generate_signal(
        "qpsk", num_symbols=8000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=30.0, seed=12
    )
    window = len(sig.samples) // 10
    faded_powers = [
        np.mean(np.abs(sig.samples[i * window : (i + 1) * window]) ** 2) for i in range(10)
    ]
    clean_powers = [
        np.mean(np.abs(no_fade.samples[i * window : (i + 1) * window]) ** 2) for i in range(10)
    ]
    # Coefficient of variation across time windows: near-zero without
    # fading, clearly nonzero with it (the channel gain is actually moving).
    faded_cv = float(np.std(faded_powers) / np.mean(faded_powers))
    clean_cv = float(np.std(clean_powers) / np.mean(clean_powers))
    assert faded_cv > clean_cv * 3


def test_timing_drift_accumulates_over_the_signal_length() -> None:
    sig, _ = generate_signal(
        "qpsk",
        num_symbols=4000,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=40.0,
        timing_drift_ppm=200.0,  # exaggerated on purpose, to be clearly measurable
        seed=13,
    )
    no_drift, _ = generate_signal(
        "qpsk", num_symbols=4000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=40.0, seed=13
    )
    start_diff = float(np.mean(np.abs(sig.samples[:200] - no_drift.samples[:200])))
    end_diff = float(np.mean(np.abs(sig.samples[-200:] - no_drift.samples[-200:])))
    # Drift accumulates with elapsed samples -- negligible near the start,
    # clearly larger by the end, for the SAME underlying modulated signal.
    assert end_diff > start_diff * 5


def test_ground_truth_records_the_new_impairment_parameters() -> None:
    _sig, gt = generate_signal(
        "qpsk",
        num_symbols=500,
        sample_rate=1_000_000.0,
        symbol_rate=100_000.0,
        snr_db=20.0,
        multipath_taps=((2, -3.0, 10.0),),
        fading_rate_hz=5.0,
        timing_drift_ppm=15.0,
        seed=14,
    )
    assert gt.multipath_taps == ((2, -3.0, 10.0),)
    assert gt.fading_rate_hz == 5.0
    assert gt.timing_drift_ppm == 15.0
