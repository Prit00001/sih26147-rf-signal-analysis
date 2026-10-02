"""Covers: FR-16 (synthetic generator with ground truth, both .IQ and .wav)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigscope.synth.generator import Modulation, generate_signal, write_pair

ALL_MODS: list[Modulation] = ["bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "8fsk", "am", "fm", "noise"]


@pytest.mark.parametrize("mod", ALL_MODS)
def test_generate_signal_produces_finite_unit_ish_power(mod: Modulation) -> None:
    sig, gt = generate_signal(mod, num_symbols=500, snr_db=20.0, seed=42)
    assert np.all(np.isfinite(sig.samples))
    assert sig.num_samples > 0
    assert gt.modulation == mod
    power = np.mean(np.abs(sig.samples) ** 2)
    assert 0.01 < power < 10.0  # RRC shaping legitimately reduces average power vs. symbol energy


def test_generate_signal_is_deterministic_with_seed() -> None:
    sig1, _ = generate_signal("qpsk", num_symbols=200, seed=7)
    sig2, _ = generate_signal("qpsk", num_symbols=200, seed=7)
    np.testing.assert_array_equal(sig1.samples, sig2.samples)


def test_snr_affects_noise_floor() -> None:
    sig_hi, _ = generate_signal("qpsk", num_symbols=2000, snr_db=30.0, seed=1)
    sig_lo, _ = generate_signal("qpsk", num_symbols=2000, snr_db=0.0, seed=1)
    # crude check: low-SNR signal has a visibly higher-variance instantaneous
    # amplitude around the constellation than high-SNR
    assert np.std(np.abs(sig_lo.samples)) > np.std(np.abs(sig_hi.samples))


def test_write_pair_creates_all_four_files(tmp_path: Path) -> None:
    sig, gt = generate_signal("bpsk", num_symbols=100, seed=3)
    paths = write_pair(sig, gt, tmp_path, "sample")
    for key in ("iq", "sigmf_meta", "wav", "ground_truth"):
        assert paths[key].is_file(), f"missing {key}"
