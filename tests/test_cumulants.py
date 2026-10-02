"""Covers: FR-06 (cumulant DSP feature extractor)."""

from __future__ import annotations

import numpy as np

from sigscope.estimate.cumulants import compute_cumulants
from sigscope.synth.generator import generate_signal


def test_bpsk_c40_more_negative_than_qpsk_at_high_snr() -> None:
    """BPSK is not rotationally symmetric (real-only constellation), so its C40
    magnitude should be clearly larger than QPSK's near-zero C40 at high SNR --
    this is the textbook basis for PSK-order discrimination via cumulants."""
    bpsk_sig, _ = generate_signal("bpsk", num_symbols=4000, snr_db=25.0, seed=1)
    qpsk_sig, _ = generate_signal("qpsk", num_symbols=4000, snr_db=25.0, seed=1)
    bpsk_c = compute_cumulants(bpsk_sig.samples)
    qpsk_c = compute_cumulants(qpsk_sig.samples)
    assert abs(bpsk_c.c40) > abs(qpsk_c.c40)


def test_cumulants_finite_for_all_classes() -> None:
    for mod in ["bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "8fsk", "am", "fm", "noise"]:
        sig, _ = generate_signal(mod, num_symbols=1000, snr_db=15.0, seed=2)  # type: ignore[arg-type]
        c = compute_cumulants(sig.samples)
        assert np.isfinite(c.as_vector()).all(), f"non-finite cumulants for {mod}"


def test_zero_signal_returns_zero_cumulants() -> None:
    c = compute_cumulants(np.zeros(100, dtype=np.complex64))
    assert c.c20 == 0j
    assert c.c40 == 0j
    assert c.c42 == 0j
