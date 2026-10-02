"""Theoretical BER formulas (standard textbook approximations, AWGN, Gray
coding assumed) used ONLY to sanity-check the measured demodulator BER --
never asserted as the tool's own claim about a real captured signal.

Covers: NFR-02 (report BER vs SNR / vs theory).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def q_function(x: float) -> float:
    return float(0.5 * erfc(x / np.sqrt(2)))


def theoretical_ber(modulation: str, es_n0_db: float) -> float:
    """Approximate theoretical bit-error rate at a given symbol-energy-to-noise
    ratio (Es/N0, in dB), for Gray-coded constellations in AWGN.

    Standard textbook approximations (e.g. Proakis, "Digital Communications"):
    BPSK/QPSK are exact; M-PSK (M>=8) and square M-QAM use the common
    high-SNR union-bound-style approximations, which slightly UNDERESTIMATE
    true BER at very low SNR -- fine for this project's "reasonable margin"
    comparison at moderate-to-high SNR, not meant as a precise low-SNR model.
    """
    es_n0 = 10 ** (es_n0_db / 10.0)
    if modulation == "bpsk":
        return q_function(np.sqrt(2 * es_n0))
    if modulation == "qpsk":
        return q_function(np.sqrt(2 * es_n0))  # per-bit BER identical to BPSK for Gray-coded QPSK
    if modulation == "8psk":
        m = 8
        ser = 2 * q_function(np.sqrt(2 * es_n0) * np.sin(np.pi / m))
        return float(ser / np.log2(m))
    if modulation in ("16qam", "64qam"):
        m = 16 if modulation == "16qam" else 64
        ser = 4 * (1 - 1 / np.sqrt(m)) * q_function(np.sqrt(3 * es_n0 / (m - 1)))
        return float(ser / np.log2(m))
    raise ValueError(f"no theoretical BER formula for modulation: {modulation}")
