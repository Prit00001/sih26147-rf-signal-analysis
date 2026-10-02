"""Covers: FR-13 support (bitstream correlation C++ kernel, used by Phase 5)."""

from __future__ import annotations

from sigscope import _native


def test_correlation_peaks_at_true_match() -> None:
    pattern = [1, 0, 1, 1, 0, 1, 0, 0]
    bits = [0, 1, 1, 0] + pattern + [1, 1, 0, 0, 1]
    scores = _native.bitstream_correlate(bits, pattern)
    best_offset = max(range(len(scores)), key=lambda i: scores[i])
    assert best_offset == 4
    assert scores[best_offset] == len(pattern)  # perfect match


def test_autocorrelation_finds_period() -> None:
    bits = [0, 1, 1, 0] * 20
    autoc = _native.bitstream_autocorrelation(bits, 10)
    best_lag = max(range(1, 11), key=lambda lag: autoc[lag - 1])
    assert best_lag == 4
