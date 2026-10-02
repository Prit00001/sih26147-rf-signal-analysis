"""Covers: FR-05 (sample-rate sanity check), FR-09 (bandwidth, CFO), FR-14 (spectrum/waterfall data)."""

from __future__ import annotations

import numpy as np

from sigscope.spectral.analysis import estimate_cfo, occupied_bandwidth, sample_rate_sanity_check, waterfall, welch_psd


def _tone(freq_hz: float, sample_rate: float, n: int) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return np.exp(1j * 2 * np.pi * freq_hz * t).astype(np.complex64)


def test_welch_psd_peaks_near_tone_frequency() -> None:
    sample_rate = 1_000_000.0
    tone_freq = 150_000.0
    samples = _tone(tone_freq, sample_rate, 8192)
    freqs, psd = welch_psd(samples, sample_rate, nperseg=1024)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(peak_freq - tone_freq) < sample_rate / 1024 * 2


def test_occupied_bandwidth_narrow_for_single_tone() -> None:
    sample_rate = 1_000_000.0
    samples = _tone(0.0, sample_rate, 8192)
    freqs, psd = welch_psd(samples, sample_rate, nperseg=1024)
    low, high, bw = occupied_bandwidth(freqs, psd, fraction=0.5)
    assert bw < sample_rate * 0.05


def test_estimate_cfo_matches_tone_offset() -> None:
    sample_rate = 1_000_000.0
    tone_freq = 80_000.0
    samples = _tone(tone_freq, sample_rate, 8192)
    freqs, psd = welch_psd(samples, sample_rate, nperseg=1024)
    cfo = estimate_cfo(freqs, psd)
    assert abs(cfo - tone_freq) < sample_rate / 1024 * 4


def test_waterfall_shape() -> None:
    sample_rate = 1_000_000.0
    samples = _tone(0.0, sample_rate, 4096)
    times, freqs, mag_db = waterfall(samples, sample_rate, nperseg=256)
    assert mag_db.shape == (len(freqs), len(times))


def test_sample_rate_sanity_flags_aliasing_risk() -> None:
    result = sample_rate_sanity_check(sample_rate=100_000.0, occupied_bw_hz=99_000.0)
    assert not result.plausible
    assert result.warnings


def test_sample_rate_sanity_ok_case() -> None:
    result = sample_rate_sanity_check(sample_rate=1_000_000.0, occupied_bw_hz=100_000.0)
    assert result.plausible
    assert result.confidence == 1.0
