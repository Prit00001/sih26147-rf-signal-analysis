"""Spectral analysis: PSD, waterfall (STFT), occupied bandwidth, CFO estimate,
and a sample-rate sanity check.

Covers: FR-05 (sample-rate sanity check backing the blind estimator), FR-09
(bandwidth, CFO), FR-14 (spectrum/waterfall data feeding the GUI plots).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.signal import stft, welch


def welch_psd(
    samples: npt.NDArray[np.complex64], sample_rate: float, *, nperseg: int = 1024
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Two-sided Welch power spectral density, frequency-centred (fftshifted)."""
    nperseg = min(nperseg, len(samples)) or 1
    freqs, psd = welch(samples, fs=sample_rate, nperseg=nperseg, return_onesided=False, detrend=False)
    order = np.argsort(freqs)
    return freqs[order], psd[order]


def waterfall(
    samples: npt.NDArray[np.complex64], sample_rate: float, *, nperseg: int = 256, noverlap: int | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Time-frequency magnitude matrix (dB) for the GUI waterfall view.

    Returns (times, freqs, magnitude_db) with freqs frequency-centred.
    """
    nperseg = min(nperseg, len(samples)) or 1
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, times, sxx = stft(samples, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    order = np.argsort(freqs)
    magnitude_db = 20 * np.log10(np.abs(sxx[order]) + 1e-12)
    return times, freqs[order], magnitude_db


def occupied_bandwidth(
    freqs: npt.NDArray[np.float64], psd: npt.NDArray[np.float64], *, fraction: float = 0.99
) -> tuple[float, float, float]:
    """Smallest contiguous (in sorted-freq order) band containing ``fraction`` of total power.

    Returns (low_freq, high_freq, bandwidth_hz).
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    total = float(np.sum(psd))
    if total <= 0:
        return float(freqs[0]), float(freqs[-1]), float(freqs[-1] - freqs[0])
    cumulative = np.cumsum(psd) / total
    target_each_side = (1.0 - fraction) / 2.0
    low_idx = int(np.searchsorted(cumulative, target_each_side))
    high_idx = int(np.searchsorted(cumulative, 1.0 - target_each_side))
    low_idx = min(low_idx, len(freqs) - 1)
    high_idx = min(max(high_idx, low_idx), len(freqs) - 1)
    return float(freqs[low_idx]), float(freqs[high_idx]), float(freqs[high_idx] - freqs[low_idx])


def estimate_cfo(freqs: npt.NDArray[np.float64], psd: npt.NDArray[np.float64]) -> float:
    """Rough blind carrier-frequency-offset proxy: the spectral power centroid.

    This assumes a modulation whose spectrum is (approximately) symmetric about
    its true carrier when centred correctly, so any offset shows up as a shifted
    centroid. It is a coarse estimate only -- fine CFO tracking/compensation is
    the Costas-loop stage in demodulation (FR-10), not this function.
    """
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * psd) / total)


@dataclass
class SampleRateSanityResult:
    plausible: bool
    confidence: float
    warnings: list[str] = field(default_factory=list)


def sample_rate_sanity_check(
    sample_rate: float, occupied_bw_hz: float, *, min_bw_fraction: float = 1e-4, max_bw_fraction: float = 0.95
) -> SampleRateSanityResult:
    """Flag a declared sample rate that is implausible given the measured occupied bandwidth.

    Covers FR-05: metadata-declared sample rate is trusted by default (confidence
    1.0 at ingestion) but this check lets the analyst see a warning when it looks
    wrong, without silently overriding it.
    """
    warnings: list[str] = []
    bw_fraction = occupied_bw_hz / sample_rate if sample_rate > 0 else float("inf")
    if bw_fraction > max_bw_fraction:
        warnings.append(
            f"occupied bandwidth ({occupied_bw_hz:.1f} Hz) is {bw_fraction:.0%} of the declared "
            f"sample rate ({sample_rate:.1f} Hz); risk of aliasing, sample rate may be too low"
        )
    if bw_fraction < min_bw_fraction:
        warnings.append(
            f"occupied bandwidth ({occupied_bw_hz:.1f} Hz) is only {bw_fraction:.4%} of the declared "
            f"sample rate ({sample_rate:.1f} Hz); sample rate may be far too high for this signal"
        )
    plausible = len(warnings) == 0
    confidence = 1.0 if plausible else 0.3
    return SampleRateSanityResult(plausible=plausible, confidence=confidence, warnings=warnings)
