"""Blind parameter estimators that don't need a trained model: symbol rate,
SNR, roll-off, and (where fundamentally possible) sample rate.

Covers: FR-05 (sample-rate estimate when metadata is absent/unreliable), FR-09
(symbol rate, SNR, roll-off), NFR-02 (these feed the accuracy-vs-SNR benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from sigscope.spectral.analysis import welch_psd


def estimate_symbol_rate_normalized(
    samples: npt.NDArray[np.complex64], *, nfft: int | None = None
) -> tuple[float, float]:
    """Blind symbol-rate estimate via the FFT of the squared envelope (cyclostationary
    feature), returned as a NORMALIZED rate in cycles/sample (i.e. symbol_rate / Fs).

    This is scale-invariant: it needs no sample rate at all, which is exactly
    the quantity you CAN extract blindly from raw samples. Returns
    (normalized_symbol_rate, confidence in [0, 1]) where confidence is the
    fraction of spectral energy (excluding DC) concentrated at the detected peak.
    """
    n = len(samples)
    if n < 16:
        return 0.0, 0.0
    nfft = nfft or int(2 ** np.ceil(np.log2(n)))
    envelope_sq = np.abs(samples) ** 2
    envelope_sq = envelope_sq - np.mean(envelope_sq)
    spectrum = np.abs(np.fft.rfft(envelope_sq, n=nfft))
    spectrum[0] = 0.0  # ignore DC
    nonzero = spectrum[spectrum > 0]
    if nonzero.size == 0:
        return 0.0, 0.0
    peak_bin = int(np.argmax(spectrum))
    normalized_rate = peak_bin / nfft
    # Confidence is peak-to-median prominence (a CFAR-style "is there really a
    # cyclostationary line here" statistic), not a fraction of total energy --
    # the latter is misleadingly small even for a very clean peak, since a
    # pulse-shaped signal's squared envelope has broadband content besides the
    # symbol-rate line. Saturates towards 1.0 as the peak dominates the noise
    # floor; this DOES stay high even at low input SNR, because squaring the
    # envelope and averaging over many symbols is itself SNR-gaining -- a
    # property of the method, not an estimation bug (see the accuracy-vs-SNR
    # benchmark for the actual measured error, not just this confidence proxy).
    median_level = float(np.median(nonzero))
    ratio = float(spectrum[peak_bin]) / median_level if median_level > 0 else 0.0
    # Under pure noise, the max of many roughly-exponential periodogram bins
    # is, by chance alone, several times the median (extreme-value statistics)
    # -- empirically around ratio ~4 for our typical FFT sizes. _RATIO_FLOOR is
    # set above that chance level so pure noise reads ~0 confidence rather than
    # a false positive; _RATIO_SATURATE is where confidence reaches 1.0 for a
    # clearly dominant, genuine spectral line. Both are heuristic constants
    # calibrated against this project's own noise/signal test vectors, not a
    # rigorously derived detection threshold.
    _RATIO_FLOOR = 5.0
    _RATIO_SATURATE = 50.0
    confidence = (ratio - _RATIO_FLOOR) / (_RATIO_SATURATE - _RATIO_FLOOR)
    return normalized_rate, float(np.clip(confidence, 0.0, 1.0))


def estimate_symbol_rate_hz(samples: npt.NDArray[np.complex64], sample_rate: float) -> tuple[float, float]:
    """Symbol-rate estimate in Hz, given a (trusted) sample rate."""
    normalized_rate, confidence = estimate_symbol_rate_normalized(samples)
    return normalized_rate * sample_rate, confidence


@dataclass
class BlindSampleRateEstimate:
    candidates_hz: list[float]
    confidence: float
    note: str


def estimate_sample_rate_blind(
    samples: npt.NDArray[np.complex64], *, candidate_symbol_rates_hz: list[float]
) -> BlindSampleRateEstimate:
    """Attempt a blind sample-rate estimate by assuming the true symbol rate is one
    of a small set of standard candidate baud rates.

    This is a genuinely underdetermined problem in general: normalized samples
    carry no absolute time reference, so a signal recorded at 1 Msps and one
    recorded at 2 Msps of the same normalized waveform are indistinguishable
    without EITHER external metadata OR an assumption like "the symbol rate is
    one of these known standard values." When metadata/SigMF gives the sample
    rate directly (FR-05's primary path), this function is not needed at all;
    it exists purely for the analyst-assisted fallback when metadata is absent.
    """
    normalized_rate, conf = estimate_symbol_rate_normalized(samples)
    if normalized_rate <= 0:
        return BlindSampleRateEstimate([], 0.0, "no detectable cyclostationary peak")
    candidates = sorted({round(rate / normalized_rate) for rate in candidate_symbol_rates_hz})
    return BlindSampleRateEstimate(
        candidates_hz=[float(c) for c in candidates],
        confidence=min(conf, 0.5),  # capped: even a clean peak leaves multiple candidates ambiguous
        note=(
            "sample rate cannot be recovered from samples alone without external reference; "
            "these candidates assume the true symbol rate is one of candidate_symbol_rates_hz"
        ),
    )


@lru_cache(maxsize=1)
def _signal_kurtosis_factors() -> dict[str, float]:
    """Measures ka_s = E[|s|^4] / E[|s|^2]^2 from THIS project's own signal
    model at a near-noise-free SNR, per modulation -- the same pattern
    classify/ensemble.py's _cumulant_prototypes() already uses (a measured
    reference constant from our own generator, not a detection result).

    Used to generalize the M2M4 SNR estimator below beyond its textbook
    derivation, which assumes ka_s=1 (ideal constant-modulus symbols sampled
    at symbol centers). A REAL, MEASURED bug this fixes: this project's
    signals are continuously-sampled, RRC-pulse-shaped waveforms, whose
    envelope fluctuates between symbol centers -- measured ka_s is ~1.21 for
    QPSK, not 1.0. Assuming ka_s=1 on a ka_s=1.21 signal produced a
    monotonically growing bias (-1 dB at 0 dB true SNR, widening to -21 dB at
    30 dB true SNR -- see git history / README for the before/after numbers).
    """
    from sigscope.synth.generator import ALL_MODULATIONS, generate_signal

    factors: dict[str, float] = {}
    for mod in ALL_MODULATIONS:
        sig, _ = generate_signal(
            mod, num_symbols=3000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=40.0, seed=99
        )
        x = sig.samples.astype(np.complex128)
        m2 = float(np.mean(np.abs(x) ** 2))
        m4 = float(np.mean(np.abs(x) ** 4))
        factors[mod] = m4 / (m2 * m2)
    return factors


def estimate_snr_m2m4(samples: npt.NDArray[np.complex64], *, modulation: str | None = None) -> tuple[float, float]:
    """Blind SNR estimate via the non-data-aided M2M4 (2nd/4th moment) method,
    generalized to an arbitrary signal kurtosis factor ka_s (see
    _signal_kurtosis_factors): for y = s + n with independent zero-mean
    circular n, M2=Ps+Pn and M4=ka_s*Ps^2+4*Ps*Pn+2*Pn^2, which solves to
    Ps^2 = (2*M2^2 - M4) / (2 - ka_s). The textbook ka_s=1 special case
    (constant-modulus symbols) recovers the original Ps=sqrt(2*M2^2-M4).

    ``modulation`` is only known AFTER classification in this project's
    pipeline (estimate -> classify order), so this is called twice in
    practice: once early with modulation=None (ka_s=1.0 fallback, lower
    confidence), once more after classification corrects it with the real
    per-modulation ka_s.
    """
    x = samples.astype(np.complex128)
    m2 = float(np.mean(np.abs(x) ** 2))
    m4 = float(np.mean(np.abs(x) ** 4))
    if m2 <= 1e-20:
        return -99.0, 0.0
    ka_s = _signal_kurtosis_factors().get(modulation, 1.0) if modulation else 1.0
    ka_denom = 2.0 - ka_s
    if ka_denom <= 1e-6:
        # ka_s >= 2 (e.g. noise-like statistics): the M2M4 system is
        # degenerate for this signal type, not solvable this way.
        return -10.0, 0.2
    inner = 2 * m2 * m2 - m4
    if inner <= 0:
        # Degenerate case (e.g. pure noise): report a low SNR with low
        # confidence rather than raising or returning a nonsensical value.
        return -10.0, 0.2
    signal_power_est = np.sqrt(inner / ka_denom)
    denom = m2 - signal_power_est
    if denom <= 1e-20:
        return 30.0, 0.3  # essentially noise-free
    snr_linear = signal_power_est / denom
    snr_db = 10 * np.log10(max(snr_linear, 1e-6))
    confidence = 0.6 if modulation else 0.45  # lower confidence for the pre-classification generic fallback
    # MEASURED (not assumed): M2M4 noise power is a difference of two large,
    # close numbers once Pn is small relative to Ps, so error/variance grows
    # sharply above ~15 dB true SNR even with the ka_s fix above -- measured
    # std 0.35 dB at 15 dB true SNR growing to 1.8+ dB by 20 dB. Discount
    # confidence once the ESTIMATE itself reads high, since that's exactly
    # the regime this method is measurably less reliable in.
    if snr_db > 12.0:
        confidence *= max(0.3, 1.0 - (snr_db - 12.0) / 20.0)
    return float(snr_db), float(confidence)


def _raised_cosine_shape(freqs: npt.NDArray[np.float64], symbol_rate_hz: float, beta: float) -> npt.NDArray[np.float64]:
    """Normalized (0..1) raised-cosine POWER spectrum -- what an RRC-pulse-
    shaped signal's own PSD shape follows, since |RRC(f)|^2 = RC(f) by
    definition (RRC is the square-root-in-magnitude of RC)."""
    f = np.abs(freqs)
    f1 = (1.0 - beta) * symbol_rate_hz / 2.0
    f2 = (1.0 + beta) * symbol_rate_hz / 2.0
    shape = np.zeros_like(f)
    shape[f <= f1] = 1.0
    if beta > 1e-6:
        transition = (f > f1) & (f <= f2)
        shape[transition] = 0.5 * (1.0 + np.cos(np.pi / (beta * symbol_rate_hz) * (f[transition] - f1)))
    return shape


def estimate_rolloff(
    samples: npt.NDArray[np.complex64], sample_rate: float, symbol_rate_hz: float, *, beta_grid_size: int = 50
) -> tuple[float, float]:
    """Roll-off estimate by fitting the measured PSD's BAND-EDGE SHAPE (not
    just a cumulative-power threshold) against the known raised-cosine
    spectral shape, grid-searching beta in (0, 1).

    REPLACES an earlier occupied-bandwidth-threshold method: measured
    (scripts/calibrate_ensemble_temperature.py) mean absolute error 0.64-0.69
    below 10 dB SNR and still 0.23-0.30 at 20-25 dB -- comparable to the
    entire plausible rolloff range, i.e. not reliable at any SNR. This
    shape-fit method is benchmarked separately (scripts/benchmark_rolloff.py)
    -- see that script's output / README for the measured error vs SNR this
    method actually achieves, and pipeline_core.py for how its confidence is
    set honestly based on that measurement (excluded from the ground-truth
    score, with a tooltip, if it does not clear +/-0.1 at 15 dB).
    """
    if symbol_rate_hz <= 0:
        return 0.0, 0.0
    freqs, psd = welch_psd(samples, sample_rate)
    passband_mask = np.abs(freqs) < 0.3 * symbol_rate_hz
    if not passband_mask.any():
        return 0.0, 0.0
    p0 = float(np.median(psd[passband_mask]))
    if p0 <= 0:
        return 0.0, 0.0
    far_mask = np.abs(freqs) > 1.2 * symbol_rate_hz
    noise_floor = float(np.median(psd[far_mask])) if far_mask.any() else 0.0
    norm_psd = np.clip((psd - noise_floor) / max(p0 - noise_floor, 1e-20), 0.0, 1.5)

    fit_mask = np.abs(freqs) < 0.9 * symbol_rate_hz
    f_fit, p_fit = freqs[fit_mask], norm_psd[fit_mask]
    if len(f_fit) == 0:
        return 0.0, 0.0

    beta_grid = np.linspace(0.02, 0.98, beta_grid_size)
    errors = np.array([np.mean((_raised_cosine_shape(f_fit, symbol_rate_hz, b) - p_fit) ** 2) for b in beta_grid])
    best_idx = int(np.argmin(errors))
    best_beta = float(beta_grid[best_idx])
    # Confidence from fit quality: how much better the best beta fits than a
    # "flat/no shape" null fit (constant 1.0, i.e. beta=0 extreme) -- a
    # genuinely RRC-shaped signal fits noticeably better than that baseline;
    # pure noise or a non-linear modulation (no RRC shape at all) won't.
    null_error = float(np.mean((1.0 - p_fit) ** 2))
    improvement = (null_error - errors[best_idx]) / max(null_error, 1e-9)
    confidence = float(np.clip(improvement, 0.0, 1.0))
    return best_beta, confidence
