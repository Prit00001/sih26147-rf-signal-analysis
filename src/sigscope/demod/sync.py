"""Synchronization primitives: matched filtering, blind CFO pre-estimate,
Gardner timing recovery, and decision-directed carrier recovery (generalizes
the Costas loop to any PSK/QAM constellation).

Covers: FR-10 (carrier recovery, timing recovery).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sigscope.synth.generator import rrc_taps


def matched_filter(
    samples: npt.NDArray[np.complex64], beta: float, span_symbols: int, sps: int
) -> npt.NDArray[np.complex64]:
    """Apply the (unit-energy) RRC matched filter used at the transmitter."""
    taps = rrc_taps(beta, span_symbols, sps)
    filtered = np.convolve(samples, taps, mode="same")
    return filtered.astype(np.complex64)


def estimate_cfo_mth_power(samples: npt.NDArray[np.complex64], sample_rate: float, order: int) -> float:
    """Blind coarse CFO estimate via the M-th power method: raising an M-ary PSK
    signal to the M-th power removes the modulation and leaves a pure tone at
    M*CFO. Approximate (not exact) for QAM, since it is not constant-modulus,
    but still useful as a coarse pre-correction before the fine tracking loop.
    """
    powered = samples.astype(np.complex128) ** order
    n = len(powered)
    nfft = int(2 ** np.ceil(np.log2(max(n, 16))))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(powered, n=nfft)))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate))
    peak_freq = float(freqs[np.argmax(spectrum)])
    return peak_freq / order


def _pll_gains(loop_bw: float, damping: float, detector_gain: float = 1.0) -> tuple[float, float]:
    """Proportional/integral gains for a discrete 2nd-order PLL from a normalized
    loop bandwidth (Bn*T) and damping factor (standard textbook formula)."""
    theta = loop_bw / (damping + 0.25 / damping)
    denom = 1.0 + 2.0 * damping * theta + theta * theta
    kp = (4.0 * damping * theta) / denom / detector_gain
    ki = (4.0 * theta * theta) / denom / detector_gain
    return kp, ki


def gardner_timing_recovery(
    samples: npt.NDArray[np.complex64], sps: float, *, gain: float = 0.002
) -> npt.NDArray[np.complex64]:
    """Symbol-timing recovery via the Gardner timing-error detector driving a
    linear fractional interpolator. Requires the signal to be comfortably
    oversampled (sps >= ~4) for the linear interpolator to be accurate enough.

    Proportional-only (no integral term) by design, found empirically during
    testing: adding an integral term (as a standard 2nd-order PLL would) let
    small steady-state biases accumulate into unbounded symbol-timing drift
    over long runs, eventually causing the sampling instant to wander into an
    adjacent symbol and produce a cascade of decision errors -- a real bug
    caught by measuring drift over long synthetic sequences, not a
    hypothetical one. A gain much above ~0.002-0.005 was also found to let the
    loop jump to a stable-but-wrong lock point offset by half a symbol
    (Gardner's detector has more than one equilibrium); ``gain`` is kept
    conservative by default for this reason -- this trades slower acquisition
    for reliably not losing lock, appropriate given this project's channel
    model has no clock drift to track, only a fixed sub-symbol offset to find.

    Returns one recovered complex sample per symbol.
    """
    n = len(samples)

    def interp(pos: float) -> complex:
        i0 = int(np.floor(pos))
        frac = pos - i0
        if i0 < 0 or i0 + 1 >= n:
            return 0j
        return complex(samples[i0]) * (1 - frac) + complex(samples[i0 + 1]) * frac

    symbols: list[complex] = []
    last_symbol = 0j
    pos = sps
    while pos + sps < n:
        mid = interp(pos - sps / 2.0)
        cur = interp(pos)
        error = (np.conj(mid) * (cur - last_symbol)).real
        symbols.append(cur)
        last_symbol = cur
        pos += sps + gain * error
    return np.array(symbols, dtype=np.complex64)


def decision_directed_carrier_recovery(
    symbols: npt.NDArray[np.complex64],
    constellation: npt.NDArray[np.complex128],
    *,
    loop_bw: float = 0.02,
    damping: float = 0.707,
) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.int64]]:
    """2nd-order decision-directed carrier phase/frequency tracking loop.

    This generalizes the classic Costas loop (normally described specifically
    for BPSK/QPSK) to any PSK/QAM constellation by using the nearest-
    constellation-point decision directly as the phase reference; the
    per-symbol computation is otherwise the same 2nd-order PLL structure.

    NOTE (rotational ambiguity): with no known preamble or differential
    encoding, this loop can lock onto any of the constellation's rotational
    symmetries (e.g. any of 4 for QPSK/square QAM) equally validly -- this is
    a fundamental property of decision-directed recovery, not a bug. Resolving
    it requires a sync word (Phase 5's bit-stream correlation) or differential
    encoding in a real deployment; see demod/pipeline.py's benchmark helper for
    how this project resolves it for BER measurement purposes only.
    """
    kp, ki = _pll_gains(loop_bw, damping)
    phase = 0.0
    freq = 0.0
    corrected = np.zeros_like(symbols)
    decisions = np.zeros(len(symbols), dtype=np.int64)
    for i, y in enumerate(symbols):
        rotated = y * np.exp(-1j * phase)
        idx = int(np.argmin(np.abs(rotated - constellation)))
        decision = constellation[idx]
        error = float(np.angle(rotated * np.conj(decision)))
        corrected[i] = rotated
        decisions[i] = idx
        freq += ki * error
        phase += freq + kp * error
    return corrected, decisions
