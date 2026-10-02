"""Higher-order cumulant features (C20, C40, C42) for modulation classification.

Covers: FR-06 (the DSP arm of the classifier ensemble -- "DSP where physics is
exact, ML where it is ambiguous"). Cumulants are computed directly on raw
(power-normalized) baseband IQ samples, not on symbol-synchronized samples, so
separation between classes is real but imperfect -- this is measured (not
asserted) in the accuracy-vs-SNR benchmark.

FIXED LIMITATION (was measured broken, now measured fixed -- see
compute_cumulants_windowed below): C20 and C40 involve x^2/x^4, which carry
absolute carrier phase. A single static phase offset only rotates them (their
magnitude, used by as_vector(), is invariant to that). But a residual CARRIER
FREQUENCY OFFSET is a phase that drifts continuously across the observation
window, which causes DESTRUCTIVE averaging of x^2/x^4 over the block -- not
just a rotation -- shrinking |C20|/|C40| towards zero the LONGER the window
is, which made a QPSK signal's cumulants converge towards an 8PSK-like
near-zero value (this was the root cause of QPSK/16QAM/64QAM classification
accuracy being <45% before this fix). compute_cumulants_windowed() fixes this
with M-th-power CFO pre-compensation plus averaging MAGNITUDES (not raw
complex values) across many short windows, which sidesteps the destructive-
averaging mechanism entirely. plain compute_cumulants() below is kept for
callers that want the raw (non-windowed) computation; classify/ensemble.py
uses the windowed version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope.demod.sync import estimate_cfo_mth_power


@dataclass
class Cumulants:
    c20: complex
    c40: complex
    c42: complex

    def as_vector(self) -> npt.NDArray[np.float64]:
        """Flatten to a real-valued, CFO-invariant feature vector: [|C20|, |C40|, Re(C42)].

        C20 and C40 involve x^2/x^4, which carry the signal's absolute carrier
        phase -- a residual CFO rotates them over the observation window and
        would corrupt any classifier trained on their raw (signed) real/
        imaginary parts, especially over the short windows used here. Taking
        the magnitude removes that rotation entirely (a phase rotation of x
        by theta rotates C20 by 2*theta and C40 by 4*theta, but leaves their
        magnitudes unchanged), which is the standard fix in the cumulant-based
        AMC literature. C42 is already phase-invariant (built from |x|^2 and
        |x|^4), so only its tiny estimation-noise imaginary part is dropped.
        """
        return np.array([abs(self.c20), abs(self.c40), self.c42.real])


def compute_cumulants(samples: npt.NDArray[np.complex64]) -> Cumulants:
    """Compute normalized 2nd/4th-order cumulants (Swami-Sadler convention).

    Samples are power-normalized to unit E[|x|^2] first (cumulants are scale
    variant otherwise, and absolute signal power is arbitrary/gain-dependent).
    """
    x = samples.astype(np.complex128)
    x = x - np.mean(x)
    power = np.mean(np.abs(x) ** 2)
    if power <= 1e-20:
        return Cumulants(0j, 0j, 0j)
    x = x / np.sqrt(power)  # now E[|x|^2] = 1 = C21

    m20 = np.mean(x**2)
    m21 = np.mean(np.abs(x) ** 2)  # == 1.0 after normalization
    m40 = np.mean(x**4)
    m42 = np.mean(np.abs(x) ** 4)

    c20 = m20
    c40 = m40 - 3 * m20**2
    c42 = m42 - abs(m20) ** 2 - 2 * m21**2
    return Cumulants(complex(c20), complex(c40), complex(c42))


def compute_cumulants_windowed(
    samples: npt.NDArray[np.complex64], *, window_size: int = 64, cfo_precompensate_order: int = 4
) -> Cumulants:
    """Fixes the KNOWN LIMITATION above: coarse CFO pre-compensation (M-th
    power method, order 4 as a generic default -- not exact for every
    constellation, but reduces residual drift for all of them) followed by
    computing cumulants over many SHORT windows and averaging the resulting
    magnitudes (not the raw complex values -- these are already phase-
    invariant per as_vector(), so averaging them across windows only reduces
    estimation variance, it does not reintroduce destructive phase
    cancellation the way averaging over one long window did).

    Median (not mean) across windows for robustness: a handful of windows
    landing on a bad noise realization would otherwise skew the mean.
    """
    n = len(samples)
    if n < window_size * 4:
        return compute_cumulants(samples)

    sample_rate_placeholder = 1.0  # CFO here is estimated/removed in NORMALIZED cycles/sample terms
    cfo_norm = estimate_cfo_mth_power(samples, sample_rate_placeholder, cfo_precompensate_order)
    if abs(cfo_norm) > 1e-9:
        idx = np.arange(n)
        samples = (samples.astype(np.complex128) * np.exp(-1j * 2 * np.pi * cfo_norm * idx)).astype(np.complex64)

    num_windows = n // window_size
    vectors = np.zeros((num_windows, 3))
    for i in range(num_windows):
        chunk = samples[i * window_size : (i + 1) * window_size]
        vectors[i] = compute_cumulants(chunk).as_vector()
    median_vec = np.median(vectors, axis=0)
    return Cumulants(complex(median_vec[0]), complex(median_vec[1]), complex(median_vec[2]))
