"""DC offset removal, IQ-imbalance correction, and power normalization.

Covers: FR-03 (recordings from different sensors/locations need normalization
before downstream analysis is comparable).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def remove_dc(samples: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Subtract the mean (DC offset) from a complex baseband signal."""
    return cast(npt.NDArray[np.complex64], (samples - np.mean(samples)).astype(np.complex64))


def correct_iq_imbalance(samples: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Blind gain/phase IQ-imbalance correction via Gram-Schmidt orthogonalization.

    Assumes the underlying modulated signal has (approximately) circularly
    symmetric statistics, which holds for the PSK/QAM/FSK signals this tool
    targets. Returns the input unchanged if the estimate is degenerate (e.g. a
    near-zero-power signal) rather than dividing by zero.
    """
    i_comp = samples.real.astype(np.float64)
    q_comp = samples.imag.astype(np.float64)
    alpha = float(np.mean(i_comp * i_comp))
    if alpha <= 1e-20:
        return samples
    gamma = float(np.mean(i_comp * q_comp))
    beta = float(np.mean(q_comp * q_comp))
    c = gamma / alpha
    residual = beta - c * c * alpha
    if residual <= 1e-20:
        return samples
    q_corrected = (q_comp - c * i_comp) / np.sqrt(residual)
    return cast(npt.NDArray[np.complex64], (i_comp + 1j * q_corrected).astype(np.complex64))


def normalize_power(samples: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Scale a signal to unit average power. No-op on an all-zero/empty signal."""
    power = float(np.mean(np.abs(samples) ** 2)) if samples.size else 0.0
    if power <= 1e-20:
        return samples
    return cast(npt.NDArray[np.complex64], (samples / np.sqrt(power)).astype(np.complex64))
