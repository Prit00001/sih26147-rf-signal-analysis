"""Blind (CMA) then decision-directed (LMS) adaptive equalizer.

Covers: FR-10 (equalization stage of the PSK/QAM demod chain). This project's
synthetic channel model (synth/generator.py) has no multipath/dispersive
distortion -- only AWGN, CFO, timing offset, and IQ imbalance -- so on these
test signals the equalizer is expected to converge close to an identity
(all-pass) response and NOT measurably change BER; it exists so the chain is
ready to correct real captures with actual channel distortion (ISI). This
expectation is checked directly by a regression test (equalizer on vs off).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def cma_lms_equalize(
    symbols: npt.NDArray[np.complex64],
    *,
    num_taps: int = 5,
    cma_step: float = 1e-3,
    lms_step: float = 1e-3,
    cma_symbols: int = 200,
    modulus: float = 1.0,
) -> npt.NDArray[np.complex64]:
    """Adaptive FIR equalizer at the symbol rate: Godard/CMA for the first
    ``cma_symbols`` symbols (blind, modulus-based cost -- ``modulus`` is the
    Godard R2 dispersion constant E[|a|^4]/E[|a|^2] for the target
    constellation), then decision-directed LMS using the equalizer's own
    (unit-modulus-projected) output as a rough decision reference.
    """
    n = len(symbols)
    if n <= num_taps:
        return symbols.copy()
    taps = np.zeros(num_taps, dtype=np.complex128)
    center = num_taps // 2
    taps[center] = 1.0 + 0j  # identity initialization: safe if no equalization is needed

    out = np.zeros(n, dtype=np.complex64)
    padded = np.concatenate(
        [
            np.zeros(center, dtype=np.complex128),
            symbols.astype(np.complex128),
            np.zeros(num_taps - center - 1, dtype=np.complex128),
        ]
    )
    for i in range(n):
        window = padded[i : i + num_taps][::-1]
        y = np.dot(taps, window)
        out[i] = y
        if i < cma_symbols:
            error = y * (np.abs(y) ** 2 - modulus)
            taps = taps - cma_step * error * np.conj(window)
        else:
            mag = max(np.abs(y), 1e-9)
            decision = y / mag * np.sqrt(modulus)
            error = y - decision
            taps = taps - lms_step * error * np.conj(window)
    return out
