"""Shared test helpers for demodulation tests: resolving the constellation
rotation + start-offset ambiguity inherent to decision-directed carrier
recovery (see sync.py's decision_directed_carrier_recovery docstring), and
computing per-bit BER against ground truth once resolved.

This brute-force resolution against known ground truth is an EVALUATION
technique only, standing in for what a real receiver would use a sync word
(Phase 5) or differential encoding to resolve.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sigscope.synth.generator import LINEAR_BITS_PER_SYMBOL, constellation_for


def tx_symbol_indices(bits: list[int], modulation: str) -> npt.NDArray[np.int64]:
    bps = LINEAR_BITS_PER_SYMBOL[modulation]
    tx_bits = np.array(bits, dtype=np.int64)
    n_full = len(tx_bits) // bps
    weights = 1 << np.arange(bps - 1, -1, -1)
    return (tx_bits[: n_full * bps].reshape(-1, bps) @ weights).astype(np.int64)


def best_aligned_ser(
    rx_symbols: npt.NDArray[np.complex64], tx_symbols: npt.NDArray[np.int64], modulation: str, *, max_shift: int = 30
) -> tuple[float, int, int, int]:
    """Search constellation rotations and start-offsets for the best match.
    Returns (symbol_error_rate, rotation, shift, matched_length)."""
    const = constellation_for(modulation)
    m = len(const)
    best: tuple[float, int, int, int] | None = None
    for rot in range(m):
        derot = rx_symbols * np.exp(-1j * rot * 2 * np.pi / m)
        rx_decisions = np.argmin(np.abs(derot[:, None] - const[None, :]), axis=1)
        for shift in range(max_shift):
            a = rx_decisions[shift:]
            b = tx_symbols[: len(a)]
            length = min(len(a), len(b))
            if length < 200:
                continue
            errs = float(np.mean(a[:length] != b[:length]))
            if best is None or errs < best[0]:
                best = (errs, rot, shift, length)
    assert best is not None, "not enough symbols to evaluate alignment"
    return best


def measured_symbol_snr_db(
    rx_symbols: npt.NDArray[np.complex64], tx_symbols: npt.NDArray[np.int64], modulation: str, rot: int, shift: int
) -> float:
    """The actual SNR delivered to the symbol detector after synchronization,
    measured directly from residual error vs. the (rotation-resolved) ideal
    constellation points -- decouples this from the generator's own SNR
    parameter, which is not a clean Es/N0 due to RRC pulse-shaping/oversampling.
    """
    const = constellation_for(modulation)
    m = len(const)
    derot = rx_symbols * np.exp(-1j * rot * 2 * np.pi / m)
    a = derot[shift:]
    b = tx_symbols[: len(a)]
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    ideal = const[b]
    error = a - ideal
    signal_power = float(np.mean(np.abs(ideal) ** 2))
    noise_power = float(np.mean(np.abs(error) ** 2))
    if noise_power <= 1e-20:
        return 60.0
    return 10 * np.log10(signal_power / noise_power)
