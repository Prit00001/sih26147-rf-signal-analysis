"""FSK demodulation: frequency discriminator, eye-opening symbol timing search,
tone-level slicing, soft/hard bit output.

Covers: FR-10 (demodulate FSK to soft and hard bits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope.synth.generator import FSK_BITS_PER_SYMBOL


@dataclass
class FskDemodResult:
    hard_bits: npt.NDArray[np.int64]
    soft_bits: npt.NDArray[np.float64]  # positive => bit 0 more likely (same convention as psk_qam)
    freq_estimates: npt.NDArray[np.float64]  # per-symbol instantaneous frequency estimate, Hz
    decided_indices: npt.NDArray[np.int64]


def _best_sampling_phase(smoothed: npt.NDArray[np.float64], sps: int) -> int:
    """Pick the decimation phase (0..sps-1) whose decimated symbol sequence has
    the highest variance -- the classic "eye opening" timing-recovery heuristic,
    appropriate here since our channel model has a fixed (non-drifting) timing
    offset rather than a slowly drifting clock.
    """
    best_phase, best_var = 0, -1.0
    for phase in range(sps):
        decimated = smoothed[phase::sps]
        if len(decimated) < 4:
            continue
        var = float(np.var(decimated))
        if var > best_var:
            best_var, best_phase = var, phase
    return best_phase


def demodulate_fsk(
    samples: npt.NDArray[np.complex64],
    modulation: str,
    sample_rate: float,
    symbol_rate: float,
    *,
    mod_index: float = 1.0,
) -> FskDemodResult:
    """Full receive chain for 2/4/8-FSK."""
    if modulation not in FSK_BITS_PER_SYMBOL:
        raise ValueError(f"unsupported FSK modulation: {modulation}")
    bps = FSK_BITS_PER_SYMBOL[modulation]
    order = 2**bps
    sps = max(1, round(sample_rate / symbol_rate))
    delta_f = symbol_rate * mod_index

    inst_freq = np.angle(samples[1:] * np.conj(samples[:-1])) * sample_rate / (2 * np.pi)
    kernel = np.ones(sps) / sps
    smoothed = np.convolve(inst_freq, kernel, mode="same")

    phase = _best_sampling_phase(smoothed, sps)
    freq_estimates = smoothed[phase::sps]

    level_indices = np.arange(order)
    level_freqs = (level_indices - (order - 1) / 2.0) * delta_f
    dist = np.abs(freq_estimates[:, None] - level_freqs[None, :])
    decided_indices = np.argmin(dist, axis=1)

    bit_positions = np.arange(bps - 1, -1, -1)
    hard_bits = ((decided_indices[:, None] >> bit_positions[None, :]) & 1).astype(np.int64).reshape(-1)

    # Soft bits: distance-in-frequency-space max-log approximation, same sign
    # convention as psk_qam's soft_bits (positive => bit 0 more likely).
    bits_table = ((level_indices[:, None] >> bit_positions[None, :]) & 1).astype(np.int64)
    dist_sq = dist**2
    soft = np.zeros((len(freq_estimates), bps), dtype=np.float64)
    for k in range(bps):
        is_one = bits_table[:, k] == 1
        d0_min = np.min(dist_sq[:, ~is_one], axis=1)
        d1_min = np.min(dist_sq[:, is_one], axis=1)
        soft[:, k] = d1_min - d0_min
    soft_bits = soft.reshape(-1)

    return FskDemodResult(
        hard_bits=hard_bits, soft_bits=soft_bits, freq_estimates=freq_estimates, decided_indices=decided_indices
    )
