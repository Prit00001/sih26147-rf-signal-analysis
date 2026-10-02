"""PSK/QAM demodulation chain: coarse CFO -> matched filter -> Gardner timing
recovery -> CMA/LMS equalizer -> decision-directed carrier recovery -> soft
and hard bit output with Gray mapping.

Covers: FR-10 (demodulate PSK, QAM to soft and hard bits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope.demod.equalizer import cma_lms_equalize
from sigscope.demod.sync import (
    decision_directed_carrier_recovery,
    estimate_cfo_mth_power,
    gardner_timing_recovery,
    matched_filter,
)
from sigscope.synth.generator import LINEAR_BITS_PER_SYMBOL, constellation_for

# M-th power order used for the coarse blind CFO pre-estimate per modulation.
# QAM does not have a clean M-th power tone (non-constant modulus), but 4th
# power still gives a usable, if noisier, coarse estimate in practice.
_MTH_POWER_ORDER = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16qam": 4, "64qam": 4}


@dataclass
class PskQamDemodResult:
    hard_bits: npt.NDArray[np.int64]
    soft_bits: npt.NDArray[np.float64]  # max-log LLR-like metric; positive => bit 0 more likely
    symbols: npt.NDArray[np.complex64]  # carrier-corrected symbols, for constellation/eye-diagram display
    decided_indices: npt.NDArray[np.int64]  # constellation-array index per symbol (== TX symbol_idx convention)
    pre_carrier_symbols: npt.NDArray[np.complex64] | None = None  # post timing-recovery, BEFORE carrier correction


def _bits_per_index(bps: int) -> npt.NDArray[np.int64]:
    """[2**bps, bps] matrix: row s, column k = bit k (MSB-first) of symbol index s."""
    indices = np.arange(2**bps)
    bit_positions = np.arange(bps - 1, -1, -1)
    return ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.int64)


def _soft_bits_from_symbols(
    symbols: npt.NDArray[np.complex64], constellation: npt.NDArray[np.complex128], bps: int
) -> npt.NDArray[np.float64]:
    bits_table = _bits_per_index(bps)  # [M, bps]
    dist_sq = np.abs(symbols[:, None] - constellation[None, :]) ** 2  # [N, M]
    llrs = np.zeros((len(symbols), bps), dtype=np.float64)
    for k in range(bps):
        is_one = bits_table[:, k] == 1
        d0_min = np.min(dist_sq[:, ~is_one], axis=1)
        d1_min = np.min(dist_sq[:, is_one], axis=1)
        llrs[:, k] = d1_min - d0_min  # positive => bit 0 (closer) more likely
    return llrs.reshape(-1)


def demodulate_psk_qam(
    samples: npt.NDArray[np.complex64],
    modulation: str,
    sample_rate: float,
    symbol_rate: float,
    *,
    rolloff: float = 0.35,
    rrc_span_symbols: int = 8,
    use_equalizer: bool = False,
    timing_gain: float = 0.002,
    carrier_loop_bw: float = 0.02,
) -> PskQamDemodResult:
    """Full receive chain for a linear (PSK/QAM) modulation.

    ``modulation`` must be one of the keys in
    ``sigscope.synth.generator.LINEAR_BITS_PER_SYMBOL``.

    ``use_equalizer`` defaults to False: measured directly on this project's
    test signals, enabling CMA/LMS equalization made higher-order QAM BER
    WORSE at low-to-moderate SNR (e.g. 64QAM at 10 dB: ~7.5% SER with it off,
    ~43% with it on), because there is no actual multipath/ISI in this
    project's channel model for CMA's constant-modulus cost to usefully
    correct, and CMA's constant-modulus assumption is a poor fit for QAM
    generally. The capability is kept available (FR-10 requires it exist) for
    the analyst-override path on real captures that DO have channel
    dispersion, but is not applied automatically.
    """
    if modulation not in LINEAR_BITS_PER_SYMBOL:
        raise ValueError(f"unsupported linear modulation: {modulation}")
    bps = LINEAR_BITS_PER_SYMBOL[modulation]
    sps = sample_rate / symbol_rate

    order = _MTH_POWER_ORDER[modulation]
    cfo_est = estimate_cfo_mth_power(samples, sample_rate, order)
    n = np.arange(len(samples))
    coarse = (samples.astype(np.complex128) * np.exp(-1j * 2 * np.pi * cfo_est * n / sample_rate)).astype(np.complex64)

    filtered = matched_filter(coarse, rolloff, rrc_span_symbols, int(round(sps)))
    symbols = gardner_timing_recovery(filtered, sps, gain=timing_gain)

    constellation = constellation_for(modulation)
    if use_equalizer and len(symbols) > 10:
        r2 = float(np.mean(np.abs(constellation) ** 4) / np.mean(np.abs(constellation) ** 2))
        symbols = cma_lms_equalize(symbols, modulus=r2)

    pre_carrier_symbols = symbols.copy()
    corrected, decisions = decision_directed_carrier_recovery(symbols, constellation, loop_bw=carrier_loop_bw)

    bits_table = _bits_per_index(bps)
    hard_bits = bits_table[decisions].reshape(-1)
    soft_bits = _soft_bits_from_symbols(corrected, constellation, bps)

    return PskQamDemodResult(
        hard_bits=hard_bits,
        soft_bits=soft_bits,
        symbols=corrected,
        decided_indices=decisions,
        pre_carrier_symbols=pre_carrier_symbols,
    )
