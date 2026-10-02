"""Demodulation pipeline stage: the auto_estimate()/apply() dual-mode contract
(this project's non-negotiable design principle #2) plus chunked streaming
for long files.

Covers: FR-10, FR-15 (pipeline stage), NFR-01 (bounded-memory chunking).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigscope.core.signal import Signal
from sigscope.demod.fsk import FskDemodResult, demodulate_fsk
from sigscope.demod.psk_qam import PskQamDemodResult, demodulate_psk_qam
from sigscope.estimate.blind import estimate_symbol_rate_hz
from sigscope.synth.generator import FSK_BITS_PER_SYMBOL, LINEAR_BITS_PER_SYMBOL

DemodResult = PskQamDemodResult | FskDemodResult


@dataclass
class DemodParams:
    modulation: str
    symbol_rate_hz: float
    use_equalizer: bool = False


@dataclass
class DemodStage:
    """auto_estimate() -> (params, confidence); apply(params) re-runs
    demodulation with analyst-supplied/corrected parameters. Modulation itself
    is not re-estimated here (that is classify.ensemble's job, Phase 2); this
    stage takes a modulation label as given and estimates/accepts symbol rate.
    """

    signal: Signal

    def auto_estimate(self, modulation: str) -> tuple[DemodParams, float]:
        symbol_rate_hz, confidence = estimate_symbol_rate_hz(self.signal.samples, self.signal.sample_rate)
        return DemodParams(modulation=modulation, symbol_rate_hz=symbol_rate_hz), confidence

    def apply(self, params: DemodParams) -> DemodResult:
        if params.modulation in LINEAR_BITS_PER_SYMBOL:
            return demodulate_psk_qam(
                self.signal.samples,
                params.modulation,
                self.signal.sample_rate,
                params.symbol_rate_hz,
                use_equalizer=params.use_equalizer,
            )
        if params.modulation in FSK_BITS_PER_SYMBOL:
            return demodulate_fsk(
                self.signal.samples, params.modulation, self.signal.sample_rate, params.symbol_rate_hz
            )
        raise ValueError(f"unsupported modulation for demodulation: {params.modulation}")


def chunked_demodulate(signal: Signal, params: DemodParams, *, chunk_seconds: float = 0.05) -> DemodResult:
    """Process a long signal in bounded-memory chunks (NFR-01), concatenating
    per-chunk results. Each chunk re-runs full synchronization independently --
    simple and definitely bounded-memory, at the cost of re-acquiring sync per
    chunk rather than tracking continuously; appropriate given the priority
    here is correctness on multi-GB files, not maximum sample efficiency.
    """
    sps = max(1, round(signal.sample_rate / params.symbol_rate_hz))
    chunk_samples = max(1, int(chunk_seconds * signal.sample_rate))
    is_linear = params.modulation in LINEAR_BITS_PER_SYMBOL

    hard_bits_parts = []
    soft_bits_parts = []
    decided_parts = []
    aux_parts: list[np.ndarray] = []
    for start in range(0, signal.num_samples, chunk_samples):
        chunk = signal.samples[start : start + chunk_samples]
        if len(chunk) < 20 * sps:
            continue  # too short a tail chunk to synchronize meaningfully
        r: DemodResult
        if is_linear:
            r = demodulate_psk_qam(
                chunk, params.modulation, signal.sample_rate, params.symbol_rate_hz, use_equalizer=params.use_equalizer
            )
            aux_parts.append(r.symbols)
        else:
            r = demodulate_fsk(chunk, params.modulation, signal.sample_rate, params.symbol_rate_hz)
            aux_parts.append(r.freq_estimates)
        hard_bits_parts.append(r.hard_bits)
        soft_bits_parts.append(r.soft_bits)
        decided_parts.append(r.decided_indices)

    hard_bits = np.concatenate(hard_bits_parts) if hard_bits_parts else np.zeros(0, dtype=np.int64)
    soft_bits = np.concatenate(soft_bits_parts) if soft_bits_parts else np.zeros(0, dtype=np.float64)
    decided = np.concatenate(decided_parts) if decided_parts else np.zeros(0, dtype=np.int64)

    if is_linear:
        symbols = np.concatenate(aux_parts).astype(np.complex64) if aux_parts else np.zeros(0, dtype=np.complex64)
        return PskQamDemodResult(hard_bits=hard_bits, soft_bits=soft_bits, symbols=symbols, decided_indices=decided)
    freqs = np.concatenate(aux_parts).astype(np.float64) if aux_parts else np.zeros(0, dtype=np.float64)
    return FskDemodResult(hard_bits=hard_bits, soft_bits=soft_bits, freq_estimates=freqs, decided_indices=decided)
