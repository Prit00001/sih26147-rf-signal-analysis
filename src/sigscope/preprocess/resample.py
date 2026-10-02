"""Polyphase resampling and bounded-memory chunked iteration.

Covers: FR-03 (varying sample rates across sensors), NFR-01 (chunked processing
with bounded memory for multi-GB files).
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly

from sigscope.core.signal import Signal


def resample_signal(signal: Signal, target_rate: float, *, max_denominator: int = 10_000) -> Signal:
    """Resample to ``target_rate`` using a rational polyphase filter.

    The resampling ratio is approximated as a fraction (bounded denominator) so
    the polyphase filter design stays a manageable size; this trades a small
    amount of rate accuracy for bounded compute/memory.
    """
    if target_rate <= 0:
        raise ValueError(f"target_rate must be positive, got {target_rate}")
    ratio = Fraction(target_rate / signal.sample_rate).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(signal.samples, up, down).astype(np.complex64)
    return Signal(
        samples=resampled,
        sample_rate=signal.sample_rate * up / down,
        center_freq=signal.center_freq,
        source_format=signal.source_format,
        source_path=signal.source_path,
        provenance=dict(signal.provenance),
        confidence=dict(signal.confidence),
        extra=dict(signal.extra),
    )


def chunk_iter(
    samples: npt.NDArray[np.complex64], chunk_size: int, *, overlap: int = 0
) -> Iterator[npt.NDArray[np.complex64]]:
    """Yield successive (optionally overlapping) chunks without copying the source.

    Bounds peak memory for streaming-capable stages regardless of total file
    length (NFR-01). ``overlap`` lets filters/estimators that need context across
    a chunk boundary (e.g. timing recovery) see a few samples of the previous chunk.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
    n = samples.shape[0]
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield samples[max(0, start - overlap) : end]
        start = end
