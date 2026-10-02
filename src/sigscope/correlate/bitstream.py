"""Bit-stream correlation: frame-length discovery (autocorrelation), sync-word
search (cross-correlation), and header/payload segmentation via per-bit-
position entropy. Wraps the C++ kernels from Phase 4 (native._native).

Covers: FR-13 (sync words, frame length, header vs payload boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope import _native  # type: ignore[attr-defined]  # compiled extension, no stub


@dataclass
class FrameLengthResult:
    period: int
    confidence: float
    autocorrelation: list[float]


def find_frame_length(bits: npt.NDArray[np.int64], max_lag: int) -> FrameLengthResult:
    """Autocorrelation-based frame-period search: the lag with the strongest
    (positive) autocorrelation peak, relative to the typical background
    level, is the most likely repeating frame length.

    A signal periodic with period P is trivially ALSO periodic with period
    2P, 3P, ... -- a REAL, measured failure mode this fixes: on a genuinely
    framed signal (period 232), the raw argmax picked lag 464 (exactly 2x)
    because its peak (0.1253) was marginally higher than the true
    fundamental's (0.1207 at lag 232) -- an essentially-tied harmonic
    ambiguity, not a clearly-wrong guess. Standard fix (the same principle
    pitch-detection algorithms use to prefer the fundamental over an
    overtone): if a smaller lag that evenly divides the argmax lag has a
    peak within 10% of it, prefer that smaller lag.
    """
    bits_list = [int(b) for b in bits]
    autoc = _native.bitstream_autocorrelation(bits_list, max_lag)
    if not autoc:
        return FrameLengthResult(0, 0.0, [])
    autoc_arr = np.array(autoc)
    best_lag = int(np.argmax(autoc_arr)) + 1
    peak = float(autoc_arr[best_lag - 1])
    for divisor in range(2, best_lag + 1):
        if best_lag % divisor != 0:
            continue
        candidate_lag = best_lag // divisor
        candidate_peak = float(autoc_arr[candidate_lag - 1])
        if candidate_peak >= peak * 0.9:
            best_lag, peak = candidate_lag, candidate_peak
    median_level = float(np.median(np.abs(autoc_arr)))
    confidence = float(np.clip((peak - median_level) / (1.0 - median_level + 1e-9), 0.0, 1.0)) if peak > 0 else 0.0
    return FrameLengthResult(best_lag, confidence, autoc)


@dataclass
class SyncWordResult:
    positions: list[int]
    scores: list[int]
    confidence: float


def find_sync_word(
    bits: npt.NDArray[np.int64], pattern: npt.NDArray[np.int64], *, min_score_fraction: float = 1.0
) -> SyncWordResult:
    """Cross-correlate ``bits`` against a known ``pattern`` (e.g. an assumed
    or analyst-supplied sync word) and report every offset scoring at least
    ``min_score_fraction`` of a perfect match (default: exact match only --
    short patterns have a real, nonzero false-alarm rate under partial-match
    thresholds, which a lower fraction would trade for higher false alarms).

    Confidence is NOT margin-to-runner-up: a periodic sync word legitimately
    repeats at many equally-good positions, so "how much better is #1 than
    #2" is the wrong question (it would misread an obviously-real, repeated
    match as zero confidence). Instead this measures how PERIODIC the found
    positions are: real sync-word hits recur at the frame length; isolated
    chance false alarms (a real, nonzero risk for a short pattern) do not
    share that common spacing.
    """
    bits_list = [int(b) for b in bits]
    pattern_list = [int(b) for b in pattern]
    scores = _native.bitstream_correlate(bits_list, pattern_list)
    if not scores:
        return SyncWordResult([], [], 0.0)
    scores_arr = np.array(scores)
    threshold = min_score_fraction * len(pattern)
    positions = [int(i) for i in np.where(scores_arr >= threshold)[0]]
    if len(positions) < 2:
        confidence = 0.0
    else:
        gaps = np.diff(positions)
        values, counts = np.unique(gaps, return_counts=True)
        confidence = float(np.max(counts) / len(gaps))
    return SyncWordResult(positions, [int(scores_arr[p]) for p in positions], confidence)


@dataclass
class SegmentationResult:
    header_length: int
    confidence: float
    entropy_per_position: list[float]


def segment_header_payload(framed_bits: npt.NDArray[np.int64], frame_length: int) -> SegmentationResult:
    """Given many frames of known ``frame_length``, compute the Shannon
    entropy at each bit position across frames: header/sync-word positions
    are typically constant (near-zero entropy) while payload positions carry
    data (near-maximum entropy, ~1 bit). The header/payload boundary is taken
    as the last position before entropy stays consistently high.
    """
    n_frames = len(framed_bits) // frame_length
    if n_frames < 2:
        return SegmentationResult(0, 0.0, [])
    matrix = framed_bits[: n_frames * frame_length].reshape(n_frames, frame_length)
    p1 = matrix.mean(axis=0)
    p0 = 1.0 - p1
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(p1 * np.log2(np.where(p1 > 0, p1, 1)) + p0 * np.log2(np.where(p0 > 0, p0, 1)))
    entropy = np.nan_to_num(entropy, nan=0.0)

    threshold = 0.5
    low_entropy_mask = entropy < threshold
    header_length = 0
    for i, is_low in enumerate(low_entropy_mask):
        if is_low:
            header_length = i + 1
        else:
            break
    mean_low = float(np.mean(entropy[:header_length])) if header_length > 0 else 1.0
    mean_high = float(np.mean(entropy[header_length:])) if header_length < frame_length else 0.0
    confidence = float(np.clip(mean_high - mean_low, 0.0, 1.0))
    return SegmentationResult(header_length, confidence, entropy.tolist())
