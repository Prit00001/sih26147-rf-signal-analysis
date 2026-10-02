"""Covers: FR-13 (sync words, frame length, header vs payload boundaries)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.correlate.bitstream import find_frame_length, find_sync_word, segment_header_payload

pytest.importorskip("sigscope._native", reason="native extension not built")

SYNC_WORD = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int64)
PAYLOAD_LEN = 40
FRAME_LEN = len(SYNC_WORD) + PAYLOAD_LEN


def _framed_bits(n_frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = [np.concatenate([SYNC_WORD, rng.integers(0, 2, PAYLOAD_LEN)]) for _ in range(n_frames)]
    return np.concatenate(frames)


def test_find_frame_length_recovers_true_period() -> None:
    bits = _framed_bits(60, seed=0)
    result = find_frame_length(bits, max_lag=FRAME_LEN * 2)
    assert result.period == FRAME_LEN
    assert result.confidence > 0.0


def test_find_sync_word_locates_true_positions() -> None:
    bits = _framed_bits(60, seed=1)
    result = find_sync_word(bits, SYNC_WORD)
    expected_positions = [i * FRAME_LEN for i in range(60)]
    assert result.positions == expected_positions
    assert result.confidence > 0.8


def test_find_sync_word_low_confidence_for_absent_pattern() -> None:
    bits = _framed_bits(60, seed=2)
    fake_pattern = np.random.default_rng(3).integers(0, 2, 12)
    result = find_sync_word(bits, fake_pattern)
    assert result.confidence < 0.5


def test_segment_header_payload_finds_boundary() -> None:
    bits = _framed_bits(80, seed=4)
    result = segment_header_payload(bits, FRAME_LEN)
    assert result.header_length == len(SYNC_WORD)
    assert result.confidence > 0.5


def test_segment_header_payload_needs_multiple_frames() -> None:
    bits = _framed_bits(1, seed=5)
    result = segment_header_payload(bits, FRAME_LEN)
    assert result.header_length == 0
    assert result.confidence == 0.0
