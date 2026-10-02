"""Covers: FR-11 (interleave/de-interleave round trips for all 4 types)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.fec.interleave import (
    block_deinterleave,
    block_interleave,
    convolutional_deinterleave,
    convolutional_interleave,
    convolutional_interleaver_delay,
    diagonal_deinterleave,
    diagonal_interleave,
    lfsr_permutation,
    pseudo_random_deinterleave,
    pseudo_random_interleave,
)


@pytest.fixture
def bits() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, 240).astype(np.int64)


def test_block_round_trip(bits: np.ndarray) -> None:
    inter = block_interleave(bits, 12, 20)
    back = block_deinterleave(inter, 12, 20)
    assert np.array_equal(back, bits)


def test_diagonal_round_trip(bits: np.ndarray) -> None:
    inter = diagonal_interleave(bits, 12, 20)
    back = diagonal_deinterleave(inter, 12, 20)
    assert np.array_equal(back, bits)


def test_convolutional_round_trip_with_pipeline_delay(bits: np.ndarray) -> None:
    inter = convolutional_interleave(bits, num_branches=4, depth_increment=3)
    back = convolutional_deinterleave(inter, num_branches=4, depth_increment=3)
    delay = convolutional_interleaver_delay(4, 3)
    assert delay < len(bits)
    assert np.array_equal(back[delay:], bits[: len(bits) - delay])


def test_pseudo_random_round_trip(bits: np.ndarray) -> None:
    perm = lfsr_permutation(len(bits), degree=7, seed=3)
    assert sorted(perm.tolist()) == list(range(len(bits)))  # a valid permutation
    inter = pseudo_random_interleave(bits, perm)
    back = pseudo_random_deinterleave(inter, perm)
    assert np.array_equal(back, bits)


def test_block_interleave_rejects_too_few_bits() -> None:
    with pytest.raises(ValueError):
        block_interleave(np.zeros(5, dtype=np.int64), 3, 3)
