"""Covers: FR-07 (interleaver identification), FR-08 (FEC identification)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sigscope.fec.identify import (
    BitStatsCentroidClassifier,
    bit_statistics_features,
    identify_block_period,
    identify_ldpc,
    identify_pseudo_random,
    identify_rs,
)
from sigscope.fec.interleave import block_interleave, lfsr_permutation, pseudo_random_interleave
from sigscope.fec.reed_solomon import RSCode


@pytest.fixture
def rs_bits() -> tuple[np.ndarray, RSCode]:
    rng = np.random.default_rng(0)
    rs = RSCode(m=4, n=15, k=9)
    symbols = np.concatenate([rs.encode(rng.integers(0, 16, rs.k)) for _ in range(40)])
    bits = ((symbols[:, None] >> np.arange(3, -1, -1)) & 1).reshape(-1).astype(np.int64)
    return bits, rs


def test_identify_rs_finds_correct_n_k(rs_bits: tuple[np.ndarray, RSCode]) -> None:
    bits, rs = rs_bits
    guess = identify_rs(bits, m=4, candidate_ns=[7, 15, 31], candidate_ks=[3, 5, 7, 9, 11, 13])
    assert guess.params == {"n": 15, "k": 9, "m": 4}
    assert guess.confidence > 0.9


def test_identify_block_period_on_interleaved_data(rs_bits: tuple[np.ndarray, RSCode]) -> None:
    bits, _rs = rs_bits
    cols = 60
    rows = len(bits) // cols
    interleaved = block_interleave(bits[: rows * cols], rows, cols)
    guess = identify_block_period(interleaved, candidate_periods=[30, 45, 60, 75, 90, 120])
    assert guess.params["period"] == 60


def test_identify_pseudo_random_with_rs_checker(rs_bits: tuple[np.ndarray, RSCode]) -> None:
    bits, rs = rs_bits
    block_len = 60
    rows = len(bits) // block_len
    perm = lfsr_permutation(block_len, degree=7, seed=3)
    interleaved = np.concatenate(
        [pseudo_random_interleave(bits[i * block_len : (i + 1) * block_len], perm) for i in range(rows)]
    )

    def checker(block_bits: np.ndarray) -> bool:
        weights = 1 << np.arange(3, -1, -1)
        symbols = (block_bits.reshape(-1, 4) @ weights).astype(np.int64)
        return all(s == 0 for s in rs.syndromes(symbols))

    guess = identify_pseudo_random(
        interleaved, block_length=block_len, checker=checker, degrees=[6, 7, 8], seeds=[1, 2, 3, 5, 7]
    )
    assert guess.params == {"degree": 7, "seed": 3}
    assert guess.confidence > 0.5


def test_identify_pseudo_random_reports_low_confidence_when_generator_unknown(
    rs_bits: tuple[np.ndarray, RSCode],
) -> None:
    """Honesty check: a generator OUTSIDE the tried library must not produce a false positive."""
    bits, rs = rs_bits
    block_len = 60
    rows = len(bits) // block_len
    unknown_perm = np.random.default_rng(99).permutation(block_len)  # not from any library LFSR
    interleaved = np.concatenate(
        [pseudo_random_interleave(bits[i * block_len : (i + 1) * block_len], unknown_perm) for i in range(rows)]
    )

    def checker(block_bits: np.ndarray) -> bool:
        weights = 1 << np.arange(3, -1, -1)
        symbols = (block_bits.reshape(-1, 4) @ weights).astype(np.int64)
        return all(s == 0 for s in rs.syndromes(symbols))

    guess = identify_pseudo_random(
        interleaved, block_length=block_len, checker=checker, degrees=[4, 5, 6, 7, 8], seeds=[1, 2, 3, 5, 7]
    )
    assert guess.confidence < 0.5


def test_identify_ldpc_library_match() -> None:
    rng = np.random.default_rng(2)
    n_bits = 12
    h = np.zeros((6, n_bits), dtype=int)
    for c in range(6):
        for col in rng.choice(n_bits, size=4, replace=False):
            h[c, col] = 1
    codeword = next(
        np.array(bits_tuple) for bits_tuple in itertools.product([0, 1], repeat=n_bits)
        if np.all((h @ np.array(bits_tuple)) % 2 == 0) and sum(bits_tuple) > 2
    )
    h_rows, h_cols = [], []
    for r in range(6):
        for c in range(n_bits):
            if h[r, c]:
                h_rows.append(r)
                h_cols.append(c)
    library = {"toy_ldpc": (h_rows, h_cols, 6, n_bits)}
    guess = identify_ldpc(np.tile(codeword, 5), library)
    assert guess.params == {"name": "toy_ldpc"}
    assert guess.confidence == 1.0


def test_bit_stats_centroid_classifier_separates_coded_from_random() -> None:
    rng = np.random.default_rng(5)
    random_examples = [rng.integers(0, 2, 200) for _ in range(10)]
    # a crude "coded-looking" example: strong runs (low transition rate)
    run_examples = [np.repeat(rng.integers(0, 2, 20), 10) for _ in range(10)]
    clf = BitStatsCentroidClassifier()
    clf.fit({"random": random_examples, "runny": run_examples})
    label, confidence = clf.predict(np.repeat(rng.integers(0, 2, 20), 10))
    assert label == "runny"
    assert confidence > 0.0


def test_bit_statistics_features_shape() -> None:
    feats = bit_statistics_features(np.array([0, 1, 0, 1, 1, 0], dtype=np.int64))
    assert feats.shape == (3,)


def test_identify_rs_prefers_true_k_over_weaker_nested_k(rs_bits: tuple[np.ndarray, RSCode]) -> None:
    """Real bug caught by P3 testing: RS(n,k') for k'>k checks a SUBSET of the
    same syndrome roots RS(n,k) does, so genuine RS(15,9) data also passes
    the weaker RS(15,10)..RS(15,14) tests as a logical consequence. Naive
    argmax always picked the weakest (largest, wrong) k. Verify the fix
    picks the true, most specific k even when many larger k's also "pass".
    """
    bits, _rs = rs_bits
    guess = identify_rs(bits, m=4, candidate_ns=[15], candidate_ks=list(range(1, 15)))
    assert guess.params["k"] == 9
