"""Blind identification: interleaver type/period, and FEC type/parameters.

Covers: FR-07 (interleaver ID), FR-08 (FEC ID). Consistent with this
project's Phase 1 documented limits: identification here is confidence-scored
and falls back to "unidentified" rather than a false positive; the analyst
override path (apply the interleaver/FEC directly with known parameters) is
always available regardless of what this module reports.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope.fec.interleave import LFSR_LIBRARY, block_deinterleave, lfsr_permutation, pseudo_random_deinterleave
from sigscope.fec.reed_solomon import RSCode


def gf2_rank(matrix: npt.NDArray[np.int64]) -> int:
    """Rank of a 0/1 matrix over GF(2) via Gaussian elimination (XOR pivoting)."""
    m = matrix.copy().astype(np.int64) & 1
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if m[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank


@dataclass
class InterleaverGuess:
    kind: str
    params: dict[str, int]
    confidence: float


def identify_block_period(bits: npt.NDArray[np.int64], candidate_periods: list[int]) -> InterleaverGuess:
    """Score each candidate column count by GF(2) rank deficiency of the bit
    matrix reshaped with that many columns: if the underlying (pre-
    interleave) data carries linear redundancy (e.g. from an FEC code), the
    correctly-aligned reshape shows more rank deficiency than a misaligned one.
    """
    scores: dict[int, float] = {}
    for period in candidate_periods:
        rows = len(bits) // period
        if rows < 2:
            continue
        # Actually undo a candidate block interleaver of this (rows, period)
        # shape, then check whether the RESULT shows GF(2) rank deficiency
        # (redundant linear structure, e.g. from an FEC code) -- reshaping
        # without properly deinterleaving first tests the wrong hypothesis.
        deinterleaved = block_deinterleave(bits[: rows * period], rows, period)
        matrix = deinterleaved.reshape(rows, period)
        rank = gf2_rank(matrix)
        deficiency = 1.0 - (rank / min(rows, period))
        scores[period] = deficiency
    if not scores:
        return InterleaverGuess("block", {"period": 0}, 0.0)
    best_period = max(scores, key=lambda p: scores[p])
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    confidence = float(np.clip(margin * 2, 0.0, 1.0))
    return InterleaverGuess("block", {"period": best_period}, confidence)


def identify_pseudo_random(
    bits: npt.NDArray[np.int64],
    block_length: int,
    checker: Callable[[npt.NDArray[np.int64]], bool],
    *,
    degrees: list[int] | None = None,
    seeds: list[int] | None = None,
) -> InterleaverGuess:
    """Match against the LFSR_LIBRARY (see interleave.py): try de-interleaving
    each block with each library generator's permutation and score by how
    often ``checker`` (e.g. "is this a valid RS/LDPC codeword?", a
    position-sensitive test) accepts the result.

    NOTE (a real mathematical fact, not a tuning issue): GF(2) rank of a
    per-block row-permutation is IDENTICAL to the un-permuted rank --
    reordering a vector's coordinates never changes the rank of the row
    space it spans. So generic rank-deficiency scoring (as used for block
    interleavers, which regroup bits across many blocks rather than merely
    reordering one) CANNOT detect an intra-block pseudo-random permutation at
    all; this function instead requires a caller-supplied, code-specific
    ``checker`` (e.g. sigscope.fec.reed_solomon.RSCode.syndromes) that is
    sensitive to which bit ended up in which position. Confidence is 0.0
    ("unidentified") if nothing in the library stands out -- consistent with
    Phase 1's documented position that blind PRNG identification outside a
    known library, or without a known code to check against, is not claimed;
    the analyst-supplied-permutation-table override always remains available.
    """
    degrees = degrees or sorted(LFSR_LIBRARY)
    seeds = seeds or [1, 2, 3, 5, 7, 11, 13]
    n_blocks = len(bits) // block_length
    if n_blocks < 2:
        return InterleaverGuess("pseudo_random", {}, 0.0)
    matrix_raw = bits[: n_blocks * block_length].reshape(n_blocks, block_length)

    best_score = -1.0
    best_params: dict[str, int] = {}
    scores: list[float] = []
    for degree in degrees:
        for seed in seeds:
            perm = lfsr_permutation(block_length, degree, seed)
            passed = sum(1 for row in matrix_raw if checker(pseudo_random_deinterleave(row, perm)))
            frac = passed / n_blocks
            scores.append(frac)
            if frac > best_score:
                best_score = frac
                best_params = {"degree": degree, "seed": seed}
    scores.sort(reverse=True)
    margin = scores[0] - (scores[1] if len(scores) > 1 else 0.0)
    confidence = float(np.clip(scores[0] * margin * 4, 0.0, 1.0))
    return InterleaverGuess("pseudo_random", best_params, confidence)


@dataclass
class FecGuess:
    kind: str
    params: dict[str, int]
    confidence: float


def identify_rs(bits: npt.NDArray[np.int64], m: int, candidate_ns: list[int], candidate_ks: list[int]) -> FecGuess:
    """RS structure test: group bits into m-bit symbols, and for each
    candidate (n,k), check whether many consecutive n-symbol blocks have an
    all-zero syndrome -- a real RS codeword.

    Two things had to be fixed here, both real bugs caught by testing against
    a signal with a KNOWN RS(15,9) code, not assumed correct:

    1. Confidence must be calibrated against the ANALYTIC chance level for
       that specific (n,k), not the raw pass fraction: a genuine RS(n,k)
       codeword satisfies n-k independent GF(2^m) syndrome equations, so by
       chance alone a random block passes with probability (2^m)^-(n-k) --
       negligible for a strongly-redundant code (e.g. 2^-24 for our test
       code) but NOT negligible for a weakly-redundant one (2^-4 = 6.25% for
       n-k=1), so a single fixed pass-rate threshold is not comparable across
       different k.

    2. Even after that fix, RS(n,k') for k'>k checks a SUBSET of the same
       syndrome roots RS(n,k) does -- so genuine RS(15,9) data will, as a
       pure logical consequence (not independent evidence), also "pass" the
       weaker RS(15,10)...RS(15,14) tests. Measured directly: the true k=9
       and every k=10..14 all scored within a few points of each other, and
       naive argmax picked k=14 (the LEAST redundant, i.e. wrong) every time.
       Fixed by preferring the SMALLEST k among those near the best
       confidence (Occam's razor: the most specific code consistent with the
       data, not the weakest one a stronger code's data trivially also fits).
    """
    n_full_symbols = len(bits) // m
    weights = 1 << np.arange(m - 1, -1, -1)
    symbols = (bits[: n_full_symbols * m].reshape(-1, m) @ weights).astype(np.int64)

    candidates: list[tuple[int, int, float]] = []
    for n in candidate_ns:
        if n > (1 << m) - 1:
            continue
        num_blocks = len(symbols) // n
        if num_blocks < 3:
            continue
        for k in sorted(set(candidate_ks)):
            if not (0 < k < n):
                continue
            try:
                rs = RSCode(m=m, n=n, k=k)
            except ValueError:
                continue
            passed = 0
            for b in range(num_blocks):
                block = symbols[b * n : (b + 1) * n]
                synd = rs.syndromes(block)
                if all(s == 0 for s in synd):
                    passed += 1
            frac = passed / num_blocks
            chance = (1 << m) ** (-(n - k))
            confidence = (frac - chance) / (1.0 - chance) if chance < 1.0 else 0.0
            candidates.append((n, k, max(0.0, min(1.0, confidence))))

    if not candidates:
        return FecGuess("rs", {}, 0.0)
    best_confidence = max(c for _, _, c in candidates)
    if best_confidence <= 0.0:
        return FecGuess("rs", {}, 0.0)
    near_best = [(n, k, c) for n, k, c in candidates if c >= best_confidence - 0.05]
    n, k, confidence = min(near_best, key=lambda t: t[1])
    return FecGuess("rs", {"n": n, "k": k, "m": m}, confidence)


def identify_ldpc(bits: npt.NDArray[np.int64], library: dict[str, tuple[list[int], list[int], int, int]]) -> FecGuess:
    """Match against a library of standard (or user-supplied) H matrices,
    given as {name: (h_rows, h_cols, num_checks, num_bits)}. Scores each by
    the fraction of num_bits-sized blocks whose syndrome H@bits is all-zero.
    """
    best_name = None
    best_frac = 0.0
    for name, (h_rows, h_cols, num_checks, num_bits) in library.items():
        num_blocks = len(bits) // num_bits
        if num_blocks < 3:
            continue
        passed = 0
        for b in range(num_blocks):
            block = bits[b * num_bits : (b + 1) * num_bits]
            syndrome = np.zeros(num_checks, dtype=np.int64)
            for r, c in zip(h_rows, h_cols, strict=True):
                syndrome[r] ^= block[c]
            if not syndrome.any():
                passed += 1
        frac = passed / num_blocks
        if frac > best_frac:
            best_frac = frac
            best_name = name
    if best_name is None:
        return FecGuess("ldpc", {}, 0.0)
    return FecGuess("ldpc", {"name": best_name}, float(best_frac))  # type: ignore[dict-item]


def bit_statistics_features(bits: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Lightweight hand-crafted feature vector for the secondary bit-
    statistics classifier: bit balance, mean run length, and lag-1
    autocorrelation. Deliberately simple (not a trained deep model) -- a
    fast, transparent secondary signal alongside the rigorous rank/syndrome
    tests above, per FR-08's "ML classifier on bit statistics as a secondary
    signal."
    """
    b = bits.astype(np.float64)
    balance = float(np.mean(b))
    changes = np.sum(np.abs(np.diff(b)))
    mean_run_length = len(b) / (changes + 1.0)
    bipolar = 1 - 2 * b
    autocorr_lag1 = float(np.mean(bipolar[:-1] * bipolar[1:])) if len(b) > 1 else 0.0
    return np.array([balance, mean_run_length, autocorr_lag1])


class BitStatsCentroidClassifier:
    """Nearest-centroid classifier over bit_statistics_features(), fit from a
    handful of labelled examples (e.g. "uncoded_random" vs "conv_coded" vs
    "rs_coded"). Intentionally simple/fast rather than a trained deep model.

    MEASURED LIMITATION: tried as a way to give FEC identification a third
    "none present" outcome (distinct from "unidentified") when no library
    candidate matches. Testing against a KNOWN-uncoded demodulated signal
    showed this feature set (bit balance, mean run length, lag-1
    autocorrelation) is not discriminative enough for that: real uncoded bits
    were nearest to the "conv_coded" centroid, not "uncoded_random". Not wired
    into pipeline_core.py's decision for that reason -- kept here only as
    available, explicitly-unreliable-for-this-purpose scaffolding per FR-08.
    """

    def __init__(self) -> None:
        self._centroids: dict[str, npt.NDArray[np.float64]] = {}

    def fit(self, examples: dict[str, list[npt.NDArray[np.int64]]]) -> None:
        for label, bit_arrays in examples.items():
            feats = np.array([bit_statistics_features(b) for b in bit_arrays])
            self._centroids[label] = feats.mean(axis=0)

    def predict(self, bits: npt.NDArray[np.int64]) -> tuple[str, float]:
        if not self._centroids:
            raise RuntimeError("call fit() before predict()")
        features = bit_statistics_features(bits)
        distances = {label: float(np.linalg.norm(features - c)) for label, c in self._centroids.items()}
        best_label = min(distances, key=lambda label: distances[label])
        sorted_d = sorted(distances.values())
        margin = (sorted_d[1] - sorted_d[0]) if len(sorted_d) > 1 else 1.0
        confidence = float(np.clip(margin / (sorted_d[0] + margin + 1e-9), 0.0, 1.0))
        return best_label, confidence
