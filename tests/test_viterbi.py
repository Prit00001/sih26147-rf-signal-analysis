"""Covers: FR-12 (soft-decision Viterbi decode, C++ kernel)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope import _native

pytestmark = pytest.mark.skipif(
    not hasattr(_native, "viterbi_decode"), reason="native extension not built with viterbi_decode"
)

K = 7
GENERATORS = [0o171, 0o133]  # rate-1/2, K=7 "NASA standard" code


def _encode(bits: list[int]) -> list[int]:
    mem = 0
    out = []
    for b in bits:
        window = (b << (K - 1)) | mem
        for g in GENERATORS:
            out.append(bin(window & g).count("1") & 1)
        mem = (b << (K - 2)) | (mem >> 1)
    return out


def test_viterbi_noiseless_perfect_recovery() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 500).tolist()
    coded = _encode(bits)
    llrs = [5.0 if b == 0 else -5.0 for b in coded]
    decoded = _native.viterbi_decode(llrs, K, GENERATORS)
    assert list(decoded) == bits


def test_viterbi_ber_improves_with_snr() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 2000).tolist()
    coded = np.array(_encode(bits))

    def ber_at(snr_db: float) -> float:
        bpsk = 1 - 2 * coded.astype(float)
        noise = rng.standard_normal(len(bpsk)) / (10 ** (snr_db / 20))
        llrs = (2 * (bpsk + noise)).tolist()
        decoded = _native.viterbi_decode(llrs, K, GENERATORS)
        return float(np.mean(np.array(decoded) != np.array(bits)))

    assert ber_at(10.0) <= ber_at(0.0)


def test_viterbi_rejects_bad_llr_length() -> None:
    with pytest.raises(ValueError):
        _native.viterbi_decode([1.0, 2.0, 3.0], K, GENERATORS)  # not a multiple of len(GENERATORS)=2
