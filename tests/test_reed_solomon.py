"""Covers: FR-08 (RS n,k), FR-12 (RS decode)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.fec.reed_solomon import RSCode, UncorrectableError


@pytest.mark.parametrize("m,n,k", [(4, 15, 9), (3, 7, 3), (5, 31, 21)])
def test_rs_corrects_up_to_t_errors(m: int, n: int, k: int) -> None:
    rng = np.random.default_rng(1)
    rs = RSCode(m=m, n=n, k=k)
    trials = 6 if n > 20 else 20  # the brute-force decoder's combinatorial search cost grows fast with n
    for _ in range(trials):
        message = rng.integers(0, 1 << m, rs.k)
        codeword = rs.encode(message)
        received = codeword.copy()
        num_err = rng.integers(0, rs.t + 1)
        positions = rng.choice(rs.n, size=int(num_err), replace=False)
        for p in positions:
            received[p] = int(received[p]) ^ int(rng.integers(1, 1 << m))
        decoded, nerr = rs.decode(received)
        assert np.array_equal(decoded, codeword)
        assert nerr == num_err


def test_rs_raises_on_uncorrectable_error_count() -> None:
    rng = np.random.default_rng(2)
    rs = RSCode(m=4, n=15, k=9)
    codeword = rs.encode(rng.integers(0, 16, rs.k))
    received = codeword.copy()
    positions = rng.choice(rs.n, size=rs.t + 1, replace=False)
    for p in positions:
        received[p] = int(received[p]) ^ int(rng.integers(1, 16))
    with pytest.raises(UncorrectableError):
        rs.decode(received)


def test_rs_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        RSCode(m=4, n=15, k=15)  # k must be < n
    with pytest.raises(ValueError):
        RSCode(m=4, n=20, k=5)  # n exceeds GF(16) max length
