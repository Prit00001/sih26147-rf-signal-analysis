"""Covers: FR-08 (RS n,k), FR-12 (RS decode)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.fec.reed_solomon import RSCode, UncorrectableError


@pytest.mark.parametrize("m,n,k", [(4, 15, 9), (3, 7, 3), (5, 31, 21), (8, 255, 223)])
def test_rs_corrects_up_to_t_errors(m: int, n: int, k: int) -> None:
    # Berlekamp-Massey + Chien + Forney (see reed_solomon.py) is O(n*t), not
    # O(C(n,t)) -- RS(255,223) (t=16) is only even testable here because of
    # that: the previous combinatorial decoder could not have finished this
    # in any reasonable time (see _decode_combinatorial's own docstring).
    rng = np.random.default_rng(1)
    rs = RSCode(m=m, n=n, k=k)
    trials = 20
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


@pytest.mark.parametrize("m,n,k,trials", [(4, 15, 9, 80), (3, 7, 3, 80), (3, 7, 5, 80), (5, 31, 21, 15)])
def test_rs_decode_agrees_with_combinatorial_cross_check(m: int, n: int, k: int, trials: int) -> None:
    """Regression test for a real bug caught exactly this way: the
    Berlekamp-Massey/Chien/Forney decoder's error MAGNITUDES (not
    locations -- those were already right) were silently wrong at the full
    correction radius t, because Lambda's formal derivative in a
    characteristic-2 field lands odd-power terms at EVEN exponents, not
    consecutive ones -- evaluating it as if consecutive gave a confidently
    wrong error value every time. 2000+ cross-validation trials against the
    kept-for-this-purpose _decode_combinatorial caught it; this test keeps
    that cross-check permanent. Trial count is lower for (31,21): MEASURED,
    the O(C(n,t)) comparison decoder alone takes ~6-8s per call at its own
    full t=5 there, so this is a suite-runtime bound, not a coverage cut --
    the smaller configs already cover t up to 3 at 80 trials each."""
    rng = np.random.default_rng(123)
    rs = RSCode(m=m, n=n, k=k)
    for _ in range(trials):
        message = rng.integers(0, 1 << m, rs.k)
        codeword = rs.encode(message)
        num_err = int(rng.integers(0, rs.t + 1))
        positions = rng.choice(rs.n, size=num_err, replace=False)
        received = codeword.copy()
        for p in positions:
            received[p] = int(received[p]) ^ int(rng.integers(1, 1 << m))

        try:
            new_corrected, new_n = rs.decode(received)
            new_result: tuple[bool, int] | None = (bool(np.array_equal(new_corrected, codeword)), new_n)
        except UncorrectableError:
            new_result = None
        try:
            old_corrected, old_n = rs._decode_combinatorial(received)
            old_result: tuple[bool, int] | None = (bool(np.array_equal(old_corrected, codeword)), old_n)
        except UncorrectableError:
            old_result = None
        assert new_result == old_result, f"m={m} n={n} k={k} num_err={num_err}: new={new_result} old={old_result}"


def test_rs_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        RSCode(m=4, n=15, k=15)  # k must be < n
    with pytest.raises(ValueError):
        RSCode(m=4, n=20, k=5)  # n exceeds GF(16) max length
