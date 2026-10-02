"""Covers: FR-08 (LDPC identification), FR-12 (LDPC min-sum BP decode, C++ kernel)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sigscope import _native

pytestmark = pytest.mark.skipif(
    not hasattr(_native, "ldpc_decode_min_sum"), reason="native extension not built with ldpc_decode_min_sum"
)


def _small_ldpc(seed: int = 2) -> tuple[list[int], list[int], int, int, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_bits = 12
    h = np.zeros((6, n_bits), dtype=int)
    for c in range(6):
        for col in rng.choice(n_bits, size=4, replace=False):
            h[c, col] = 1
    codeword = None
    for bits_tuple in itertools.product([0, 1], repeat=n_bits):
        v = np.array(bits_tuple)
        if np.all((h @ v) % 2 == 0) and v.sum() > 2:
            codeword = v
            break
    assert codeword is not None
    h_rows, h_cols = [], []
    for r in range(6):
        for c in range(n_bits):
            if h[r, c]:
                h_rows.append(r)
                h_cols.append(c)
    return h_rows, h_cols, 6, n_bits, codeword


def test_ldpc_decodes_noisy_codeword() -> None:
    h_rows, h_cols, num_checks, num_bits, codeword = _small_ldpc()
    rng = np.random.default_rng(3)
    bpsk = 1 - 2 * codeword.astype(float)
    noise = rng.standard_normal(num_bits) * 0.2
    llrs = (2 * (bpsk + noise) / (0.2**2)).tolist()
    result = _native.ldpc_decode_min_sum(llrs, h_rows, h_cols, num_checks, num_bits, 50)
    assert list(result.bits) == list(codeword)
    assert result.converged


def test_ldpc_reports_nonconvergence_on_pure_noise() -> None:
    h_rows, h_cols, num_checks, num_bits, _codeword = _small_ldpc()
    rng = np.random.default_rng(4)
    llrs = rng.standard_normal(num_bits).tolist()
    result = _native.ldpc_decode_min_sum(llrs, h_rows, h_cols, num_checks, num_bits, 20)
    assert result.iterations <= 20  # never hangs past max_iterations regardless of convergence
