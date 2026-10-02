"""Reed-Solomon encode/decode over GF(2^m).

Covers: FR-08 (RS n,k parameters), FR-12 (RS decode).

Decoding strategy: rather than the classical Berlekamp-Massey/Chien-search
pipeline, this searches candidate error-location subsets directly and solves
a small Vandermonde linear system per candidate (guaranteed non-singular for
any set of distinct locations, since a Vandermonde matrix over a field is
singular only when its evaluation points coincide). This is O(C(n, t)) in the
worst case -- fine at this project's test/demo code lengths (n <= ~63) but a
known ceiling: it would not scale to the large block lengths a production RS
codec (e.g. n=255) needs. Berlekamp-Massey (O(n*t)) is the documented upgrade
path if that scale is ever required. Chosen deliberately for lower bug risk
in a from-scratch implementation over hand-deriving BM's index bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

from sigscope.core.exceptions import SigscopeError
from sigscope.fec.gf import GF2m


class UncorrectableError(SigscopeError):
    """Raised when RS decoding cannot find a consistent error pattern within
    the code's guaranteed correction radius t = (n-k)//2."""


@dataclass
class RSCode:
    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        self.gf = GF2m(self.m)
        if not (0 < self.n <= self.gf.size - 1):
            raise ValueError(f"n={self.n} must be in (0, {self.gf.size - 1}] for GF(2^{self.m})")
        if not (0 < self.k < self.n):
            raise ValueError(f"need 0 < k < n, got k={self.k}, n={self.n}")
        self.num_parity = self.n - self.k
        self.t = self.num_parity // 2
        self.generator = self._build_generator()

    def _build_generator(self) -> list[int]:
        """Builds g(x) = product_{i=1}^{n-k} (x + alpha^i) in increasing-degree
        order (g[0] = constant term), then reverses to the decreasing-degree
        (g[0] = leading/monic term) convention _poly_mod/_codeword_eval use."""
        gf = self.gf
        g = [1]
        for i in range(1, self.num_parity + 1):
            root = gf.pow(2, i)
            new_g = [0] * (len(g) + 1)
            for j, coeff in enumerate(g):
                new_g[j] ^= gf.mul(coeff, root)
                new_g[j + 1] ^= coeff
            g = new_g
        return list(reversed(g))

    def _poly_mod(self, dividend: list[int], divisor: list[int]) -> list[int]:
        gf = self.gf
        remainder = list(dividend)
        for i in range(len(dividend) - len(divisor) + 1):
            coeff = remainder[i]
            if coeff != 0:
                for j, d in enumerate(divisor):
                    remainder[i + j] ^= gf.mul(coeff, d)
        return remainder[-(len(divisor) - 1) :] if len(divisor) > 1 else []

    def encode(self, message: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """Systematic encode: k message symbols -> n codeword symbols
        (message symbols first, followed by parity)."""
        if len(message) != self.k:
            raise ValueError(f"message must have exactly k={self.k} symbols, got {len(message)}")
        shifted = list(int(v) for v in message) + [0] * self.num_parity
        parity = self._poly_mod(shifted, self.generator)
        return np.array(list(message) + parity, dtype=np.int64)

    def _codeword_eval(self, codeword: npt.NDArray[np.int64], x: int) -> int:
        gf = self.gf
        result = 0
        for c in codeword:
            result = gf.mul(result, x) ^ int(c)
        return result

    def syndromes(self, received: npt.NDArray[np.int64]) -> list[int]:
        """S_i = received(alpha^i) for i = 1..n-k; all zero iff no detected errors."""
        gf = self.gf
        return [self._codeword_eval(received, gf.pow(2, i)) for i in range(1, self.num_parity + 1)]

    def decode(self, received: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], int]:
        """Returns (corrected_codeword, num_errors_corrected). Raises
        UncorrectableError if no error pattern of weight <= t is consistent
        with the observed syndromes."""
        gf = self.gf
        if len(received) != self.n:
            raise ValueError(f"received must have exactly n={self.n} symbols, got {len(received)}")
        synd = self.syndromes(received)
        if all(s == 0 for s in synd):
            return np.array(received, dtype=np.int64), 0

        # Location value for codeword index i (0-indexed from the front):
        # _codeword_eval's Horner evaluation processes codeword[0] first, so
        # codeword[0] ends up multiplied by x^(n-1) and codeword[n-1] by x^0
        # -- i.e. index i corresponds to exponent (n-1-i), not i itself.
        for num_errors in range(1, self.t + 1):
            for positions in combinations(range(self.n), num_errors):
                locations = [gf.pow(2, self.n - 1 - pos) for pos in positions]
                # First `num_errors` syndrome equations: sum_j e_j * X_j^r = S_{r} for r=1..num_errors.
                matrix = [[gf.pow(locations[j], r) for j in range(num_errors)] for r in range(1, num_errors + 1)]
                rhs = synd[:num_errors]
                solution = gf.solve_linear_system(matrix, rhs)
                if solution is None or any(v == 0 for v in solution):
                    continue  # a "zero error" at a claimed position isn't a real error
                # Verify against ALL remaining syndrome equations too.
                consistent = True
                for r in range(num_errors + 1, self.num_parity + 1):
                    predicted = 0
                    for j in range(num_errors):
                        predicted ^= gf.mul(solution[j], gf.pow(locations[j], r))
                    if predicted != synd[r - 1]:
                        consistent = False
                        break
                if consistent:
                    corrected = np.array(received, dtype=np.int64)
                    for pos, err_val in zip(positions, solution, strict=True):
                        corrected[pos] ^= err_val
                    return corrected, num_errors
        raise UncorrectableError(
            f"no error pattern of weight <= t={self.t} is consistent with the observed syndromes"
        )
