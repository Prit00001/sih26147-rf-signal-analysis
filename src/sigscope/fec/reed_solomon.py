"""Reed-Solomon encode/decode over GF(2^m).

Covers: FR-08 (RS n,k parameters), FR-12 (RS decode).

Decoding: Berlekamp-Massey (finds the error-locator polynomial in O(n*t)),
Chien search (finds its roots = error locations by brute-force evaluation
at each of the n field elements, O(n*t)), and Forney's algorithm (computes
error magnitudes from the error-evaluator polynomial, O(t)) -- the standard
RS decoding pipeline, replacing an earlier from-scratch implementation that
searched candidate error-location subsets directly (O(C(n, t)), fine at
small n but did not scale to n=255; see decode()'s docstring for the
measured before/after).
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

    def _berlekamp_massey(self, synd: list[int]) -> tuple[list[int], int]:
        """Standard Berlekamp-Massey iteration: finds the minimal-degree
        error-locator polynomial Lambda(x) (ascending-power coefficients,
        Lambda[0]=1) consistent with every syndrome, in O(n*t) instead of
        the O(C(n,t)) the previous combinatorial search needed."""
        gf = self.gf
        c = [1]  # Lambda(x), current candidate
        b = [1]  # previous candidate used on the last length-change step
        deg = 0  # current degree
        shift = 1  # symbols since b was last updated
        last_discrepancy = 1
        for i in range(len(synd)):
            delta = synd[i]
            for j in range(1, deg + 1):
                delta ^= gf.mul(c[j], synd[i - j])
            if delta == 0:
                shift += 1
            elif 2 * deg <= i:
                t_poly = c[:]
                coef = gf.div(delta, last_discrepancy)
                while len(c) < len(b) + shift:
                    c.append(0)
                for j, bj in enumerate(b):
                    c[j + shift] ^= gf.mul(coef, bj)
                deg = i + 1 - deg
                b = t_poly
                last_discrepancy = delta
                shift = 1
            else:
                coef = gf.div(delta, last_discrepancy)
                while len(c) < len(b) + shift:
                    c.append(0)
                for j, bj in enumerate(b):
                    c[j + shift] ^= gf.mul(coef, bj)
                shift += 1
        return c[: deg + 1], deg

    def _eval_poly(self, poly: list[int], x: int) -> int:
        gf = self.gf
        result = 0
        power = 1
        for coeff in poly:
            result ^= gf.mul(coeff, power)
            power = gf.mul(power, x)
        return result

    def decode(self, received: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], int]:
        """Returns (corrected_codeword, num_errors_corrected). Raises
        UncorrectableError if no error pattern of weight <= t is consistent
        with the observed syndromes.

        MEASURED (scripts/benchmark_rs_decode.py, full correction radius t
        errors per trial, reports/rs_decode_time_before_after.csv):
        replacing the previous O(C(n,t)) combinatorial search with
        Berlekamp-Massey + Chien search + Forney's algorithm (all O(n*t) or
        better):
          RS(15,9)   t=3:  old 6.24 ms  -> new 0.11 ms  (~57x)
          RS(31,21)  t=5:  old 5881 ms  -> new 0.34 ms  (~17,000x)
          RS(255,223) t=16: old not feasible to run (C(255,16) is
            astronomically large) -> new 6.42 ms mean (6.71 ms worst case).
        """
        gf = self.gf
        if len(received) != self.n:
            raise ValueError(f"received must have exactly n={self.n} symbols, got {len(received)}")
        synd = self.syndromes(received)
        if all(s == 0 for s in synd):
            return np.array(received, dtype=np.int64), 0

        lam, deg = self._berlekamp_massey(synd)
        if deg == 0 or deg > self.t:
            raise UncorrectableError(f"error-locator degree {deg} exceeds correction radius t={self.t}")

        # Chien search: Lambda's roots are the INVERSES of the error
        # locations. Location value for codeword index i (0-indexed from
        # the front): _codeword_eval's Horner evaluation processes
        # codeword[0] first, so codeword[0] ends up multiplied by x^(n-1)
        # and codeword[n-1] by x^0 -- i.e. index i corresponds to exponent
        # (n-1-i), not i itself.
        error_positions: list[int] = []
        error_locations: list[int] = []
        for pos in range(self.n):
            location = gf.pow(2, self.n - 1 - pos)
            if self._eval_poly(lam, gf.inv(location)) == 0:
                error_positions.append(pos)
                error_locations.append(location)
        if len(error_positions) != deg:
            # Lambda's degree says deg roots must exist; Chien search found a
            # different count -- a detected-but-uncorrectable error pattern
            # (more errors than t actually occurred), not a real deg-error
            # solution. Same honest-fallback discipline as everywhere else
            # in this project: report it, don't guess.
            raise UncorrectableError(
                f"error-locator degree {deg} does not match {len(error_positions)} Chien-search roots "
                "found -- likely more than t errors occurred"
            )

        # Forney's algorithm: Omega(x) = (S(x) * Lambda(x)) mod x^deg is the
        # error-evaluator polynomial; error magnitude at location X is
        # Omega(X^-1) / Lambda'(X^-1), where Lambda' is the formal
        # derivative (odd-power terms only survive in char-2 fields).
        omega = [0] * deg
        for i in range(deg):
            value = 0
            for j in range(i + 1):
                if j < len(lam) and (i - j) < len(synd):
                    value ^= gf.mul(lam[j], synd[i - j])
            omega[i] = value
        # Lambda'(x) in a characteristic-2 field: d/dx(c_i x^i) = i*c_i*x^{i-1},
        # and i*c_i (repeated addition in GF(2^m)) is 0 for even i, c_i for
        # odd i -- so only ODD-power terms of Lambda survive, EACH LANDING
        # AT AN EVEN EXPONENT (lam[1]*x^1 -> lam[1]*x^0, lam[3]*x^3 ->
        # lam[3]*x^2, ...), not consecutive ones. MEASURED bug this fixes:
        # evaluating lam_prime's coefficients as if they were consecutive
        # powers of x_inv (instead of consecutive powers of x_inv^2)
        # produced a confidently wrong error magnitude at every trial with
        # 3 simultaneous errors on RS(15,9) -- caught by cross-validating
        # every call against the original combinatorial decoder.
        lam_prime = [lam[i] for i in range(1, len(lam), 2)]

        corrected = np.array(received, dtype=np.int64)
        for pos, location in zip(error_positions, error_locations, strict=True):
            x_inv = gf.inv(location)
            denom = self._eval_poly(lam_prime, gf.mul(x_inv, x_inv))
            if denom == 0:
                raise UncorrectableError("degenerate Forney step (zero derivative) -- likely more than t errors")
            err_val = gf.div(self._eval_poly(omega, x_inv), denom)
            corrected[pos] ^= err_val
        return corrected, deg

    def _decode_combinatorial(self, received: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], int]:
        """The ORIGINAL decoder (O(C(n,t)) candidate error-location search +
        small Vandermonde solve per candidate), kept only for cross-
        validation of the Berlekamp-Massey/Chien/Forney replacement above
        (tests/test_reed_solomon.py) and for the before/after timing
        comparison (scripts/benchmark_rs_decode.py) -- not used by the live
        pipeline. Infeasible at RS(255,223) scale by design; do not call it
        there."""
        gf = self.gf
        if len(received) != self.n:
            raise ValueError(f"received must have exactly n={self.n} symbols, got {len(received)}")
        synd = self.syndromes(received)
        if all(s == 0 for s in synd):
            return np.array(received, dtype=np.int64), 0

        for num_errors in range(1, self.t + 1):
            for positions in combinations(range(self.n), num_errors):
                locations = [gf.pow(2, self.n - 1 - pos) for pos in positions]
                matrix = [[gf.pow(locations[j], r) for j in range(num_errors)] for r in range(1, num_errors + 1)]
                rhs = synd[:num_errors]
                solution = gf.solve_linear_system(matrix, rhs)
                if solution is None or any(v == 0 for v in solution):
                    continue
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
