"""GF(2^m) finite-field arithmetic via log/antilog tables.

Covers: FR-08/FR-12 (Reed-Solomon over GF(2^m)).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Standard primitive polynomials, one per supported field size.
_PRIMITIVE_POLYS: dict[int, int] = {
    3: 0b1011,
    4: 0b10011,
    5: 0b100101,
    6: 0b1000011,
    7: 0b10000011,
    8: 0b100011101,
}


class GF2m:
    """GF(2^m) with addition = XOR, multiplication via log/antilog tables."""

    def __init__(self, m: int) -> None:
        if m not in _PRIMITIVE_POLYS:
            raise ValueError(
                f"no primitive polynomial in the library for GF(2^{m}); supported m: {sorted(_PRIMITIVE_POLYS)}"
            )
        self.m = m
        self.size = 1 << m
        self.poly = _PRIMITIVE_POLYS[m]
        self.exp: npt.NDArray[np.int64] = np.zeros(2 * self.size, dtype=np.int64)
        self.log: npt.NDArray[np.int64] = np.zeros(self.size, dtype=np.int64)
        x = 1
        for i in range(self.size - 1):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.size:
                x ^= self.poly
        for i in range(self.size - 1, 2 * self.size):
            self.exp[i] = self.exp[i - (self.size - 1)]

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def div(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("division by zero in GF(2^m)")
        return int(self.exp[(self.log[a] - self.log[b]) % (self.size - 1)])

    def pow(self, a: int, k: int) -> int:
        if a == 0:
            return 0
        return int(self.exp[(int(self.log[a]) * k) % (self.size - 1)])

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("no inverse of zero in GF(2^m)")
        return int(self.exp[self.size - 1 - self.log[a]])

    def solve_linear_system(self, matrix: list[list[int]], rhs: list[int]) -> list[int] | None:
        """Solve A @ x = b over this field via Gaussian elimination with
        pivoting. Returns None if the matrix is singular."""
        n = len(rhs)
        aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
        for col in range(n):
            pivot_row = next((r for r in range(col, n) if aug[r][col] != 0), None)
            if pivot_row is None:
                return None
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot_val = aug[col][col]
            inv_pivot = self.inv(pivot_val)
            aug[col] = [self.mul(v, inv_pivot) for v in aug[col]]
            for r in range(n):
                if r != col and aug[r][col] != 0:
                    factor = aug[r][col]
                    aug[r] = [aug[r][c] ^ self.mul(factor, aug[col][c]) for c in range(n + 1)]
        return [aug[i][n] for i in range(n)]
