"""Concatenated FEC chain: Reed-Solomon outer code + convolutional inner code
(configurable), the classic construction for combining burst-error correction
(RS, after interleaving) with random-error correction (Viterbi).

Covers: FR-12 (concatenated codes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sigscope import _native  # type: ignore[attr-defined]  # compiled extension, no stub
from sigscope.fec.convolutional import conv_encode
from sigscope.fec.reed_solomon import RSCode


def _bits_to_symbols(bits: npt.NDArray[np.int64], m: int) -> npt.NDArray[np.int64]:
    n_full = len(bits) // m
    weights = 1 << np.arange(m - 1, -1, -1)
    return (bits[: n_full * m].reshape(-1, m) @ weights).astype(np.int64)


def _symbols_to_bits(symbols: npt.NDArray[np.int64], m: int) -> npt.NDArray[np.int64]:
    bit_positions = np.arange(m - 1, -1, -1)
    return ((symbols[:, None] >> bit_positions[None, :]) & 1).astype(np.int64).reshape(-1)


@dataclass
class ConcatenatedCode:
    rs: RSCode
    constraint_length: int
    generators: list[int]

    def encode(self, message_bits: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """message_bits -> RS-encode (outer) -> convolutional-encode (inner) -> coded bits."""
        m = self.rs.m
        if len(message_bits) != self.rs.k * m:
            raise ValueError(f"message_bits must have exactly k*m={self.rs.k * m} bits, got {len(message_bits)}")
        message_symbols = _bits_to_symbols(message_bits, m)
        rs_codeword = self.rs.encode(message_symbols)
        outer_bits = _symbols_to_bits(rs_codeword, m)
        return conv_encode(outer_bits, self.constraint_length, self.generators)

    def decode(self, llrs: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], int]:
        """Soft LLRs (coded-bit domain) -> Viterbi decode (inner) -> RS decode
        (outer) -> message bits. Returns (message_bits, rs_errors_corrected)."""
        m = self.rs.m
        raw = _native.viterbi_decode(llrs.tolist(), self.constraint_length, self.generators)
        outer_bits = np.array(raw, dtype=np.int64)
        rs_codeword = _bits_to_symbols(outer_bits, m)
        if len(rs_codeword) != self.rs.n:
            raise ValueError(f"decoded outer bitstream yields {len(rs_codeword)} RS symbols, expected n={self.rs.n}")
        corrected, num_errors = self.rs.decode(rs_codeword)
        message_bits = _symbols_to_bits(corrected[: self.rs.k], m)
        return message_bits, num_errors
