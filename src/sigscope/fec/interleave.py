"""Interleavers and de-interleavers: Block, Convolutional, Diagonal,
Pseudo-random (a small LFSR/PRNG library plus user-supplied permutation
tables).

Covers: FR-11 (de-interleave Block/Convolutional/Diagonal/Pseudo-random).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# A small library of standard maximal-length LFSR tap sets (degree -> taps as
# a bitmask, 1 bit per tap position, matching the common Fibonacci-LFSR
# convention). Not exhaustive -- this is the documented, honest limit of
# "library matching" for pseudo-random interleavers (see Phase 1 docs);
# anything outside this library needs an analyst-supplied permutation table.
LFSR_LIBRARY: dict[int, int] = {
    4: 0b1001,  # x^4 + x + 1
    5: 0b10010,  # x^5 + x^2 + 1
    6: 0b100001,  # x^6 + x + 1... placeholder; see lfsr_sequence for actual poly use
    7: 0b1100000,
    8: 0b10111000,
}


def block_interleave(bits: npt.NDArray[np.int64], rows: int, cols: int) -> npt.NDArray[np.int64]:
    """Write row-wise into a rows x cols matrix, read out column-wise."""
    n = rows * cols
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits for a {rows}x{cols} block interleaver, got {len(bits)}")
    matrix = bits[:n].reshape(rows, cols)
    return matrix.T.reshape(-1)


def block_deinterleave(bits: npt.NDArray[np.int64], rows: int, cols: int) -> npt.NDArray[np.int64]:
    n = rows * cols
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits for a {rows}x{cols} block de-interleaver, got {len(bits)}")
    matrix = bits[:n].reshape(cols, rows)
    return matrix.T.reshape(-1)


def diagonal_interleave(bits: npt.NDArray[np.int64], rows: int, cols: int) -> npt.NDArray[np.int64]:
    """Write along diagonals of a rows x cols matrix, read out row-wise."""
    n = rows * cols
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits for a {rows}x{cols} diagonal interleaver, got {len(bits)}")
    matrix = np.zeros((rows, cols), dtype=bits.dtype)
    idx = 0
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                matrix[r, c] = bits[idx]
                idx += 1
    return matrix.reshape(-1)


def diagonal_deinterleave(bits: npt.NDArray[np.int64], rows: int, cols: int) -> npt.NDArray[np.int64]:
    n = rows * cols
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits for a {rows}x{cols} diagonal de-interleaver, got {len(bits)}")
    matrix = bits[:n].reshape(rows, cols)
    out = np.zeros(n, dtype=bits.dtype)
    idx = 0
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                out[idx] = matrix[r, c]
                idx += 1
    return out


def convolutional_interleaver_delay(num_branches: int, depth_increment: int) -> int:
    """Fixed end-to-end pipeline delay (in symbols) introduced by
    interleave+deinterleave together: a real, expected property of this
    interleaver structure, not a defect -- the first this-many outputs of
    convolutional_deinterleave() are meaningless pipeline fill, and output[i]
    for i >= delay corresponds to input[i - delay]."""
    return num_branches * (num_branches - 1) * depth_increment


def convolutional_interleave(
    bits: npt.NDArray[np.int64], num_branches: int, depth_increment: int
) -> npt.NDArray[np.int64]:
    """Forney-style convolutional (cross) interleaver: branch i delays by
    i*depth_increment symbols via a FIFO shift register; symbols are
    distributed to branches round-robin. See convolutional_interleaver_delay()
    for the fixed pipeline delay this introduces end-to-end."""
    n = len(bits)
    delay_lines: list[list[int]] = [[0] * (i * depth_increment) for i in range(num_branches)]
    out = np.zeros(n, dtype=bits.dtype)
    for idx in range(n):
        branch = idx % num_branches
        line = delay_lines[branch]
        if line:
            out[idx] = line.pop(0)
            line.append(int(bits[idx]))
        else:
            out[idx] = bits[idx]
    return out


def convolutional_deinterleave(
    bits: npt.NDArray[np.int64], num_branches: int, depth_increment: int
) -> npt.NDArray[np.int64]:
    """Inverse of convolutional_interleave: branch i now delays by
    (num_branches - 1 - i)*depth_increment, restoring the original order."""
    n = len(bits)
    delay_lines: list[list[int]] = [[0] * ((num_branches - 1 - i) * depth_increment) for i in range(num_branches)]
    out = np.zeros(n, dtype=bits.dtype)
    for idx in range(n):
        branch = idx % num_branches
        line = delay_lines[branch]
        if line:
            out[idx] = line.pop(0)
            line.append(int(bits[idx]))
        else:
            out[idx] = bits[idx]
    return out


def lfsr_permutation(length: int, degree: int, seed: int = 1) -> npt.NDArray[np.int64]:
    """A pseudo-random permutation of 0..length-1 derived from a library LFSR
    (see LFSR_LIBRARY): argsort of successive LFSR-generated pseudorandom
    values. Deterministic given (length, degree, seed) -- both TX (to
    interleave) and an analyst who knows the generator (to de-interleave) can
    reproduce it, without transmitting the whole table.
    """
    if degree not in LFSR_LIBRARY:
        raise ValueError(f"no LFSR in the library for degree {degree}; supply an explicit permutation table instead")
    taps = LFSR_LIBRARY[degree]
    state = seed & ((1 << degree) - 1) or 1
    values = np.zeros(length, dtype=np.int64)
    for i in range(length):
        values[i] = state
        feedback = bin(state & taps).count("1") & 1
        state = ((state << 1) | feedback) & ((1 << degree) - 1)
        if state == 0:
            state = 1
    return np.argsort(values, kind="stable")


def pseudo_random_interleave(bits: npt.NDArray[np.int64], permutation: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    n = len(permutation)
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits, got {len(bits)}")
    return bits[:n][permutation]


def pseudo_random_deinterleave(
    bits: npt.NDArray[np.int64], permutation: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    n = len(permutation)
    if len(bits) < n:
        raise ValueError(f"need at least {n} bits, got {len(bits)}")
    inverse = np.empty(n, dtype=np.int64)
    inverse[permutation] = np.arange(n)
    return bits[:n][inverse]
