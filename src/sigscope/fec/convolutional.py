"""Shared convolutional-code encode primitive (rate 1/n, constraint length K),
used by concatenated.py, blind identification, and tests -- pulled out to one
place instead of being reimplemented ad hoc in several.

Covers: FR-08/FR-12 (convolutional codes).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def conv_encode(bits: npt.NDArray[np.int64], constraint_length: int, generators: list[int]) -> npt.NDArray[np.int64]:
    """Encode ``bits`` with a rate-1/n convolutional code. Matches the exact
    trellis convention used by native._native.viterbi_decode (see
    native/src/viterbi.hpp): encoder starts in the all-zero memory state.
    """
    mem = 0
    out = np.zeros(len(bits) * len(generators), dtype=np.int64)
    idx = 0
    for b in bits:
        window = (int(b) << (constraint_length - 1)) | mem
        for g in generators:
            out[idx] = bin(window & g).count("1") & 1
            idx += 1
        mem = (int(b) << (constraint_length - 2)) | (mem >> 1)
    return out


def puncture(bits: npt.NDArray[np.int64], pattern: list[int]) -> npt.NDArray[np.int64]:
    """Keep only positions where pattern[i % len(pattern)] == 1."""
    period = len(pattern)
    mask = np.array([pattern[i % period] for i in range(len(bits))], dtype=bool)
    return cast(npt.NDArray[np.int64], bits[mask])
