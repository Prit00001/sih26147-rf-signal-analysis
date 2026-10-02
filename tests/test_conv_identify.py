"""Covers: FR-08 (blind convolutional code identification).

Real bug caught and fixed while building this: Viterbi decoding finds the
best-fitting trellis path for WHATEVER it is given, so re-encode agreement is
substantial even on pure noise (measured ~0.81-0.95 depending on code rate --
worse for punctured/higher-rate codes, which have more decoded-bit degrees of
freedom per transmitted bit). A single fixed agreement threshold produced
false positives on random data and on a code not in the library. Fixed by
reporting confidence as excess agreement over each candidate's own
calibrated (measured, not hand-picked) noise floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.fec.conv_identify import CONV_CODE_LIBRARY, identify_convolutional_code
from sigscope.fec.convolutional import conv_encode, puncture

pytest.importorskip("sigscope._native", reason="native extension not built")


def _encode_with_errors(spec, num_bits: int, ber: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, num_bits)
    coded = conv_encode(bits, spec.constraint_length, list(spec.generators))
    if spec.puncture_pattern is not None:
        coded = puncture(coded, list(spec.puncture_pattern))
    flip_mask = rng.random(len(coded)) < ber
    noisy = coded.copy()
    noisy[flip_mask] = 1 - noisy[flip_mask]
    return noisy


@pytest.mark.parametrize("spec", CONV_CODE_LIBRARY, ids=lambda s: s.name)
def test_identifies_every_library_code_at_2pct_ber(spec) -> None:  # type: ignore[no-untyped-def]
    noisy = _encode_with_errors(spec, 3000, ber=0.02, seed=hash(spec.name) % (2**31))
    result = identify_convolutional_code(noisy)
    assert result.name == spec.name, f"expected {spec.name}, got {result.name} (confidence {result.confidence:.3f})"
    assert result.confidence > 0.3


def test_reports_unidentified_for_pure_random_data() -> None:
    """Negative control: no convolutional structure at all -- must not false-positive."""
    rng = np.random.default_rng(7)
    random_bits = rng.integers(0, 2, 3000)
    result = identify_convolutional_code(random_bits)
    assert result.name is None
    assert result.confidence < 0.2


def test_reports_unidentified_for_code_outside_library() -> None:
    """Negative control: a real convolutional code, but not one we have in the library."""
    rng = np.random.default_rng(8)
    bits = rng.integers(0, 2, 3000)
    coded = conv_encode(bits, 9, [0o561, 0o753])  # K=9, not in CONV_CODE_LIBRARY
    result = identify_convolutional_code(coded)
    assert result.name is None
    assert result.confidence < 0.2


def test_confidence_degrades_with_worse_ber() -> None:
    spec = CONV_CODE_LIBRARY[2]  # rate1/2_K7
    low_ber = identify_convolutional_code(_encode_with_errors(spec, 3000, ber=0.01, seed=1))
    high_ber = identify_convolutional_code(_encode_with_errors(spec, 3000, ber=0.08, seed=1))
    assert low_ber.confidence >= high_ber.confidence
